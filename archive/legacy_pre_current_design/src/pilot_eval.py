#!/usr/bin/env python3
"""Coverage landscape: does an oracle regressor beat the plug-in as coverage drops?

This replaces the old fixed-budget GO/NO-GO. With a size sweep, every
(substrate, n, budget) cell sits at a different walk COVERAGE (fraction of edges
seen). The question that decides whether an LLM task even exists is:

    as coverage drops, does a gap open between the deployable plug-in estimator
    (the floor) and a supervised regressor that uses ALL walk features (the
    oracle ceiling)?

  - no gap (both track each other, or both collapse together): the plug-in is
    already as good as the information allows -> nothing to reason about.
  - gap opens at low coverage (plug-in degrades, regressor still recovers the
    target from other features): there IS extractable signal a naive read-off
    misses -> that gap is the room a zero-shot LLM could occupy.

Run for TWO targets: rho_headline (threshold persistence) and mean_span_frac
(mean occupancy = Hidalgo P_ij). For each, three estimator tiers:
  mean_floor   train-fold mean (information-free floor)
  plugin       target-specific walk plug-in, training-free:
                 rho_headline  -> walk_rho_plugin
                 mean_span_frac-> mean_windows_per_observed_edge / W
  plugin_cond  walk_rho_conditional (rho only; bias-aware conditioning)
  ridge_full / xgb_full        supervised, ALL walk features (oracle ceiling)
  ridge_noplugin / xgb_noplugin supervised, features MINUS the near-direct
                 estimates (walk_rho_plugin, walk_rho_conditional,
                 mean_windows_per_observed_edge) -> tests whether RAW ingredients
                 alone recover the target (the plug-in-ablated regressor).

Estimators are out-of-fold via GroupKFold by family (no family in train+test).

Usage: python pilot_eval.py --summaries summaries.csv --out-prefix grid
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# time_agnostic has no temporal features (all-NaN columns); the imputer warns
# and skips them, which is the intended behaviour. Silence the noise.
warnings.filterwarnings("ignore", message="Skipping features without any observed values")
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    def make_booster():
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.08,
                            subsample=0.9, colsample_bytree=0.9,
                            random_state=0, n_jobs=2, verbosity=0)
    BOOSTER_NAME = "xgboost"
except ImportError:
    from sklearn.ensemble import HistGradientBoostingRegressor
    def make_booster():
        return HistGradientBoostingRegressor(max_iter=300, max_depth=4,
                                             learning_rate=0.08, random_state=0)
    BOOSTER_NAME = "histgb (xgboost not installed)"

# everything that is metadata or a label, never a feature
META = ["substrate", "n", "family", "rho_target", "rep", "hub_bias",
        "strategy", "walk_seed", "budget", "coverage", "total_edges"]
LABELS = ["rho_headline", "mean_span_frac"]
# near-direct estimates ablated to test raw-ingredient recovery
ABLATE = ["walk_rho_plugin", "walk_rho_conditional", "mean_windows_per_observed_edge"]
# precomputed analytical estimates carried as columns by run_pilot_walks; used as
# their own estimator tier, NEVER fed to the regressors as features.
PRECOMPUTED_EST = ["rho_mle", "occ_mle"]
W_DEFAULT = 5


def feature_columns(df, exclude=()):
    drop = set(META) | set(LABELS) | set(PRECOMPUTED_EST) | set(exclude)
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def plugin_prediction(d, label, W=W_DEFAULT):
    """Target-specific training-free plug-in prediction array (or None)."""
    if label == "rho_headline":
        if "walk_rho_plugin" in d.columns:
            return d["walk_rho_plugin"].to_numpy(dtype=float)
    elif label == "mean_span_frac":
        if "mean_windows_per_observed_edge" in d.columns:
            return d["mean_windows_per_observed_edge"].to_numpy(dtype=float) / W
    return None


def mle_prediction(d, label):
    """Precomputed occupancy-MLE prediction array for the target (or None).

    rho_headline  -> rho_mle column (1 - pi_1)
    mean_span_frac-> occ_mle column ((sum_k k*pi_k)/W), the SAME pi fit.
    These are training-free, label-free; the tier is the analytical baseline
    whose collapse under forward-time bias the whole study is about.
    """
    if label == "rho_headline" and "rho_mle" in d.columns:
        return d["rho_mle"].to_numpy(dtype=float)
    if label == "mean_span_frac" and "occ_mle" in d.columns:
        return d["occ_mle"].to_numpy(dtype=float)
    return None


def evaluate_cell(d, label, n_splits=5):
    """Out-of-fold predictions for one cell, for a given target label."""
    y = d[label].to_numpy(dtype=float)
    groups = d["family"].to_numpy()
    ng = len(np.unique(groups))
    if ng < 2:
        return []
    n_splits = min(n_splits, ng)

    feats_full = feature_columns(d)
    feats_noplug = feature_columns(d, exclude=ABLATE)
    if not feats_full:
        return []
    Xf = d[feats_full].to_numpy(dtype=float)
    Xn = d[feats_noplug].to_numpy(dtype=float)

    preds = {k: np.full(len(d), np.nan) for k in
             ["mean_floor", "plugin", "plugin_cond", "mle",
              "ridge_full", "ridge_noplugin", "xgb_full", "xgb_noplugin"]}

    plug = plugin_prediction(d, label)
    mle = mle_prediction(d, label)
    cond = (d["walk_rho_conditional"].to_numpy(dtype=float)
            if label == "rho_headline" and "walk_rho_conditional" in d.columns
            else None)

    def ridge():
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 2, 20)))

    for tr, te in GroupKFold(n_splits=n_splits).split(Xf, y, groups):
        floor = float(np.mean(y[tr]))
        preds["mean_floor"][te] = floor
        if plug is not None:
            p = plug[te]
            preds["plugin"][te] = np.where(np.isfinite(p), np.clip(p, 0, 1), floor)
        if mle is not None:
            mv = mle[te]
            preds["mle"][te] = np.where(np.isfinite(mv), np.clip(mv, 0, 1), floor)
        if cond is not None:
            c = cond[te]
            base = preds["plugin"][te]
            base = np.where(np.isfinite(base), base, floor)
            preds["plugin_cond"][te] = np.where(np.isfinite(c), np.clip(c, 0, 1), base)
        rf = ridge(); rf.fit(Xf[tr], y[tr]); preds["ridge_full"][te] = rf.predict(Xf[te])
        rn = ridge(); rn.fit(Xn[tr], y[tr]); preds["ridge_noplugin"][te] = rn.predict(Xn[te])
        xf = make_booster(); xf.fit(Xf[tr], y[tr]); preds["xgb_full"][te] = xf.predict(Xf[te])
        xn = make_booster(); xn.fit(Xn[tr], y[tr]); preds["xgb_noplugin"][te] = xn.predict(Xn[te])

    rows = []
    for name, p in preds.items():
        ok = np.isfinite(p)
        if not ok.any():
            continue
        rows.append({"estimator": name,
                     "mae": mean_absolute_error(y[ok], p[ok]),
                     "r2": r2_score(y[ok], p[ok]) if ok.sum() > 1 else float("nan"),
                     "spearman": pd.Series(p[ok]).corr(pd.Series(y[ok]), method="spearman"),
                     "n_rows": int(ok.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="summaries.csv")
    ap.add_argument("--out-prefix", default="grid")
    ap.add_argument("--W", type=int, default=W_DEFAULT)
    args = ap.parse_args()

    df = pd.read_csv(args.summaries)
    print(f"booster: {BOOSTER_NAME} | {len(df)} rows | "
          f"sizes {sorted(df.n.unique())} | budgets {sorted(df.budget.unique())}")

    results = []
    cells = df.groupby(["substrate", "strategy", "n", "budget"])
    for (sub, strat, n, budget), d in cells:
        d = d.reset_index(drop=True)
        cov = float(d["coverage"].mean())
        for label in LABELS:
            for row in evaluate_cell(d, label):
                results.append({"target": label, "substrate": sub,
                                "strategy": strat, "n": int(n), "budget": int(budget),
                                "coverage": cov, **row})
    res = pd.DataFrame(results)
    res.to_csv(f"{args.out_prefix}_results.csv", index=False)
    print(f"wrote {args.out_prefix}_results.csv ({len(res)} rows)")

    # ---- gap summary: low- vs high-coverage, plug-in vs oracle (xgb_full) ----
    print("\n================  COVERAGE-GAP SUMMARY  ================")
    print("(mean MAE over cells in the lowest vs highest coverage third;\n"
          " gap = plugin_MAE - xgb_full_MAE. A gap that GROWS at low coverage\n"
          " is the room a reasoner could occupy.)\n")
    for label in LABELS:
        sub = res[res.target == label]
        if sub.empty:
            continue
        cov_by_cell = sub.groupby(["substrate", "strategy", "n", "budget"])["coverage"].first()
        if len(cov_by_cell) < 3:
            continue
        lo_thr, hi_thr = cov_by_cell.quantile([1/3, 2/3])
        def band(r):
            return "low" if r <= lo_thr else ("high" if r > hi_thr else "mid")
        sub = sub.assign(cov_band=sub["coverage"].map(band))
        print(f"--- target: {label} ---")
        for est in ["plugin", "mle", "plugin_cond", "ridge_full", "ridge_noplugin",
                    "xgb_full", "xgb_noplugin", "mean_floor"]:
            e = sub[sub.estimator == est]
            if e.empty:
                continue
            lo = e[e.cov_band == "low"]["mae"].mean()
            hi = e[e.cov_band == "high"]["mae"].mean()
            print(f"  {est:15s}  MAE low-cov {lo:6.4f}   high-cov {hi:6.4f}")
        # the headline gap, plus the floor -> mle -> oracle band per coverage band
        for band_name in ["low", "high"]:
            fl = sub[(sub.estimator == "mean_floor") & (sub.cov_band == band_name)]["mae"].mean()
            ml = sub[(sub.estimator == "mle") & (sub.cov_band == band_name)]["mae"].mean()
            pl = sub[(sub.estimator == "plugin") & (sub.cov_band == band_name)]["mae"].mean()
            xg = sub[(sub.estimator == "xgb_full") & (sub.cov_band == band_name)]["mae"].mean()
            if np.isfinite(pl) and np.isfinite(xg):
                print(f"  >> {band_name:4s}-cov gap (plugin - xgb_full) = {pl - xg:+.4f}")
            if np.isfinite(fl) and np.isfinite(ml) and np.isfinite(xg):
                frac = (fl - ml) / (fl - xg) if (fl - xg) > 1e-9 else float("nan")
                print(f"     {band_name:4s}-cov band: floor {fl:.4f} -> mle {ml:.4f} "
                      f"-> oracle {xg:.4f}   (mle captures {frac:+.2f} of recoverable signal)")
        print()
    print("=======================================================\n")

    # ---- plots: MAE vs coverage, one panel per target ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ests = ["mean_floor", "plugin", "mle", "ridge_full", "ridge_noplugin", "xgb_full", "xgb_noplugin"]
    colors = {e: c for e, c in zip(ests, plt.cm.tab10(np.linspace(0, 1, len(ests))))}
    fig, axes = plt.subplots(1, len(LABELS), figsize=(7 * len(LABELS), 5), squeeze=False)
    for ax, label in zip(axes[0], LABELS):
        sub = res[res.target == label]
        for est in ests:
            e = sub[sub.estimator == est].sort_values("coverage")
            if e.empty:
                continue
            ax.scatter(e["coverage"], e["mae"], s=18, alpha=0.35, color=colors[est])
            # binned-mean trend over coverage deciles
            q = pd.qcut(e["coverage"], q=min(8, e["coverage"].nunique()), duplicates="drop")
            tr = e.groupby(q, observed=True).agg(c=("coverage", "mean"), m=("mae", "mean"))
            ax.plot(tr["c"], tr["m"], marker="o", color=colors[est], label=est, lw=2)
        ax.set_xscale("log")
        ax.set_xlabel("walk coverage (unique edges seen / total edges, log)")
        ax.set_ylabel("MAE vs true target")
        ax.set_title(f"target: {label}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_mae_vs_coverage.png", dpi=150)
    print(f"wrote {args.out_prefix}_mae_vs_coverage.png")


if __name__ == "__main__":
    main()
