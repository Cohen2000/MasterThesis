#!/usr/bin/env python3
"""Pilot step 3: estimators, error-vs-budget curves, go/no-go evaluation.

Estimators (per strategy, per budget, out-of-fold via GroupKFold by family;
no family ever appears in train and test simultaneously):
  mean_floor   predict the train-fold mean of rho (information-free floor)
  plugin       walk_rho_plugin used directly (training-free)
  plugin_cond  walk_rho_conditional (bias-corrected: only edges with >= 2
               observations had a chance to reveal a revisit), training-free;
               falls back to the fold floor where undefined
  ridge        linear probe on all summary features
  xgboost      gradient-boosted probe (falls back to sklearn HistGB if
               xgboost is not installed)

Pre-registered go/no-go at the top budget:
  G1  negative control: xgboost R^2 on time_agnostic summaries <= 0.05
  G2  signal:           xgboost R^2 on time_respecting summaries >= 0.50
  G3  headroom:         xgboost beats the best training-free direct estimator
                        by >= 20% MAE on each temporal strategy

Usage: python pilot_eval.py --summaries summaries.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
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

META = ["substrate", "family", "rho_target", "rep", "hub_bias",
        "strategy", "walk_seed", "budget"]
LABEL = "rho_headline"
GONOGO = {"control_r2_max": 0.05, "signal_r2_min": 0.50, "headroom_min": 0.20}


def feature_columns(df):
    drop = set(META + [LABEL])
    return [c for c in df.columns
            if c not in drop and pd.api.types.is_numeric_dtype(df[c])]


def evaluate_cell(d, feats, n_splits=5):
    """Out-of-fold predictions for one (strategy, budget) cell."""
    y = d[LABEL].to_numpy()
    groups = d["family"].to_numpy()
    n_splits = min(n_splits, len(np.unique(groups)))
    preds = {name: np.full(len(d), np.nan) for name in
             ["mean_floor", "plugin", "plugin_cond", "ridge", "xgboost"]}

    usable = [c for c in feats if d[c].notna().any()]
    X = d[usable].to_numpy(dtype=float)
    has_plugin = "walk_rho_plugin" in d.columns and d["walk_rho_plugin"].notna().any()

    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        floor = float(np.mean(y[tr]))
        preds["mean_floor"][te] = floor
        if has_plugin:
            p = d["walk_rho_plugin"].to_numpy(dtype=float)[te]
            preds["plugin"][te] = np.where(np.isfinite(p), np.clip(p, 0, 1), floor)
            c = d["walk_rho_conditional"].to_numpy(dtype=float)[te] \
                if "walk_rho_conditional" in d.columns else np.full(len(te), np.nan)
            preds["plugin_cond"][te] = np.where(np.isfinite(c), np.clip(c, 0, 1),
                                                preds["plugin"][te])
        ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                              RidgeCV(alphas=np.logspace(-3, 2, 20)))
        ridge.fit(X[tr], y[tr])
        preds["ridge"][te] = ridge.predict(X[te])
        xgb = make_booster()
        xgb.fit(X[tr], y[tr])
        preds["xgboost"][te] = xgb.predict(X[te])

    rows = []
    for name, p in preds.items():
        ok = np.isfinite(p)
        if not ok.any():
            continue
        rows.append({"estimator": name,
                     "mae": mean_absolute_error(y[ok], p[ok]),
                     "r2": r2_score(y[ok], p[ok]),
                     "spearman": pd.Series(p[ok]).corr(pd.Series(y[ok]),
                                                       method="spearman"),
                     "n": int(ok.sum())})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="summaries.csv")
    ap.add_argument("--out-prefix", default="pilot")
    args = ap.parse_args()

    df = pd.read_csv(args.summaries)
    feats = feature_columns(df)
    print(f"booster: {BOOSTER_NAME} | {len(df)} rows | {len(feats)} features")

    results = []
    for (strat, budget), d in df.groupby(["strategy", "budget"]):
        for row in evaluate_cell(d.reset_index(drop=True), feats):
            results.append({"strategy": strat, "budget": int(budget), **row})
    res = pd.DataFrame(results)
    res.to_csv(f"{args.out_prefix}_results.csv", index=False)
    print(f"wrote {args.out_prefix}_results.csv ({len(res)} rows)")

    top = int(res["budget"].max())
    at_top = res[res["budget"] == top]

    def get(strat, est, col):
        m = at_top[(at_top.strategy == strat) & (at_top.estimator == est)]
        return float(m[col].iloc[0]) if len(m) else float("nan")

    print(f"\n================ GO / NO-GO (top budget = {top}) ================")
    r2_ctrl = get("time_agnostic", "xgboost", "r2")
    g1 = r2_ctrl <= GONOGO["control_r2_max"]
    print(f"G1 negative control  xgboost R2[time_agnostic] = {r2_ctrl:+.3f} "
          f"(<= {GONOGO['control_r2_max']})  ->  {'PASS' if g1 else 'FAIL'}")
    r2_sig = get("time_respecting", "xgboost", "r2")
    g2 = r2_sig >= GONOGO["signal_r2_min"]
    print(f"G2 temporal signal   xgboost R2[time_respecting] = {r2_sig:+.3f} "
          f"(>= {GONOGO['signal_r2_min']})  ->  {'PASS' if g2 else 'FAIL'}")
    g3_all = True
    temporal = sorted(s for s in at_top.strategy.unique() if s != "time_agnostic")
    for strat in temporal:
        best_direct = min(get(strat, "plugin", "mae"), get(strat, "plugin_cond", "mae"))
        mae_xgb = get(strat, "xgboost", "mae")
        impr = 1.0 - mae_xgb / best_direct
        ok = impr >= GONOGO["headroom_min"]
        g3_all &= ok
        print(f"G3 headroom          {strat:16s} best-direct MAE {best_direct:.4f} "
              f"-> xgboost {mae_xgb:.4f}  ({impr:+.0%}, >= 20%)  ->  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"VERDICT: {'GO' if (g1 and g2 and g3_all) else 'REVIEW NEEDED'}")
    print("=================================================================\n")

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, est, title in [(axes[0], "plugin_cond", "training-free (plugin_cond)"),
                           (axes[1], "xgboost", "supervised probe (xgboost)")]:
        for strat, d in res[res.estimator == est].groupby("strategy"):
            ax.plot(d.budget, d.mae, marker="o", label=strat)
        floor = res[res.estimator == "mean_floor"].groupby("budget")["mae"].mean()
        ax.plot(floor.index, floor.values, "k--", label="mean floor")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("budget (observed events, log)")
        ax.set_title(title)
    axes[0].set_ylabel("MAE vs true rho")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_mae_vs_budget.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for strat, d in res[res.estimator == "xgboost"].groupby("strategy"):
        ax.plot(d.budget, d.r2, marker="o", label=strat)
    ax.axhline(GONOGO["signal_r2_min"], color="grey", ls=":", label="G2 threshold")
    ax.axhline(GONOGO["control_r2_max"], color="grey", ls="--", label="G1 threshold")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("budget (observed events, log)")
    ax.set_ylabel("out-of-fold R2 (xgboost)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_r2_vs_budget.png", dpi=150)
    print(f"wrote {args.out_prefix}_mae_vs_budget.png, {args.out_prefix}_r2_vs_budget.png")


if __name__ == "__main__":
    main()
