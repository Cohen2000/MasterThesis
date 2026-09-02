"""The primary slope, reported the way freeze (i) and (j) require.

(j) stratifies the panel: the primary claim runs on the **clean** arms, where
the sign of the required correction follows from the mechanism description
qualitatively. The **opposed** arms are an extension, because there the sign is
a net quantity nobody could derive from the text without the magnitudes.

(i) requires every slope to be printed beside the slope a model that has stopped
reading the sample would produce -- N0 (both legs move together, slope 0) and N1
(the hidden leg pinned to the plug-in, the mechanism leg falling back to a
constant, slope not zero) -- with R^2 and residual RMSE, and a skill score that
the artefact cannot enter.

Nothing here divides by, differences against, or rescales by a null. The nulls
are lines to read the estimate against.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = 4000
SEED = 20260901

CLEAN = ("event_sample_then_full_history", "time_agnostic_t",
         "node_panel_full_history")
OPPOSED = ("recent_history_k20", "time_respecting")
# node_panel's Var(delta) is 0.0026; freeze (b) prespecifies that its within-arm
# slope is not a point estimate. It is reported with the value blanked.
NO_SLOPE = ("node_panel_full_history",)


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.allclose(x, x[0]):
        return {"slope": np.nan, "r2": np.nan, "rmse": np.nan, "n": len(x)}
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss = float(((y - y.mean()) ** 2).sum())
    return {"slope": float(b),
            "r2": float(1 - (resid ** 2).sum() / ss) if ss > 0 else np.nan,
            "rmse": float(np.sqrt((resid ** 2).mean())),
            "n": int(len(x))}


def _boot_slope(frame, n=BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = frame.group_id.to_numpy()
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    x, y = frame.delta_i.to_numpy(float), frame.Delta_i.to_numpy(float)
    draws = np.empty(n)
    for d in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        draws[d] = _fit(x[idx], y[idx])["slope"]
    return (float(np.nanpercentile(draws, 2.5)),
            float(np.nanpercentile(draws, 97.5)),
            float(np.mean(np.asarray(draws) > 0)))


def n1_slope(cases: pd.DataFrame) -> float:
    """1 - Cov(true, delta)/Var(delta), the prior-fallback reference."""
    true = cases.rho_W5_k2.to_numpy(float)
    delta = cases.delta_i.to_numpy(float)
    if len(true) < 3:
        return np.nan
    return float(1.0 - np.cov(true, delta, ddof=1)[0, 1] / np.var(delta, ddof=1))


def slope_table(paired: pd.DataFrame, seed_slot: int = 0) -> pd.DataFrame:
    if "seed_slot" in paired:
        paired = paired[paired.seed_slot == seed_slot]
    rows = []
    for model, part in sorted(paired.groupby("model")):
        scopes = [("clean_arms (primary)", part[part.strategy.isin(CLEAN)]),
                  ("opposed_arms (extension)", part[part.strategy.isin(OPPOSED)]),
                  ("all_arms", part)]
        scopes += [(arm, sub) for arm, sub in sorted(part.groupby("strategy"))]
        for scope, sub in scopes:
            if len(sub) < 4:
                continue
            fit = _fit(sub.delta_i, sub.Delta_i)
            lo, hi, pos = _boot_slope(sub)
            blanked = scope in NO_SLOPE
            rows.append({
                "model": model, "scope": scope,
                "arms": sub.strategy.nunique(),
                "graph_groups": sub.group_id.nunique(),
                "n": fit["n"],
                "slope": np.nan if blanked else fit["slope"],
                "ci_lo": np.nan if blanked else lo,
                "ci_hi": np.nan if blanked else hi,
                "share_positive_draws": np.nan if blanked else pos,
                "r2": np.nan if blanked else fit["r2"],
                "rmse": np.nan if blanked else fit["rmse"],
                "null_N0": 0.0,
                "null_N1": n1_slope(sub),
                "slope_not_reportable": blanked,
            })
    return pd.DataFrame(rows)


def skill_table(cell: pd.DataFrame) -> pd.DataFrame:
    """1 - MSE(model) / MSE(best constant), against the truth.

    The best constant is the in-sample mean of the truth over the same scope --
    the most generous denominator available, and exactly what N1 emits, so N1
    scores 0 by construction and the artefact cannot enter.
    """
    averaged = (cell.groupby(["model", "case_id", "strategy", "condition",
                              "rho_W5_k2"], as_index=False)
                .agg(rho_k2=("rho_k2", "mean")))
    rows = []
    for (model, arm, condition), part in averaged.groupby(
            ["model", "strategy", "condition"]):
        pred = part.rho_k2.to_numpy(float)
        true = part.rho_W5_k2.to_numpy(float)
        ok = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[ok], true[ok]
        if len(true) < 4:
            continue
        mse_const = float(((true.mean() - true) ** 2).mean())
        mse_model = float(((pred - true) ** 2).mean())
        rows.append({
            "model": model, "strategy": arm, "condition": condition,
            "n": len(true), "mse_model": mse_model,
            "mse_best_constant": mse_const,
            "skill_score": float(1 - mse_model / mse_const)
            if mse_const > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["model", "strategy", "condition"])


def interval_table(cell: pd.DataFrame) -> pd.DataFrame:
    """Stated interval width and its empirical coverage, per (h).

    The other operationalization of properness. Reported beside the realized
    dispersion, never merged with it: a model whose stated intervals widen with
    missing information while its answers do not is calibrated in rhetoric only.
    """
    frame = cell.dropna(subset=["lo90", "hi90"]).copy()
    if frame.empty:
        return pd.DataFrame()
    frame["width"] = frame.hi90 - frame.lo90
    frame["covers"] = ((frame.lo90 <= frame.rho_W5_k2)
                       & (frame.rho_W5_k2 <= frame.hi90))
    frame["log_coverage"] = np.log10(frame.coverage.where(frame.coverage > 0))
    rows = []
    for (model, arm), part in frame.groupby(["model", "strategy"]):
        if len(part) < 8:
            continue
        rows.append({
            "model": model, "strategy": arm, "n": len(part),
            "median_width": float(part.width.median()),
            "empirical_coverage": float(part.covers.mean()),
            "width_vs_log_coverage_slope": _fit(part.log_coverage,
                                                part.width)["slope"],
            "inverted_share": float((part.width < 0).mean()),
        })
    return pd.DataFrame(rows).sort_values(["model", "strategy"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired", default=str(REPO / "results_summary/g4/g4_paired.csv"))
    ap.add_argument("--cell", default=str(REPO / "results_summary/g4/g4_cell.csv"))
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    slopes = slope_table(pd.read_csv(args.paired))
    slopes.to_csv(out / "primary_slope.csv", index=False)
    print(slopes.round(4).to_string(index=False))

    cell = pd.read_csv(args.cell)
    skill = skill_table(cell)
    skill.to_csv(out / "skill_scores.csv", index=False)
    print()
    print(skill.round(4).to_string(index=False))

    intervals = interval_table(cell)
    if not intervals.empty:
        intervals.to_csv(out / "stated_intervals.csv", index=False)
        print()
        print(intervals.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
