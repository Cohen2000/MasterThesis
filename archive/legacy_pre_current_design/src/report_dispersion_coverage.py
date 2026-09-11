"""Freeze (h): realized dispersion across generations against dyad coverage.

Multiple imputation is proper when reported uncertainty scales with missing
information. The suite operationalizes that twice: the *stated* interval
(`lo90`/`hi90`) and, here, the **realized** spread of `rho_k2` across
independent generations of the identical prompt.

Everything in (h) that guards the result is implemented, not optional:

* the regressor is `log10(coverage)`, fixed in advance because coverage is
  heavily right-skewed;
* dispersion is reported as **both** sd and IQR, because with three draws they
  behave differently under an outlier;
* the slope is computed on cases with all three generations complete **and** on
  cases with at least two, and both are reported -- the dropout is
  coverage-dependent, which is the same variable the regression is about;
* an adjusted slope controls for the case's mean predicted `rho_k2`, because
  dispersion is compressed near 0 and 1 and a mean-variance relation alone
  could manufacture a coverage slope;
* `n` and the coverage distribution of included against excluded cases are
  reported, so the size of the selection is visible rather than trusted.

Properness predicts dispersion rising as coverage falls: a **negative** slope.
The readout is the cluster-bootstrap interval over graph groups, not a p-value.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = 4000
SEED = 20260901


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _fit_adjusted(x, y, z):
    """Slope of y on x, holding z linear. Two-regressor least squares."""
    x, y, z = (np.asarray(v, float) for v in (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if len(x) < 4:
        return np.nan
    design = np.column_stack([np.ones_like(x), x, z])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(beta[1])


def dispersion(cell: pd.DataFrame, min_generations: int) -> pd.DataFrame:
    """One row per (model, case, condition) with at least `min_generations`."""
    rows = []
    keys = ["model", "case_id", "group_id", "strategy", "condition", "coverage"]
    for key, part in cell.groupby(keys, dropna=False):
        values = part.rho_k2.to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) < min_generations:
            continue
        rows.append(dict(zip(keys, key), **{
            "generations": len(values),
            "sd": float(np.std(values, ddof=1)),
            "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)),
            "mean_rho2": float(values.mean()),
        }))
    return pd.DataFrame(rows)


def _bootstrap(frame, statistic, n=BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = frame.group_id.to_numpy()
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    draws = np.empty(n)
    for d in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        draws[d] = statistic(frame.iloc[idx])
    return (float(np.nanpercentile(draws, 2.5)),
            float(np.nanpercentile(draws, 97.5)),
            float(np.mean(np.asarray(draws) < 0)))


def slopes(frame: pd.DataFrame, measure: str) -> pd.DataFrame:
    frame = frame[frame.coverage > 0].copy()
    frame["log_coverage"] = np.log10(frame.coverage)
    rows = []
    scopes = [("pooled", frame)] + sorted(frame.groupby("strategy"))
    for scope, part in scopes:
        for model, sub in sorted(part.groupby("model")):
            if len(sub) < 8:
                continue
            raw = _fit(sub.log_coverage, sub[measure])
            adj = _fit_adjusted(sub.log_coverage, sub[measure], sub.mean_rho2)
            lo, hi, share_neg = _bootstrap(
                sub, lambda f: _fit(f.log_coverage, f[measure]))
            rows.append({
                "scope": scope, "model": model, "measure": measure,
                "n_cases": len(sub), "graph_groups": sub.group_id.nunique(),
                "slope": raw, "ci_lo": lo, "ci_hi": hi,
                "share_negative_draws": share_neg,
                "slope_adjusted_for_mean": adj,
            })
    return pd.DataFrame(rows)


def dropout_report(cell: pd.DataFrame, kept: pd.DataFrame) -> pd.DataFrame:
    """Coverage of included against excluded cases. The threat (h) names."""
    keys = ["model", "case_id", "condition"]
    all_cells = cell[keys + ["coverage"]].drop_duplicates(keys)
    kept_keys = set(map(tuple, kept[keys].to_numpy()))
    all_cells = all_cells.assign(
        included=[tuple(r) in kept_keys for r in all_cells[keys].to_numpy()])
    rows = []
    for (model, included), part in all_cells.groupby(["model", "included"]):
        rows.append({
            "model": model, "included": bool(included), "n": len(part),
            "coverage_median": float(part.coverage.median()),
            "coverage_q25": float(part.coverage.quantile(0.25)),
            "coverage_q75": float(part.coverage.quantile(0.75)),
        })
    return pd.DataFrame(rows).sort_values(["model", "included"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cell", required=True,
                    help="CSV with model, case_id, group_id, strategy, "
                         "condition, coverage, generation, rho_k2")
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    cell = pd.read_csv(args.cell)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tables = []
    for label, minimum in (("all_three", 3), ("at_least_two", 2)):
        frame = dispersion(cell, minimum)
        frame.to_csv(out / f"dispersion_cases_{label}.csv", index=False)
        for measure in ("sd", "iqr"):
            table = slopes(frame, measure)
            table.insert(0, "completeness", label)
            tables.append(table)
        if label == "all_three":
            dropout_report(cell, frame).to_csv(
                out / "dispersion_dropout.csv", index=False)

    result = pd.concat(tables, ignore_index=True)
    result.to_csv(out / "dispersion_vs_coverage.csv", index=False)
    print(result.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
