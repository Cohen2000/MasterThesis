"""Degenerate-reference check for the `Delta_i ~ delta_i` regression.

A model that ignores the sample and answers with a constant is not neutral in
this regression. Which value it produces depends entirely on how `Delta_i` is
defined, and the suite defines it two ways:

  position   Delta_i(condition) = rho2_model(condition) - rho2_naive_i
             -- freeze section (a), the axis of the main figure
  paired     Delta_i            = rho2_model(mechanism) - rho2_model(hidden)
             -- freeze section (a), and the headline slope in step1_slopes.csv

Under `position`, a constant predictor P_i = c gives Delta_i = c - naive_i, so

    slope = [Var(naive) - Cov(naive, true)] / Var(delta)
          = 1 + Cov(naive - true, true) / Var(delta)

which is near 1 whenever the naive bias is roughly uncorrelated with the truth.
Under `paired` the same constant gives Delta_i = c - c = 0 identically, and the
slope is exactly zero with nothing left to explain.

So the artefact is real, but it is a property of one of the two definitions.
This module computes the reference for both, so no reported slope has to be
read against an assumed zero.

Everything here except the two model-dependent tables is analytic: it needs the
ground truth and the plug-in estimate, no model output at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = 2000
SEED = 20260901


def _fit(x: np.ndarray, y: np.ndarray) -> dict:
    """Slope, intercept, R^2 and residual RMSE of y on x."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2 or np.allclose(x, x[0]):
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "rmse": np.nan, "n": len(x)}
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "slope": float(b),
        "intercept": float(a),
        # A constant y (the paired null) has no variance to explain; R^2 is
        # undefined there rather than zero, and saying so beats printing 0.
        "r2": float(1 - (resid ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan,
        "rmse": float(np.sqrt((resid ** 2).mean())),
        "n": int(len(x)),
    }


def variance_table(cases: pd.DataFrame) -> pd.DataFrame:
    """Var(true) against Var(delta), which is what sets the degenerate slope.

    The ratio is the whole story: the constant-predictor slope under
    `position` is 1 + Cov(naive - true, true)/Var(delta), so a Var(delta) that
    is small relative to Var(true) makes the degenerate reference both large
    and unstable.
    """
    rows = []
    for scope, part in [("pooled", cases)] + sorted(cases.groupby("strategy")):
        true = part.rho_W5_k2.to_numpy(float)
        naive = part.est__plugin_rho_k2.to_numpy(float)
        delta = true - naive
        rows.append({
            "scope": scope,
            "n": len(part),
            "var_true": float(np.var(true, ddof=1)),
            "var_naive": float(np.var(naive, ddof=1)),
            "var_delta": float(np.var(delta, ddof=1)),
            "var_ratio_true_over_delta": float(np.var(true, ddof=1)
                                               / np.var(delta, ddof=1)),
            "cov_naive_true": float(np.cov(naive, true, ddof=1)[0, 1]),
            "corr_naive_true": float(np.corrcoef(naive, true)[0, 1]),
            "mean_delta": float(delta.mean()),
        })
    return pd.DataFrame(rows)


def null_baselines(cases: pd.DataFrame, n_perm: int = 1000) -> pd.DataFrame:
    """The three falsification baselines, under both Delta definitions.

    Two properties of this panel make the constant baselines simpler than they
    look, and both are worth stating rather than discovering later.

    The slope of `c - naive_i` on `delta_i` does not depend on `c` at all: the
    constant is absorbed into the intercept. So `constant_pooled` and
    `constant_arm` cannot differ in slope by construction, and the degenerate
    reference is robust to *which* constant a model happens to emit -- which
    makes the artefact stronger, not weaker.

    They also cannot differ in intercept here, because the truth is a property
    of the instance and every arm sees the same 32 instances; the pooled mean
    of `rho_W5_k2` and its per-arm mean are the same number. Both rows are kept
    anyway, so the table shows this rather than asserting it.

    The permutation baseline needs a response to permute against. Analytically
    the strongest available response is the perfect corrector, whose `position`
    shift is exactly `delta_i`; permuting `delta_i` within arm destroys the
    within-case pairing while leaving each arm's marginal distribution intact.
    What survives is the between-arm component, and on a pooled fit that is
    not zero -- which is the real reference for the pooled slope.

    The `paired` definition has no analytic permutation counterpart: the
    mechanism-minus-hidden contrast is not determined by the ground truth, so
    that row belongs with the model-dependent tables and is not invented here.
    """
    rng = np.random.default_rng(SEED)
    rows = []
    for scope, part in [("pooled", cases)] + sorted(cases.groupby("strategy")):
        true = part.rho_W5_k2.to_numpy(float)
        naive = part.est__plugin_rho_k2.to_numpy(float)
        delta = true - naive
        arm = part.strategy.to_numpy()

        for name, const in (("constant_pooled", cases.rho_W5_k2.mean()),
                            ("constant_arm", part.rho_W5_k2.mean())):
            rows.append({"scope": scope, "baseline": name,
                         "definition": "position", "constant": float(const),
                         **_fit(delta, const - naive)})
            # Both conditions return the same constant, so the contrast is
            # identically zero whatever the case looks like.
            rows.append({"scope": scope, "baseline": name,
                         "definition": "paired", "constant": float(const),
                         **_fit(delta, np.zeros_like(delta)),
                         })

        # The degenerate model this panel actually contains. Under `hidden`
        # the measured correlation between prediction and plug-in is 0.95 to
        # 1.00 in every arm and model, so the empirically relevant "ignores
        # the mechanism" predictor is not a constant -- it is the plug-in
        # itself. Its position shift is identically zero, and so is its slope.
        rows.append({"scope": scope, "baseline": "plugin_reproducer",
                     "definition": "position", "constant": np.nan,
                     **_fit(delta, naive - naive)})

        # Reference at the other end: a model that corrects perfectly.
        rows.append({"scope": scope, "baseline": "perfect_corrector",
                     "definition": "position", "constant": np.nan,
                     **_fit(delta, delta)})

        slopes = []
        for _ in range(n_perm):
            shuffled = np.empty_like(delta)
            for a in np.unique(arm):
                idx = np.flatnonzero(arm == a)
                shuffled[idx] = rng.permutation(delta[idx])
            slopes.append(_fit(shuffled, delta)["slope"])
        slopes = np.asarray(slopes, float)
        rows.append({
            "scope": scope,
            "baseline": "permutation_within_arm_perfect_corrector",
            "definition": "position", "constant": np.nan,
            "slope": float(np.nanmean(slopes)),
            "intercept": np.nan, "r2": np.nan, "rmse": np.nan,
            "n": len(part),
            "perm_lo": float(np.nanpercentile(slopes, 2.5)),
            "perm_hi": float(np.nanpercentile(slopes, 97.5)),
        })
    return pd.DataFrame(rows)


def between_within_arm(frame: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """How much of a pooled slope is carried by the arm means alone.

    Freeze section (b) already says `delta_i` is a three-level contrast by arm.
    This puts a number on it: the slope through the arm means, against the
    slope with every arm centred on its own mean. A pooled slope close to the
    between-arm value is a statement about arms; a within-arm slope near zero
    means the pooled number is not evidence about cases.

    It needs a real response. Run against the perfect corrector it returns 1
    everywhere by construction and says nothing, which is why it takes the
    observed columns rather than deriving them.
    """
    means = frame.groupby("strategy")[[x, y]].mean()
    within = frame.copy()
    for col in (x, y):
        within[col] = within[col] - within.groupby("strategy")[col].transform("mean")
    return pd.DataFrame([
        {"component": "pooled", **_fit(frame[x], frame[y])},
        {"component": "between_arm_means", **_fit(means[x], means[y])},
        {"component": "within_arm_centred", **_fit(within[x], within[y])},
    ])


def skill_scores(cell: pd.DataFrame) -> pd.DataFrame:
    """1 - MSE(model) / MSE(best constant), against the truth.

    Artefact-free in the sense that matters here: it never differences against
    the naive baseline, so a constant predictor scores 0 by construction and a
    model has to beat the best achievable constant to score above it. The best
    constant is the in-sample mean of the truth over the same scope, which is
    the most generous constant available and therefore the honest denominator.
    """
    rows = []
    for (model, arm, condition), part in cell.groupby(
            ["model", "strategy", "condition"], dropna=False):
        pred = part.rho_k2.to_numpy(float)
        true = part.rho_W5_k2.to_numpy(float)
        ok = np.isfinite(pred) & np.isfinite(true)
        pred, true = pred[ok], true[ok]
        if len(true) < 2:
            continue
        mse_model = float(((pred - true) ** 2).mean())
        mse_const = float(((true.mean() - true) ** 2).mean())
        rows.append({
            "model": model, "strategy": arm, "condition": condition,
            "n": len(true),
            "mse_model": mse_model, "mse_best_constant": mse_const,
            "skill_score": float(1 - mse_model / mse_const)
            if mse_const > 0 else np.nan,
        })
    return pd.DataFrame(rows).sort_values(["model", "strategy", "condition"])


def hidden_prior_correlation(cell: pd.DataFrame) -> pd.DataFrame:
    """Corr(prediction under `hidden`, naive plug-in), per arm and model.

    This is how strongly the artefact can bite at all. The degenerate slope
    assumes the prediction carries no information about the case; a `hidden`
    prediction that tracks the naive plug-in closely is close to that
    degenerate case, one that does not is far from it.
    """
    rows = []
    part = cell[cell.condition == "hidden"]
    for (model, arm), grp in part.groupby(["model", "strategy"], dropna=False):
        pred = grp.rho_k2.to_numpy(float)
        naive = grp.est__plugin_rho_k2.to_numpy(float)
        true = grp.rho_W5_k2.to_numpy(float)
        ok = np.isfinite(pred) & np.isfinite(naive)
        if ok.sum() < 3:
            continue
        rows.append({
            "model": model, "strategy": arm, "n": int(ok.sum()),
            "corr_hidden_naive": float(np.corrcoef(pred[ok], naive[ok])[0, 1]),
            "corr_hidden_true": float(np.corrcoef(pred[ok], true[ok])[0, 1]),
            "sd_hidden_prediction": float(np.std(pred[ok], ddof=1)),
            "sd_naive": float(np.std(naive[ok], ddof=1)),
        })
    return pd.DataFrame(rows).sort_values(["model", "strategy"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(REPO / "results/final_run_g2/final_cases.csv.gz"))
    ap.add_argument("--seed-slot", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g3"))
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--paired-cases", default=None,
                    help="a paired-cases CSV (case_id, strategy, delta_i, "
                         "Delta_i); adds the observed decomposition, which "
                         "cannot be derived from the ground truth alone")
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    cases = cases[cases.seed_slot == args.seed_slot]
    cases = cases[["case_id", "instance_id", "group_id", "strategy",
                   "coverage", "rho_W5_k2", "est__plugin_rho_k2"]]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    var = variance_table(cases)
    var.to_csv(out / "null_variance_decomposition.csv", index=False)
    print(var.to_string(index=False))

    nulls = null_baselines(cases, args.permutations)
    nulls.to_csv(out / "null_baseline_slopes.csv", index=False)
    print()
    print(nulls.to_string(index=False))

    if args.paired_cases:
        observed = pd.read_csv(args.paired_cases)
        parts = between_within_arm(observed, "delta_i", "Delta_i")
        parts.insert(0, "source", Path(args.paired_cases).name)
        parts.to_csv(out / "null_between_within_arm.csv", index=False)
        print()
        print(parts.to_string(index=False))


if __name__ == "__main__":
    main()
