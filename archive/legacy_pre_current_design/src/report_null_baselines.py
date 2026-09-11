"""Degenerate-reference check for the `Delta_i ~ delta_i` regression.

A model that stops reading the sample is not neutral in this regression, and
there are two ways for it to stop, with different consequences.

  N0  no movement.      P_hidden = P_mechanism, whatever that common value is
                        -- the plug-in, a prior, anything.
                        Delta_i = 0 identically, slope = 0.

  N1  prior fallback.   P_hidden = naive_i, P_mechanism = c.
                        Delta_i = c - naive_i = (c - true_i) + delta_i
                        slope = 1 + Cov(delta, c - true) / Var(delta)
                              = 1 - Cov(true, delta) / Var(delta)

N1 is the one that matters, and it is not a straw man. It is the strategy "the
text tells me this sample is biased, so I distrust it and answer something
generic" -- exactly what a model trained to treat bias warnings as a cue to back
off would do. It is also the empirically live option: the measured correlation
between the `hidden` prediction and the plug-in is 0.81 to 1.00 across every arm
and model, so the hidden leg is pinned to the plug-in rather than free.

N1's slope is not zero **under the paired contrast either**. `naive_i` appears
inside both `delta_i` and `Delta_i`, so the two share a term and the regression
inherits it. Only N0, where both legs move together, collapses to zero.

The constant `c` cancels in N1: it is absorbed into the intercept. So the
reference does not depend on which constant a model emits, and a
pooled-mean and an arm-mean constant give the same slope by construction.

Everything here except the model-dependent tables is analytic: it needs the
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


def null_baselines(cases: pd.DataFrame) -> pd.DataFrame:
    """N0 and N1 per scope, analytically and numerically.

    Reported for the paired contrast, which is what the headline slope uses.
    N1 applies there too -- see the module docstring -- and the two columns
    exist so that no slope is ever read against an assumed zero.
    """
    rows = []
    for scope, part in [("pooled", cases)] + sorted(cases.groupby("strategy")):
        true = part.rho_W5_k2.to_numpy(float)
        naive = part.est__plugin_rho_k2.to_numpy(float)
        delta = true - naive
        # The constant cancels out of the slope; the pooled mean of the truth
        # is used so the intercept is reported at a meaningful value.
        const = float(cases.rho_W5_k2.mean())

        rows.append({"scope": scope, "null": "N0_no_movement",
                     "constant": np.nan,
                     **_fit(delta, np.zeros_like(delta)),
                     "slope_analytic": 0.0})

        analytic = 1.0 - (np.cov(true, delta, ddof=1)[0, 1]
                          / np.var(delta, ddof=1))
        rows.append({"scope": scope, "null": "N1_prior_fallback",
                     "constant": const,
                     **_fit(delta, const - naive),
                     "slope_analytic": float(analytic)})

        rows.append({"scope": scope, "null": "perfect_corrector",
                     "constant": np.nan,
                     **_fit(delta, delta),
                     "slope_analytic": 1.0})
    return pd.DataFrame(rows)


def permutation_null(frame: pd.DataFrame, x: str, y: str, stratum: str,
                     draws: int = 4000, seed: int = SEED) -> pd.DataFrame:
    """Permutation reference for a pooled slope. Schema fixed, not implied.

    Permuted:   the predictor `x` (`delta_i`), never the response.
    Strata:     `stratum` (the arm). Permuting within arm destroys the
                within-case pairing and leaves each arm's marginal
                distribution and its mean exactly where they were.
    Response:   the *observed* `y`. A synthetic response -- the perfect
                corrector, say -- answers a different question and must not be
                substituted; an earlier version of this module did exactly
                that on a different panel and reported 0.559 for what is
                0.519 here.
    Draws:      `draws`, seeded.
    Reported:   the mean, with the standard deviation and the 2.5/97.5
                percentiles of the null distribution. The mean is the
                reference line; the band is what "near it" means.

    Why it is not zero: what survives permutation is the between-stratum
    structure, and the closed form is

        slope_between x Var_between / (Var_between + Var_within)

    which is reported alongside so a reader need not rerun the resampling.
    """
    rng = np.random.default_rng(seed)
    xv = frame[x].to_numpy(float)
    yv = frame[y].to_numpy(float)
    groups = frame[stratum].to_numpy()
    idx = [np.flatnonzero(groups == g) for g in np.unique(groups)]

    slopes = np.empty(draws)
    for d in range(draws):
        shuffled = np.empty_like(xv)
        for i in idx:
            shuffled[i] = rng.permutation(xv[i])
        slopes[d] = _fit(shuffled, yv)["slope"]

    means = frame.groupby(stratum)[[x, y]].mean()
    centred = xv - frame.groupby(stratum)[x].transform("mean").to_numpy(float)
    var_between = float(np.var(xv - centred, ddof=0))
    var_within = float(np.var(centred, ddof=0))
    slope_between = _fit(means[x], means[y])["slope"]

    return pd.DataFrame([{
        "permuted": x, "response": y, "stratum": stratum,
        "draws": draws, "seed": seed,
        "mean": float(slopes.mean()),
        "sd": float(slopes.std(ddof=1)),
        "lo2.5": float(np.percentile(slopes, 2.5)),
        "hi97.5": float(np.percentile(slopes, 97.5)),
        "slope_between": slope_between,
        "var_between_share": var_between / (var_between + var_within),
        "analytic": slope_between * var_between / (var_between + var_within),
        "n": len(frame),
        "strata": int(len(idx)),
    }])


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
    ap.add_argument("--permutations", type=int, default=4000)
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

    nulls = null_baselines(cases)
    nulls.to_csv(out / "null_baseline_slopes.csv", index=False)
    print()
    print(nulls.to_string(index=False))

    if args.paired_cases:
        observed = pd.read_csv(args.paired_cases)
        perm = permutation_null(observed, "delta_i", "Delta_i", "strategy",
                                draws=args.permutations)
        perm.insert(0, "source", Path(args.paired_cases).name)
        perm.to_csv(out / "null_permutation.csv", index=False)
        print()
        print(perm.to_string(index=False))

        parts = between_within_arm(observed, "delta_i", "Delta_i")
        parts.insert(0, "source", Path(args.paired_cases).name)
        parts.to_csv(out / "null_between_within_arm.csv", index=False)
        print()
        print(parts.to_string(index=False))


if __name__ == "__main__":
    main()
