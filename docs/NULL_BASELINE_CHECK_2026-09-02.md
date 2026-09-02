# Constant-output null: verification of an external review

An external review, working only from the committed summary CSVs, reported that
the primary regression `Delta_i ~ delta_i` has a degenerate reference far from
zero, and that a model ignoring the sample would produce roughly the slope that
is read as mechanism evidence.

Both of the review's number sets reproduce **exactly**. Its algebra is correct.
Its conclusion about the headline statistic does not follow, for a reason the
summary CSVs could not have shown.

## What reproduces

Recomputed from `results_summary/g3/step1_paired_cases.csv` joined to
`results/final_run_g2/final_cases.csv.gz`:

| quantity | review | recomputed |
|---|---|---|
| observed pooled slope / R^2 | 0.826 / 0.797 | **0.826 / 0.797** |
| constant-null pooled slope / R^2 | 0.872 / 0.538 | **0.872 / 0.538** |
| constant-null, `time_agnostic_t` | 0.252 | **0.252** |
| constant-null, `event_sample_then_full_history` | 1.580 | **1.580** |

The match is to three decimals on every entry, which identifies the
construction: a constant predictor, `hidden` used as the naive baseline, under
the position definition of `Delta_i`.

Replacing the proxy with the real column, `est__plugin_rho_k2`, moves the nulls
by almost nothing — 0.883 pooled, 0.210 and 1.588 within arm. The proxy was not
the problem.

## Where the first reading of this check went wrong

The check originally concluded that because the headline slope uses the paired
contrast `Delta_i = rho2_model(mechanism) - rho2_model(hidden)`, a
sample-ignoring model gives `c - c = 0` and the artefact does not reach it.

That is true of only one of the two ways to ignore the sample, and not the one
this data supports. It assumes the model is constant in **both** legs. The
measured correlation between the `hidden` prediction and the plug-in is 0.81 to
1.00 across every arm and model, so the hidden leg is pinned to the plug-in, not
free. A model that backs off only under `mechanism` therefore produces

    Delta_i = c - naive_i = (c - true_i) + delta_i

whose slope is `1 - Cov(true, delta)/Var(delta)`, not zero, because `naive_i`
sits inside both the predictor and the response. Freeze (i) now carries both
nulls under the names N0 (no movement, slope 0) and N1 (prior fallback), and
reports N1 for the paired contrast as well as the position axis.

The review's calculation was N1. It was labelled here as the position axis,
which was wrong; the two are algebraically the same expression, so the numbers
were right and the label was not.

## What separates N1 from the observed result

Not the slope: +0.826 observed against +0.883 for N1 on the Step 1 slice. The
discriminators are the fit and the skill score.

| | slope | R^2 | RMSE | skill score |
|---|---|---|---|---|
| observed | +0.826 | 0.797 | 0.117 | **+0.721** |
| N1 | +0.883 | 0.543 | 0.228 | **0.000** |

N1 scores exactly zero by construction: its prediction is the best constant, so
it is the denominator of the skill score. The observed +0.721 is out of reach
for any model that has stopped reading the sample. N1 is rejected, on the
statistic built to reject it rather than on the slope.

## Permutation reference

Reproduced under the schema now fixed in (i) -- predictor permuted within arm,
observed response, 4,000 draws, seed 20260901: mean **+0.519**, sd 0.051, 95%
band [0.420, 0.619]. Closed form `slope_between x Var_between/(Var_between +
Var_within)` = 0.836 x 0.6218 = **+0.520**.

An earlier figure of 0.559 from this module is withdrawn: it was computed on the
five-arm panel with a synthetic perfect-corrector response, which answers a
different question than a reference line for an observed slope.

## Reproducing

```bash
python src/report_null_baselines.py \
    --paired-cases results_summary/g3/step1_paired_cases.csv
```

Writes `null_variance_decomposition.csv`, `null_baseline_slopes.csv` and
`null_between_within_arm.csv` to `results_summary/g3/`.
