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

## Where it goes wrong

The two numbers being compared come from two different regressions.

The observed 0.826 is the **paired** contrast, `Delta_i = rho2_model(mechanism)
- rho2_model(hidden)`. This is verifiable in the file itself: across all 64
Step 1 cases, `|Delta_i - (mechanism - hidden)|` has a maximum of 1.4e-16.

The null 0.872 is the **position** definition, `Delta_i = c - rho2_naive_i`.

Under the definition the headline slope actually uses, a constant predictor
returns the same value in both conditions, so `Delta_i = c - c = 0` for every
case. The constant-output null for the reported 0.826 is **exactly zero**, not
0.872. The summary CSV contains `hidden`, `mechanism` and `delta_i` but neither
the truth nor the plug-in, so this could not have been checked from it.

## Where it is right, and matters

Freeze section (a) also defines a position axis, `rho2_model(condition) -
rho2_naive_i`, and makes it the axis of the main figure. There the artefact is
real, large, and sign-varying across arms: +1.59, +1.13, +0.21, -0.11, -0.21.
A figure on that axis without its null would be misleading. That is what
amendment (i) fixes.

## The empirical correction that changes the reading

The artefact requires a model that does not read the sample. Measured
correlation between the `hidden` prediction and the naive plug-in, across arms:
0.960–0.988 for Qwen thinking, 0.812–0.995 for Qwen non-thinking, 0.953–1.000
for Codex — which reaches exactly 1.000 on two arms with matching standard
deviations to six digits.

These models are not emitting a prior under `hidden`. They are computing the
plug-in. The plug-in reproducer's slope is 0 on both definitions, so the
degenerate reference that actually applies to this data is zero, and the
constant-output slope is a worst-case bound rather than the expected null.

## Reproducing

```bash
python src/report_null_baselines.py \
    --paired-cases results_summary/g3/step1_paired_cases.csv
```

Writes `null_variance_decomposition.csv`, `null_baseline_slopes.csv` and
`null_between_within_arm.csv` to `results_summary/g3/`.
