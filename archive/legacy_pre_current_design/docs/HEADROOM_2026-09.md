# G0 headroom audit: does arm identity change the correction?

Prepared: **2026-08-31** for the September 2026 freeze  
Gate status: **G0 complete; blocked pending confirmation. No G1 work has begun.**

## Scope and provenance

This audit used no LLM calls and did not modify a frozen artifact. The frozen
V2 shards have no `oracle__` columns and retain only aggregated observed
histograms, which cannot identify the true K of individual dyads. The event
streams were therefore regenerated in a new directory from the archived V2
plan and the eight raw datasets present when the 745-instance archive was
built. The exact archived walk RNG seeds were replayed at budget 800.

- frozen benchmark cases: `results/benchmark_v2/results/cases_shard_*.csv.gz`
- frozen benchmark manifest: `results/benchmark_v2/data/manifest.csv`
- regenerated event manifest: `results/g0_headroom_2026_09/regenerated_benchmark_v2/manifest.csv`
- panel probe (read-only, walk seed 0): `results/panel_seed_probe/cases.csv.gz`
- compact joint counts and numerical tables: `results/g0_headroom_2026_09`
- frozen cases selected: 8,940 (745 instances, 38 groups, four seeds, three arms)
- replayed `(n,mask)` histograms verified exactly: 8,940/8,940
- observation-model groups by arm: {"recent_history_k20": 38, "time_agnostic_t": 38, "time_respecting": 38}
- panel cases: 96 (32 graphs × 3 arms; 12 groups)
- panel groups also present in benchmark training universe and held out: 12/12

The observation model includes mask `0`, meaning that a population dyad was
not observed. Probabilities are normalized within group first and then averaged
over groups, matching the primary group-macro estimand. The reported K-weighted
TV uses the corresponding group-macro distribution of true K.

## G0.1 — Empirical observation models

Total-variation distance for the full `P(m | K, arm)`, including mask 0:

| arm_a           | arm_b              |      1 |      2 |      3 |      4 |      5 |   K-weighted |
|:----------------|:-------------------|-------:|-------:|-------:|-------:|-------:|-------------:|
| time_agnostic_t | recent_history_k20 | 0.0676 | 0.0526 | 0.0481 | 0.0788 | 0.1293 |       0.0676 |
| time_agnostic_t | time_respecting    | 0.0931 | 0.0931 | 0.0926 | 0.1110 | 0.1928 |       0.1009 |
| time_respecting | recent_history_k20 | 0.0580 | 0.0856 | 0.1205 | 0.1599 | 0.2203 |       0.0858 |

Supplementary diagnostic conditional on the dyad being observed (`m != 0`):

| arm_a           | arm_b              |      1 |      2 |      3 |      4 |      5 |   K-weighted |
|:----------------|:-------------------|-------:|-------:|-------:|-------:|-------:|-------------:|
| time_agnostic_t | recent_history_k20 | 0.2109 | 0.2329 | 0.2499 | 0.2709 | 0.3270 |       0.2301 |
| time_agnostic_t | time_respecting    | 0.4496 | 0.4931 | 0.5236 | 0.5348 | 0.6270 |       0.4820 |
| time_respecting | recent_history_k20 | 0.6008 | 0.6119 | 0.6396 | 0.6842 | 0.7456 |       0.6209 |

The first table is the requested population observation model. The conditional
table separates differences in the masks of discovered dyads from differences
in arm-specific inclusion probabilities.

## Parameterization check

`src/corrected_estimator.py:rho_mle` is parameterized only by `(n, w, W,
iters)` and uses one uniform-occupancy likelihood. `src/mask_estimator.py:mask_mle`
is parameterized only by `(n, mask, W, iters, prior, weights)` and likewise uses
one uniform-within-active-windows likelihood. Neither accepts an access arm or
an arm-specific propensity model. Evaluating those columns on a particular arm
therefore does **not** make them mechanism-parameterized.

G0.2 consequently uses the empirical label-assisted `P(m | K, arm)` above.
For each panel group and assumed arm, the model is refitted after excluding the
entire matching benchmark source/family. The MLE conditions on observing a
nonzero mask and divides out the learned `P(observed | K, arm)`.

## G0.2 — Wrong-mechanism estimator matrix

Group-macro ProfileMAE (rows: sample-producing arm; columns: assumed arm):

| sample_arm         |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:-------------------|---------------------:|------------------:|------------------:|
| recent_history_k20 |               0.0983 |            0.1059 |            0.1229 |
| time_agnostic_t    |               0.0972 |            0.0874 |            0.1159 |
| time_respecting    |               0.1318 |            0.1140 |            0.0925 |

Group-macro signed bias on `rho_2`:

| sample_arm         |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:-------------------|---------------------:|------------------:|------------------:|
| recent_history_k20 |              -0.0849 |           -0.0717 |           -0.1976 |
| time_agnostic_t    |              -0.1468 |           -0.1171 |           -0.2034 |
| time_respecting    |              -0.1986 |           -0.1497 |           -0.0833 |

Off-diagonal penalties relative to the correct diagonal assumption:

| sample_arm         | assumed_arm        |   profile_mae_penalty |   rho2_bias_shift |   abs_rho2_bias_shift |
|:-------------------|:-------------------|----------------------:|------------------:|----------------------:|
| time_agnostic_t    | time_respecting    |                0.0285 |           -0.0863 |                0.0863 |
| time_agnostic_t    | recent_history_k20 |                0.0098 |           -0.0296 |                0.0296 |
| time_respecting    | time_agnostic_t    |                0.0215 |           -0.0665 |                0.0665 |
| time_respecting    | recent_history_k20 |                0.0394 |           -0.1153 |                0.1153 |
| recent_history_k20 | time_agnostic_t    |                0.0076 |            0.0133 |                0.0133 |
| recent_history_k20 | time_respecting    |                0.0246 |           -0.1126 |                0.1126 |

**This matrix uses true dyad labels to fit each observation model and is a
ceiling, not a label-free estimator.** It measures whether arm identity offers
potentially usable headroom; it is not a proposed deployable correction.

## G0.3 — Estimator ladder on the panel

All rows use the same 96 seed-0 panel cases at budget 800. The mean floor is
leave-one-panel-group-out. ExtraTrees uses the frozen benchmark V2 cases at
budget 800 and the observable `combined = occ + pat + crawl` feature set; for
every panel prediction its complete source/family is excluded from training.
The occupancy and mask MLE rows are the existing uniform likelihoods described
above, not newly mechanism-parameterized estimators.

| arm                | estimator                                  |   profile_mae |   rho2_bias |   pooled_profile_mae |   pooled_rho2_bias |
|:-------------------|:-------------------------------------------|--------------:|------------:|---------------------:|-------------------:|
| time_agnostic_t    | mean floor (panel LOGO)                    |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| time_agnostic_t    | naive read-off                             |        0.1562 |     -0.3085 |               0.1437 |            -0.3031 |
| time_agnostic_t    | occupancy MLE (uniform)                    |        0.0597 |     -0.0657 |               0.0578 |            -0.0721 |
| time_agnostic_t    | mask MLE (uniform)                         |        0.0579 |     -0.0490 |               0.0558 |            -0.0554 |
| time_agnostic_t    | supervised ExtraTrees (benchmark transfer) |        0.0412 |     -0.0139 |               0.0431 |            -0.0165 |
| time_respecting    | mean floor (panel LOGO)                    |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| time_respecting    | naive read-off                             |        0.1573 |     -0.3098 |               0.1435 |            -0.3001 |
| time_respecting    | occupancy MLE (uniform)                    |        0.1169 |     -0.1475 |               0.1089 |            -0.1449 |
| time_respecting    | mask MLE (uniform)                         |        0.1139 |     -0.1150 |               0.1032 |            -0.1071 |
| time_respecting    | supervised ExtraTrees (benchmark transfer) |        0.0556 |      0.0141 |               0.0558 |             0.0127 |
| recent_history_k20 | mean floor (panel LOGO)                    |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| recent_history_k20 | naive read-off                             |        0.1493 |     -0.2868 |               0.1367 |            -0.2807 |
| recent_history_k20 | occupancy MLE (uniform)                    |        0.0893 |     -0.0625 |               0.0864 |            -0.0743 |
| recent_history_k20 | mask MLE (uniform)                         |        0.0851 |     -0.0424 |               0.0787 |            -0.0588 |
| recent_history_k20 | supervised ExtraTrees (benchmark transfer) |        0.0519 |      0.0019 |               0.0509 |            -0.0000 |

Naive-to-mask-MLE ProfileMAE headroom by arm:

| arm                |   naive_profile_mae |   mask_mle_profile_mae |   naive_to_mask_headroom |
|:-------------------|--------------------:|-----------------------:|-------------------------:|
| time_agnostic_t    |              0.1562 |                 0.0579 |                   0.0983 |
| time_respecting    |              0.1573 |                 0.1139 |                   0.0434 |
| recent_history_k20 |              0.1493 |                 0.0851 |                   0.0641 |

## G0.4 — Mechanical decisions

- Median off-diagonal ProfileMAE penalty: **0.0230** (survival threshold: > 0.02).
- Median absolute off-diagonal shift in signed `rho_2` bias: **0.0764** (survival threshold: > 0.03).
- `mismatched`: **SURVIVES**.
- Prespecified bidirectional mismatch pair if retained: **`time_respecting` ↔ `recent_history_k20`**.
- Arms with naive-to-mask-MLE headroom below 0.02: **none**.

Conditions surviving G0: **`hidden`, `mechanism`, and
`mechanism_direction`**; also `mismatched` for the pair `time_respecting` ↔ `recent_history_k20`.

## What this shows, and what it does not

The arm comparison is estimated from the archived benchmark population and
evaluated on a separate fixed panel with matching source/family exclusions. It
directly tests whether substituting one learned arm likelihood for another can
matter at the planned budget.

It does not prove that an LLM can recover the correction. The empirical model
uses labels, collapses the exact `(n,mask)` sample to mask frequencies, and
averages over heterogeneous coverage, topology and graph size within an arm.
Fixed-budget walk observations are dependent, whereas this MLE uses their
marginal empirical distribution. The 3×3 matrix is therefore a deliberately
optimistic diagnostic with model misspecification, not an information-theoretic
bound. G0 also uses an existing seed-0 development-panel sample; G2, if
authorized later, is still responsible for the fresh final sample.

## Decision

**STOP at G0. Await explicit confirmation before writing the G1 prompt
contract or generating any final prompt.**
