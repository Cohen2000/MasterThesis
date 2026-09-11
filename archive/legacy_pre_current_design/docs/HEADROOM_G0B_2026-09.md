# G0b headroom audit: widening the signed-bias axis

Prepared: **2026-08-31**  
Gate status: **G0 + G0b complete; blocked pending joint confirmation. No G1 work has begun.**

## Scope and implementation checks

No LLM was called and no frozen artifact was modified. G0b added new case
files only below `results/g0b_headroom_2026_09`. Candidate A samples a random priority ordering of
all event-active nodes and reveals a selected node's complete incident event
record. On this panel the event-active universe equals the manifest node
universe for all 32 graphs (32/32).
It stops before the first whole-node response that would exceed the budget;
there are no partial node responses, but the adaptive stopping time means the
fixed-p identity `1-(1-p)^2` is only a conceptual reference, not the exact
inclusion probability of this implementation. Candidate B uses an exact
exponential-race PPS ordering by full dyad event count. Whole histories that do
not fit the remaining budget are skipped, never truncated.

Candidate C was not run: it was optional, both priority candidates already qualified, and the rule permits adding at most one arm.

## G0b.1–2 — Candidate bias and mechanical selection

Signed naive read-off bias on `rho_2` (primary group-macro; candidate means
average the eight prespecified sample seeds):

| arm                                     |   group_macro_bias_mean_8seeds |   group_macro_bias_sd_8seeds |   seed0_group_macro_bias |   pooled_case_bias |   positive_case_share |   negative_case_share | qualifies_near_zero   | qualifies_opposite_to_walks   |
|:----------------------------------------|-------------------------------:|-----------------------------:|-------------------------:|-------------------:|----------------------:|----------------------:|:----------------------|:------------------------------|
| activity_proportional_dyad_full_history |                         0.1404 |                       0.0199 |                   0.1564 |             0.1383 |                0.8750 |                0.1250 | False                 | True                          |
| node_panel_full_history                 |                        -0.0107 |                       0.0149 |                   0.0123 |            -0.0114 |                0.3438 |                0.5000 | True                  | False                         |

Bias spread for each candidate set:

| arm_set                               |   minimum_bias |   maximum_bias |   signed_bias_spread |
|:--------------------------------------|---------------:|---------------:|---------------------:|
| three walks                           |        -0.3098 |        -0.2868 |               0.0231 |
| three walks + candidate A             |        -0.3098 |        -0.0107 |               0.2991 |
| three walks + candidate B             |        -0.3098 |         0.1404 |               0.4502 |
| three walks + A + B (diagnostic only) |        -0.3098 |         0.1404 |               0.4502 |

Both A (within 0.05 of zero) and B (opposite aggregate sign) qualify. The
prespecified priority `B > A > C` therefore selects **`activity_proportional_dyad_full_history`**. Its
aggregate correction on this panel is downward, whereas all three walk arms
require an upward correction.

## G0b.3 — Viability of the selected arm

Coverage and budget diagnostics:

| scope                          |   median |     mean |      p10 |      p90 |      min |      max |
|:-------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| winner coverage, seed 0        |   0.0076 |   0.0159 |   0.0007 |   0.0307 |   0.0006 |   0.0947 |
| winner coverage, all 8 seeds   |   0.0063 |   0.0157 |   0.0007 |   0.0334 |   0.0002 |   0.0947 |
| winner realized event budget   | 800.0000 | 800.0000 | 800.0000 | 800.0000 | 800.0000 | 800.0000 |
| winner event-budget slack      |   0.0000 |   0.0000 |   0.0000 |   0.0000 |   0.0000 |   0.0000 |
| candidate A coverage, seed 0   |   0.0170 |   0.0233 |   0.0000 |   0.0717 |   0.0000 |   0.1030 |
| candidate A event-budget slack | 130.5000 | 277.4180 |   6.5000 | 800.0000 |   0.0000 | 800.0000 |
| walk coverage, seed 0          |   0.0555 |   0.1047 |   0.0174 |   0.2651 |   0.0028 |   0.4978 |

At seed 0 the selected arm's coverage is substantially below the walk panel's
historical median 0.056. The following is a sufficient whole-dyad event budget
for each graph to include the first 5.6% of dyads in the same PPS order; it is
reported only as a parity diagnostic and does **not** change budget 800:

| scope                                     |    median |       mean |       p10 |         p90 |      min |         max |
|:------------------------------------------|----------:|-----------:|----------:|------------:|---------:|------------:|
| sufficient budget for 0.056 dyad coverage | 9721.0000 | 34295.0625 | 1482.0000 | 100477.2000 | 488.0000 | 141662.0000 |

Static mask-input length (`INPUT_MASK` plus exact histogram, before any G1
mechanism text):

| arm                                     | measure         |   median |     mean |      p10 |       p90 |      min |       max |
|:----------------------------------------|:----------------|---------:|---------:|---------:|----------:|---------:|----------:|
| activity_proportional_dyad_full_history | characters      | 669.0000 | 765.1562 | 608.1000 | 1011.2000 | 553.0000 | 1337.0000 |
| activity_proportional_dyad_full_history | utf8_bytes      | 669.0000 | 765.1562 | 608.1000 | 1011.2000 | 553.0000 | 1337.0000 |
| activity_proportional_dyad_full_history | portable_tokens | 162.0000 | 205.0000 | 138.0000 |  309.6000 | 114.0000 |  462.0000 |
| recent_history_k20                      | characters      | 876.0000 | 836.2812 | 630.7000 | 1047.4000 | 573.0000 | 1152.0000 |
| recent_history_k20                      | utf8_bytes      | 876.0000 | 836.2812 | 630.7000 | 1047.4000 | 573.0000 | 1152.0000 |
| recent_history_k20                      | portable_tokens | 254.0000 | 236.5000 | 146.4000 |  327.2000 | 122.0000 |  374.0000 |
| time_agnostic_t                         | characters      | 666.0000 | 710.6562 | 622.2000 |  868.7000 | 604.0000 | 1089.0000 |
| time_agnostic_t                         | utf8_bytes      | 666.0000 | 710.6562 | 622.2000 |  868.7000 | 604.0000 | 1089.0000 |
| time_agnostic_t                         | portable_tokens | 162.0000 | 181.5000 | 142.0000 |  251.6000 | 134.0000 |  350.0000 |
| time_respecting                         | characters      | 744.0000 | 729.3438 | 618.2000 |  843.7000 | 581.0000 |  850.0000 |
| time_respecting                         | utf8_bytes      | 744.0000 | 729.3438 | 618.2000 |  843.7000 | 581.0000 |  850.0000 |
| time_respecting                         | portable_tokens | 196.0000 | 189.8750 | 142.0000 |  238.0000 | 126.0000 |  242.0000 |

`portable_tokens` is an explicitly tokenizer-independent lexical count, not a
Qwen/Codex token claim. Exact Qwen3.6-27B tokenizer counts for the same static
mask inputs (cached snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
`add_special_tokens=False`) are:

| arm                                     |   median |     mean |      p10 |      p90 |      min |      max |
|:----------------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| activity_proportional_dyad_full_history | 271.0000 | 345.6562 | 222.1000 | 539.4000 | 179.0000 | 789.0000 |
| recent_history_k20                      | 432.0000 | 401.0312 | 240.5000 | 566.8000 | 195.0000 | 648.0000 |
| time_agnostic_t                         | 268.0000 | 302.9062 | 234.2000 | 425.9000 | 220.0000 | 597.0000 |
| time_respecting                         | 329.0000 | 317.4062 | 230.2000 | 407.7000 | 201.0000 | 412.0000 |

These are token counts for the fixed input block only. Exact **full-prompt**
counts cannot exist until G1 fixes the mechanism texts.

Eight-seed variance decomposition for the selected arm, using the same
implementation as `src/report_seed_variance.py`:

| method                                                       |   mean |   n |   n_group |   n_inst |   n_seed |   s2_group |   s2_inst |   s2_seed |   seed_variance_share |   se_1seed |   se_8seeds |
|:-------------------------------------------------------------|-------:|----:|----------:|---------:|---------:|-----------:|----------:|----------:|----------------------:|-----------:|------------:|
| naive read-off                                               | 0.0989 | 256 |   12.0000 |   2.6667 |   8.0000 |     0.0000 |    0.0134 |    0.0020 |                0.1304 |     0.0220 |      0.0207 |
| occupancy MLE (uniform; censoring-aware, mechanism-agnostic) | 0.2532 | 256 |   12.0000 |   2.6667 |   8.0000 |     0.0389 |    0.0220 |    0.0035 |                0.0540 |     0.0635 |      0.0628 |
| mask MLE (uniform; censoring-aware, mechanism-agnostic)      | 0.2027 | 256 |   12.0000 |   2.6667 |   8.0000 |     0.0324 |    0.0077 |    0.0026 |                0.0614 |     0.0550 |      0.0543 |

## G0b.4 — Widened observation-model audit

The new benchmark cases were replayed from regenerated event streams and
**2,980/2,980** exact `(n,mask)` histograms matched. Existing
walk counts and 3×3 cells were reused rather than regenerated.

Full `P(m | K, arm)` total-variation distances (mask 0 included):

| arm_a              | arm_b                                   |      1 |      2 |      3 |      4 |      5 |   K-weighted |
|:-------------------|:----------------------------------------|-------:|-------:|-------:|-------:|-------:|-------------:|
| recent_history_k20 | activity_proportional_dyad_full_history | 0.0559 | 0.1181 | 0.1767 | 0.2267 | 0.2926 |       0.1047 |
| time_agnostic_t    | activity_proportional_dyad_full_history | 0.1184 | 0.1375 | 0.1628 | 0.1702 | 0.1782 |       0.1331 |
| time_agnostic_t    | recent_history_k20                      | 0.0676 | 0.0526 | 0.0481 | 0.0788 | 0.1293 |       0.0676 |
| time_agnostic_t    | time_respecting                         | 0.0931 | 0.0931 | 0.0926 | 0.1110 | 0.1928 |       0.1009 |
| time_respecting    | activity_proportional_dyad_full_history | 0.0554 | 0.1070 | 0.1502 | 0.1955 | 0.2763 |       0.0970 |
| time_respecting    | recent_history_k20                      | 0.0580 | 0.0856 | 0.1205 | 0.1599 | 0.2203 |       0.0858 |

Conditional on observation (`m != 0`):

| arm_a              | arm_b                                   |      1 |      2 |      3 |      4 |      5 |   K-weighted |
|:-------------------|:----------------------------------------|-------:|-------:|-------:|-------:|-------:|-------------:|
| recent_history_k20 | activity_proportional_dyad_full_history | 0.2172 | 0.8727 | 0.9745 | 0.9927 | 0.9961 |       0.5139 |
| time_agnostic_t    | activity_proportional_dyad_full_history | 0.0184 | 0.8517 | 0.9712 | 0.9902 | 0.9937 |       0.3930 |
| time_agnostic_t    | recent_history_k20                      | 0.2109 | 0.2329 | 0.2499 | 0.2709 | 0.3270 |       0.2301 |
| time_agnostic_t    | time_respecting                         | 0.4496 | 0.4931 | 0.5236 | 0.5348 | 0.6270 |       0.4820 |
| time_respecting    | activity_proportional_dyad_full_history | 0.4590 | 0.9178 | 0.9891 | 0.9988 | 0.9999 |       0.6662 |
| time_respecting    | recent_history_k20                      | 0.6008 | 0.6119 | 0.6396 | 0.6842 | 0.7456 |       0.6209 |

Label-assisted arm-likelihood ceiling, group-macro ProfileMAE:

| sample_arm                              |   activity_proportional_dyad_full_history |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:----------------------------------------|------------------------------------------:|---------------------:|------------------:|------------------:|
| activity_proportional_dyad_full_history |                                    0.0701 |               0.4938 |            0.4761 |            0.3153 |
| recent_history_k20                      |                                    0.1529 |               0.0983 |            0.1059 |            0.1229 |
| time_agnostic_t                         |                                    0.1583 |               0.0972 |            0.0874 |            0.1159 |
| time_respecting                         |                                    0.1593 |               0.1318 |            0.1140 |            0.0925 |

The same matrix as signed `rho_2` bias:

| sample_arm                              |   activity_proportional_dyad_full_history |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:----------------------------------------|------------------------------------------:|---------------------:|------------------:|------------------:|
| activity_proportional_dyad_full_history |                                    0.1117 |               0.5329 |            0.5425 |            0.4568 |
| recent_history_k20                      |                                   -0.2976 |              -0.0849 |           -0.0717 |           -0.1976 |
| time_agnostic_t                         |                                   -0.3153 |              -0.1468 |           -0.1171 |           -0.2034 |
| time_respecting                         |                                   -0.3168 |              -0.1986 |           -0.1497 |           -0.0833 |

All off-diagonal changes relative to the matching diagonal:

| sample_arm                              | assumed_arm                             |   profile_mae_penalty |   rho2_bias_shift |   abs_rho2_bias_shift |
|:----------------------------------------|:----------------------------------------|----------------------:|------------------:|----------------------:|
| time_agnostic_t                         | time_respecting                         |                0.0285 |           -0.0863 |                0.0863 |
| time_agnostic_t                         | recent_history_k20                      |                0.0098 |           -0.0296 |                0.0296 |
| time_respecting                         | time_agnostic_t                         |                0.0215 |           -0.0665 |                0.0665 |
| time_respecting                         | recent_history_k20                      |                0.0394 |           -0.1153 |                0.1153 |
| recent_history_k20                      | time_agnostic_t                         |                0.0076 |            0.0133 |                0.0133 |
| recent_history_k20                      | time_respecting                         |                0.0246 |           -0.1126 |                0.1126 |
| time_agnostic_t                         | activity_proportional_dyad_full_history |                0.0710 |           -0.1982 |                0.1982 |
| time_respecting                         | activity_proportional_dyad_full_history |                0.0668 |           -0.2335 |                0.2335 |
| recent_history_k20                      | activity_proportional_dyad_full_history |                0.0546 |           -0.2127 |                0.2127 |
| activity_proportional_dyad_full_history | time_agnostic_t                         |                0.4060 |            0.4308 |                0.4308 |
| activity_proportional_dyad_full_history | time_respecting                         |                0.2451 |            0.3451 |                0.3451 |
| activity_proportional_dyad_full_history | recent_history_k20                      |                0.4236 |            0.4212 |                0.4212 |

This remains a **label-assisted diagnostic ceiling**, not a deployable
estimator and not an information-theoretic bound.

## Estimator ladder and naming correction

The uniform occupancy and mask MLEs are **censoring-aware,
mechanism-agnostic**. They receive no arm parameter and are not design-aware.

| arm                                     | estimator                                                    |   n |   profile_mae |   rho2_bias |   pooled_profile_mae |   pooled_rho2_bias |
|:----------------------------------------|:-------------------------------------------------------------|----:|--------------:|------------:|---------------------:|-------------------:|
| time_agnostic_t                         | mean floor (panel LOGO)                                      |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| time_agnostic_t                         | naive read-off                                               |  32 |        0.1562 |     -0.3085 |               0.1437 |            -0.3031 |
| time_agnostic_t                         | occupancy MLE (uniform; censoring-aware, mechanism-agnostic) |  32 |        0.0597 |     -0.0657 |               0.0578 |            -0.0721 |
| time_agnostic_t                         | mask MLE (uniform; censoring-aware, mechanism-agnostic)      |  32 |        0.0579 |     -0.0490 |               0.0558 |            -0.0554 |
| time_agnostic_t                         | supervised ExtraTrees (benchmark transfer)                   |  32 |        0.0412 |     -0.0139 |               0.0431 |            -0.0165 |
| time_respecting                         | mean floor (panel LOGO)                                      |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| time_respecting                         | naive read-off                                               |  32 |        0.1573 |     -0.3098 |               0.1435 |            -0.3001 |
| time_respecting                         | occupancy MLE (uniform; censoring-aware, mechanism-agnostic) |  32 |        0.1169 |     -0.1475 |               0.1089 |            -0.1449 |
| time_respecting                         | mask MLE (uniform; censoring-aware, mechanism-agnostic)      |  32 |        0.1139 |     -0.1150 |               0.1032 |            -0.1071 |
| time_respecting                         | supervised ExtraTrees (benchmark transfer)                   |  32 |        0.0556 |      0.0141 |               0.0558 |             0.0127 |
| recent_history_k20                      | mean floor (panel LOGO)                                      |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| recent_history_k20                      | naive read-off                                               |  32 |        0.1493 |     -0.2868 |               0.1367 |            -0.2807 |
| recent_history_k20                      | occupancy MLE (uniform; censoring-aware, mechanism-agnostic) |  32 |        0.0893 |     -0.0625 |               0.0864 |            -0.0743 |
| recent_history_k20                      | mask MLE (uniform; censoring-aware, mechanism-agnostic)      |  32 |        0.0851 |     -0.0424 |               0.0787 |            -0.0588 |
| recent_history_k20                      | supervised ExtraTrees (benchmark transfer)                   |  32 |        0.0519 |      0.0019 |               0.0509 |            -0.0000 |
| activity_proportional_dyad_full_history | mean floor (panel LOGO)                                      |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| activity_proportional_dyad_full_history | naive read-off                                               |  32 |        0.1011 |      0.1564 |               0.0952 |             0.1548 |
| activity_proportional_dyad_full_history | occupancy MLE (uniform; censoring-aware, mechanism-agnostic) |  32 |        0.2785 |      0.3361 |               0.2499 |             0.3161 |
| activity_proportional_dyad_full_history | mask MLE (uniform; censoring-aware, mechanism-agnostic)      |  32 |        0.2261 |      0.3294 |               0.1999 |             0.3091 |
| activity_proportional_dyad_full_history | supervised ExtraTrees (benchmark transfer)                   |  32 |        0.0271 |      0.0120 |               0.0295 |             0.0120 |

Signed-bias denominator of the renamed `CensoringRecovery` scale:

| arm                                     |   naive_rho2_bias |   mask_mle_rho2_bias |   bias_denominator | normalization_status                            |
|:----------------------------------------|------------------:|---------------------:|-------------------:|:------------------------------------------------|
| activity_proportional_dyad_full_history |            0.1564 |               0.3294 |            -0.1730 | raw bias only: anchor does not move toward zero |
| recent_history_k20                      |           -0.2868 |              -0.0424 |            -0.2444 | defined                                         |
| time_agnostic_t                         |           -0.3085 |              -0.0490 |            -0.2594 | defined                                         |
| time_respecting                         |           -0.3098 |              -0.1150 |            -0.1949 | defined                                         |

`CensoringRecovery = 1` anchors to the existing uniform mask MLE: it means
matching that estimator's recovery of uniform occupancy censoring on the arm,
not recovering the arm's observation design. A near-zero denominator or an
anchor that moves farther from zero is left unnormalized, and raw signed bias
must be used.

## G0b.5 — Mismatch pairing

Only pairs with opposite required correction directions are eligible:

| walk_arm           | other_arm                               |   median_profile_mae_penalty |   median_abs_rho2_bias_shift |   min_abs_rho2_bias_shift |   max_abs_rho2_bias_shift |
|:-------------------|:----------------------------------------|-----------------------------:|-----------------------------:|--------------------------:|--------------------------:|
| time_agnostic_t    | activity_proportional_dyad_full_history |                       0.2385 |                       0.3145 |                    0.1982 |                    0.4308 |
| time_respecting    | activity_proportional_dyad_full_history |                       0.1559 |                       0.2893 |                    0.2335 |                    0.3451 |
| recent_history_k20 | activity_proportional_dyad_full_history |                       0.2391 |                       0.3170 |                    0.2127 |                    0.4212 |

The best new pair is **`recent_history_k20` ↔ `activity_proportional_dyad_full_history`**, with a
median bidirectional absolute `rho_2`-bias shift of **0.3170**.
The old walk-only pair scored **0.1140** on the same bias-axis
summary. Therefore the new pair **replaces**
the old pair. Final mismatch pair: **`recent_history_k20` ↔ `activity_proportional_dyad_full_history`**.

## Decision and limitations

**Arms for the main run: [`time_agnostic_t`, `time_respecting`,
`recent_history_k20`, `activity_proportional_dyad_full_history`].** Conditions retained: `hidden`,
`mechanism`, `mechanism_direction`, `metadata_only`, and `mismatched` only for
the final pair above.

The new arm cleanly widens the aggregate bias direction, but it is deliberately
synthetic and oracle-like: selection weights require full dyad event counts and
selected dyads return full histories. It is a specificity stimulus/control,
not a claimed real-world limited-access mechanism. Its much lower dyad
coverage at equal event budget is a genuine arm difference.

**Crucial qualification for the revised G4 wording:** positive recurrence bias
is not a label-free theorem of activity-PPS. In the with-replacement idealization
its sign is the sign of `Cov(event_count, I[K>=2])`; a K=1 dyad can have many
events, so the covariance can be negative. The observed direction is an
empirical panel-level fact, and individual cases can have the opposite sign
(see `negative_case_share` above). The planned direction contrast is therefore
an aggregate panel specificity test, not a casewise label-free ground truth.
G0b does not show that an LLM can use the mechanism description, and no prompt
wording has yet been approved.

## Stop

**STOP at G0b. Await explicit confirmation of G0 and G0b together before any
G1 prompt-contract work.**
