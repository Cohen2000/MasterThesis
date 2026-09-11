# G0c: estimand pin and two-phase arm-B audit

Prepared: **2026-08-31**  
Gate status: **G0c complete. No LLM calls were made. STOP before G1.**

## Scope and decision

G0c replaces the oracle-weighted G0b arm with `event_sample_then_full_history` and keeps both
full-history controls in the planned five-arm set. All new artifacts live
below `results/g0c_headroom_2026_09`; no frozen benchmark case, panel truth, or LLM artifact was
modified.

The arms are a spanning set over the two named error sources, not a sampler
catalogue:

| arm | censoring | selection bias | required correction |
|---|---|---|---|
| `time_agnostic_t` | strong | weak | upward |
| `time_respecting` | strong | weak | upward |
| `recent_history_k20` | strong | weak | upward |
| `node_panel_full_history` | none | none | none |
| `event_sample_then_full_history` | none | strong | downward |

The three walk arms are near-replicates on the signed-bias axis; they differ
mainly in observable mask shape rather than in the direction or size of the
required correction.

The rebuilt arm's eight-seed group-macro `rho_2` bias is
**+0.1625** (seed SD
0.0282). The prespecified
acceptance threshold `> +0.05` therefore **passes**.

## G0c.1 — Pinned estimand

- **Node universe:** all node IDs appearing as an endpoint of at least one
  event in the complete normalized stream. Isolated nodes outside the event
  stream are not in the manifest universe.
- **Dyad universe:** all canonical undirected dyads `(u,v)`, `u < v`, with at
  least one event in that complete stream. It is **not** all `n choose 2`
  possible pairs.
- **Denominator:** exactly the number of those full-stream event-active dyads
  (`n_pairs` in `census.py`, stored as `n_edges` in the manifests). For each
  such dyad, `K` is the number of distinct active windows and
  `rho_k = mean(1[K >= k])` over that fixed denominator.
- **Across arms:** the universe and truth columns are identical for all five
  arms. The samplers copy truth from the same manifest row, and the replay
  audit initializes every full-population dyad at observation mask 0 before
  moving observed dyads to a nonzero mask.

Direct panel-file checks: node counts match in **32/32**
graphs, active-dyad counts in **32/32**,
event counts in **32/32**, and canonical
undirected endpoint order in **32/32**.

**Sample-dependent conditioning exists only in estimators, not in the
estimand.** In particular, the naive read-off in `benchmark_features.py`
averages over the dyads present in the sample. That observed-dyad denominator
is intentionally sample-dependent and is the source of the selection/censoring
bias being measured. The frozen truth denominator does not change by arm.

## G0c.2 — Rebuilt arm B at 800 unique events

Implementation: phase 1 traverses a uniform random-priority ordering of event
records. The first occurrence of a dyad triggers phase 2, which adds that
dyad's complete event history. The prefix stops before the first new lookup
whose complete history would exceed the remaining unique-event budget. No
full-stream event count is used as a sampling weight. More active dyads are
discovered more often because they own more event records in phase 1.

Every emitted dyad history is complete and no partial response is retained.
The stop can leave slack; if the first event belongs to a dyad with more than
800 events it emits an empty sample. This happened in
**9/256**
eight-seed cases. Those cases are reported as undefined, not silently redrawn
or treated as `rho_2=0`. Seed 0, used for the estimator ladder, has
**0** empty cases.

Bias by seed (undefined empty cases excluded from signed bias):

|   sample_seed |   valid_cases |   empty_cases |   group_macro_rho2_bias |
|--------------:|--------------:|--------------:|------------------------:|
|        0.0000 |       32.0000 |        0.0000 |                  0.1291 |
|        1.0000 |       29.0000 |        3.0000 |                  0.1705 |
|        2.0000 |       31.0000 |        1.0000 |                  0.1788 |
|        3.0000 |       31.0000 |        1.0000 |                  0.1857 |
|        4.0000 |       30.0000 |        2.0000 |                  0.1593 |
|        5.0000 |       31.0000 |        1.0000 |                  0.1129 |
|        6.0000 |       32.0000 |        0.0000 |                  0.1688 |
|        7.0000 |       31.0000 |        1.0000 |                  0.1952 |

Aggregate bias and case-wise direction spread:

|   group_macro_bias_mean |   group_macro_bias_sd |   seed0_group_macro_bias |   total_cases |   valid_cases |   empty_cases |   seed0_empty_cases |   delta_lt_zero_share_valid |   delta_gt_zero_share_valid |   delta_eq_zero_share_valid |
|------------------------:|----------------------:|-------------------------:|--------------:|--------------:|--------------:|--------------------:|----------------------------:|----------------------------:|----------------------------:|
|                  0.1625 |                0.0282 |                   0.1291 |      256.0000 |      247.0000 |        9.0000 |              0.0000 |                      0.8421 |                      0.1579 |                      0.0000 |

Coverage and budget diagnostics:

| scope                            |   median |     mean |      p10 |      p90 |    min |      max |
|:---------------------------------|---------:|---------:|---------:|---------:|-------:|---------:|
| rebuilt B coverage, seed 0       |   0.0030 |   0.0143 |   0.0003 |   0.0289 | 0.0001 |   0.0927 |
| rebuilt B coverage, all 8 seeds  |   0.0031 |   0.0145 |   0.0002 |   0.0337 | 0.0000 |   0.0947 |
| rebuilt B realized unique events | 773.0000 | 666.8867 | 394.5000 | 800.0000 | 0.0000 | 800.0000 |
| rebuilt B budget slack           |  27.0000 | 133.1133 |   0.0000 | 405.5000 | 0.0000 | 800.0000 |
| node panel A coverage, seed 0    |   0.0170 |   0.0233 |   0.0000 |   0.0717 | 0.0000 |   0.1030 |
| three walks coverage, seed 0     |   0.0555 |   0.1047 |   0.0174 |   0.2651 | 0.0028 |   0.4978 |

Unique-event phase split (`phase1 + phase2 additional = realized`):

| scope                            |   median |     mean |      p10 |      p90 |    min |      max |
|:---------------------------------|---------:|---------:|---------:|---------:|-------:|---------:|
| phase 1 uniform event sample     |  36.0000 | 129.6328 |   2.0000 | 450.0000 | 0.0000 | 788.0000 |
| phase 2 additional lookup events | 625.0000 | 537.2539 |  72.0000 | 750.0000 | 0.0000 | 787.0000 |
| realized unique events           | 773.0000 | 666.8867 | 394.5000 | 800.0000 | 0.0000 | 800.0000 |

Static mask-input lengths (`INPUT_MASK` plus exact histogram, before G1 text):

| arm                            | measure         |   median |     mean |      p10 |       p90 |      min |       max |
|:-------------------------------|:----------------|---------:|---------:|---------:|----------:|---------:|----------:|
| event_sample_then_full_history | characters      | 622.5000 | 714.1562 | 525.0000 |  986.0000 | 513.0000 | 1309.0000 |
| event_sample_then_full_history | utf8_bytes      | 622.5000 | 714.1562 | 525.0000 |  986.0000 | 513.0000 | 1309.0000 |
| event_sample_then_full_history | portable_tokens | 142.0000 | 183.3750 | 102.0000 |  300.4000 |  98.0000 |  450.0000 |
| node_panel_full_history        | characters      | 765.5000 | 775.3438 | 504.0000 | 1162.8000 | 504.0000 | 1224.0000 |
| node_panel_full_history        | utf8_bytes      | 765.5000 | 775.3438 | 504.0000 | 1162.8000 | 504.0000 | 1224.0000 |
| node_panel_full_history        | portable_tokens | 206.0000 | 211.0938 |  95.0000 |  378.8000 |  95.0000 |  410.0000 |
| recent_history_k20             | characters      | 876.0000 | 836.2812 | 630.7000 | 1047.4000 | 573.0000 | 1152.0000 |
| recent_history_k20             | utf8_bytes      | 876.0000 | 836.2812 | 630.7000 | 1047.4000 | 573.0000 | 1152.0000 |
| recent_history_k20             | portable_tokens | 254.0000 | 236.5000 | 146.4000 |  327.2000 | 122.0000 |  374.0000 |
| time_agnostic_t                | characters      | 666.0000 | 710.6562 | 622.2000 |  868.7000 | 604.0000 | 1089.0000 |
| time_agnostic_t                | utf8_bytes      | 666.0000 | 710.6562 | 622.2000 |  868.7000 | 604.0000 | 1089.0000 |
| time_agnostic_t                | portable_tokens | 162.0000 | 181.5000 | 142.0000 |  251.6000 | 134.0000 |  350.0000 |
| time_respecting                | characters      | 744.0000 | 729.3438 | 618.2000 |  843.7000 | 581.0000 |  850.0000 |
| time_respecting                | utf8_bytes      | 744.0000 | 729.3438 | 618.2000 |  843.7000 | 581.0000 |  850.0000 |
| time_respecting                | portable_tokens | 196.0000 | 189.8750 | 142.0000 |  238.0000 | 126.0000 |  242.0000 |

Exact Qwen3.6-27B tokenizer counts for the same block
(cached snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
`add_special_tokens=False`):

| arm                            |   median |     mean |      p10 |      p90 |      min |      max |
|:-------------------------------|---------:|---------:|---------:|---------:|---------:|---------:|
| event_sample_then_full_history | 234.5000 | 305.4688 | 157.0000 | 518.8000 | 147.0000 | 767.0000 |
| node_panel_full_history        | 345.5000 | 352.6875 | 139.0000 | 656.4000 | 139.0000 | 702.0000 |
| recent_history_k20             | 432.0000 | 401.0312 | 240.5000 | 566.8000 | 195.0000 | 648.0000 |
| time_agnostic_t                | 268.0000 | 302.9062 | 234.2000 | 425.9000 | 220.0000 | 597.0000 |
| time_respecting                | 329.0000 | 317.4062 | 230.2000 | 407.7000 | 201.0000 | 412.0000 |

Eight-seed variance decomposition uses successful cases only and is therefore
slightly unbalanced where the whole-lookup stop produced an empty sample:

| method                                                       |   mean |   n |   n_group |   n_inst |   n_seed |   s2_group |   s2_inst |   s2_seed |   missing_cases |   seed_variance_share |   se_1seed |   se_8seeds |
|:-------------------------------------------------------------|-------:|----:|----------:|---------:|---------:|-----------:|----------:|----------:|----------------:|----------------------:|-----------:|------------:|
| naive read-off                                               | 0.1349 | 247 |   12.0000 |   2.6667 |   7.7188 |     0.0000 |    0.0204 |    0.0070 |               9 |                0.2565 |     0.0293 |      0.0258 |
| occupancy MLE (uniform; censoring-aware, mechanism-agnostic) | 0.2701 | 247 |   12.0000 |   2.6667 |   7.7188 |     0.0324 |    0.0242 |    0.0077 |               9 |                0.1193 |     0.0608 |      0.0590 |
| mask MLE (uniform; censoring-aware, mechanism-agnostic)      | 0.2222 | 247 |   12.0000 |   2.6667 |   7.7188 |     0.0265 |    0.0114 |    0.0073 |               9 |                0.1610 |     0.0528 |      0.0509 |

Full reference ladder at seed 0. `occ_mle` and `mask_mle` are
**censoring-aware, mechanism-agnostic**; ExtraTrees is a **label-informed
performance reference**, trained with the matching panel backbone held out:

| arm                            | estimator                                                                              |   n |   profile_mae |   rho2_bias |   pooled_profile_mae |   pooled_rho2_bias |
|:-------------------------------|:---------------------------------------------------------------------------------------|----:|--------------:|------------:|---------------------:|-------------------:|
| time_agnostic_t                | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| time_agnostic_t                | naive read-off                                                                         |  32 |        0.1562 |     -0.3085 |               0.1437 |            -0.3031 |
| time_agnostic_t                | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.0597 |     -0.0657 |               0.0578 |            -0.0721 |
| time_agnostic_t                | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.0579 |     -0.0490 |               0.0558 |            -0.0554 |
| time_agnostic_t                | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0412 |     -0.0139 |               0.0431 |            -0.0165 |
| time_respecting                | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| time_respecting                | naive read-off                                                                         |  32 |        0.1573 |     -0.3098 |               0.1435 |            -0.3001 |
| time_respecting                | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.1169 |     -0.1475 |               0.1089 |            -0.1449 |
| time_respecting                | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.1139 |     -0.1150 |               0.1032 |            -0.1071 |
| time_respecting                | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0556 |      0.0141 |               0.0558 |             0.0127 |
| recent_history_k20             | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| recent_history_k20             | naive read-off                                                                         |  32 |        0.1493 |     -0.2868 |               0.1367 |            -0.2807 |
| recent_history_k20             | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.0893 |     -0.0625 |               0.0864 |            -0.0743 |
| recent_history_k20             | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.0851 |     -0.0424 |               0.0787 |            -0.0588 |
| recent_history_k20             | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0519 |      0.0019 |               0.0509 |            -0.0000 |
| node_panel_full_history        | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| node_panel_full_history        | naive read-off                                                                         |  25 |        0.0303 |      0.0123 |               0.0246 |             0.0055 |
| node_panel_full_history        | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  25 |        0.3333 |      0.3350 |               0.2898 |             0.3003 |
| node_panel_full_history        | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  25 |        0.2106 |      0.3168 |               0.1823 |             0.2820 |
| node_panel_full_history        | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0342 |      0.0055 |               0.0367 |             0.0043 |
| event_sample_then_full_history | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| event_sample_then_full_history | naive read-off                                                                         |  32 |        0.1304 |      0.1291 |               0.1271 |             0.1251 |
| event_sample_then_full_history | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.2731 |      0.2740 |               0.2492 |             0.2495 |
| event_sample_then_full_history | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.2264 |      0.2758 |               0.2048 |             0.2524 |
| event_sample_then_full_history | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0312 |     -0.0012 |               0.0334 |            -0.0027 |

Arm A has **7/32** empty seed-0 samples under its existing
whole-node stop, hence `n=25` for its analytical rows.
The supervised reference still produces 32 predictions by treating the empty
observable feature vector as a valid input. This difference in `n` must not be
hidden in later comparisons.

## G0c.3 — Coverage-matched sensitivity

A common diagnostic budget of **10,500** unique events gives rebuilt B median
coverage 0.0553,
against 0.0555
for the walks. This is median matching, not graph-by-graph matching; the main
run remains fixed at 800.

| scope                           |   median |   mean |    p10 |    p90 |    min |    max |
|:--------------------------------|---------:|-------:|-------:|-------:|-------:|-------:|
| rebuilt B, budget 800, seed 0   |   0.0030 | 0.0143 | 0.0003 | 0.0289 | 0.0001 | 0.0927 |
| rebuilt B, budget 10500, seed 0 |   0.0553 | 0.2088 | 0.0025 | 0.9370 | 0.0011 | 1.0000 |
| three walks, budget 800, seed 0 |   0.0555 | 0.1047 | 0.0174 | 0.2651 | 0.0028 | 0.4978 |

Estimator ladder at the sensitivity budget:

| arm                            | estimator                                                                              |   n |   profile_mae |   rho2_bias |   pooled_profile_mae |   pooled_rho2_bias |
|:-------------------------------|:---------------------------------------------------------------------------------------|----:|--------------:|------------:|---------------------:|-------------------:|
| event_sample_then_full_history | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| event_sample_then_full_history | naive read-off                                                                         |  32 |        0.0949 |      0.1493 |               0.0987 |             0.1591 |
| event_sample_then_full_history | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.2721 |      0.3292 |               0.2480 |             0.3119 |
| event_sample_then_full_history | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.2286 |      0.3267 |               0.2065 |             0.3095 |
| event_sample_then_full_history | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0264 |      0.0107 |               0.0283 |             0.0118 |

The positive naive-bias sign **and rough magnitude hold**
at matched median coverage. This check does not make the access arms
information-equivalent: B still observes fewer dyads with complete records,
whereas the walks observe more dyads with truncated/repeated records.

## G0c.4 — Mismatch pair over all five arms

The empirical observation models for A and rebuilt B were learned from the
regenerated 745-instance benchmark with complete source/family holdout. The
new `(n,mask)` samples were deterministically replayed and
**5,960/5,960** histograms matched. As in G0/G0b, this
arm-likelihood exercise is label-assisted and is neither a deployable
estimator nor a theoretical bound.

Group-macro ProfileMAE (rows produce the sample; columns supply the assumed
arm likelihood):

| sample_arm                     |   event_sample_then_full_history |   node_panel_full_history |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:-------------------------------|---------------------------------:|--------------------------:|---------------------:|------------------:|------------------:|
| event_sample_then_full_history |                           0.1065 |                    0.1398 |               0.4454 |            0.4433 |            0.3131 |
| node_panel_full_history        |                           0.0390 |                    0.0368 |               0.3422 |            0.3529 |            0.2229 |
| recent_history_k20             |                           0.1531 |                    0.1460 |               0.0983 |            0.1059 |            0.1229 |
| time_agnostic_t                |                           0.1585 |                    0.1542 |               0.0972 |            0.0874 |            0.1159 |
| time_respecting                |                           0.1594 |                    0.1553 |               0.1318 |            0.1140 |            0.0925 |

The same matrix as signed `rho_2` bias:

| sample_arm                     |   event_sample_then_full_history |   node_panel_full_history |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:-------------------------------|---------------------------------:|--------------------------:|---------------------:|------------------:|------------------:|
| event_sample_then_full_history |                           0.0919 |                    0.1532 |               0.5159 |            0.5148 |            0.4092 |
| node_panel_full_history        |                          -0.0343 |                    0.0436 |               0.4365 |            0.4473 |            0.3462 |
| recent_history_k20             |                          -0.2985 |                   -0.2762 |              -0.0849 |           -0.0717 |           -0.1976 |
| time_agnostic_t                |                          -0.3159 |                   -0.3016 |              -0.1468 |           -0.1171 |           -0.2034 |
| time_respecting                |                          -0.3174 |                   -0.3026 |              -0.1986 |           -0.1497 |           -0.0833 |

All directed off-diagonal penalties:

| sample_arm                     | assumed_arm                    |   profile_mae_penalty |   rho2_bias_shift |   abs_rho2_bias_shift |
|:-------------------------------|:-------------------------------|----------------------:|------------------:|----------------------:|
| time_agnostic_t                | time_respecting                |                0.0285 |           -0.0863 |                0.0863 |
| time_agnostic_t                | recent_history_k20             |                0.0098 |           -0.0296 |                0.0296 |
| time_agnostic_t                | node_panel_full_history        |                0.0668 |           -0.1845 |                0.1845 |
| time_agnostic_t                | event_sample_then_full_history |                0.0711 |           -0.1988 |                0.1988 |
| time_respecting                | time_agnostic_t                |                0.0215 |           -0.0665 |                0.0665 |
| time_respecting                | recent_history_k20             |                0.0394 |           -0.1153 |                0.1153 |
| time_respecting                | node_panel_full_history        |                0.0628 |           -0.2193 |                0.2193 |
| time_respecting                | event_sample_then_full_history |                0.0669 |           -0.2342 |                0.2342 |
| recent_history_k20             | time_agnostic_t                |                0.0076 |            0.0133 |                0.0133 |
| recent_history_k20             | time_respecting                |                0.0246 |           -0.1126 |                0.1126 |
| recent_history_k20             | node_panel_full_history        |                0.0476 |           -0.1912 |                0.1912 |
| recent_history_k20             | event_sample_then_full_history |                0.0548 |           -0.2135 |                0.2135 |
| node_panel_full_history        | time_agnostic_t                |                0.3162 |            0.4037 |                0.4037 |
| node_panel_full_history        | time_respecting                |                0.1861 |            0.3026 |                0.3026 |
| node_panel_full_history        | recent_history_k20             |                0.3054 |            0.3928 |                0.3928 |
| node_panel_full_history        | event_sample_then_full_history |                0.0022 |           -0.0780 |                0.0780 |
| event_sample_then_full_history | time_agnostic_t                |                0.3367 |            0.4229 |                0.4229 |
| event_sample_then_full_history | time_respecting                |                0.2066 |            0.3173 |                0.3173 |
| event_sample_then_full_history | recent_history_k20             |                0.3389 |            0.4240 |                0.4240 |
| event_sample_then_full_history | node_panel_full_history        |                0.0333 |            0.0613 |                0.0613 |

Eligible bidirectional pairs must have different correction classes. `none`
means an eight-seed naive bias within +/-0.05. Observable distinguishability
uses only `(n,mask)`-histogram-derived features: a logistic arm classifier is
trained with the entire graph group held out. AUC 0.5 / distance 0 means
indistinguishable; AUC 1 / distance 1 means perfectly distinguishable.

Measured bias classes used for eligibility:

| arm                            |   naive_group_macro_rho2_bias | required_correction   |
|:-------------------------------|------------------------------:|:----------------------|
| time_agnostic_t                |                       -0.3085 | upward                |
| time_respecting                |                       -0.3098 | upward                |
| recent_history_k20             |                       -0.2868 | upward                |
| node_panel_full_history        |                       -0.0107 | none                  |
| event_sample_then_full_history |                        0.1625 | downward              |

| arm_a                   | arm_b                          | correction_a   | correction_b   |   median_profile_mae_penalty |   median_abs_rho2_bias_shift |   min_abs_rho2_bias_shift |   max_abs_rho2_bias_shift |   observable_auc_logo |   observable_distance_2auc_minus1 |   observable_balanced_accuracy |   observable_feature_count |
|:------------------------|:-------------------------------|:---------------|:---------------|-----------------------------:|-----------------------------:|--------------------------:|--------------------------:|----------------------:|----------------------------------:|-------------------------------:|---------------------------:|
| time_agnostic_t         | node_panel_full_history        | upward         | none           |                       0.1915 |                       0.2941 |                    0.1845 |                    0.4037 |                0.8154 |                            0.6309 |                         0.8281 |                        447 |
| time_agnostic_t         | event_sample_then_full_history | upward         | downward       |                       0.2039 |                       0.3108 |                    0.1988 |                    0.4229 |                0.8105 |                            0.6211 |                         0.8125 |                        447 |
| time_respecting         | node_panel_full_history        | upward         | none           |                       0.1245 |                       0.2609 |                    0.2193 |                    0.3026 |                0.9971 |                            0.9941 |                         0.9688 |                        447 |
| time_respecting         | event_sample_then_full_history | upward         | downward       |                       0.1368 |                       0.2758 |                    0.2342 |                    0.3173 |                0.9355 |                            0.8711 |                         0.8594 |                        447 |
| recent_history_k20      | node_panel_full_history        | upward         | none           |                       0.1765 |                       0.2920 |                    0.1912 |                    0.3928 |                0.6826 |                            0.3652 |                         0.7188 |                        447 |
| recent_history_k20      | event_sample_then_full_history | upward         | downward       |                       0.1968 |                       0.3188 |                    0.2135 |                    0.4240 |                0.6660 |                            0.3320 |                         0.7188 |                        447 |
| node_panel_full_history | event_sample_then_full_history | none           | downward       |                       0.0177 |                       0.0696 |                    0.0613 |                    0.0780 |                0.6875 |                            0.3750 |                         0.6875 |                        447 |

Largest bidirectional bias penalty: **`recent_history_k20` <->
`event_sample_then_full_history`**, median absolute `rho_2`-bias shift
**0.3188**. Its held-group-out
observable AUC is **0.6660** (distance
0.3320). The chosen pair is only
moderately distinguishable under this held-group-out diagnostic, not grossly
separable. Text distrust and fallback toward `hidden` remain possible, but the
observable statistics do not make that alternative explanation automatic.

## What remains uncertain

- Positive selection bias is a panel-level empirical result, not a per-case
  theorem. The case-wise `delta_i` sign distribution above is why G4 must use
  the prespecified slope rather than assign one arm-level sign to every case.
- A hard unique-event cap and complete-history requirement make oversize first
  lookups genuinely undefined. G2 must freeze how these cases enter response
  and failure-rate reporting for both full-history arms; G0c did not redraw
  them.
- Coverage matching controls one scalar only. It does not match number of
  dyads, event multiplicities, temporal masks, or sample coherence.
- No language model has been tested here, and this report says nothing about
  whether a model can operationalize the mechanism text.

arms for the main run: [`time_agnostic_t`, `time_respecting`,
`recent_history_k20`, `node_panel_full_history`,
`event_sample_then_full_history`]
