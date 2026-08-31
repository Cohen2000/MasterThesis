# G0d: budget parity by coverage, not by events

Prepared: **2026-08-31**  
Gate status: **G0d.1-G0d.4 complete and passing. G0d.5 does not clear its
own bar and hands G1 an open decision (see below). No LLM calls were made.
STOP before G1.**

## Why the budgets moved

G0c gave every arm 800 unique events.  For a random walk that is a generous
budget; for whole-entity retrieval it is not, because one response is a whole
node's or a whole dyad's complete history.  The consequence was not a
smaller sample but no sample: **7/32** empty cases on arm A at seed 0 and
**9/256** on arm B across eight seeds.  An arm that fails to produce a sample
on a fifth of the panel is not an arm.

The arms do **not** need to be information-matched.  The primary contrast in
G4 is `mechanism - hidden` within a case on an identical sample, so the arm is
a facet, not a competitor, and cross-arm comparisons of absolute accuracy are
descriptive only.  What the arms do need is to produce a usable sample on
every case.  Coverage parity is the budget rule that delivers that while
keeping the arms on a comparable footing.

**The walks are unchanged at 800 unique events.**  Every frozen number, the
budget probe, the MDD calculation and the historical comparison depend on
that, and nothing here touches them.

## G0d.1 — Budgets set by dyad coverage

Target: the three walk arms' pooled median dyad coverage at 800 events,
**0.0548** over the same eight seed slots
(**0.0555** at seed 0 alone).

Budgets were selected with `src/g0d_budget_ladder.py` over a 28-point grid.
Both whole-entity samplers stop before the first entity that does not fit, so
the sample at any budget is a prefix of one fixed random entity order and a
single cumulative pass prices the whole grid exactly.  The ladder was verified
against the production samplers at budget 2,500.

The rungs around each adopted budget, so the choice is auditable rather than
asserted (`empty_draw_rate` is over all 8 x 32 draws at that rung):

| arm                            |   target_budget |   median_dyad_coverage |   empty_draw_rate |   median_realized_events |   median_natural_units |   distance_to_target | adopted   |
|:-------------------------------|----------------:|-----------------------:|------------------:|-------------------------:|-----------------------:|---------------------:|:----------|
| node_panel_full_history        |            2200 |                 0.0443 |            0.0195 |                2093.5000 |                37.0000 |               0.0104 | False     |
| node_panel_full_history        |            2300 |                 0.0477 |            0.0156 |                2165.0000 |                37.5000 |               0.0070 | False     |
| node_panel_full_history        |            2400 |                 0.0522 |            0.0156 |                2298.5000 |                40.5000 |               0.0026 | False     |
| node_panel_full_history        |            2500 |                 0.0566 |            0.0156 |                2381.0000 |                41.0000 |               0.0019 | True      |
| node_panel_full_history        |            2600 |                 0.0588 |            0.0117 |                2484.0000 |                43.5000 |               0.0040 | False     |
| node_panel_full_history        |            2700 |                 0.0597 |            0.0117 |                2566.5000 |                46.5000 |               0.0049 | False     |
| node_panel_full_history        |            2800 |                 0.0604 |            0.0117 |                2668.5000 |                46.5000 |               0.0057 | False     |
| event_sample_then_full_history |            9000 |                 0.0505 |            0.0000 |                8973.0000 |               564.5000 |               0.0043 | False     |
| event_sample_then_full_history |            9200 |                 0.0514 |            0.0000 |                9175.0000 |               581.0000 |               0.0034 | False     |
| event_sample_then_full_history |            9400 |                 0.0527 |            0.0000 |                9380.0000 |               595.0000 |               0.0021 | False     |
| event_sample_then_full_history |            9600 |                 0.0544 |            0.0000 |                9579.0000 |               611.5000 |               0.0003 | True      |
| event_sample_then_full_history |            9800 |                 0.0558 |            0.0000 |                9765.0000 |               626.5000 |               0.0010 | False     |
| event_sample_then_full_history |           10000 |                 0.0584 |            0.0000 |                9977.0000 |               643.5000 |               0.0036 | False     |
| event_sample_then_full_history |           10200 |                 0.0601 |            0.0000 |               10168.5000 |               654.5000 |               0.0054 | False     |

**Arm B's G0c budget does not confirm at eight seeds.** G0c read 10,500 ->
coverage 0.0553 from **seed 0 alone**; that value reproduces exactly here, but
the eight-seed median at 10,500 is **0.0610**, about 11% above the walks.
Seed 0 sat at the low end of B's seed spread.  The eight-seed parity point is
**9,600**, which is what this report adopts as primary.  10,500 is carried
through the tables below as a measured alternative so the G0c choice remains
available; the two differ little and both clear the bias gate.

Realized access at the adopted budgets, with each arm's natural access unit
beside the common event count:

| arm                            |   target_budget | natural_unit    |   median_natural_units |   median_realized_events |   median_dyad_coverage |   median_node_coverage |   median_windows_touched |   median_temporal_evenness |   median_dyads_observed |   cases |
|:-------------------------------|----------------:|:----------------|-----------------------:|-------------------------:|-----------------------:|-----------------------:|-------------------------:|---------------------------:|------------------------:|--------:|
| event_sample_then_full_history |            9600 | dyads looked up |               611.5000 |                9579.0000 |                 0.0544 |                 0.5326 |                   1.0000 |                     0.6909 |                611.5000 |     256 |
| node_panel_full_history        |            2500 | nodes recruited |                41.0000 |                2381.0000 |                 0.0575 |                 0.2844 |                   1.0000 |                     0.6892 |                335.5000 |     256 |
| recent_history_k20             |             800 | walk steps      |               800.0000 |                 800.0000 |                 0.0540 |                 0.4090 |                   1.0000 |                     0.5800 |                471.0000 |     256 |
| time_agnostic_t                |             800 | walk steps      |               800.0000 |                 800.0000 |                 0.0901 |                 0.5170 |                   1.0000 |                     0.8176 |                715.0000 |     256 |
| time_respecting                |             800 | walk steps      |               800.0000 |                 800.0000 |                 0.0518 |                 0.5241 |                   1.0000 |                     0.2660 |                393.5000 |     256 |

Events are the common descriptive; the natural unit is what a real study would
actually budget, and the three differ by more than an order of magnitude -- a
walk spends 800 steps, arm A recruits tens of nodes, arm B looks up hundreds of
dyads.  These are the eight accepted seed slots, which is the basis parity was
set on.

The coverage axes move differently, which is the point: matching dyad coverage
does **not** match node coverage, and it does not match temporal spread either.
Every arm touches all five windows on every case, so `windows_touched` is 1.00
throughout and separates nothing; `temporal_evenness` (1 = the sample's events
are spread uniformly over the five windows, 0 = all in one) does separate them.
The arms remain information-distinct by construction, and G4 must say so rather
than read cross-arm accuracy differences as arm difficulty.

## G0d.2 — The empty-sample rule, and why arm A still needs it

A whole-entity sample is empty exactly when the *first* entity drawn does not
fit on its own, so the empty probability is available in closed form and needs
no simulation.  Arm A draws nodes uniformly, giving the share of nodes whose
incident-event count exceeds the budget.  Arm B reaches a dyad through a
uniformly drawn event, giving the event-weighted share of oversized dyads.

**Arm B at 9,600 is structurally safe: P(empty) = 0.0000 on all 32
graphs.** The panel's longest single dyad history is 4,992 events, so no first
lookup can overflow the budget.  The nine empty G0c cases are gone by
construction, not by luck.

**Arm A is not, and cannot be at any parity budget.** The panel's largest node
carries **10,571** incident events, so a whole-node response is
only guaranteed to fit above budget 12,000 -- nearly five times the parity
budget, at median coverage 0.31 instead of 0.055.  At 2,500 the exact empty
probability is nonzero on 15 of 32 graphs, worst case **0.0765**.  The
brief expected a zero empty rate at the new budgets; for arm A that
expectation does not hold, and the prespecified rule is load-bearing rather
than a contingency.

The rule, applied and logged: the seed sequence for a case is the prespecified
index order 0, 1, 2, ...; a draw whose whole-entity stop left the sample empty
is skipped and the next index is used.  Nothing is redrawn at random and no
case is dropped, so the accepted set is a deterministic function of the
sequence.  Panels ran 16 seed slots to give the rule headroom.  Because the
worst-case empty probability is 0.0765, far below 1, the rule
terminates almost surely and cheaply.

Across the accepted panel, **4** seed advances were needed in
total, affecting **4** of 64 (case, arm) pairs.
Per-case counts are in `seed_advance_log.csv`; exact per-graph probabilities
are in `exact_empty_probability.csv`.

## G0d.3 — Re-measurement at the new budgets

Group-macro `rho_2` bias by seed slot:

| arm                            |   seed_slot |   valid_cases |   empty_cases |   group_macro_rho2_bias |
|:-------------------------------|------------:|--------------:|--------------:|------------------------:|
| event_sample_then_full_history |           0 |            32 |             0 |                  0.1494 |
| event_sample_then_full_history |           1 |            32 |             0 |                  0.1429 |
| event_sample_then_full_history |           2 |            32 |             0 |                  0.1522 |
| event_sample_then_full_history |           3 |            32 |             0 |                  0.1505 |
| event_sample_then_full_history |           4 |            32 |             0 |                  0.1500 |
| event_sample_then_full_history |           5 |            32 |             0 |                  0.1555 |
| event_sample_then_full_history |           6 |            32 |             0 |                  0.1532 |
| event_sample_then_full_history |           7 |            32 |             0 |                  0.1556 |
| node_panel_full_history        |           0 |            32 |             0 |                 -0.0065 |
| node_panel_full_history        |           1 |            32 |             0 |                 -0.0005 |
| node_panel_full_history        |           2 |            32 |             0 |                  0.0020 |
| node_panel_full_history        |           3 |            32 |             0 |                 -0.0042 |
| node_panel_full_history        |           4 |            32 |             0 |                 -0.0105 |
| node_panel_full_history        |           5 |            32 |             0 |                 -0.0105 |
| node_panel_full_history        |           6 |            32 |             0 |                 -0.0010 |
| node_panel_full_history        |           7 |            32 |             0 |                 -0.0115 |

Eight-slot aggregate, with the case-wise spread of the correct correction
direction:

| arm                            |   target_budget |   group_macro_bias_mean |   group_macro_bias_sd |   slot0_group_macro_bias |   accepted_cases |   empty_cases_after_rule |   delta_gt_zero_share |   delta_lt_zero_share |
|:-------------------------------|----------------:|------------------------:|----------------------:|-------------------------:|-----------------:|-------------------------:|----------------------:|----------------------:|
| event_sample_then_full_history |            9600 |                  0.1512 |                0.0041 |                   0.1494 |              256 |                        0 |                0.0742 |                0.8945 |
| node_panel_full_history        |            2500 |                 -0.0053 |                0.0052 |                  -0.0065 |              256 |                        0 |                0.5664 |                0.4336 |

**Acceptance.** Arm A's bias must stay within +/-0.05 of zero and arm B's
above +0.05.  Arm A: **-0.0053**.  Arm B: **+0.1512**.  Result:
**PASS for both arms**.  No budget was tuned after seeing a sign; the two budgets
come from the coverage ladder alone.

Per-case sign distribution of `delta_i = rho2_true_i - rho2_naive_i`.  This is
why G4 must regress on the per-case `delta_i` rather than assign one
arm-level sign to every case:

| arm                            |   cases |   share_delta_gt_0_upward |   share_delta_lt_0_downward |   share_delta_eq_0 |   median_delta |   mean_delta |
|:-------------------------------|--------:|--------------------------:|----------------------------:|-------------------:|---------------:|-------------:|
| event_sample_then_full_history |     256 |                    0.0742 |                      0.8945 |             0.0312 |        -0.1192 |      -0.1607 |
| node_panel_full_history        |     256 |                    0.5664 |                      0.4336 |             0.0000 |         0.0042 |       0.0057 |

Full reference ladder at the accepted first seed slot.  `occ_mle` and
`mask_mle` are **censoring-aware but mechanism-agnostic** -- one uniform
likelihood on every arm, no arm parameter anywhere -- and ExtraTrees is a
**label-informed performance reference**, trained on the benchmark population
with the matching panel backbones held out.  Neither is design-aware and
neither is an upper bound.  The walk rows are the unchanged G0 numbers at 800.

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
| node_panel_full_history        | naive read-off                                                                         |  32 |        0.0169 |     -0.0065 |               0.0182 |            -0.0069 |
| node_panel_full_history        | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.2620 |      0.3086 |               0.2392 |             0.2917 |
| node_panel_full_history        | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.1908 |      0.2905 |               0.1674 |             0.2720 |
| node_panel_full_history        | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0186 |     -0.0046 |               0.0205 |            -0.0055 |
| event_sample_then_full_history | mean floor (panel LOGO)                                                                |  32 |        0.1413 |     -0.0116 |               0.1304 |             0.0010 |
| event_sample_then_full_history | naive read-off                                                                         |  32 |        0.0943 |      0.1494 |               0.0978 |             0.1589 |
| event_sample_then_full_history | occupancy MLE (uniform; censoring-aware, mechanism-agnostic)                           |  32 |        0.2707 |      0.3279 |               0.2464 |             0.3104 |
| event_sample_then_full_history | mask MLE (uniform; censoring-aware, mechanism-agnostic)                                |  32 |        0.2277 |      0.3255 |               0.2053 |             0.3080 |
| event_sample_then_full_history | supervised ExtraTrees (label-informed performance reference; panel backbones held out) |  32 |        0.0264 |      0.0123 |               0.0282 |             0.0136 |

Four readings of this ladder matter for G2 and G4.

First, **arm A now has n = 32 on every row.** In G0c its analytical rows ran on
n = 25 because seven cases had no sample, which meant arm A's numbers were not
comparable to the others'.  That gap is closed, and it was the main thing the
re-budget had to fix.

Second, **arm A has almost no headroom left, and that is what makes it the
control.** Its naive read-off reaches ProfileMAE 0.0169 with bias -0.0065 --
better than the supervised reference on the same cases.  With no censoring and
no selection, the plug-in is already very nearly the right answer, so there is
no correction for a model to discover.  A `mechanism > hidden` effect on arm A
would therefore have nowhere to come from, and G4 should read arm A as the
negative control it is: the arm where the correct answer to "how should I
adjust?" is "do not".  This is also why G1.3's `direction_only` text for arm A
has to say that the naive estimate is approximately unbiased rather than name a
direction.

Third, `occ_mle` and `mask_mle` are **worse than the naive read-off on both
non-walk arms**, and worse by a wide margin on arm B.  That is the expected
behaviour of a censoring correction applied where there is no censoring: both
arms return complete histories, so the estimators inflate a profile that was
already right, and on arm B they add to a selection bias that already points
the same way.  This is the concrete evidence that they are mechanism-agnostic,
and it is why they must never be described as design-aware.

Fourth, the `CensoringRecovery` normalization degenerates on both arms, as
G0b already found for its own arm B: the anchor moves *away* from zero rather
than toward it, so the denominator is negative and the ratio is meaningless.
G2 must report raw signed bias for arms A and B and say why, exactly as the
amendment anticipates.

Eight-slot variance decomposition:

| arm                            | method                                                       |   mean |   n |   s2_group |   s2_inst |   s2_seed |   seed_variance_share |   se_1seed |   se_8seeds |
|:-------------------------------|:-------------------------------------------------------------|-------:|----:|-----------:|----------:|----------:|----------------------:|-----------:|------------:|
| event_sample_then_full_history | naive read-off                                               | 0.1004 | 256 |     0.0000 |    0.0163 |    0.0005 |                0.0279 |     0.0229 |      0.0226 |
| event_sample_then_full_history | occupancy MLE (uniform; censoring-aware, mechanism-agnostic) | 0.2514 | 256 |     0.0348 |    0.0275 |    0.0005 |                0.0081 |     0.0614 |      0.0613 |
| event_sample_then_full_history | mask MLE (uniform; censoring-aware, mechanism-agnostic)      | 0.2094 | 256 |     0.0325 |    0.0140 |    0.0005 |                0.0098 |     0.0562 |      0.0561 |
| node_panel_full_history        | naive read-off                                               | 0.0172 | 256 |     0.0001 |    0.0001 |    0.0002 |                0.4959 |     0.0039 |      0.0032 |
| node_panel_full_history        | occupancy MLE (uniform; censoring-aware, mechanism-agnostic) | 0.2350 | 256 |     0.0372 |    0.0245 |    0.0029 |                0.0442 |     0.0629 |      0.0623 |
| node_panel_full_history        | mask MLE (uniform; censoring-aware, mechanism-agnostic)      | 0.1624 | 256 |     0.0367 |    0.0042 |    0.0011 |                0.0267 |     0.0568 |      0.0565 |

## G0d.4 — Prompt size at the new budgets

The frozen input contract is `INPUT_MASK` plus an exact `(n, mask)` histogram
rendered as a sparse JSON map.  It is **neither** of the two shapes the brief
anticipated.  It is not a fixed-size histogram over `K = 1..5`, so length is
not constant.  But it is not per-dyad rows either, so length does not track
the dyad count: the map is keyed by *distinct* `(n, mask)` combinations, the
mask has only 31 nonzero values, and `n` repeats heavily across dyads, so the
key count saturates long before it reaches one key per dyad.

Distinct histogram keys actually rendered:

| arm                            |   median |     mean |     p10 |      p90 |     min |      max |
|:-------------------------------|---------:|---------:|--------:|---------:|--------:|---------:|
| event_sample_then_full_history |  91.5000 | 108.2500 | 15.0000 | 215.9000 | 11.0000 | 323.0000 |
| node_panel_full_history        |  67.0000 |  64.9375 | 19.5000 | 117.1000 | 11.0000 | 139.0000 |
| recent_history_k20             |  40.0000 |  35.6250 | 13.1000 |  58.3000 |  7.0000 |  70.0000 |
| time_agnostic_t                |  17.0000 |  21.8750 | 12.0000 |  39.4000 | 10.0000 |  64.0000 |
| time_respecting                |  25.5000 |  23.9688 | 12.0000 |  36.0000 |  8.0000 |  37.0000 |

Rendered block length, against the same measure at budget 800 in G0c:

| arm                            | measure         |    median |      mean |      p10 |       p90 |      min |       max |   median_at_800 |   growth_factor |
|:-------------------------------|:----------------|----------:|----------:|---------:|----------:|---------:|----------:|----------------:|----------------:|
| event_sample_then_full_history | characters      | 1384.5000 | 1566.8125 | 664.0000 | 2564.3000 | 624.0000 | 3629.0000 |        622.5000 |          2.2241 |
| event_sample_then_full_history | utf8_bytes      | 1384.5000 | 1566.8125 | 664.0000 | 2564.3000 | 624.0000 | 3629.0000 |        622.5000 |          2.2241 |
| event_sample_then_full_history | portable_tokens |  460.0000 |  527.0000 | 154.0000 |  957.6000 | 138.0000 | 1386.0000 |        142.0000 |          3.2394 |
| node_panel_full_history        | characters      | 1138.5000 | 1117.1562 | 689.1000 | 1578.3000 | 616.0000 | 1797.0000 |        765.5000 |          1.4873 |
| node_panel_full_history        | utf8_bytes      | 1138.5000 | 1117.1562 | 689.1000 | 1578.3000 | 616.0000 | 1797.0000 |        765.5000 |          1.4873 |
| node_panel_full_history        | portable_tokens |  362.0000 |  353.7500 | 172.0000 |  562.4000 | 138.0000 |  650.0000 |        206.0000 |          1.7573 |

**The input does grow, and far more slowly than coverage does.** Arm B's dyad
coverage rose roughly seventeenfold (0.0031 -> 0.0544) while its median
rendered block grew **3.2x**; arm A's coverage rose fourfold while its block
grew **1.8x**.  That gap is the saturation described above.  The three walk
arms are byte-identical to G0c, as they must be.

Absolute size is what decides whether this blocks.  Exact Qwen3.6 counts
cannot be recomputed here -- the tokenizer snapshot G0c used is not present in
this environment -- so they are calibrated from G0c's stored exact counts and
the same rendered texts.  The ratio of exact Qwen tokens to the
tokenizer-independent portable count is tight on those cases (median 1.664,
p10 1.579, p90 1.725), so the conversion below is good to roughly +/-5% and is
an estimate, not a measurement:

| arm                            |   calibration_ratio |   median |   mean |   p10 |   p90 |   min |   max |
|:-------------------------------|--------------------:|---------:|-------:|------:|------:|------:|------:|
| event_sample_then_full_history |              1.6637 |      765 |    877 |   256 |  1593 |   230 |  2306 |
| node_panel_full_history        |              1.6637 |      602 |    589 |   286 |   936 |   230 |  1081 |
| recent_history_k20             |              1.6637 |      423 |    393 |   244 |   544 |   203 |   622 |
| time_agnostic_t                |              1.6637 |      270 |    302 |   236 |   419 |   223 |   582 |
| time_respecting                |              1.6637 |      326 |    316 |   236 |   396 |   210 |   403 |

**Verdict: not blocking.** The largest input in the panel is about 2,300
tokens on arm B, against 268-432 median for the walks at 800.  That is a 2-3x
input, not the order of magnitude that would break the token-limited models:
the historical V2.1 open-weights runs were truncated on the *output* side at
8,192 tokens, and an input of this size is far inside every context window in
the model matrix.  The growth is real and belongs in the G1 length-band
accounting, where the sections must be matched across arms, but it does not
threaten the run.

One consequence does carry into G1: arm B's block is now the longest in the
suite and its spread is the widest (p90 about 1,600 estimated tokens against
about 400 for `time_respecting`).  G1.5 asks for the same length band across
arms for the *prose* sections, and that is still achievable, but the data
block cannot be equalized without changing the input contract.  Report the
data-block length per arm alongside the prose length rather than claiming one
band for the whole prompt.

### Arm B at the G0c budget of 10,500, for comparison

Same eight seed slots, same rule, same panel:

| arm                            |   target_budget |   median_dyad_coverage |   coverage_target |   group_macro_bias_mean |   group_macro_bias_sd |   seed_advances |   empty_draws |
|:-------------------------------|----------------:|-----------------------:|------------------:|------------------------:|----------------------:|----------------:|--------------:|
| event_sample_then_full_history |           10500 |                 0.0610 |            0.0548 |                  0.1495 |                0.0034 |               0 |             0 |

It clears the bias gate as comfortably as 9,600 does and is equally free of
empty samples; it simply sits further above the walks on coverage.  If the
G0c budget is preferred for continuity, nothing in the acceptance decision
changes -- only the parity claim weakens.

## G0d.5 — The mismatch pair at the new budgets: a blocking finding

Group-macro ProfileMAE (rows produce the sample; columns supply the assumed
arm likelihood).  As in G0/G0b/G0c this matrix is label-assisted: each
observation model is fitted from true dyad labels on the benchmark population
with the whole matching group held out.  It is a diagnostic ceiling, not a
deployable estimator.

| sample_arm                     |   event_sample_then_full_history |   node_panel_full_history |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:-------------------------------|---------------------------------:|--------------------------:|---------------------:|------------------:|------------------:|
| event_sample_then_full_history |                           0.0892 |                    0.1041 |               0.4669 |            0.4891 |            0.2981 |
| node_panel_full_history        |                           0.0206 |                    0.0209 |               0.3791 |            0.3989 |            0.2177 |
| recent_history_k20             |                           0.1503 |                    0.1469 |               0.0983 |            0.1059 |            0.1229 |
| time_agnostic_t                |                           0.1568 |                    0.1547 |               0.0972 |            0.0874 |            0.1159 |
| time_respecting                |                           0.1578 |                    0.1559 |               0.1318 |            0.1140 |            0.0925 |

The same matrix as signed `rho_2` bias:

| sample_arm                     |   event_sample_then_full_history |   node_panel_full_history |   recent_history_k20 |   time_agnostic_t |   time_respecting |
|:-------------------------------|---------------------------------:|--------------------------:|---------------------:|------------------:|------------------:|
| event_sample_then_full_history |                           0.1378 |                    0.1702 |               0.5123 |            0.5291 |            0.4076 |
| node_panel_full_history        |                          -0.0198 |                    0.0175 |               0.4576 |            0.4707 |            0.3261 |
| recent_history_k20             |                          -0.2897 |                   -0.2792 |              -0.0849 |           -0.0717 |           -0.1976 |
| time_agnostic_t                |                          -0.3103 |                   -0.3036 |              -0.1468 |           -0.1171 |           -0.2034 |
| time_respecting                |                          -0.3116 |                   -0.3046 |              -0.1986 |           -0.1497 |           -0.0833 |

Measured bias classes used for pair eligibility (`none` = eight-slot naive
bias within +/-0.05):

| arm                            |   target_budget |   naive_group_macro_rho2_bias | required_correction   |
|:-------------------------------|----------------:|------------------------------:|:----------------------|
| time_agnostic_t                |             800 |                       -0.3085 | upward                |
| time_respecting                |             800 |                       -0.3098 | upward                |
| recent_history_k20             |             800 |                       -0.2868 | upward                |
| node_panel_full_history        |            2500 |                       -0.0053 | none                  |
| event_sample_then_full_history |            9600 |                        0.1512 | downward              |

All candidate pairs.  Observable distinguishability uses only
`(n,mask)`-derived columns -- all of them shares, not raw counts -- with the
entire graph group held out:

| arm_a                   | arm_b                          | correction_a   | correction_b   |   median_profile_mae_penalty |   median_abs_rho2_bias_shift |   min_abs_rho2_bias_shift |   max_abs_rho2_bias_shift |   observable_auc_logo |   observable_distance_2auc_minus1 |   observable_balanced_accuracy |   observable_feature_count |
|:------------------------|:-------------------------------|:---------------|:---------------|-----------------------------:|-----------------------------:|--------------------------:|--------------------------:|----------------------:|----------------------------------:|-------------------------------:|---------------------------:|
| time_agnostic_t         | node_panel_full_history        | upward         | none           |                       0.2227 |                       0.3198 |                    0.1864 |                    0.4532 |                0.8760 |                            0.7520 |                         0.8906 |                        447 |
| time_agnostic_t         | event_sample_then_full_history | upward         | downward       |                       0.2346 |                       0.2922 |                    0.1931 |                    0.3913 |                0.8584 |                            0.7168 |                         0.9062 |                        447 |
| time_respecting         | node_panel_full_history        | upward         | none           |                       0.1301 |                       0.2650 |                    0.2214 |                    0.3086 |                1.0000 |                            1.0000 |                         0.9844 |                        447 |
| time_respecting         | event_sample_then_full_history | upward         | downward       |                       0.1371 |                       0.2491 |                    0.2284 |                    0.2698 |                0.9951 |                            0.9902 |                         0.9531 |                        447 |
| recent_history_k20      | node_panel_full_history        | upward         | none           |                       0.2034 |                       0.3172 |                    0.1943 |                    0.4401 |                0.8369 |                            0.6738 |                         0.8281 |                        447 |
| recent_history_k20      | event_sample_then_full_history | upward         | downward       |                       0.2149 |                       0.2896 |                    0.2048 |                    0.3745 |                0.9102 |                            0.8203 |                         0.8594 |                        447 |
| node_panel_full_history | event_sample_then_full_history | none           | downward       |                       0.0073 |                       0.0349 |                    0.0324 |                    0.0373 |                0.6445 |                            0.2891 |                         0.6094 |                        447 |

`pair_summary` treats `none` as its own correction class, so it admits
`upward` <-> `none` pairs.  G0d.5 asks for a pair whose arms need corrections
in different *directions*, and an arm needing no correction has no direction,
so only these are eligible:

| arm_a              | arm_b                          | correction_a   | correction_b   |   median_profile_mae_penalty |   median_abs_rho2_bias_shift |   min_abs_rho2_bias_shift |   max_abs_rho2_bias_shift |   observable_auc_logo |   observable_distance_2auc_minus1 |   observable_balanced_accuracy |   observable_feature_count |
|:-------------------|:-------------------------------|:---------------|:---------------|-----------------------------:|-----------------------------:|--------------------------:|--------------------------:|----------------------:|----------------------------------:|-------------------------------:|---------------------------:|
| time_agnostic_t    | event_sample_then_full_history | upward         | downward       |                       0.2346 |                       0.2922 |                    0.1931 |                    0.3913 |                0.8584 |                            0.7168 |                         0.9062 |                        447 |
| time_respecting    | event_sample_then_full_history | upward         | downward       |                       0.1371 |                       0.2491 |                    0.2284 |                    0.2698 |                0.9951 |                            0.9902 |                         0.9531 |                        447 |
| recent_history_k20 | event_sample_then_full_history | upward         | downward       |                       0.2149 |                       0.2896 |                    0.2048 |                    0.3745 |                0.9102 |                            0.8203 |                         0.8594 |                        447 |

The best eligible pair by bias penalty is **`time_agnostic_t` <->
`event_sample_then_full_history`**, median absolute `rho_2`-bias shift
**0.2922**, held-group-out observable
AUC **0.8584**.

### This does not clear the bar, and the re-budget is why

The requirement was that the chosen pair's AUC stay **well below 1.0**, so that
a model cannot detect the mismatch from the sample statistics alone.  It does
not.  At **0.8584** a held-group-out logistic
classifier reading only the mask histogram tells the two arms apart most of the
time, and one ineligible pair (`time_respecting` <-> `node_panel_full_history`)
is now at AUC 1.0000, i.e. perfectly separable.

The re-budget caused this.  On the identical pair, G0c measured AUC
**0.6660** with both arms at 800 events; here the same pair sits at
**0.9102**.  Nothing about the arms' definitions changed -- only arm B's
budget.  The mechanism is visible in the input: with complete histories at
9,600 events, arm B's `n` distribution is the true per-dyad event count over
611 dyads, while a walk at 800 steps sees each dyad once or twice.  Those are
different-shaped histograms, and the classifier reads the shape, not the size.

**Bias penalty and observable detectability are now positively coupled**, which
is the worst possible arrangement: the pairs worth contrasting are the ones a
model can most easily tell apart without reading the mechanism text.  Across
all seven candidate pairs the only one under AUC 0.75 is
`node_panel_full_history` <-> `event_sample_then_full_history` at
**0.6445** -- and its bias penalty is
**0.0349**, an order of magnitude
below the others, because both of its arms return complete histories and
neither needs an upward correction.  It is ineligible in any case: arm A needs
no correction at all, so the pair has no opposing directions to contrast.
Among the three eligible pairs the *lowest* AUC is
**0.8584**.

### What this means for `mismatched`

This is a G1 design decision, not something G0d should settle by picking the
least-bad number.  The options, with what each costs:

1. **Demote `mismatched` to exploratory.** Keep it, run it, and state up front
   that a specificity effect on this pair has a live rival explanation -- the
   model may be reacting to a sample that does not look like the described
   process rather than to the mechanism text.  Cheapest, and honest, but the
   condition stops being able to support the claim it was added for.
2. **Take the AUC as a measured covariate.** Run the pair, and report the
   effect against the per-case detectability rather than pooled.  If the effect
   is flat in detectability, the rival explanation is weakened empirically
   instead of by design.  Costs nothing extra to run and is the strongest
   version of option 1.
3. **Add a sixth configuration purely for the mismatch contrast**, with the two
   arms event-matched so their histograms are the same shape.  This restores
   the G0c AUC but breaks the within-case pairing that the primary contrast
   depends on, because `mismatched` would then run on a different sample from
   `hidden` and `mechanism` for the same case.  Expensive and it damages the
   primary design.
4. **Drop `mismatched`.** The 2x2 over {process described} x {direction
   stated} plus `metadata_only` still answers the main question.  `mismatched`
   was always the specificity check, not the effect.

Recommendation: **option 2**, falling back to option 1 if the detectability
covariate turns out to have no spread.  Option 3 should not be taken -- the
within-case pairing is worth more than this one condition.  None of this is
G0d's to decide; it is recorded here because G1 cannot write the `mismatched`
text without choosing.

## Final arm configuration

| arm                            |   budget_unique_events | natural_unit    |   median_natural_units |   median_dyad_coverage |   coverage_parity_target |   rho2_naive_bias |   expected_empty_rate_per_draw |   observed_empty_rate_per_draw |   empty_rate_after_seed_rule |   median_prompt_tokens_portable |
|:-------------------------------|-----------------------:|:----------------|-----------------------:|-----------------------:|-------------------------:|------------------:|-------------------------------:|-------------------------------:|-----------------------------:|--------------------------------:|
| time_agnostic_t                |                    800 | walk steps      |               800.0000 |                 0.0901 |                   0.0548 |           -0.3085 |                         0.0000 |                         0.0000 |                       0.0000 |                        162.0000 |
| time_respecting                |                    800 | walk steps      |               800.0000 |                 0.0518 |                   0.0548 |           -0.3098 |                         0.0000 |                         0.0000 |                       0.0000 |                        196.0000 |
| recent_history_k20             |                    800 | walk steps      |               800.0000 |                 0.0540 |                   0.0548 |           -0.2868 |                         0.0000 |                         0.0000 |                       0.0000 |                        254.0000 |
| node_panel_full_history        |                   2500 | nodes recruited |                41.0000 |                 0.0575 |                   0.0548 |           -0.0053 |                         0.0198 |                         0.0176 |                       0.0000 |                        362.0000 |
| event_sample_then_full_history |                   9600 | dyads looked up |               611.5000 |                 0.0544 |                   0.0548 |            0.1512 |                         0.0000 |                         0.0000 |                       0.0000 |                        460.0000 |

Three empty-rate columns rather than one, because they answer different
questions.  `expected_empty_rate_per_draw` is the closed-form probability
averaged over the 32 graphs; `observed_empty_rate_per_draw` is what the 16
drawn seed slots actually produced; `empty_rate_after_seed_rule` is what
survives into the analysis, and it must be zero or the rule has failed.  The
walks are exempt: a walk always returns a sample.

Prompt tokens are the tokenizer-independent portable count, so the column is
comparable across rows; the calibrated exact-tokenizer figures are in G0d.4.

Replay verification: 5960/5960 benchmark cases reproduced
their stored `(n,mask)` histogram exactly.  All new artifacts live below
`results/g0d_headroom_2026_09`; no frozen benchmark case, panel truth, walk artifact or LLM
artifact was modified.

## What remains uncertain

- Coverage parity matches one scalar.  Node coverage, temporal coverage, dyad
  multiplicity and sample coherence remain different across arms, deliberately
  so.  G4 must state that the arms are not information-matched.
- Arm A's empty cases are handled by the seed rule, not eliminated.  The
  accepted sample for an affected case is drawn from a seed index further
  along the sequence than its neighbours; the rule is deterministic and
  logged, but it is a conditioning on non-emptiness and should be named as
  such in the write-up.
- The wrong-mechanism matrix uses labels and collapses the exact sample to
  mask frequencies.  It bounds nothing.
- No language model has been tested here.  Nothing in this report says whether
  a model can operationalize a mechanism description.

Arms and budgets for the main run:

- `node_panel_full_history` (arm A): **2,500** unique events
- `event_sample_then_full_history` (arm B): **9,600** unique events
- `time_agnostic_t`, `time_respecting`, `recent_history_k20`: **800**
  walk steps, unchanged.
