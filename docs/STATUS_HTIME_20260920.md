# Status: cells10-htime60-20260920

Offline implementation and computation completed on 2026-09-20. New Qwen H
generation is running; its retrieval and final evaluation remain outstanding.
The previous current state is preserved under `archive/pre_h_time_20260920/`;
its local bulk results were moved without changing their contents.

## Completed

- Timestamp-based uniform-node H, primary h=0.60, generic JSON input/output.
- 280 main observations, 320 real training observations; no empty main inputs.
- All main-source H coverage targets reachable and matched within tolerance;
  five draws per source, no saturated H panels.
- 500 unchanged pool graphs reproduced; new H observations for all of them.
- 7 pipeline control fits and 14 pooled/real-only reference folds refitted with
  134 features. Hyperparameters, weighting, LOSO and graph labels unchanged.
- 450 real/main R/S/B observation blocks and 7500 pool R/S/B blocks verified
  unchanged. 560 learned-reference predictions independently recomputed.
- 1260 previous Qwen R/S/B answers copied byte-for-byte after full request,
  seed, payload, prompt, runner and generation-binding checks. Old H excluded.
- 108 tests passed, no skips; main offline, revision and mock audits passed.
- GPT/Sol and DeepSeek ledgers empty; no API generation started.

The inherited pool stage report still labels its pool as a "new pool version";
that descriptive field predates this revision and is not the provenance claim
for this run. The pool definition/seeds are unchanged, as verified explicitly
by `verify_htime_revision.py` and the [revision audit](results/htime_20260920/revision_audit.json).

## h sensitivity

H homogeneous ZT-Binomial reference, fixed-panel equal-source means:

| h | Real MAE2 | Real ProfileMAE | Synthetic MAE2 | Synthetic ProfileMAE |
| --- | ---: | ---: | ---: | ---: |
| 0.40 | 0.183558 | 0.083670 | 0.095799 | 0.086515 |
| 0.60 | 0.149511 | 0.061015 | 0.090043 | 0.067902 |
| 0.80 | 0.101384 | 0.045721 | 0.078846 | 0.056229 |

This is a working-model reference, not an unbiased estimator. These results do
not change the primary choice h=0.60. Plugin, median and both ExtraTrees variants
are included in the [sensitivity report](results/htime_20260920/sensitivity.json).

Descriptive net-history share of the sum of absolute selection/history components:

| h | Real rho2 | Real profile | Synthetic rho2 | Synthetic profile |
| --- | ---: | ---: | ---: | ---: |
| 0.40 | 89.52% | 90.68% | 93.81% | 95.63% |
| 0.60 | 70.50% | 81.33% | 86.35% | 91.59% |
| 0.80 | 44.27% | 65.08% | 71.75% | 82.64% |

At primary h=0.60 net history exceeds node selection in 5/6 real and 8/8
synthetic sources, for both rho2 and profile absolute components. Thus H is
history-dominated in these panel averages, not universally per source. At h=0.80
the real-panel rho2 comparison no longer supports average history domination.
The decomposition includes entirely history-invisible dyads; it does not
misattribute their disappearance to node selection. Shares are descriptive,
not additive fractions of total absolute error or generalization claims.

## Pending completion

420 new primary Qwen H answers are planned: 210 thinking, 210 non-thinking.
Pre-generation freeze: `e1b25a69d010db7c66150855c851a3c9c43a5b1e`.
The bundle was verified on uc3 (`BUNDLE_OK`, 280 observations), then submitted:

| Stage | Slurm job |
| --- | --- |
| Round 1 (2 shards) | 7065832 |
| Round 2 | 7065833 |
| Round 3 | 7065834 |
| Archive | 7065835 |

At handoff both round-1 shards were RUNNING; successors waited on dependencies.
Initial H status: planned/found/valid/invalid/missing = **420/0/0/0/420**.
These are uncompleted requests, not observed model failures. Do not submit a
second chain. This job chain continues independently of the interactive session.

Finish the independent H cluster chain, retrieve and verify its archive, merge
only its H files with the proven R/S/B collection, and run the current evaluator
against the newly trained references. The whole Qwen study then has 1680
planned answers. Until retrieval, no final H found/valid/invalid count is claimed.

Commands and paths: [runbook](RUNBOOK_HTIME_20260920.md). The local offline chain
has already completed; do not start it from scratch or change immutable bindings.
Run `scripts/verify_htime_revision.py` for the revision audit and
`scripts/analyze_htime.py` for the reproducible offline sensitivity.
