# v11 extension report (2026-09-28)

Additive to design `panel888-access-v11-20260923`. No v11 observation, prompt, estimator, seed or result file was changed. Code: `src/v11_ext/`, `scripts/v11_ext.py`, `config/v11_ext.yaml`, `cluster/v11_ext_task.sbatch`.

## What ran

| Stage | Tasks | Content |
|---|---:|---|
| source | 4 | Parse, W=5 window check, canonical graph, 10% calibration of all arms, 3 main draws per arm (no H panel saturated), v11 R/H panel release, plugin/median/design/MLE, ET features |
| testset | 1 | 360 v11 main ET rows (frozen caches) plus 60 new-source rows |
| draw_real | 800 | Replicates 1-10: 16 real training sources x 5 arms x 5 draws, stream domain `training_r<k>` |
| draw_pool | 200 | Replicates 1-10: 400 pool-train graphs (20 per task) x 5 arms x 5 draws, domain `pool_train_r<k>` |
| select | 505 | Nested leave-one-real-training-source-out selection: replicate 0 only for the new `nr_radoslaw_email` fold, replicates 1-10 for all 10 folds x 5 arms (seed domain `v10_et_nested_r<k>`) |
| train | 550 | Final fits, 11 replicates x 5 arms x 10 folds (seed domain `v10_et_final_r<k>`) |
| qwen_bundle + Qwen | 1 + 2 rounds x 4 H100 shards | 60 observations x {thinking, non-thinking} x 3 repeats = 360 requests |
| report | 1 | Tables in this folder |

Orchestration: `python scripts/v11_ext.py --dry-run | --status | --submit [--attach]`. Every task output is stored under a key that hashes stage code, configuration, parameters, the frozen v10/v11 input manifests and upstream keys; finished keys are skipped. Tests: 114 passed, 7 skipped locally; 113 passed, 8 skipped on uc3 (skips need local raw data), run before submission.

## What was reused

- v11 observations, prompts and all v11 predictions (`panel888_v11_main_20260923/PREDICTIONS.csv`), unchanged.
- v10 graphs, calibrations and training draws of the 16 real sources; the 500-graph pool definition, budgets and draws; the v10 and R/H panel-release ET caches, choices and pickled models.
- nr_radoslaw_email: the v10 graph and calibration (already parsed as a training source).
- The v11 Qwen runner, sbatch, venv, model and decoding settings; all existing Qwen answers (none rerun).

## Checks

- Current code reproduces sealed v10 draws bit for bit: training draw 1 of every real source and arm, and pool-train draw 1 of all 400 pool graphs and arms (checked inside the draw tasks). Regenerated pool graphs match N, D, M and truth.
- v11 ET features recomputed from the released blocks: max |difference| 8.9e-16 (R), 0 elsewhere.
- Replicate 0: refitting the 45 v11 (arm, fold) models with the sealed choices and seeds gives max |difference| 1.7e-16 from the v11 ET predictions. The pickled production models themselves differ from their stored predictions by up to 1.1e-16, so the residual is floating-point summation order in parallel tree averaging. The sealed v11 values are kept for v11 rows (replicate 0 equals v11 exactly); the three new sources on the `synthetic` fold are predicted with the pickled production models.
- Real-8 summary recomputed from the long table equals the v11 summary (max |difference| 0.0).

## New sources

| Source | Records | Cleaning | N | D | M | rho_2 | Window shares | ET fold |
|---|---|---|---:|---:|---:|---:|---|---|
| reality_mining | 1,086,404 rows, weight 1 throughout | proximity: identical (u, v, t) records dropped as for copenhagen_bluetooth (78% of rows are same-timestamp duplicates) | 96 | 2,539 | 234,757 | 0.607 | .282 .348 .114 .157 .100 | 16-source (`synthetic`) |
| lkml_reply | 1,096,440 rows, weight 1 throughout | 68,207 self-replies removed; same-time duplicates kept as events (as for snap/nr sources) | 26,885 | 159,996 | 1,028,233 | 0.153 | .189 .207 .193 .189 .221 | 16-source (`synthetic`) |
| sp_malawi | 102,293 rows | already one row per 20 s contact (`contact_time` in multiples of 20 s); no onset/duration expansion needed | 86 | 347 | 102,293 | 0.507 | .227 .192 .148 .237 .197 | 16-source (`synthetic`) |
| nr_radoslaw_email | v10 graph | unchanged | 167 | 3,250 | 82,876 | 0.551 | .237 .199 .173 .195 .198 | own LOSO fold |

No window has less than 1% of the events, so the trimming rule (`config/v11_ext.yaml`) was not applied and no source is flagged. All five arms of all four sources are budget-matched within 5%. The netzschleuder weight column is 1 for every record, so one record is one event. Medians use the 16 training truths (15 for nr_radoslaw_email).

## Qwen on the new sources

360 of 360 requests completed with no technical failure; 355 valid. The 5 invalid answers are non-thinking S (2) and S_obs (3), all rejected for non-monotone profiles. Details: `QWEN_NEW_SOURCES.csv`.

## Wall time and compute

First job start 03:03:21, report finished 03:26:45 (uc3 time), about 24 minutes. The dry run estimated 1.7 h and about 410 CPU hours plus 4 GPU hours; 10 replicates were therefore kept (limit 12 h). Measured: about 232 CPU hours (selection 202, fits 13, draws 17) and about 1.1 H100 hours (round 1: 15-17 minutes per shard including a 6-minute model load; round 2 found nothing to do). Per-array times: `WALL_TIMES.csv`.

## Anomalies and deviations

- The first Qwen bundle job failed because the orchestrator had created the Qwen log directory before the bundle install; the install was made resumable and the bundle rebuilt (no generation had started). The report job was resubmitted twice (missing Qwen dependency, then a fix in the report code); tasks already finished were not rerun.
- Qwen waited a few minutes on the group CPU limit held by the selection arrays; pending selection elements were held until Qwen started, then released.
- Deviation: pool-dev observations were not redrawn, because v10 ET training uses pool-train rows only.
- Deviation: ET replicate statistics use all 11 replicates (production plus 10), since replicate 0 is a draw of the same procedure.
- Paid API models were not run on the new sources; GPT repeat SDs refer to the v11 R/S/H/B observations only.
- Fifteen stale v10-era jobs pending with `DependencyNeverSatisfied` were already in the queue and were left untouched.
