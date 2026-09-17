# Handoff: design `cells10-20260917`

Purpose: anyone (a person or another model) can see where the run stands and
continue it without the session that started it. Specification: last section of
[MAIN_EXPERIMENT_IMPLEMENTATION.md](MAIN_EXPERIMENT_IMPLEMENTATION.md).

## Scope, fixed by the user

* Arms matched on expected observed active dyad-windows (10 % of sum_e K_e), one level.
* Qwen final answers constrained to `common.ANSWER_REGEX`; non-thinking is a direct estimate.
* Everything regenerated; Sol and DeepSeek stay disabled; no additional experiments.
* The Word overview is updated only after the results exist.

## Steps and how to tell whether each is done

| # | Step | Command | Done when |
|---|---|---|---|
| 1 | Spec committed and pushed | `git log` | commit "spec: match the arms on active dyad-windows …" exists |
| 2 | Offline chain | `nohup bash scripts/run_cells10_offline.sh > results/cells10_offline_logs/driver.log 2>&1 &` | `results/cells10_offline_logs/STATUS` starts with `ALL_OFFLINE_DONE`; rerun the same command after any interruption |
| 3 | Upload to the cluster | `bash scripts/cluster_bundle.sh cells10 results/main_experiment/cells10_20260917` | prints `BUNDLE_OK` |
| 4 | Smoke test (8 requests, never evaluated) | on uc3: `cd $WS/cells10/mainexp && sbatch --partition=gpu_h100_short,gpu_h100 --time=00:30:00 --array=0 qwen_engine.sbatch cells10 6 16 all 8 answers_smoke 900 1500` | `python status.py . answers_smoke`: every final text matches the regex, thinking answers have `reasoning_closed`, `engine_vs_own_input_tokens_mismatch` is 0 |
| 5 | Production chain | on uc3: `cd $WS/cells10/mainexp && bash submit_production.sh cells10 6` | `production_jobs.txt` lists four job ids; the chain finishes on its own |
| 6 | Monitoring | on uc3: `squeue -u $USER`; `python status.py .` | `status_final.json` exists with `"complete": true`; `archive/ARCHIVE_REPORT.json` has no read-back mismatches |
| 7 | Collect and evaluate | `bash scripts/finish_cells10.sh` | prints `FINISH_DONE` |
| 8 | Report and docs | write `results/baseline_revision_cells10_20260917/REVISION_REPORT.md`, update README, commit, push | — |
| 9 | Word overview | targeted edits of `~/Downloads/Masterarbeit_Studienuebersicht_20260917.docx` | — |

`$WS` is `/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot` (expires 2026-10-24).
Cluster commands run in a login shell (`bash -l`); Python on the login node needs
`module load devel/python/3.12.3-gnu-14.2; source $WS/venv_mainexp/bin/activate`.
ssh to `uc3` needs an interactive OTP: the user opens a ControlMaster (`ssh uc3`),
after which scripted `ssh`/`scp`/`rsync` calls work for up to eight idle hours. Keep
to one polling loop at a time; the server refuses extra multiplexed sessions.

## If something fails

* Offline step fails: read `results/cells10_offline_logs/<step>.log`, fix, rerun step 2.
  A completed step is skipped; a changed dependency makes the pipeline refuse to
  resume, which is intended (use a new output directory or delete that step's
  output together with its `.done` marker).
* Smoke test shows unconstrained output or an engine error about structured
  outputs: do not start production. Fall back only after writing the decision into
  the specification (`--no-structured` exists but changes the protocol).
* Production jobs cut by the wall time: rounds 2 and 3 resume automatically. If
  answers are still missing afterwards, resubmit `submit_production.sh`; finished
  answers are never regenerated.
* A single answer that ends at the output limit is kept and evaluated under the
  fixed replacement rule; it is not retried.

## State log

Newest last. Each entry: date, what was done, what is next.

* 2026-09-17: design implemented and unit-tested; next: commit spec, run step 2.
* 2026-09-17 12:55: spec committed (d225bff) and pushed; offline chain started
  detached (`results/cells10_offline_logs/`). A structured-output probe (job
  7011649, 8 requests of the previous design's observations, directory
  `$WS/cells10_probe`, never evaluated) checks the answer constraint before any
  production data exist.
* 2026-09-17 13:10: probe 7011649 passed: 8/8 answers match the regex, the 5
  non-thinking answers are the bare object (44-46 tokens), the 2 thinking answers
  reason freely and emit the object right after `</think>`, engine and runner
  token counts agree. Offline step `offline` done: 280 main / 320 training
  observations, all 24 graphs matched on every arm, no saturated H, prompts at
  most 1 429 tokens.
* 2026-09-17 13:12: the first offline pass wrote fold manifests under the stale
  training label `baseline-revision-3-hrecent5-20260917`; the label was fixed
  (commit "fix: training revision label for cells10"), the pass discarded (logs kept
  in `results/cells10_offline_logs/first_pass/`) and the chain restarted; the pool
  (label-independent) was kept. In the first pass the 20-draw H check showed one
  |z| = 2.25 and one SD ratio 1.43; 400-draw rechecks of those graphs gave means
  within 1.4 MCSE of zero and SD ratios 0.95-1.04, i.e. chance.
  Step 4 (smoke test in `$WS/cells10`) is replaced by the probe above, which ran
  the same runner and job script; production is watched during its first minutes.
* 2026-09-17 13:28: offline chain rerun complete (ALL_OFFLINE_DONE, 115 tests),
  results committed (84ae6d1) and pushed; bundle uploaded to `$WS/cells10`
  (BUNDLE_OK, SPEC_COMMIT 84ae6d1). Production chain submitted:
  round1=7012311, round2=7012312, round3=7012314, archive=7012315 (6 shards,
  `$WS/cells10/mainexp/production_jobs.txt`). Next: wait for the archive job to
  finish, then `bash scripts/finish_cells10.sh`, then report and Word overview.
* 2026-09-17 13:36: round 1 shards 0-2 finished (all 840 non-thinking answers,
  840/840 valid under parser v2, median 46 output tokens). Sharding by sorted
  request id puts each mode in its own shards, so shards 3-5 carry all thinking
  requests; they started at about 13:34 and are expected to finish within their
  two-hour limit, otherwise rounds 2-3 resume them.
* 2026-09-17 14:55: production chain complete (1 680/1 680, all regular ends,
  round 1 sufficed, rounds 2-3 empty, archive 1 990 files without mismatches);
  `scripts/finish_cells10.sh` done (FINISH_DONE); report written in
  `results/baseline_revision_cells10_20260917/REVISION_REPORT.md`. Remaining: the
  Word overview (step 9).
