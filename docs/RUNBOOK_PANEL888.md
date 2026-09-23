# Runbook

**v10 status (2026-09-23):** the [amended interaction-walk gate](results/panel888_v10_walk_gate_20260923/WALK_GATE.md)
permits production, with sp_hospital__pwt flagged as not correctable at the 10%
budget. Jobs 7143567 and 7143568 produced the audit and confirmation evidence;
7143619 prepared v10 observations and 7143620 builds the synthetic pool.
The steps below describe the sealed v9 predecessor.

One offline entry point, one Qwen production path, one integration path.
The scientific design is in [PROTOCOL_PANEL888_20260921.md](PROTOCOL_PANEL888_20260921.md).

## 1. Offline study (no model calls, about 25 minutes; on uc3: `cluster/offline_study.sbatch`)

Requirements: the pinned `.venv`, raw sources in `data/raw/`, tokenizers in
`data/tokenizers/` (`scripts/fetch_tokenizers.py`), and `g++` for the walk kernel.

```
bash scripts/run_offline.sh
```

Refuses to start if `results/panel888/` exists; every run recomputes from scratch.
Steps and outputs (logs in `results/panel888/logs/`):

| Step | Script | Output in `results/panel888/` |
|---|---|---|
| graphs, surrogates, budgets, observations, prompts, requests | `prepare_study.py` | `prepared/` |
| synthetic training/development pool | `build_references.py pool` | `references/pool/` |
| 9 LOSO folds x {pooled, real_only} ExtraTrees | `build_references.py train` | `references/models_*/` |
| development check and main reference predictions | `build_references.py references` | `references/primary_baselines.json` |
| oracle selection/history decomposition | `diagnose_decomposition.py` | `diagnostics/decomposition/` |
| H at h = .40/.60/.80 | `diagnose_history.py` | `diagnostics/history/` |
| S1/S2 walk construct validity, 1000 walks at L and 4L | `diagnose_walk.py` | `diagnostics/walk/` |
| 99 offline P[w,t] null shuffles per parent | `diagnose_null_model.py` | `diagnostics/null_model/` |
| W = 2..20, {4,5,8}, thresholds, census descriptors | `diagnose_windows.py` | `diagnostics/windows/` |
| B mixture bound widening | `diagnose_mixture_bounds.py` | `diagnostics/mixture_bounds/` |
| Sol/DeepSeek ledger (dispatch disabled) | `run_api.py prepare` | `api/` |
| paired controls of the references | `evaluate_paired_controls.py` | `paired_control_references/` |
| evaluator check with fake answers | `check_evaluation_mocks.py` | `audit/mock_evaluation/` |
| independent audit | `audit_offline.py` | `audit/offline_audit.json` |
| unit tests | `unittest discover -s tests` | `logs/tests.log` |
| seal | `seal_offline.py` | `docs/results/panel888_offline/` |

## 2. Freeze

Commit the sources and `docs/results/panel888_offline/`, push to `origin/master`,
and record the commit SHA. The working tree must be clean.

## 3. Qwen production (cluster uc3; needs the open ssh ControlMaster)

Before anything: `squeue -u $USER`, and check that `$WS/panel888_final_main` does not
exist yet (or reconcile its `mainexp/production_jobs.txt`; never submit twice).

```
bash scripts/cluster_bundle.sh panel888_final_main          # verifies the seal against HEAD
ssh uc3 'cd /pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/panel888_final_main/mainexp && \
         bash submit_production.sh panel888_final_main 10 all <commit>'
```

Answers to requests whose request record is byte-identical to one of an earlier
production workspace may be copied into `mainexp/answers/` before submission (with
`answers/REUSED_ANSWERS.json` listing each file; `cluster/reuse_answers.py`); the runner then admits only the
remaining requests and the archive verification checks every answer's ID, prompt,
payload, seed and runner hash. Answers identical under this rule are listed in
the archive's `answers/REUSED_ANSWERS.json`.

`submit_production.sh` verifies the bundle checksums, the commit and the clean
source status, takes a lock, refuses if a chain was ever submitted, and writes each
job ID to `production_jobs.txt` immediately: three rounds of a sharded GPU array
(later rounds only admit never-started requests; no answer is regenerated) and a
CPU archive job. Monitor with `squeue` and `python status.py . answers`.

## 4. Integration (local, after the archive job has finished)

```
bash scripts/integrate_qwen.sh <commit>
```

Downloads the archive, verifies checksums, request/observation identity, runner,
engine and model identity (`verify_qwen_archive.py`), collects one answer per
request (`collect_qwen_answers.py`), evaluates validity and conditional accuracy
(`evaluate_responses.py`), the paired temporal control (`evaluate_paired_controls.py`),
re-parses every answer independently (`audit_qwen.py`) and copies the compact
evidence to `docs/results/panel888_qwen/` (`report_qwen.py`).

## Sol and DeepSeek

Their 864 requests each are prepared in `prepared/requests.jsonl` and a disabled
ledger in `api/`. Dispatch requires an explicit technical release
(`execution.validate_release`) and is not part of this run.
