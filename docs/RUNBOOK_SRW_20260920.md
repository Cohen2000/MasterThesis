# Runbook: cells10-srw-htime60-20260920

[Protocol](PROTOCOL_SRW_20260920.md). Old event-weighted code, configuration,
tests, docs and results: `archive/pre_srw_20260920/` (source `96214c0`).

```bash
bash scripts/run_srw_offline.sh
.venv/bin/python scripts/analyze_srw.py
.venv/bin/python scripts/verify_srw_revision.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

This copies only verified graphs. All SRW calibration checkpoints, S observations,
134-feature vectors, 7 control fits and 14 learned-reference folds are new. Pool
graph definitions/seeds and R/B/H observation blocks must remain identical.
The API prepare action is offline and leaves its ledger empty and unauthorized.

Qwen workspace: `uc3:/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_srw_htime60_20260920/mainexp`.
Create and verify the bundle after the pre-generation commit, then submit only S:

```bash
bash scripts/cluster_bundle.sh cells10_srw_htime60_20260920 results/main_experiment/cells10_srw_htime60_20260920
ssh uc3 'cd /pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_srw_htime60_20260920/mainexp && test ! -e production_jobs.txt && bash submit_production.sh cells10_srw_htime60_20260920 6 S'
```

Never repeat an already-submitted chain. Follow-up rounds only admit unstarted
requests; a begun attempt is never regenerated. H's separate existing chain is
`7065832/7065833/7065834/7065835`. Retrieve its finished archive into
`results/imported_h_20260920/`. The S archive goes into
`results/imported_srw_20260920/`; verify both original CHECKSUMS.json manifests.

Integration and collection use `scripts/integrate_srw_qwen.py`, which checks
complete generation identity and raw bytes for R/B/H and accepts only new S IDs.
All evaluation outputs are separate from the immutable offline preparation:

```bash
.venv/bin/python scripts/integrate_srw_qwen.py
.venv/bin/python scripts/collect_qwen_answers.py \
  --run results/main_experiment/cells10_srw_htime60_20260920 \
  --answers results/main_experiment/cells10_srw_htime60_20260920_qwen/answers \
  --out results/main_experiment/cells10_srw_htime60_20260920_qwen/responses.jsonl
.venv/bin/python scripts/evaluate_main_responses.py \
  --run results/main_experiment/cells10_srw_htime60_20260920 \
  --baselines results/baseline_revision_cells10_srw_htime60_20260920/primary_baselines.json \
  --responses results/main_experiment/cells10_srw_htime60_20260920_qwen/responses.jsonl \
  --out results/main_experiment/cells10_srw_htime60_20260920_qwen/evaluation
```

Unstarted API configurations keep complete_main_result=false even when Qwen
is fully finished. No extrapolation from incomplete Qwen cells is reported.
