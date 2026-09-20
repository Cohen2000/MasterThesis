# Execution: cells10-htime60-20260920

Scientific definitions: [protocol](PROTOCOL_HTIME_20260920.md).
Old code, docs, tests and outputs: `archive/pre_h_time_20260920/` (source commit
`08f1e5088e0d2378a8de917758b369f861f30600`). Old production artifacts are immutable.

```bash
bash scripts/run_htime_offline.sh
.venv/bin/python scripts/analyze_htime.py
.venv/bin/python scripts/reuse_htime_qwen.py
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

The first script seeds only unchanged graph and walk checkpoints, then prepares
new observations, pool observations, models, baselines and an unauthorized API
ledger. Reusing R/S/B Qwen verifies full input/generation identity and copies raw
bytes with a reuse manifest. No old H answer is admitted.

New cluster workspace:
`/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_htime60_20260920/mainexp`.
After offline verification and the pre-generation commit:

**Already executed:** freeze `e1b25a6`, rounds `7065832/7065833/7065834`,
archive `7065835`. The following submission commands document the execution;
do not run them again for this workspace.

```bash
bash scripts/cluster_bundle.sh cells10_htime60_20260920 results/main_experiment/cells10_htime60_20260920
ssh uc3 'cd /pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/cells10_htime60_20260920/mainexp && bash submit_production.sh cells10_htime60_20260920 2 H'
```

Two shards cover the 420 primary H requests; every generation has one attempt.
Later rounds resume only unbegun requests. Do not resubmit this chain when it
already exists. No GPT/Sol or DeepSeek calls, including technical smokes.

After the archive job completes, copy its archive into a separate local
`results/main_experiment/cells10_htime60_20260920_qwen_h_archive/`, verify every
checksum, then copy its H answer and attempt files into the combined collection
`results/main_experiment/cells10_htime60_20260920_qwen/answers/`.
Keep the new H engine binding in its own archive; it must not replace the old
binding recorded by the R/S/B reuse manifest.

Read-only status:

```bash
ssh uc3 'squeue -j 7065832,7065833,7065834,7065835; sacct -j 7065832,7065833,7065834,7065835 --format=JobID,State,ExitCode,Elapsed -P'
```

```bash
.venv/bin/python scripts/collect_qwen_answers.py \
  --run results/main_experiment/cells10_htime60_20260920 \
  --answers results/main_experiment/cells10_htime60_20260920_qwen/answers \
  --out results/main_experiment/cells10_htime60_20260920_qwen/responses.jsonl
.venv/bin/python scripts/evaluate_main_responses.py \
  --run results/main_experiment/cells10_htime60_20260920 \
  --baselines results/baseline_revision_cells10_htime60_20260920/primary_baselines.json \
  --responses results/main_experiment/cells10_htime60_20260920_qwen/responses.jsonl \
  --out results/main_experiment/cells10_htime60_20260920_qwen/evaluation
```

Report Qwen completion separately from the four-configuration main experiment,
which remains incomplete while the two API configurations are unstarted.
