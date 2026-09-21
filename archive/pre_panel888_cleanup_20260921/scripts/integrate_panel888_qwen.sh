#!/bin/bash
# Run only after the single production archive job has completed.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP=panel888_pwt_srw_20260921
RUN=results/main_experiment/$EXP
REV=results/baseline_revision_$EXP
IMPORTED=results/imported_$EXP
QWEN=${RUN}_qwen
mkdir -p "$IMPORTED" "$QWEN"
rsync -a "uc3:/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/$EXP/mainexp/archive/" "$IMPORTED/"
.venv/bin/python scripts/verify_panel888_archive.py --archive "$IMPORTED"
.venv/bin/python scripts/collect_qwen_answers.py --run "$RUN" --answers "$IMPORTED/answers" --out "$QWEN/responses.jsonl"
.venv/bin/python scripts/evaluate_main_responses.py --run "$RUN" --baselines "$REV/primary_baselines.json" --responses "$QWEN/responses.jsonl" --out "$QWEN/evaluation"
.venv/bin/python scripts/evaluate_panel888_pairs.py --responses "$QWEN/responses.jsonl" --out "$QWEN/paired_control"
.venv/bin/python scripts/audit_panel888.py
.venv/bin/python scripts/report_panel888.py
