#!/bin/bash
# After the single production chain (including its archive job) has finished:
# download, verify, collect, evaluate, audit and report the Qwen answers.
#   bash scripts/integrate_qwen.sh <frozen source commit>
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT="${1:?frozen source commit}"
EXP=panel888_main
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
PY=.venv/bin/python
OUT=results/panel888
ARCHIVE=$OUT/qwen_archive
if [ -e "$OUT/qwen" ]; then echo "$OUT/qwen exists; integration already ran" >&2; exit 1; fi
mkdir -p "$ARCHIVE" "$OUT/qwen" "$OUT/logs"
rsync -a "uc3:$WS/$EXP/mainexp/archive/" "$ARCHIVE/"
$PY scripts/verify_qwen_archive.py --archive "$ARCHIVE" --commit "$COMMIT" | tee "$OUT/logs/qwen_verify.log"
$PY scripts/collect_qwen_answers.py --answers "$ARCHIVE/answers" --out "$OUT/qwen/responses.jsonl" 
$PY scripts/evaluate_responses.py --responses "$OUT/qwen/responses.jsonl" --out "$OUT/qwen/evaluation"
$PY scripts/evaluate_paired_controls.py --responses "$OUT/qwen/responses.jsonl" --out "$OUT/qwen/paired_control"
$PY scripts/audit_qwen.py | tee "$OUT/logs/qwen_audit.log"
$PY scripts/report_qwen.py
