#!/bin/bash
# After the budget-sensitivity production chain (including its archive job) has finished:
# download, verify, collect and evaluate. Never touches results/panel888.
#   bash scripts/integrate_budget_sensitivity.sh <frozen source commit>
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT="${1:?frozen source commit}"
EXP=panel888_budget
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
PY=.venv/bin/python
OUT=results/panel888_budget_sensitivity
if [ -e "$OUT/qwen" ]; then echo "$OUT/qwen exists; integration already ran" >&2; exit 1; fi
mkdir -p "$OUT/qwen_archive" "$OUT/qwen"
rsync -a "uc3:$WS/$EXP/mainexp/archive/" "$OUT/qwen_archive/"
$PY scripts/verify_qwen_archive.py --archive "$OUT/qwen_archive" --commit "$COMMIT" --run "$OUT/run"
$PY scripts/collect_qwen_answers.py --run "$OUT/run" --answers "$OUT/qwen_archive/answers" --out "$OUT/qwen/responses.jsonl"
$PY scripts/budget_sensitivity.py evaluate
