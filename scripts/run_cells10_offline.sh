#!/bin/bash
# Resumable offline chain for design cells10-20260917. Every step is idempotent
# and leaves a .done marker; rerunning the script continues after the last
# finished step. Nothing here calls an LLM or needs the cluster.
#   nohup bash scripts/run_cells10_offline.sh > results/cells10_offline_logs/driver.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=results/main_experiment/cells10_20260917
REV=results/baseline_revision_cells10_20260917
LOG=results/cells10_offline_logs
mkdir -p "$LOG" "$REV"
step() {
  local name=$1; shift
  if [ -f "$LOG/$name.done" ]; then echo "skip $name"; return 0; fi
  echo "$(date -Is) start $name"
  "$@" > "$LOG/$name.log" 2>&1
  touch "$LOG/$name.done"
  echo "$(date -Is) done  $name"
}
echo "running" > "$LOG/STATUS"
step offline   .venv/bin/python scripts/run_main_offline.py --out "$RUN"
step verify    .venv/bin/python scripts/verify_main_offline.py --run "$RUN"
for st in pool train dev main decompose_main decompose; do
  step "rev_$st" env MAIN_RUN="$RUN" .venv/bin/python scripts/run_baseline_revision.py --out "$REV" --stage "$st"
done
step hcheck    .venv/bin/python scripts/check_h_recent.py
step reuse     .venv/bin/python scripts/check_qwen_reuse.py --old-run results/main_experiment/hrecent5_20260917 \
                 --new-run "$RUN" --old-answers results/main_experiment/hrecent5_20260917_qwen/answers \
                 --out results/main_experiment/cells10_20260917_qwen/reuse_report.json
step tests     env PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main
echo "ALL_OFFLINE_DONE $(date -Is)" > "$LOG/STATUS"
