#!/bin/bash
# Drive one API run to completion; safe to rerun at any time (the runner skips collected IDs,
# never relaunches a started request and refuses a second open Batch chunk).
#   GPT:      scripts/api_cycle.sh openai RUN_DIR [submit options...]
#   DeepSeek: scripts/api_cycle.sh deepseek RUN_DIR [submit options...]   (waits for off-peak windows)
set -euo pipefail
cd "$(dirname "$0")/.."
PROVIDER=$1; OUT=$2; shift 2
PY=.venv/bin/python
remaining() { $PY scripts/api_runner.py cost --provider "$PROVIDER" --output "$OUT" "$@" | $PY -c 'import json,sys; print(json.load(sys.stdin)["remaining_requests"])'; }
if [ "$PROVIDER" = deepseek ]; then
  while [ "$(remaining "$@")" -gt 0 ]; do
    if $PY scripts/api_runner.py window --provider deepseek | grep -q '"deepseek_start_allowed_now": true'; then
      $PY scripts/api_runner.py submit --provider deepseek --output "$OUT" --execute "$@" || true
    fi
    sleep 300
  done
  echo ALL DONE; exit 0
fi
TOOLS=(); for a in "$@"; do [ "$a" = --tools ] && TOOLS=(--tools); done
while true; do
  last=$(ls "$OUT"/chunk_*.batch.json 2>/dev/null | tail -1 || true)
  if [ -n "$last" ] && [ ! -e "${last%.batch.json}.collected.json" ]; then
    s=$($PY scripts/api_runner.py status --provider openai "${TOOLS[@]}" --output "$OUT" --execute | tr -d ' \n')
    echo "$(date +%H:%M:%S) $s"
    if grep -qE '"status":"(completed|expired|cancelled|failed)"' <<<"$s"; then
      $PY scripts/api_runner.py collect --provider openai "${TOOLS[@]}" --output "$OUT" --execute
    else
      sleep 120
    fi
  elif [ "$(remaining "$@")" -eq 0 ]; then
    echo ALL DONE; exit 0
  else
    $PY scripts/api_runner.py submit --provider openai --output "$OUT" --execute "$@"
  fi
done
