#!/bin/bash
# Drive one API run to completion; safe to rerun at any time (the runner skips collected IDs,
# relaunches a lost request at most once and refuses a second open Batch chunk).
# In plain words: a loop around scripts/api_runner.py. For DeepSeek it waits for the off-peak
# window, submits, and repeats every minute until every answer is in (or the budget or the
# single retry of a lost request is used up). For GPT it submits one Batch chunk, polls it
# until it is finished, collects the answers and submits the next chunk.
#   GPT:      scripts/api_cycle.sh openai RUN_DIR [submit options...]
#   DeepSeek: scripts/api_cycle.sh deepseek RUN_DIR [submit options...]   (waits for off-peak windows)
set -euo pipefail
cd "$(dirname "$0")/.."
PROVIDER=$1; OUT=$2; shift 2
PY=.venv/bin/python
remaining() { $PY scripts/api_runner.py cost --provider "$PROVIDER" --output "$OUT" "$@" | $PY -c 'import json,sys; print(json.load(sys.stdin)["remaining_requests"])'; }
if [ "$PROVIDER" = deepseek ]; then
  while true; do
    n=$(remaining "$@")  # a failing cost check (e.g. uncertain launches) aborts instead of ending as done
    [ "$n" -gt 0 ] || break
    if $PY scripts/api_runner.py window --provider deepseek | grep -q '"deepseek_start_allowed_now": true'; then
      $PY scripts/api_runner.py submit --provider deepseek --output "$OUT" --execute "$@" || result=$?
      if [ "${result:-0}" -eq 3 ]; then echo BUDGET STOP; exit 0; fi
      if [ "${result:-0}" -eq 4 ]; then echo RETRIES EXHAUSTED; exit 0; fi
      result=0
    fi
    sleep 60
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
