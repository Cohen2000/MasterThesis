#!/usr/bin/env bash
# Wait for the thinking wrong-direction jobs, then sync and recompute once.
#
# Read-only on the cluster side; the only local writes are the synced answer
# files and the derived tables under results_summary/g4/. Exits after one
# successful rebuild. Stop it with the pid in
# results/final_run_g2/pids/wrongdir_watch.pid -- not with `pkill -f`.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
LOG=results/final_run_g2/logs/wrongdir_watch.log
INTERVAL="${INTERVAL:-600}"

while :; do
    n=$(timeout 90 ssh -o BatchMode=yes uc3 'bash -s' <<'REMOTE' 2>/dev/null
WS=$(ws_find llm_pilot); cd "$WS" || exit 1
total=0
for g in 0 1 2; do
  f=llm_g3/answers_vllm_wrongdir_qwen36-27b_think_g${g}.jsonl
  n=$( [ -f "$f" ] && wc -l < "$f" || echo 0 )
  total=$((total + n))
done
echo "$total"
REMOTE
)
    echo "$(date '+%H:%M:%S') think wrongdir records: ${n:-unreachable}/192" >> "$LOG"
    if [ "${n:-0}" -ge 192 ]; then
        echo "$(date '+%H:%M:%S') complete -- syncing and rebuilding" >> "$LOG"
        bash scripts/g4_sync.sh >> "$LOG" 2>&1
        bash scripts/g4_rebuild.sh >> "$LOG" 2>&1
        echo "$(date '+%H:%M:%S') rebuild finished (exit $?)" >> "$LOG"
        exit 0
    fi
    sleep "$INTERVAL"
done
