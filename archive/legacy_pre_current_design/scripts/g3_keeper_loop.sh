#!/usr/bin/env bash
# Run the G3 keeper on a fixed interval until stopped.
#
# The keeper itself is one-shot and idempotent; this only supplies the cadence
# and a timestamped log.  Kill it by the pid in results/final_run_g2/pids/,
# never with `pkill -f` -- a pattern broad enough to match this loop also
# matches the shell that runs the pkill, which is how the last watcher was
# lost with its state unverifiable afterwards.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
INTERVAL="${INTERVAL:-900}"
LOG=results/final_run_g2/logs/keeper.log

while :; do
    echo "--- $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG"
    bash scripts/g3_keeper.sh all >> "$LOG" 2>&1
    sleep "$INTERVAL"
done
