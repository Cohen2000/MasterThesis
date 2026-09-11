#!/usr/bin/env bash
# Fast, budget-safe DeepSeek V4 Pro probe.
#
# Non-thinking runs the full frozen 288-prompt design in four shards. Thinking
# runs a reduced 60-prompt design: 12 graphs, with three identical generations
# and three walk seeds per graph (one shared cell), in twelve shards. All
# workers reserve money in one locked ledger before every request.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PROBE="$REPO/results/llm_noise_probe"
FULL_PROMPTS="$PROBE/prompts.jsonl"
THINK_PROMPTS="$PROBE/prompts_subset60.jsonl"
BUDGET_FILE="$PROBE/deepseek_pro_parallel_budget.json"
PID_DIR="$PROBE/deepseek_pro_parallel_pids"
LOG_DIR="$PROBE/deepseek_pro_parallel_logs"
BUDGET_USD="${DEEPSEEK_PRO_PARALLEL_BUDGET_USD:-2.69}"
NONTHINK_SHARDS="${DEEPSEEK_PRO_NONTHINK_SHARDS:-4}"
THINK_SHARDS="${DEEPSEEK_PRO_THINK_SHARDS:-12}"

OLD_NONTHINK="$PROBE/answers_deepseek-v4-pro_nothink_official_noise.jsonl"
OLD_THINK="$PROBE/answers_deepseek-v4-pro_think_high_official_noise.jsonl"

check_inputs() {
    if [[ ! -x "$PYTHON_BIN" ]]; then
        echo "Python environment not found: $PYTHON_BIN" >&2
        exit 2
    fi
    if [[ ! -f "$FULL_PROMPTS" ]]; then
        echo "Frozen noise prompts not found: $FULL_PROMPTS" >&2
        exit 2
    fi
    if [[ "${DEEPSEEK_API_KEY+x}" != x || -z "$DEEPSEEK_API_KEY" ]]; then
        echo "Set DEEPSEEK_API_KEY first; the script never prints it." >&2
        exit 2
    fi
    if ! awk -v cap="$BUDGET_USD" 'BEGIN { exit !(cap > 0) }'; then
        echo "DEEPSEEK_PRO_PARALLEL_BUDGET_USD must be positive" >&2
        exit 2
    fi
    if [[ ! "$NONTHINK_SHARDS" =~ ^[1-9][0-9]*$ ||
          ! "$THINK_SHARDS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Shard counts must be positive integers" >&2
        exit 2
    fi
}

make_thinking_subset() {
    if [[ ! -f "$THINK_PROMPTS" ]]; then
        "$PYTHON_BIN" src/make_noise_subset.py --max-draw 3 --graphs 12
    fi
}

common_args=(
    --backend api
    --base-url https://api.deepseek.com
    --api-key-env DEEPSEEK_API_KEY
    --model deepseek-v4-pro
    --api-thinking-format deepseek
    --temperature 1.0
    --top-p 1.0
    --stream
    --timeout 1800
    --retries 5
    --sleep 0
    --rate-limit-total-wait 900
    --shared-budget-file "$BUDGET_FILE"
    --shared-budget-usd "$BUDGET_USD"
    --price-input-cache-hit 0.003625
    --price-input-cache-miss 0.435
    --price-output 0.87
)
worker_pids=()

worker_is_live() {
    local pid_file="$1"
    [[ -f "$pid_file" ]] || return 1
    local pid
    pid=$(<"$pid_file")
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || return 1
    kill -0 "$pid" 2>/dev/null
}

launch_worker() {
    local arm="$1"
    local shard="$2"
    local shard_count="$3"
    shift 3
    local pid_file="$PID_DIR/${arm}_${shard}.pid"
    local log_file="$LOG_DIR/${arm}_${shard}.log"
    if worker_is_live "$pid_file"; then
        echo "$arm shard $shard already running (PID $(<"$pid_file"))"
        return
    fi
    nohup env PYTHONPATH=src "$PYTHON_BIN" src/run_llm_v2.py \
        "${common_args[@]}" \
        --shard-index "$shard" --shard-count "$shard_count" \
        "$@" >>"$log_file" 2>&1 </dev/null &
    local pid=$!
    worker_pids+=("$pid")
    printf '%s\n' "$pid" >"$pid_file"
    echo "started $arm shard $((shard + 1))/$shard_count (PID $pid)"
}

start() {
    check_inputs
    make_thinking_subset
    mkdir -p "$PID_DIR" "$LOG_DIR"

    # Initialize or validate the shared ledger before background workers race
    # to it. An existing ledger is intentionally resumed, never reset.
    PYTHONPATH=src "$PYTHON_BIN" - "$BUDGET_FILE" "$BUDGET_USD" <<'PY'
import sys
from run_llm_v2 import SharedApiBudget

state = SharedApiBudget(sys.argv[1], float(sys.argv[2])).snapshot()
print(f"shared budget: ${state['spent_usd']:.6f}/${state['limit_usd']:.2f} spent")
PY

    local shard
    for ((shard=0; shard<NONTHINK_SHARDS; shard++)); do
        launch_worker nothink "$shard" "$NONTHINK_SHARDS" \
            --prompts "$FULL_PROMPTS" \
            --out "$PROBE/answers_deepseek-v4-pro_nothink_parallel.shard${shard}.jsonl" \
            --resume-from "$OLD_NONTHINK" \
            --thinking off --max-tokens 8192 --max-length-attempts 1
    done
    for ((shard=0; shard<THINK_SHARDS; shard++)); do
        launch_worker think "$shard" "$THINK_SHARDS" \
            --prompts "$THINK_PROMPTS" \
            --out "$PROBE/answers_deepseek-v4-pro_think_high_parallel.shard${shard}.jsonl" \
            --resume-from "$OLD_THINK" \
            --thinking on --reasoning-effort high \
            --max-tokens 131072 --max-length-attempts 2
    done
    echo "All workers started; keeping the supervisor attached until they finish."
    local failures=0
    local pid
    for pid in "${worker_pids[@]}"; do
        if ! wait "$pid"; then
            failures=$((failures + 1))
        fi
    done
    echo "All workers finished ($failures non-zero exits)."
    status
}

status() {
    make_thinking_subset
    PYTHONPATH=src "$PYTHON_BIN" - \
        "$FULL_PROMPTS" "$THINK_PROMPTS" "$BUDGET_FILE" "$BUDGET_USD" \
        "$OLD_NONTHINK" "$OLD_THINK" "$PID_DIR" <<'PY'
import glob
import os
import sys
from pathlib import Path

import run_llm_v2

full_path, think_path, budget_path, budget_usd, old_nt, old_t, pid_dir = sys.argv[1:]
full = run_llm_v2.load_prompts(full_path, 0, 1, [])
think = run_llm_v2.load_prompts(think_path, 0, 1, [])

def union_done(patterns):
    return run_llm_v2.resume_done_ids(patterns)

nt_done = union_done([old_nt, str(Path(full_path).parent /
                                  "answers_deepseek-v4-pro_nothink_parallel.shard*.jsonl")])
t_done = union_done([old_t, str(Path(full_path).parent /
                               "answers_deepseek-v4-pro_think_high_parallel.shard*.jsonl")])
nt_ids = {row["prompt_id"] for row in full}
t_ids = {row["prompt_id"] for row in think}

graphs = {}
for row in think:
    graphs.setdefault(row["instance_id"], set()).add(row["prompt_id"])
complete_graphs = sum(ids <= t_done for ids in graphs.values())

if Path(budget_path).exists():
    state = run_llm_v2.SharedApiBudget(
        budget_path, float(budget_usd)).snapshot()
else:
    state = {"spent_usd": 0.0, "limit_usd": float(budget_usd),
             "reservations": {}}
reserved = sum(float(x.get("usd") or 0.0)
               for x in state["reservations"].values())

print(f"non-thinking: {len(nt_done & nt_ids)}/{len(full)} complete")
print(f"thinking:     {len(t_done & t_ids)}/{len(think)} complete "
      f"({complete_graphs}/{len(graphs)} whole graphs)")
print(f"new spend:    ${state['spent_usd']:.4f}/${state['limit_usd']:.2f} "
      f"(${reserved:.4f} currently reserved)")
print(f"in-flight API requests: {len(state['reservations'])}")
PY
}

case "${1:-start}" in
    start)
        start
        ;;
    status)
        status
        ;;
    *)
        echo "usage: $0 {start|status}" >&2
        exit 2
        ;;
esac
