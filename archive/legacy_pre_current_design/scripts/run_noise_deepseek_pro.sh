#!/usr/bin/env bash
# Official DeepSeek V4 Pro arm of the frozen LLM noise probe.
#
# The 288 prompt records cover 32 graphs. Per graph, five generations of one
# identical prompt measure response noise and five walk seeds measure input
# noise; the first record is shared between both arms. The answer file is
# append-only and resume-safe by prompt_id.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
PROBE="$REPO/results/llm_noise_probe"
PROMPTS="${DEEPSEEK_PRO_NOISE_PROMPTS:-$PROBE/prompts.jsonl}"
MODE="${DEEPSEEK_PRO_NOISE_MODE:-nothink}"
CAP="${DEEPSEEK_PRO_NOISE_MAX_USD:-0.84}"
MAX_TOKENS="${DEEPSEEK_PRO_NOISE_MAX_TOKENS:-8192}"
MAX_LENGTH_ATTEMPTS="${DEEPSEEK_PRO_NOISE_MAX_LENGTH_ATTEMPTS:-1}"
SMOKE_IDS="0a8e5989118b,486603a64ee3,b4e772220a2a"

case "$MODE" in
    nothink)
        TAG="nothink"
        THINKING="off"
        REASONING_ARGS=()
        OTHER_OUT="$PROBE/answers_deepseek-v4-pro_think_high_official_noise.jsonl"
        ;;
    think)
        TAG="think_high"
        THINKING="on"
        REASONING_ARGS=(--reasoning-effort high)
        OTHER_OUT="$PROBE/answers_deepseek-v4-pro_nothink_official_noise.jsonl"
        ;;
    *)
        echo "DEEPSEEK_PRO_NOISE_MODE must be nothink or think" >&2
        exit 2
        ;;
esac

OUT="${DEEPSEEK_PRO_NOISE_OUT:-$PROBE/answers_deepseek-v4-pro_${TAG}_official_noise.jsonl}"
LOG="${DEEPSEEK_PRO_NOISE_LOG:-$PROBE/deepseek_v4_pro_${TAG}_official_noise.log}"

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Python environment not found: $PYTHON_BIN" >&2
    exit 2
fi
if [[ ! -f "$PROMPTS" ]]; then
    echo "Noise prompts not found: $PROMPTS" >&2
    exit 2
fi
if [[ "${DEEPSEEK_API_KEY+x}" != x || -z "$DEEPSEEK_API_KEY" ]]; then
    echo "Set DEEPSEEK_API_KEY first; the script never prints it." >&2
    exit 2
fi
if [[ ! "$MAX_TOKENS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSEEK_PRO_NOISE_MAX_TOKENS must be a positive integer" >&2
    exit 2
fi
if [[ ! "$MAX_LENGTH_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "DEEPSEEK_PRO_NOISE_MAX_LENGTH_ATTEMPTS must be a positive integer" >&2
    exit 2
fi
if ! awk -v cap="$CAP" 'BEGIN { exit !(cap > 0 && cap <= 0.84) }'; then
    echo "DEEPSEEK_PRO_NOISE_MAX_USD must be in (0, 0.84]" >&2
    exit 2
fi

mkdir -p "$PROBE"

cost_of() {
    "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
from pathlib import Path

total = 0.0
path = Path(sys.argv[1])
if path.exists():
    for line in path.open():
        try:
            total += float(json.loads(line).get("est_cost_usd") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
print(f"{total:.9f}")
PY
}

run() {
    local ids="${1:-}"
    local prior_cost local_cap
    prior_cost=$(cost_of "$OTHER_OUT")
    local_cap=$(awk -v cap="$CAP" -v prior="$prior_cost" \
        'BEGIN { printf "%.9f", cap - prior }')
    if ! awk -v cap="$local_cap" 'BEGIN { exit !(cap > 0) }'; then
        echo "Global cost cap already exhausted by the other Pro arm: \$$prior_cost" >&2
        return 0
    fi
    local args=(
        --backend api
        --prompts "$PROMPTS"
        --out "$OUT"
        --base-url https://api.deepseek.com
        --api-key-env DEEPSEEK_API_KEY
        --model deepseek-v4-pro
        --thinking "$THINKING"
        --api-thinking-format deepseek
        --temperature 1.0
        --top-p 1.0
        --stream
        --max-tokens "$MAX_TOKENS"
        --max-length-attempts "$MAX_LENGTH_ATTEMPTS"
        --timeout 1200
        --retries 5
        --sleep 1
        --max-usd "$local_cap"
        --price-input-cache-hit 0.003625
        --price-input-cache-miss 0.435
        --price-output 0.87
    )
    args+=("${REASONING_ARGS[@]}")
    if [[ -n "$ids" ]]; then
        args+=(--ids "$ids")
    fi
    echo "DeepSeek V4 Pro noise probe: mode=$MODE prompts=$PROMPTS"
    echo "output=$OUT  global cap=\$$CAP  prior other-arm cost=\$$prior_cost"
    echo "this-arm cumulative cap=\$$local_cap  max_tokens=$MAX_TOKENS"
    PYTHONPATH=src "$PYTHON_BIN" src/run_llm_v2.py "${args[@]}"
}

status() {
    PYTHONPATH=src "$PYTHON_BIN" - "$PROMPTS" "$OUT" "$OTHER_OUT" "$CAP" "$MODE" <<'PY'
import json
import sys
from pathlib import Path

import run_llm_v2

prompts = run_llm_v2.load_prompts(sys.argv[1], 0, 1, [])
out = Path(sys.argv[2])
other = Path(sys.argv[3])
done = run_llm_v2.done_ids(str(out))
attempts = 0
cost = 0.0
if out.exists():
    for line in out.open():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        attempts += 1
        cost += float(row.get("est_cost_usd") or 0.0)
other_cost = 0.0
if other.exists():
    for line in other.open():
        try:
            other_cost += float(json.loads(line).get("est_cost_usd") or 0.0)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
print(f"mode={sys.argv[5]} complete={len(done)}/{len(prompts)} attempts={attempts} "
      f"arm_cost=${cost:.4f} total_pro_cost=${cost + other_cost:.4f}/"
      f"${float(sys.argv[4]):.2f}")
PY
}

case "${1:-run}" in
    smoke)
        run "$SMOKE_IDS" 2>&1 | tee -a "$LOG"
        ;;
    run)
        run 2>&1 | tee -a "$LOG"
        ;;
    status)
        status
        ;;
    *)
        echo "usage: $0 {smoke|run|status}" >&2
        exit 2
        ;;
esac
