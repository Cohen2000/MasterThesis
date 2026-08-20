#!/usr/bin/env bash
# Fairness pilot: the frozen V2.1 prompt through the plain OpenAI API.
#
# Why this exists. The two strongest rows in the evidence suite (Codex GPT-5.6
# and Claude Code Opus) run inside a product harness that injects instructions
# and cannot be version-pinned. Every row that IS pinnable is a small or mid
# model -- Flash Lite, Mistral Small, Qwen-27B. So "Codex is better" currently
# confounds three things: model tier, harness prompt, and tool use. Running the
# same frozen prompt against the same model family with no harness separates
# the first from the other two.
#
#   OPENAI_MODEL=<id> bash scripts/run_llm_openai.sh <tag> [runner args...]
#
# The model id is deliberately NOT defaulted: take it from your own console,
# because ids and prices change and a stale guess silently bills the wrong
# model. Check platform.openai.com for the current list and pricing.
#
# Cost control:
#   OPENAI_MAXTOK   cap on completion tokens (default 16384; 0 = no cap,
#                   which on a paid endpoint is a decision, not a default)
#   OPENAI_EFFORT   reasoning effort, passed as reasoning_effort
#   OPENAI_TEMP     -1 (default) omits temperature, which reasoning endpoints
#                   generally require; set a value only if yours accepts one
#
# Never echo, log, or commit the key. It is read from OPENAI_API_KEY.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TAG="${1:?usage: $0 <tag> [runner args...]}"
shift || true

: "${OPENAI_MODEL:?set OPENAI_MODEL to an id from your console}"
: "${OPENAI_API_KEY:?set OPENAI_API_KEY first (do not echo it)}"

PROMPTS="${PROMPTS_FILE:-results/llm_v2/prompts.jsonl}"
OUTDIR="${OUTDIR:-results/llm_openai_pilot}"
LOGDIR="$OUTDIR/logs"
OUT="${OUT_FILE:-$OUTDIR/answers_${TAG}.jsonl}"
LOG="${LOG_FILE:-$LOGDIR/${TAG}.log}"
mkdir -p "$OUTDIR" "$LOGDIR"

# reasoning_effort is only valid for reasoning models; a plain chat model
# rejects it with HTTP 400. OPENAI_EFFORT="" omits the flag entirely.
EFFORT_ARGS=()
if [ -n "${OPENAI_EFFORT-medium}" ]; then
    EFFORT_ARGS=(--reasoning-effort "${OPENAI_EFFORT:-medium}")
fi

echo "model=$OPENAI_MODEL tag=$TAG -> $OUT (log: $LOG)"
python3 src/run_llm_v2.py --backend api \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --base-url https://api.openai.com/v1 \
    --api-key-env OPENAI_API_KEY \
    --model "$OPENAI_MODEL" \
    --temperature "${OPENAI_TEMP:--1}" \
    --max-tokens "${OPENAI_MAXTOK:-16384}" \
    --max-tokens-param max_completion_tokens \
    ${EFFORT_ARGS[@]+"${EFFORT_ARGS[@]}"} \
    --stream \
    --sleep "${OPENAI_SLEEP:-1}" \
    --rate-limit-max-wait "${OPENAI_MAX_WAIT:-120}" \
    --rate-limit-total-wait "${OPENAI_TOTAL_WAIT:-600}" \
    "$@" 2>&1 | tee -a "$LOG"

echo
echo "spend so far in $OUT:"
python3 - "$OUT" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
pin = pout = 0
for r in rows:
    u = r.get("usage") or {}
    pin += u.get("prompt_tokens") or 0
    pout += u.get("completion_tokens") or 0
print(f"  {len(rows)} records | input {pin:,} tok | output {pout:,} tok")
print("  cost = input/1e6*<in-price> + output/1e6*<out-price>; "
      "take prices from your console")
PY
