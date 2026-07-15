#!/bin/bash
# V2.1 LLM runs over the Gemini API free tier (OpenAI-compatible endpoint).
# Run from the repo root. Needs GEMINI_API_KEY in the environment; never
# echo or log the key.
#
#   bash scripts/run_llm_v21_gemini.sh <mode> [--smoke] [extra runner args...]
#
# Modes (each writes its own answers file and log under results/llm_v21/):
#   think     reasoning_effort=high + include_thoughts (thought summaries
#             land in "reasoning", the final reply in "answer")
#   minimal   reasoning_effort=minimal (lowest thinking level; gemini-3
#             models cannot fully disable reasoning, only 2.5 supports none)
#
# Free-tier design: requests are paced below the RPM limit (GEMINI_SLEEP,
# default 7s -> ~8 req/min) and HTTP 429 waits patiently with doubling
# backoff up to GEMINI_MAX_WAIT (default 3600s) instead of failing. Daily
# quotas reset at midnight Pacific, so a full 420-prompt mode can take
# multiple days -- that is expected. The outer loop reruns the runner until
# every prompt has a complete record (resume per prompt_id, append-only
# answers). Safe to Ctrl+C / kill / suspend the laptop at any time;
# restarting the same command continues where it stopped.
#
# Model override (check your free-tier models in AI Studio first):
#   GEMINI_MODEL=gemini-2.5-flash bash scripts/run_llm_v21_gemini.sh think
set -euo pipefail

PROMPTS="results/llm_v2/prompts.jsonl"   # frozen V2.1 suite (420 prompts)
OUTDIR="results/llm_v21"
LOGDIR="$OUTDIR/logs"
SMOKE_IDS="0b394cf2d923,3106eb7c74bb,90e26b753383"
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"

MODEL="${GEMINI_MODEL:-gemini-3-flash}"
SLEEP_S="${GEMINI_SLEEP:-7}"            # pace below the free-tier RPM limit
MAX_WAIT="${GEMINI_MAX_WAIT:-3600}"     # cap for the 429 quota backoff
MAX_PASSES="${GEMINI_MAX_PASSES:-40}"   # stop retrying incompletes forever

MODE="${1:-}"
shift || true

case "$MODE" in
  think)
    EFFORT="high"
    MAXTOK=16384
    THOUGHTS=(--include-thoughts)
    ;;
  minimal)
    EFFORT="minimal"
    MAXTOK=8192
    THOUGHTS=()
    ;;
  *)
    echo "usage: $0 {think|minimal} [--smoke] [extra args]" >&2
    exit 1
    ;;
esac

TAG="${MODEL##*/}_${MODE}"
OUT="$OUTDIR/answers_${TAG}.jsonl"
LOG="$LOGDIR/${TAG}.log"

# --smoke selects the three fixed smoke prompts and separate smoke files
SMOKE=0
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--smoke" ]; then
        SMOKE=1
        OUT="$OUTDIR/answers_${TAG}_smoke.jsonl"
        LOG="$LOGDIR/${TAG}_smoke.log"
        EXTRA+=(--ids "$SMOKE_IDS")
    else
        EXTRA+=("$a")
    fi
done

: "${GEMINI_API_KEY:?set GEMINI_API_KEY first (do not echo it)}"
mkdir -p "$OUTDIR" "$LOGDIR"

run_pass() {
    python3 src/run_llm_v2.py --backend api \
        --prompts "$PROMPTS" \
        --out "$OUT" \
        --base-url "$BASE_URL" \
        --api-key-env GEMINI_API_KEY \
        --model "$MODEL" \
        --reasoning-effort "$EFFORT" \
        ${THOUGHTS[@]+"${THOUGHTS[@]}"} \
        --temperature 1.0 \
        --max-tokens "$MAXTOK" \
        --sleep "$SLEEP_S" \
        --rate-limit-max-wait "$MAX_WAIT" \
        ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
}

remaining() {
    python3 - "$PROMPTS" "$OUT" <<'PY'
import sys
sys.path.insert(0, "src")
import run_llm_v2 as r
rows = r.load_prompts(sys.argv[1], 0, 1, [])
done = r.done_ids(sys.argv[2])
print(sum(1 for x in rows if x["prompt_id"] not in done))
PY
}

echo "mode=$MODE model=$MODEL -> $OUT (log: $LOG)"

if [ "$SMOKE" -eq 1 ]; then
    run_pass
    exit 0
fi

pass=1
while true; do
    echo "=== pass $pass $(date '+%F %T') ===" | tee -a "$LOG"
    run_pass
    left=$(remaining)
    echo "pass $pass done, $left prompts remaining" | tee -a "$LOG"
    if [ "$left" -eq 0 ]; then
        echo "all prompts complete: $OUT" | tee -a "$LOG"
        break
    fi
    if [ "$pass" -ge "$MAX_PASSES" ]; then
        echo "stopping after $MAX_PASSES passes with $left prompts left" \
             "(rerun the same command to continue)" | tee -a "$LOG"
        exit 1
    fi
    pass=$((pass + 1))
    sleep 60
done
