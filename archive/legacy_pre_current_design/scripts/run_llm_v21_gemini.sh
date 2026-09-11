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
#
# Output budget: GEMINI_MAXTOK=0 omits max_tokens so the model may think as
# long as it needs (server default applies). Truncation is a real outcome on
# the frozen suite -- most of the spread in the failure-penalized numbers came
# from non-responses, not worse answers -- so removing the cap removes a
# confound rather than flattering the model. Verify on a smoke that
# finish_reason is "stop" and not "length".
#
# Prompt/output override for probes and ablation cells (env, not flags):
#   PROMPTS_FILE=results/llm_noise_probe/prompts.jsonl \
#   OUT_FILE=results/llm_noise_probe/answers_gemini_minimal.jsonl \
#   bash scripts/run_llm_v21_gemini.sh minimal
set -euo pipefail

# Overridable so probes and ablation cells can reuse this runner. Both the
# generation pass and the completion check below read these variables, which
# is why --prompts/--out must NOT be passed as extra args: `remaining()` would
# then count against the wrong file and the outer loop would never terminate.
PROMPTS="${PROMPTS_FILE:-results/llm_v2/prompts.jsonl}"  # default: frozen 420
OUTDIR="results/llm_v21"
LOGDIR="$OUTDIR/logs"
# Frozen-suite defaults (one per access strategy). Override for probes:
# the ids must exist in $PROMPTS_FILE or the run finds nothing to do.
SMOKE_IDS="${SMOKE_IDS:-0b394cf2d923,3106eb7c74bb,90e26b753383}"
BASE_URL="https://generativelanguage.googleapis.com/v1beta/openai"

# default chosen from the project's actual free-tier quotas (AI Studio
# rate-limit page, 2026-07-15): gemini-3.1-flash-lite has 500 requests/day
# at 15 RPM -- the only text model with enough daily quota for a 420-prompt
# mode. The bigger flash models (3.5/3/2.5) allow only 20 requests/day.
# NOTE: both modes share the model's daily quota -> run one mode per day.
MODEL="${GEMINI_MODEL:-gemini-3.1-flash-lite}"
SLEEP_S="${GEMINI_SLEEP:-7}"            # pace below the free-tier RPM limit
MAX_WAIT="${GEMINI_MAX_WAIT:-3600}"     # cap for the 429 quota backoff
MAX_PASSES="${GEMINI_MAX_PASSES:-40}"   # stop retrying incompletes forever

MODE="${1:-}"
shift || true

case "$MODE" in
  think)
    EFFORT="high"
    # Gemini counts thinking tokens against max_tokens; the smoke run hit
    # "length" at 16384 with effort=high -> 32768 gives headroom.
    # GEMINI_MAXTOK=0 removes the cap entirely (the server default applies).
    MAXTOK="${GEMINI_MAXTOK:-32768}"
    THOUGHTS=(--include-thoughts)
    ;;
  minimal)
    EFFORT="minimal"
    MAXTOK="${GEMINI_MAXTOK:-8192}"
    THOUGHTS=()
    ;;
  *)
    echo "usage: $0 {think|minimal} [--smoke] [extra args]" >&2
    exit 1
    ;;
esac

TAG="${MODEL##*/}_${MODE}"
OUT="${OUT_FILE:-$OUTDIR/answers_${TAG}.jsonl}"
LOG="${LOG_FILE:-$LOGDIR/${TAG}.log}"

# --smoke selects the three fixed smoke prompts and separate smoke files
SMOKE=0
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--smoke" ]; then
        SMOKE=1
        # An explicit OUT_FILE/LOG_FILE wins: a probe smoke must not be
        # appended to the frozen suite's smoke file just because --smoke was
        # passed. Only derive the _smoke name when nothing was requested.
        [ -n "${OUT_FILE:-}" ] || OUT="$OUTDIR/answers_${TAG}_smoke.jsonl"
        [ -n "${LOG_FILE:-}" ] || LOG="$LOGDIR/${TAG}_smoke.log"
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

# --limit caps a single pass, not the run. Looping would re-enter it up to
# GEMINI_MAX_PASSES times and quietly generate MAX_PASSES*limit records, so a
# limited invocation is deliberately one pass only.
for a in ${EXTRA[@]+"${EXTRA[@]}"}; do
    if [ "$a" = "--limit" ]; then
        echo "--limit given: running a single pass, not the completion loop"
        run_pass
        exit 0
    fi
done

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
