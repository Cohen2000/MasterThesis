#!/bin/bash
# V2.1 LLM runs over NVIDIA NIM (OpenAI-compatible API). Run from the repo
# root. Needs NVIDIA_API_KEY in the environment; never echo or log the key.
#
#   bash scripts/run_llm_v21_nim.sh <mode> [extra runner args...]
#
# Modes (each writes its own answers file and log under results/llm_v21/):
#   mistral-none    Mistral Small 4, reasoning_effort=none
#   mistral-high    Mistral Small 4, reasoning_effort=high
#   dsv4-think      DeepSeek V4 Pro, chat_template_kwargs.thinking=true
#   dsv4-nothink    DeepSeek V4 Pro, chat_template_kwargs.thinking=false
#                   NOTE: deepseek-v4-pro reached end of life 2026-08-07 and
#                   now answers HTTP 410. Both dsv4-* modes are kept for the
#                   record and no longer run.
#   dsv4-flash      DeepSeek V4 Flash (0731), non-thinking -- the available
#                   DeepSeek on NIM as of 2026-08-20, and version-pinned in
#                   its own model id
#
# A probe or ablation can override PROMPTS_FILE, OUT_FILE and LOG_FILE in the
# environment.  Keeping all three explicit prevents a special run from ever
# appending to the frozen 420-prompt answer file.
#
# Smoke test (3 real prompts, one per access strategy, separate out/log):
#   bash scripts/run_llm_v21_nim.sh mistral-high --smoke
# Reruns resume by prompt_id: only complete final-JSON records count as done.
set -euo pipefail

PROMPTS="${PROMPTS_FILE:-results/llm_v2/prompts.jsonl}" # frozen default (420)
OUTDIR="results/llm_v21"
LOGDIR="$OUTDIR/logs"
SMOKE_IDS="0b394cf2d923,3106eb7c74bb,90e26b753383"

MODE="${1:-}"
shift || true

case "$MODE" in
  mistral-none)
    # fast instant-reply mode, equivalent to Mistral Small 3.2 style
    # (Mistral Small 3.x guidance: low temperature ~0.15)
    # Full run 2026-07-15: 59% of the first 175 records hit the 4096 cap
    # (the model enumerates edges/windows verbosely before the JSON; stop
    # records had median 3210 / p90 3963 tokens) and temp 0.15 makes
    # retries near-deterministic -> budget 4096 -> 16384 + --stream.
    # Records completed under the old cap are unaffected (a cap does not
    # change sampling below it).
    TAG="mistral-small-4_none"
    ARGS=(--model mistralai/mistral-small-4-119b-2603
          --reasoning-effort none
          --temperature 0.15
          --max-tokens 16384
          --stream)
    ;;
  mistral-high)
    # deep reasoning mode, Magistral-equivalent verbosity
    # (Magistral guidance: temperature 0.7, top_p 0.95; needs a large budget)
    # Smoke 2026-07-15: reasoning alone used ~15.5k tokens; the third smoke
    # prompt filled 16384 AND (re-smoke, streamed, 390s) 24576 entirely with
    # reasoning -> one more raise to 32768. Records that still finish with
    # "length" after a retry are a model outcome (non-converging reasoning),
    # not a pipeline problem; never repair them.
    TAG="mistral-small-4_high"
    ARGS=(--model mistralai/mistral-small-4-119b-2603
          --reasoning-effort high
          --temperature 0.7 --top-p 0.95
          --max-tokens 32768
          --stream)
    ;;
  dsv4-think)
    # NIM guidance for DeepSeek V4: temperature 1.0, top_p 1.0 (sampling
    # parameters are ignored in thinking mode anyway); thinking needs budget.
    # Smoke 2026-07-15: non-streaming requests died in NIM gateway timeouts
    # (HTTP 504, empty body) because thinking generations run longer than
    # the gateway allows -> --stream keeps the connection alive.
    TAG="deepseek-v4-pro_think"
    ARGS=(--model deepseek-ai/deepseek-v4-pro
          --thinking on
          --temperature 1.0
          --max-tokens 16384
          --stream)
    ;;
  dsv4-flash)
    # Successor arm after deepseek-v4-pro was retired. The date suffix pins
    # the version, which the -pro id never did. Same non-thinking sampling as
    # the retired mode so the rows stay comparable in character.
    # If NIM rejects chat_template_kwargs for this model (HTTP 400), rerun
    # with `--thinking none`: that sends no toggle at all and lets the model
    # default, at the cost of a larger token budget.
    TAG="deepseek-v4-flash-0731_nothink"
    ARGS=(--model deepseek-ai/deepseek-v4-flash-0731
          --thinking off
          --temperature 1.0
          # NIM_MAXTOK=0 omits max_tokens so the model may think as long as it
          # needs; the server default then applies. Check finish_reason on a
          # smoke before trusting it.
          --max-tokens "${NIM_MAXTOK:-8192}"
          # Same gateway-timeout lesson as dsv4-think: a non-streaming request
          # dies with HTTP 504 when the generation outlasts the gateway, and
          # dropping the token cap makes long generations more likely, not
          # less. Streaming keeps bytes flowing so the connection survives.
          --stream
          # The free endpoint answers HTTP 529 when it is busy. With a wait
          # budget that is backpressure to sit out, not a failed prompt --
          # but bounded, so a permanently overloaded endpoint fails visibly
          # after NIM_TOTAL_WAIT instead of parking the run on prompt 1.
          --rate-limit-max-wait "${NIM_MAX_WAIT:-120}"
          --rate-limit-total-wait "${NIM_TOTAL_WAIT:-600}"
          # 504s consume retry attempts (they are not backpressure), so a
          # flaky gateway needs more headroom than the default six.
          --retries "${NIM_RETRIES:-12}")
    ;;
  dsv4-nothink)
    TAG="deepseek-v4-pro_nothink"
    ARGS=(--model deepseek-ai/deepseek-v4-pro
          --thinking off
          --temperature 1.0
          --max-tokens 8192)
    ;;
  *)
    echo "usage: $0 {mistral-none|mistral-high|dsv4-think|dsv4-nothink|dsv4-flash} [extra args]" >&2
    exit 1
    ;;
esac

OUT="${OUT_FILE:-$OUTDIR/answers_${TAG}.jsonl}"
LOG="${LOG_FILE:-$LOGDIR/${TAG}.log}"

# --smoke selects the three fixed smoke prompts and separate smoke files
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--smoke" ]; then
        [ -n "${OUT_FILE:-}" ] || OUT="$OUTDIR/answers_${TAG}_smoke.jsonl"
        [ -n "${LOG_FILE:-}" ] || LOG="$LOGDIR/${TAG}_smoke.log"
        EXTRA+=(--ids "$SMOKE_IDS")
    else
        EXTRA+=("$a")
    fi
done

: "${NVIDIA_API_KEY:?set NVIDIA_API_KEY first (do not echo it)}"
mkdir -p "$OUTDIR" "$LOGDIR"

echo "mode=$MODE -> $OUT (log: $LOG)"
python src/run_llm_v2.py --backend api \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --base-url https://integrate.api.nvidia.com/v1 \
    --api-key-env NVIDIA_API_KEY \
    "${ARGS[@]}" \
    ${EXTRA[@]+"${EXTRA[@]}"} 2>&1 | tee -a "$LOG"
