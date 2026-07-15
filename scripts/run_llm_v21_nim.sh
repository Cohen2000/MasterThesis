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
#
# Smoke test (3 real prompts, one per access strategy, separate out/log):
#   bash scripts/run_llm_v21_nim.sh mistral-high --smoke
# Reruns resume by prompt_id: only complete final-JSON records count as done.
set -euo pipefail

PROMPTS="results/llm_v2/prompts.jsonl"   # frozen V2.1 suite (420 prompts)
OUTDIR="results/llm_v21"
LOGDIR="$OUTDIR/logs"
SMOKE_IDS="0b394cf2d923,3106eb7c74bb,90e26b753383"

MODE="${1:-}"
shift || true

case "$MODE" in
  mistral-none)
    # fast instant-reply mode, equivalent to Mistral Small 3.2 style
    # (Mistral Small 3.x guidance: low temperature ~0.15)
    TAG="mistral-small-4_none"
    ARGS=(--model mistralai/mistral-small-4-119b-2603
          --reasoning-effort none
          --temperature 0.15
          --max-tokens 4096)
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
  dsv4-nothink)
    TAG="deepseek-v4-pro_nothink"
    ARGS=(--model deepseek-ai/deepseek-v4-pro
          --thinking off
          --temperature 1.0
          --max-tokens 8192)
    ;;
  *)
    echo "usage: $0 {mistral-none|mistral-high|dsv4-think|dsv4-nothink} [extra args]" >&2
    exit 1
    ;;
esac

OUT="$OUTDIR/answers_${TAG}.jsonl"
LOG="$LOGDIR/${TAG}.log"

# --smoke selects the three fixed smoke prompts and separate smoke files
EXTRA=()
for a in "$@"; do
    if [ "$a" = "--smoke" ]; then
        OUT="$OUTDIR/answers_${TAG}_smoke.jsonl"
        LOG="$LOGDIR/${TAG}_smoke.log"
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
