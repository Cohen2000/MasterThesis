#!/bin/bash
#SBATCH --job-name=nw_q36_screen
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=nonwalk_qwen36_screen_%A_%a.out

# Bounded exploratory screen: array tasks 0..3 run non-thinking and 4..7
# run thinking. Each mode has four stable shards. One pass is intentional;
# length-truncated prompts are recorded but not retried at the same budget.
set -euo pipefail

MODEL="Qwen/Qwen3.6-27B"
# SHARDS is per mode; the array must cover 0..2*SHARDS-1. Override both
# together, e.g.  SHARDS=8 sbatch --array=0-15 <this file>
SHARDS="${SHARDS:-4}"
TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}
if [ "$TASK_ID" -lt "$SHARDS" ]; then
    MODE="nothink"
    THINKING="off"
    SHARD_INDEX=$TASK_ID
    TEMPERATURE="0.7"
    TOP_P="0.8"
    MAX_NEW_TOKENS="${NOTHINK_MAX_NEW_TOKENS:-8192}"
    # Smoke 2026-08-20: the single non-thinking generation degenerated into
    # `+1+1+1...` and consumed all 8192 tokens without emitting JSON --
    # trigram diversity fell from 0.44 to 0.001 over the answer. The model
    # card prescribes presence_penalty 1.5 for exactly this; transformers has
    # no presence_penalty, so repetition_penalty stands in. It rescales logits
    # rather than subtracting, so the useful range is near 1.05-1.15.
    # A deviation from the card's recipe is a run condition: it is echoed into
    # every answer record.
    REP_PENALTY="${NOTHINK_REP_PENALTY:-1.1}"
else
    MODE="think"
    THINKING="on"
    SHARD_INDEX=$((TASK_ID - SHARDS))
    TEMPERATURE="1.0"
    TOP_P="0.95"
    # Smoke 2026-08-20: thinking did NOT degenerate (trigram diversity 0.47 ->
    # 0.37, it was still doing correct per-mask arithmetic) -- it simply ran
    # out of budget at 16384. That is a budget problem, not a sampling one, so
    # this mode gets more room and no penalty.
    MAX_NEW_TOKENS="${THINK_MAX_NEW_TOKENS:-32768}"
    REP_PENALTY="${THINK_REP_PENALTY:-}"
fi

RUN_MODE="${RUN_MODE:-both}"
if [ "$RUN_MODE" != "both" ] && [ "$RUN_MODE" != "$MODE" ]; then
    echo "RUN_MODE=$RUN_MODE: skipping $MODE task $TASK_ID"
    exit 0
fi

COMMON=""
for directory in "${SLURM_SUBMIT_DIR:-.}" \
                 "${SLURM_SUBMIT_DIR:-.}/slurm" \
                 "$(dirname "$0")"; do
    if [ -f "$directory/llm_v21_common.sh" ]; then
        COMMON="$directory/llm_v21_common.sh"
        break
    fi
done
if [ -z "$COMMON" ]; then
    echo "llm_v21_common.sh not found" >&2
    exit 1
fi
source "$COMMON"

PROMPTS="${PROMPTS_FILE:-nonwalk_qwen36_screen_prompts.jsonl}"
OUT="answers_nonwalk_qwen36_${MODE}.shard${SHARD_INDEX}.jsonl"
EXTRA=(--shard-index "$SHARD_INDEX" --shard-count "$SHARDS")

if [ -n "${SMOKE:-}" ]; then
    if [ "$SHARD_INDEX" -ne 0 ]; then
        echo "smoke mode: shard $SHARD_INDEX skipped"
        exit 0
    fi
    OUT="answers_nonwalk_qwen36_${MODE}_smoke.jsonl"
    EXTRA=(--limit 1 --max-new-tokens "${SMOKE_MAX_NEW_TOKENS:-$MAX_NEW_TOKENS}")
fi

if [ ! -f "$PROMPTS" ]; then
    echo "prompt file not found: $PWD/$PROMPTS" >&2
    exit 1
fi

PENALTY_ARGS=()
if [ -n "$REP_PENALTY" ]; then
    PENALTY_ARGS=(--repetition-penalty "$REP_PENALTY")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking "$THINKING" \
    --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k 20 --seed 0 \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    ${PENALTY_ARGS[@]+"${PENALTY_ARGS[@]}"} \
    --max-length-attempts 1 \
    "${EXTRA[@]}"

echo "done: $MODE shard $SHARD_INDEX; $(wc -l < "$OUT") records in $OUT"
