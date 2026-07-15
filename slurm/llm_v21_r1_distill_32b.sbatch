#!/bin/bash
#SBATCH --job-name=llmv21_r1_32b
#SBATCH --partition=gpu_h100_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=llmv21_r1_distill_32b_%A_%a.out

# DeepSeek-R1-Distill-Qwen-32B, continuity baseline from the V2 round.
# Official sampling (DeepSeek model card): temperature 0.6, top_p 0.95.
# NOTE: the old llm_v2_r1_32b.sbatch ran greedy (temperature 0); this V2.1
# run follows the official recommendation, so results are a new series.
# The template always thinks -> --thinking none (no enable_thinking kwarg);
# the runner splits <think>...</think> into the "reasoning" field.
set -e
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# sbatch spools the script away, so $0 does not point at slurm/ -- find
# llm_v21_common.sh relative to the submit directory instead
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

OUT="answers_r1-distill-32b.shard${SLURM_ARRAY_TASK_ID}.jsonl"
EXTRA=(--shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 4)
if [ -n "${SMOKE:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
        echo "smoke mode: array task ${SLURM_ARRAY_TASK_ID} skipped"; exit 0
    fi
    OUT="answers_r1-distill-32b_smoke.jsonl"
    EXTRA=(--ids "$SMOKE_IDS")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking none \
    --temperature 0.6 --top-p 0.95 --seed 0 \
    --max-new-tokens 12288 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
