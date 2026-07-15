#!/bin/bash
#SBATCH --job-name=llmv21_q36_think
#SBATCH --partition=gpu_h100_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=llmv21_qwen36_27b_think_%A_%a.out

# Qwen3.6-27B, thinking mode (main comparison, replaces Qwen2.5).
# Official thinking-mode sampling (model card, general tasks):
#   temperature 1.0, top_p 0.95, top_k 20, min_p 0.
# Thinking output is long -> 8 shards, 16384 new tokens; truncated or
# incomplete answers do not count as done, so resubmitting the same job
# continues where it stopped.
set -e
MODEL="Qwen/Qwen3.6-27B"
# sbatch spools the script away, so $0 does not point at slurm/ -- find
# llm_v21_common.sh relative to the submit directory instead
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

OUT="answers_qwen36-27b_think.shard${SLURM_ARRAY_TASK_ID}.jsonl"
EXTRA=(--shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 8)
if [ -n "${SMOKE:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
        echo "smoke mode: array task ${SLURM_ARRAY_TASK_ID} skipped"; exit 0
    fi
    OUT="answers_qwen36-27b_think_smoke.jsonl"
    EXTRA=(--ids "$SMOKE_IDS")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 0 \
    --max-new-tokens 16384 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
