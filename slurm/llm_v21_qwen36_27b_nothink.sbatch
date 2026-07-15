#!/bin/bash
#SBATCH --job-name=llmv21_q36_nothink
#SBATCH --partition=gpu_h100_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=llmv21_qwen36_27b_nothink_%j.out

# Qwen3.6-27B, non-thinking mode (enable_thinking=False via chat template).
# Official instruct-mode sampling (model card):
#   temperature 0.7, top_p 0.8, top_k 20, min_p 0
#   (presence_penalty 1.5 is also recommended, but HF transformers generate()
#   has no presence_penalty -- documented open point in docs/llm_v21_run_plan.md)
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

OUT="answers_qwen36-27b_nothink.jsonl"
EXTRA=()
if [ -n "${SMOKE:-}" ]; then
    OUT="answers_qwen36-27b_nothink_smoke.jsonl"
    EXTRA=(--ids "$SMOKE_IDS")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking off \
    --temperature 0.7 --top-p 0.8 --top-k 20 --seed 0 \
    --max-new-tokens 3072 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
