#!/bin/bash
#SBATCH --job-name=llmv21_r1_8b
#SBATCH --partition=gpu_h100_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --output=llmv21_r1_0528_qwen3_8b_%j.out

# OPTIONAL: DeepSeek-R1-0528-Qwen3-8B (Qwen3 architecture, DeepSeek R1 tuning).
# Official sampling (DeepSeek model card): temperature 0.6, top_p 0.95.
# Requires a transformers version with Qwen3 support (>= 4.51); the pinned
# pilot venv has 4.46.3 -> run the compat check in docs/llm_v21_run_plan.md
# BEFORE submitting. The template thinks by default -> --thinking none.
set -e
MODEL="deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"
# sbatch spools the script away, so $0 does not point at slurm/ -- find
# llm_v21_common.sh relative to the submit directory instead
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

OUT="answers_r1-0528-qwen3-8b.jsonl"
EXTRA=()
if [ -n "${SMOKE:-}" ]; then
    OUT="answers_r1-0528-qwen3-8b_smoke.jsonl"
    EXTRA=(--ids "$SMOKE_IDS")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking none \
    --temperature 0.6 --top-p 0.95 --seed 0 \
    --max-new-tokens 16384 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
