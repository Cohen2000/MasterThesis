#!/bin/bash
#SBATCH --job-name=llmv21_r1_64k
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=llmv21_r1_distill_32b_64k_%A_%a.out

# Adaptive second tier: rerun every incomplete/missing 32k primary case from
# scratch with 65536 new tokens. Results stay separate from the primary tier.
set -e
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

IDS=$(python3 select_llm_escalation.py --prompts "$PROMPTS" \
    --answers 'answers_r1-distill-32b.shard*.jsonl')
if [ -z "$IDS" ]; then echo "no incomplete primary cases"; exit 0; fi

OUT="answers_r1-distill-32b_64k.shard${SLURM_ARRAY_TASK_ID}.jsonl"
python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" --out "$OUT" --model "$MODEL" \
    --ids "$IDS" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 8 \
    --thinking none \
    --temperature 0.6 --top-p 0.95 --seed 0 \
    --max-new-tokens 65536 --max-length-attempts 1
echo "done: $(wc -l < "$OUT") records in $OUT"
