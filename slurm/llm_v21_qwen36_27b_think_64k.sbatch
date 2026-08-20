#!/bin/bash
#SBATCH --job-name=llmv21_q36_t64k
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --output=llmv21_qwen36_27b_think_64k_%A_%a.out

# Adaptive second tier: rerun every incomplete/missing 32k primary case from
# scratch with 65536 new tokens. Results stay separate from the primary tier.
set -e
MODEL="Qwen/Qwen3.6-27B"
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

IDS=$(python3 select_llm_escalation.py --prompts "$PROMPTS" \
    --answers 'answers_qwen36-27b_think.shard*.jsonl')
if [ -z "$IDS" ]; then echo "no incomplete primary cases"; exit 0; fi

OUT="answers_qwen36-27b_think_64k.shard${SLURM_ARRAY_TASK_ID}.jsonl"
python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" --out "$OUT" --model "$MODEL" \
    --ids "$IDS" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 16 \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 0 \
    --max-new-tokens 65536 --max-length-attempts 1
echo "done: $(wc -l < "$OUT") records in $OUT"
