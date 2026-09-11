#!/bin/bash
#SBATCH --job-name=llmv21_q36_n16k
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=llmv21_qwen36_27b_nothink_16k_%A_%a.out

# Adaptive second tier: rerun every incomplete/missing 8k primary case from
# scratch with 16384 new tokens. Results stay separate from the primary tier.
set -e
MODEL="Qwen/Qwen3.6-27B"
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

IDS=$(python3 select_llm_escalation.py --prompts "$PROMPTS" \
    --answers 'answers_qwen36-27b_nothink.shard*.jsonl')
if [ -z "$IDS" ]; then echo "no incomplete primary cases"; exit 0; fi

OUT="answers_qwen36-27b_nothink_16k.shard${SLURM_ARRAY_TASK_ID}.jsonl"
python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" --out "$OUT" --model "$MODEL" \
    --ids "$IDS" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 8 \
    --thinking off \
    --temperature 0.7 --top-p 0.8 --top-k 20 --seed 0 \
    --max-new-tokens 16384 --max-length-attempts 1
echo "done: $(wc -l < "$OUT") records in $OUT"
