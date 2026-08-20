#!/bin/bash
#SBATCH --job-name=target_iso_qwen
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=target_iso_qwen_%A_%a.out

set -euo pipefail
THINKING="${THINKING:-off}"
LABEL="${LABEL:-nothink}"
MODEL="${MODEL:-Qwen/Qwen3.6-27B}"

COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

python3 run_llm_v2.py --backend hf \
  --prompts target_isolation_prompts.jsonl \
  --out "answers_target_isolation_qwen_${LABEL}.shard${SLURM_ARRAY_TASK_ID}.jsonl" \
  --model "$MODEL" --thinking "$THINKING" \
  --temperature 0.7 --top-p 0.8 --top-k 20 --seed 0 \
  --max-new-tokens 16384 \
  --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 4
