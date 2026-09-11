#!/bin/bash
#SBATCH --job-name=g3_tokens
#SBATCH --partition=dev_cpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=g3_token_count_%j.out

# Exact Qwen3.6 token counts for the frozen G3 prompts, replacing the
# calibrated estimates (portable count x 1.664) carried since G1. CPU only:
# this is a tokenizer pass, not inference.
set -e
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/g3_common.sh" ]; then COMMON="$d/g3_common.sh"; break; fi
done
GEN=0 source "$COMMON"

python3 "$WS/count_g3_qwen_tokens.py" \
    --prompts prompts_qwen36-27b_think_g0.jsonl \
    --out "$WS/g3_qwen_token_counts.csv"
