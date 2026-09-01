#!/bin/bash
#SBATCH --job-name=g3v_think
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=g3v_think_%A_%a.out

# Qwen3.6-27B thinking on vLLM. One generation per submission, chosen by GEN:
# generations must not share an answers file, because resume is per prompt_id
# and a second generation written into the first one's file would be skipped.
#
#   GEN=0 sbatch slurm/g3_vllm_think.sbatch
#
# The reasoning budget is deliberately generous (65,536 tokens). On the HF
# path half of all thinking generations ran into a 32k cap and produced
# nothing, wasting 364 of 787 GPU-hours; the successful answers have p90
# 65,563 completion tokens. vLLM makes that budget affordable, so the cap is
# set where the answers actually finish rather than where the clock forced it.
set -e
WS=$(ws_find llm_pilot)
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
source "$WS/venv_vllm/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
cd "$WS/llm_g3"

GEN="${GEN:?set GEN to the generation index}"
SHARDS="${SHARDS:-4}"
TAG="qwen36-27b_think"
OUT="answers_vllm_${TAG}_g${GEN}.shard${SLURM_ARRAY_TASK_ID}.jsonl"

python3 run_llm_vllm_g3.py \
    --prompts "prompts_${TAG}_g${GEN}.jsonl" \
    --out "$OUT" \
    --model "Qwen/Qwen3.6-27B" \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 \
    --max-new-tokens "${MAXTOK:-65536}" \
    --max-num-seqs "${SEQS:-32}" \
    --chunk "${CHUNK:-256}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "$SHARDS" \
    ${LIMIT:+--limit $LIMIT}
echo "done: $(wc -l < "$OUT") records in $OUT"
