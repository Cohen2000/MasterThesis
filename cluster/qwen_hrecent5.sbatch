#!/bin/bash -l
#SBATCH --job-name=hrecent5_qwen
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/hrecent5_%A_%a.out
#
# Qwen generation for the H revision (design budget10-hrecent5-20260917).
#
#   cd $WS/hrecent5/mainexp && sbatch --array=0-3 qwen_hrecent5.sbatch 4 16 H
#
# All six passes (two modes x three repeats) run in one engine per shard, and every
# finished answer is written at once (run_qwen_engine.py). Only gpu_h100: the 80 GB
# cards of gpu_h100_il cannot hold one request at the native 262144 context next to
# the weights (commit 2ab7327), and the context is not shortened.
#
# Two hours with admission control instead of thirty-minute shards: a request is
# admitted only during the first 80 minutes, so every admitted request has at least
# 30 minutes -- about 90k tokens at the measured 51 tokens/s per sequence under 16
# concurrent sequences, above the longest answer of the previous run (79k). The job
# stops at 110 minutes at the latest; anything still running then is lost and
# picked up by the next submission, finished answers never are.
#
# A login shell is mandatory (Lmod modules); positional arguments instead of
# --export (see cluster/STATE.md).
set -euo pipefail

WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
SHARDS="${1:-4}"
MAXSEQ="${2:-16}"
ARMS="${3:-H}"
LIMIT="${4:-}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_mainexp/bin/activate"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1
export TMPDIR="$WS/tmp"
export CUDA_HOME="$WS/venv_mainexp/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_HAS_FLASHINFER_CUBIN=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
cd "$WS/hrecent5/mainexp"
mkdir -p logs

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python run_qwen_engine.py \
    --run  "$WS/hrecent5/mainexp/run" \
    --out  "$WS/hrecent5/mainexp/answers" \
    --model "$WS/models/Qwen3.6-35B-A3B" \
    --arms "$ARMS" \
    --shard-index "$IDX" --shard-count "$SHARDS" \
    --max-num-seqs "$MAXSEQ" \
    --admit-seconds 4800 --stop-seconds 6600 \
    ${LIMIT:+--limit "$LIMIT"}
