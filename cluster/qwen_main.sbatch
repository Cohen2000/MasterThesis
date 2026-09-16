#!/bin/bash -l
#SBATCH --job-name=mainexp_qwen
#SBATCH --partition=gpu_h100_short,gpu_h100_il,gpu_h100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/qwen_%x_%A_%a.out
#
# One Qwen generation pass of the main experiment.
#
#   MODE=thinking REPEAT=1 sbatch --array=0-3 cluster/qwen_main.sbatch
#
# A login shell is mandatory: the modulefiles are Lmod .lua and a non-interactive
# shell loads classic Tcl modules, which fails silently and leaves the system
# Python 3.9 in place.
#
# Eight hours rather than a long reservation: the earlier experience on this
# cluster was that long wall times sit behind shorter ones at equal priority and
# never get backfilled. Resume is per request id and costs one model reload, so
# several short passes beat one long one.
#
# One repeat per submission, each writing into its own directory. Resume matches
# on request id, so a second repeat written into the first one's directory would
# see every id as done and silently collapse the repeat-to-repeat variation the
# study measures.
set -euo pipefail

WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
# Positional arguments, not --export: passing variables through --export makes
# Slurm build a fresh login environment for the job, which fails on this cluster
# with "user env retrieval failed requeued held". The default (inherit) works.
#   sbatch --array=0-3 qwen_main.sbatch <mode> <repeat> [shards] [maxseq] [chunk] [limit]
MODE="${1:?set mode: thinking or nonthinking}"
REPEAT="${2:?set repeat: 1, 2 or 3}"
SHARDS="${3:-4}"
MAXSEQ="${4:-16}"
CHUNK="${5:-16}"
LIMIT="${6:-}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_mainexp/bin/activate"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1
export TMPDIR="$WS/tmp"
# The Qwen Triton kernels are JIT-compiled and need nvcc. The matching CUDA 13.4
# toolchain ships inside the venv as nvidia-cuda-nvcc; the cluster modules only go
# up to 12.8, and torch here is cu130. Without this the engine dies at warmup with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
export CUDA_HOME="$WS/venv_mainexp/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
# Single GPU, so no all-reduce: flashinfer's trtllm communication kernel is
# JIT-compiled against its own bundled CCCL headers, which are incompatible with
# the CUDA 13.4 nvcc this environment has, and it is only reached when TP > 1.
# The weights need about 66 GB and one H100 here has 94 GB.
# flashinfer 0.6.18 bundles its own CCCL headers and JIT-compiles against them.
# They are incompatible with the CUDA 13.4 nvcc available here ("CUDA compiler and
# CUDA toolkit headers are incompatible"), so every flashinfer JIT path dies at
# engine start. vLLM's native PyTorch sampler needs no compilation.
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_HAS_FLASHINFER_CUBIN=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
mkdir -p "$WS/mainexp/logs"
cd "$WS/mainexp"

# Leave room for the model load and for in-flight requests to finish writing.
LIMIT_S=$(( 30*60 - 480 ))

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python run_qwen_batch.py \
    --run  "$WS/mainexp/run" \
    --out  "$WS/mainexp/answers" \
    --model "$WS/models/Qwen3.6-35B-A3B" \
    --mode "$MODE" --repeat "$REPEAT" \
    --shard-index "$IDX" --shard-count "$SHARDS" \
    --tensor-parallel-size "${TP:-1}" \
    --max-num-seqs "$MAXSEQ" \
    --chunk "$CHUNK" \
    --deadline-seconds "$LIMIT_S" \
    ${LIMIT:+--limit "$LIMIT"}
