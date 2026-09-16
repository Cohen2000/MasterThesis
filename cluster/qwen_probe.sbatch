#!/bin/bash -l
#SBATCH --job-name=mainexp_probe
#SBATCH --partition=gpu_h100_short
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --output=logs/probe_%j.out
#
# Configuration probe on a handful of development inputs: how many GPUs, how many
# concurrent sequences, and how long a full answer actually takes. Thirty minutes
# on the development partition, so it schedules immediately.
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_mainexp/bin/activate"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1 TMPDIR="$WS/tmp"
# The Qwen Triton kernels are JIT-compiled and need nvcc. The matching CUDA 13.4
# toolchain ships inside the venv as nvidia-cuda-nvcc; the cluster modules only go
# up to 12.8, and torch here is cu130. Without this the engine dies at warmup with
# "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
export CUDA_HOME="$WS/venv_mainexp/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
mkdir -p "$WS/mainexp/logs"; cd "$WS/mainexp"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
python probe_configs.py --model "$WS/models/Qwen3.6-35B-A3B" --prompts "$WS/mainexp/dev_prompts.jsonl"
