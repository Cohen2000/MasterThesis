#!/bin/bash
#SBATCH --job-name=g3_vllm_bench
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=g3_vllm_bench_%j.out
set -e
WS=$(ws_find llm_pilot)
module load devel/python/3.12.3-gnu-14.2
# vLLM JIT-compiles kernels at engine start and needs a real toolkit; without
# CUDA_HOME it fails with "Could not find nvcc". 12.8 matches the driver's
# CUDA 13.2 runtime compatibility and the cu124 torch wheels vLLM pulls.
module load devel/cuda/12.8
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
echo "CUDA_HOME=$CUDA_HOME  nvcc=$(command -v nvcc)"
source "$WS/venv_vllm/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
cd "$WS/llm_g3"
python "$WS/vllm_bench.py" prompts_qwen36-27b_think_g0.jsonl "${N:-16}" "${MAXTOK:-4096}" "${SEQS:-64}"
