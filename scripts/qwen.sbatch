#!/bin/bash -l
#SBATCH --job-name=qwen
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --array=0-5
#SBATCH --output=logs/qwen_%A_%a.out
# Qwen answers on bwUniCluster: six parallel jobs, each on one H100 GPU (94 GB; the model needs its
# full context of 262144 tokens next to the weights). New requests start during the first 80 minutes
# of the two hours. Submitting the job again only starts requests that were never started.
#   sbatch scripts/qwen.sbatch <workspace with the virtual environment and the model>
set -euo pipefail
WS="${1:?workspace}"
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_mainexp/bin/activate"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1 TMPDIR="$WS/tmp"
export CUDA_HOME="$WS/venv_mainexp/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_HAS_FLASHINFER_CUBIN=0 VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
srun python scripts/qwen.py --samples results/samples --out results/qwen --model "$WS/models/Qwen3.6-35B-A3B" \
    --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 6 --admit-seconds 4800 --stop-seconds 6600
