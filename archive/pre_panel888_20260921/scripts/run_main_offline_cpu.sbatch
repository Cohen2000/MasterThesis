#!/bin/bash
#SBATCH --job-name=thesis-offline
#SBATCH --partition=cpu
#SBATCH --account=tu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=offline-%j.log
set -euo pipefail
# Run from a copy of this repository containing only the required data/tokenizers.
# No network access, model weights, GPUs, API clients or submission of LLM jobs.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYTHON_BIN="${MAIN_OFFLINE_PYTHON:-.venv/bin/python}"
srun "$PYTHON_BIN" scripts/run_main_offline.py --out "${MAIN_OFFLINE_OUT:-results/main_experiment/run}" "$@"
