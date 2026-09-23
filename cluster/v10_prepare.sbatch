#!/bin/bash -l
#SBATCH --job-name=v10_prepare
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=logs/v10_prepare_%j.out
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_offline/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/prepare_study.py
