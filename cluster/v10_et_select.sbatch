#!/bin/bash -l
#SBATCH --job-name=v10_et_select
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/v10_et_select_%j.out
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_offline/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python scripts/build_v10_et.py select
