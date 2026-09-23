#!/bin/bash -l
#SBATCH --job-name=budget_prepare
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --output=logs/budget_prepare_%A_%a.out
# Offline preparation of one budget of the budget-sensitivity study (no model calls).
# Run from a repository copy that already contains results/panel888 (the offline study):
#   sbatch --array=0-5 cluster/budget_sensitivity_prepare.sbatch
# Array index i prepares budget BUDGETS[i]; index "check" verifies the environment.
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_offline/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
BUDGETS=(0.025 0.05 0.2 0.3 0.4 0.5)
if [ "${1:-}" = check ]; then
    python scripts/check_offline_environment.py
else
    python scripts/budget_sensitivity.py prepare "${BUDGETS[$SLURM_ARRAY_TASK_ID]}"
fi
