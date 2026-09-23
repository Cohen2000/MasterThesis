#!/bin/bash -l
#SBATCH --job-name=v10_confirm
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --array=0-2
#SBATCH --output=logs/v10_confirm_%A_%a.out
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_offline/bin/activate"
python scripts/confirm_v10_walk.py \
  --graphs "$WS/panel888_access_v9_cpu/results/panel888/prepared/graphs" \
  --old-gate "$WS/panel888_v10_walk_audit/results/walk_gate.csv" \
  --build "$WS/panel888_v10_walk_audit/build_confirm_$SLURM_ARRAY_TASK_ID" \
  --output "$WS/panel888_v10_walk_audit/confirmation" \
  --index "$SLURM_ARRAY_TASK_ID"
