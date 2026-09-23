#!/bin/bash -l
#SBATCH --job-name=v10_walk_audit
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/v10_walk_audit_%j.out
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_offline/bin/activate"
python scripts/audit_v10_walk.py \
  --graphs "$WS/panel888_access_v9_cpu/results/panel888/prepared/graphs" \
  --build "$WS/panel888_v10_walk_audit/build_amended" \
  --output "$WS/panel888_v10_walk_audit/results_amended_v2"
