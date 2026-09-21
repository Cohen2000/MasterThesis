#!/bin/bash -l
#SBATCH --job-name=offline_study
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=96G
#SBATCH --output=logs/offline_study_%j.out
# The complete offline study (scripts/run_offline.sh) on a CPU node; no model calls.
# Run from a repository copy that contains data/raw and data/tokenizers:
#   sbatch cluster/offline_study.sbatch
set -euo pipefail
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
mkdir -p .venv/bin && ln -sf "$WS/venv_offline/bin/python" .venv/bin/python
bash scripts/run_offline.sh
