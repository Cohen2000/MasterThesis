#!/bin/bash
#SBATCH --job-name=pilot_walks
#SBATCH --partition=cpu
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=02_walks_%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
echo "host $(hostname) | start $(date)"
python run_pilot_walks.py --manifest data_grid/manifest.csv \
    --out summaries.csv --walk-seeds 2 --mle-iters 200
echo "WALKS DONE | $(date)"; wc -l summaries.csv
