#!/bin/bash
#SBATCH --job-name=pilot_eval
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=03_eval_%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host $(hostname) | start $(date)"
python pilot_eval.py --summaries summaries.csv --out-prefix grid
echo "EVAL DONE | $(date)"; ls -la grid_results.csv grid_mae_vs_coverage.png
