#!/bin/bash
#SBATCH --job-name=pilot_smoke
#SBATCH --partition=dev_cpu
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=smoke_%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host $(hostname) | $(date)"
python make_pilot_data.py --out data_smoke --sizes 400,2000 \
    --substrates ba,er --families 2 --reps 2 --targets 0.1,0.3,0.5
python run_pilot_walks.py --manifest data_smoke/manifest.csv \
    --out summaries_smoke.csv --walk-seeds 1 --budgets 200,800,1600 --mle-iters 120
python pilot_eval.py --summaries summaries_smoke.csv --out-prefix smoke
echo "SMOKE DONE -> summaries_smoke.csv, smoke_results.csv, smoke_mae_vs_coverage.png"
