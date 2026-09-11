#!/bin/bash
#SBATCH --job-name=pilot_make
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=01_make_%j.out
set -e
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "host $(hostname) | start $(date)"
# Backbone grid. sizes up to 50k give walk coverage down to ~0.02 (the partial-
# access regime). continuous targets = a spread of rho in [0.02,0.58] per family.
python make_pilot_data.py --out data_grid \
    --sizes 400,2000,10000,50000 --substrates ba,er \
    --families 4 --reps 3 --targets-mode continuous --n-targets 8
echo "MAKE DONE | $(date)"; wc -l data_grid/manifest.csv
