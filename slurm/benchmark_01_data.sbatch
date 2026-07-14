#!/bin/bash
#SBATCH --job-name=bench_data
#SBATCH --partition=cpu
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=benchmark_01_data_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python src/build_benchmark_data.py --preset full --overwrite
python src/validate_benchmark.py --manifest data/benchmark_full/manifest.csv

