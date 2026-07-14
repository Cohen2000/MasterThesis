#!/bin/bash
#SBATCH --job-name=bench_walks
#SBATCH --partition=cpu
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --array=0-11
#SBATCH --output=benchmark_02_walks_%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
python src/run_benchmark_walks.py --preset full \
  --num-shards 12 --shard-id "$SLURM_ARRAY_TASK_ID"

