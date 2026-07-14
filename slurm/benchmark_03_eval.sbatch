#!/bin/bash
#SBATCH --job-name=bench_eval
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --output=benchmark_03_eval_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source venv/bin/activate
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/thesis_mpl_cache"
python src/validate_benchmark.py --cases 'results/benchmark_full/cases_shard_*.csv.gz'
python src/evaluate_benchmark.py \
  --cases 'results/benchmark_full/cases_shard_*.csv.gz' \
  --out-dir results/benchmark_full --jobs "$SLURM_CPUS_PER_TASK"
python src/collect_benchmark_results.py --preset full \
  --logs 'benchmark_01_data_*.out' 'benchmark_02_walks_*.out' 'benchmark_03_eval_*.out'

