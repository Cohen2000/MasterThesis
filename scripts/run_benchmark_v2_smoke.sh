#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/thesis_mpl_cache"
rm -rf data/benchmark_v2_smoke results/benchmark_v2_smoke
python src/build_benchmark_data.py --preset v2_smoke --overwrite
python src/validate_benchmark.py --manifest data/benchmark_v2_smoke/manifest.csv
python src/run_benchmark_walks.py --preset v2_smoke
python src/validate_benchmark.py --cases results/benchmark_v2_smoke/cases.csv.gz
python src/evaluate_benchmark.py --preset v2_smoke \
  --cases results/benchmark_v2_smoke/cases.csv.gz \
  --out-dir results/benchmark_v2_smoke --jobs 1
python -m unittest discover -s tests -v
python src/collect_benchmark_results.py --preset v2_smoke
printf '\nV2 SMOKE OK: %s\n' "$ROOT/results/benchmark_v2_smoke/SCREEN_SUMMARY.md"
