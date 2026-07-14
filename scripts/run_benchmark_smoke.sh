#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/thesis_mpl_cache}"

python src/build_benchmark_data.py --preset smoke --overwrite
python src/validate_benchmark.py --manifest data/benchmark_smoke/manifest.csv
python src/run_benchmark_walks.py --preset smoke
python src/validate_benchmark.py --cases results/benchmark_smoke/cases.csv.gz
python src/evaluate_benchmark.py \
  --cases results/benchmark_smoke/cases.csv.gz \
  --out-dir results/benchmark_smoke
python -m unittest discover -s tests -v

echo "SMOKE OK: results/benchmark_smoke/SCREEN_SUMMARY.md"
