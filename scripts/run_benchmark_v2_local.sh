#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source "$ROOT/.venv/bin/activate"

SHARDS="${SHARDS:-12}"
JOBS="${JOBS:-1}"
RESET="${RESET:-0}"
LOGDIR="local_logs_v2"
mkdir -p "$LOGDIR" results/benchmark_v2

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/thesis_mpl_cache"

exec > >(tee -a "$LOGDIR/master.log") 2>&1

echo "============================================================"
echo "V2 benchmark started: $(date)"
echo "Root: $ROOT"
echo "Python: $(python --version 2>&1)"
echo "SHARDS=$SHARDS JOBS=$JOBS RESET=$RESET"
free -h
echo "============================================================"

python src/check_real_data.py --preset v2 | tee "$LOGDIR/00_real_data_status.log"

if [[ "$RESET" == "1" ]]; then
  rm -rf data/benchmark_v2 results/benchmark_v2
  mkdir -p results/benchmark_v2
fi

if [[ ! -s data/benchmark_v2/manifest.csv ]]; then
  { /usr/bin/time -v nice -n 10 python src/build_benchmark_data.py \
      --preset v2 --overwrite; } 2>&1 | tee "$LOGDIR/01_build_data.log"
else
  echo "Reusing existing data/benchmark_v2/manifest.csv"
fi
python src/validate_benchmark.py --manifest data/benchmark_v2/manifest.csv \
  | tee "$LOGDIR/01_validate_manifest.log"

python - <<'PYMANIFEST'
import pandas as pd
d = pd.read_csv("data/benchmark_v2/manifest.csv")
print(f"Instances: {len(d):,}")
print(f"Independent groups: {d['group_id'].nunique():,}")
print(d.groupby('data_block').size().sort_values(ascending=False).to_string())
PYMANIFEST
python src/estimate_benchmark_scale.py --preset v2 \
  --manifest data/benchmark_v2/manifest.csv

for shard in $(seq 0 $((SHARDS - 1))); do
  sf=$(printf '%03d' "$shard")
  out="results/benchmark_v2/cases_shard_${sf}.csv.gz"
  log="$LOGDIR/02_walks_shard_${sf}.log"
  if [[ -s "$out" ]] && gzip -t "$out" 2>/dev/null; then
    echo "Skipping valid shard $sf"
    continue
  fi
  rm -f "$out"
  echo "Running shard $sf/$((SHARDS - 1)) at $(date)"
  { /usr/bin/time -v nice -n 10 python src/run_benchmark_walks.py \
      --preset v2 --num-shards "$SHARDS" --shard-id "$shard"; } \
      2>&1 | tee "$log"
  gzip -t "$out"
done

python src/validate_benchmark.py \
  --cases 'results/benchmark_v2/cases_shard_*.csv.gz' \
  | tee "$LOGDIR/02_validate_cases.log"

{ /usr/bin/time -v nice -n 10 python src/evaluate_benchmark.py \
    --preset v2 --cases 'results/benchmark_v2/cases_shard_*.csv.gz' \
    --out-dir results/benchmark_v2 --jobs "$JOBS"; } \
    2>&1 | tee "$LOGDIR/03_evaluation.log"

python src/analyze_v2_results.py | tee "$LOGDIR/04_diagnostics.log"
python src/collect_benchmark_results.py --preset v2 \
  --logs "$LOGDIR/*.log"
unzip -t benchmark_v2_results_to_share.zip

echo "============================================================"
echo "V2 BENCHMARK COMPLETE: $(date)"
echo "Result: $ROOT/benchmark_v2_results_to_share.zip"
echo "============================================================"
