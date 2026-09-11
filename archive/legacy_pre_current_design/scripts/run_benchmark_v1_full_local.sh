#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$HOME/Dokumente/MasterArbeit_benchmark_v1"
cd "$ROOT"

source venv/bin/activate

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="/tmp/thesis_mpl_cache"

mkdir -p local_logs
mkdir -p results/benchmark_full

echo "=================================================="
echo "Full benchmark started: $(date)"
echo "Project: $ROOT"
echo "Python: $(python --version 2>&1)"
echo "Available memory:"
free -h
echo "=================================================="

echo
echo "=== Checking real datasets ==="
python src/check_real_data.py

echo
echo "=== Phase 1: Building benchmark data ==="

if [[ -f data/benchmark_full/manifest.csv ]]; then
    echo "Existing manifest found; validating and reusing it."
    python src/validate_benchmark.py \
        --manifest data/benchmark_full/manifest.csv
else
    {
        /usr/bin/time -v nice -n 10 \
            python src/build_benchmark_data.py \
                --preset full \
                --overwrite
    } 2>&1 | tee local_logs/01_data.log

    python src/validate_benchmark.py \
        --manifest data/benchmark_full/manifest.csv \
        2>&1 | tee local_logs/01_validate_manifest.log
fi

echo
echo "Manifest summary:"
python - <<'PY'
import pandas as pd

d = pd.read_csv("data/benchmark_full/manifest.csv")
print(f"Instances: {len(d)}")
print(f"Independent groups: {d['group_id'].nunique()}")
print()
print(d.groupby("data_block").size().to_string())
PY

echo
echo "=== Phase 2: Running 12 walk shards serially ==="

for shard in $(seq 0 11); do
    shard_fmt=$(printf "%03d" "$shard")
    output="results/benchmark_full/cases_shard_${shard_fmt}.csv.gz"
    log="local_logs/02_walks_shard_${shard_fmt}.log"

    echo
    echo "--- Shard ${shard}/11: $(date) ---"

    if [[ -s "$output" ]] && gzip -t "$output" 2>/dev/null; then
        echo "Valid existing output found; skipping $output"
        continue
    fi

    rm -f "$output"

    {
        /usr/bin/time -v nice -n 10 \
            python src/run_benchmark_walks.py \
                --preset full \
                --num-shards 12 \
                --shard-id "$shard"
    } 2>&1 | tee "$log"

    gzip -t "$output"
done

echo
echo "Generated shard files:"
ls -lh results/benchmark_full/cases_shard_*.csv.gz

echo
echo "=== Validating all cases ==="

python src/validate_benchmark.py \
    --cases 'results/benchmark_full/cases_shard_*.csv.gz' \
    2>&1 | tee local_logs/02_validate_cases.log

echo
echo "=== Phase 3: Evaluating estimators and ML models ==="

rm -f \
    results/benchmark_full/predictions.csv.gz \
    results/benchmark_full/metrics.csv \
    results/benchmark_full/rankings.csv \
    results/benchmark_full/SCREEN_SUMMARY.md \
    results/benchmark_full/headline_ranking.png

{
    /usr/bin/time -v nice -n 10 \
        python src/evaluate_benchmark.py \
            --cases 'results/benchmark_full/cases_shard_*.csv.gz' \
            --out-dir results/benchmark_full \
            --jobs 1
} 2>&1 | tee local_logs/03_evaluation.log

echo
echo "=== Phase 4: Collecting results ==="

python src/collect_benchmark_results.py \
    --preset full \
    --logs 'local_logs/*.log'

echo
echo "=== Final validation ==="

unzip -t benchmark_full_results_to_share.zip

echo
echo "=================================================="
echo "FULL BENCHMARK COMPLETE: $(date)"
echo "Result: $ROOT/benchmark_full_results_to_share.zip"
echo "=================================================="
