#!/usr/bin/env bash
# Rebuild the five G4 analysis modules from the current answer files.
#
# Idempotent and read-only with respect to the frozen artifacts: it reads the
# answer JSONL plus the case table and rewrites only the derived tables under
# results_summary/g4/.  Run it after a generation finishes syncing from the
# cluster; there is no background watcher, on purpose -- the earlier one was
# killed by a self-matching pkill and its state was unverifiable afterwards.
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

OUT=results_summary/g4
PAIRED="$OUT/g4_paired.csv"
CELL="$OUT/g4_cell.csv"

echo "== 1/5 build_g4_cell"
python src/build_g4_cell.py --out-dir "$OUT"

echo "== 2/5 report_primary_slope"
python src/report_primary_slope.py --paired "$PAIRED" --cell "$CELL" --out-dir "$OUT"

echo "== 3/5 report_twin_arms"
python src/report_twin_arms.py --paired "$PAIRED" --out-dir "$OUT"

echo "== 4/5 report_dispersion_coverage"
python src/report_dispersion_coverage.py --cell "$CELL" --out-dir "$OUT"

echo "== 5/5 report_missingness"
PYTHONPATH=src python src/report_missingness.py --out-dir "$OUT"

echo
echo "rebuilt:"
ls -la "$OUT"
