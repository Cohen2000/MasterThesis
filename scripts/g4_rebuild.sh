#!/usr/bin/env bash
# Rebuild the seven G4 analysis modules from the current answer files.
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

echo "== 1/7 build_g4_cell"
python src/build_g4_cell.py --out-dir "$OUT"

echo "== 2/7 report_primary_slope"
python src/report_primary_slope.py --paired "$PAIRED" --cell "$CELL" --out-dir "$OUT"

echo "== 3/7 report_twin_arms"
python src/report_twin_arms.py --paired "$PAIRED" --out-dir "$OUT"

echo "== 4/7 report_dispersion_coverage"
python src/report_dispersion_coverage.py --cell "$CELL" --out-dir "$OUT"

echo "== 5/7 report_missingness"
PYTHONPATH=src python src/report_missingness.py --out-dir "$OUT"

# Freeze (k). Skips a model whose wrong-direction answers are not synced yet.
echo "== 6/7 report_wrong_direction"
PYTHONPATH=src python src/report_wrong_direction.py --out-dir "$OUT"

# Freeze (l): the profile, not just the headline component.
echo "== 7/7 report_profile_components"
PYTHONPATH=src python src/report_profile_components.py --out-dir "$OUT"

# The slide sheet must never drift from the tables it quotes.
echo "== checking docs/PRESENTATION_NUMBERS.md against the tables"
PYTHONPATH=src python src/check_presentation_numbers.py

echo
echo "rebuilt:"
ls -la "$OUT"
