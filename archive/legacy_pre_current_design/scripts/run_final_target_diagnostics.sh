#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/masterarbeit-matplotlib}"

"$PYTHON" src/build_final_target_panel.py

"$PYTHON" src/analyze_target_robustness.py \
  --panel results/final_target_panel/panel32_final.csv \
  --data-root results/final_target_panel \
  --windows 4,5,8 \
  --out-dir results/target_diagnostics/w_robustness_final32

"$PYTHON" src/analyze_C_complementarity.py \
  --manifest results/final_target_panel/panel32_final.csv \
  --out-dir results/target_diagnostics/C_complementarity_final32 \
  --skip-input --jobs -1

"$PYTHON" src/report_target_diagnostics.py

echo
echo "Complete. Shareable report:"
echo "results/target_diagnostics/shareable/TARGET_DIAGNOSTICS_REPORT.md"
