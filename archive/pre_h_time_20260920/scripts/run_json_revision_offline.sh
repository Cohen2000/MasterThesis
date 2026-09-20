#!/bin/bash
# Offline only. Stage implementations validate dependencies; no .done shortcuts.
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=results/main_experiment/cells10_json_20260918
REV=results/baseline_revision_cells10_json_20260918
LOG=results/json_revision_20260918_logs
mkdir -p "$LOG" "$REV"
.venv/bin/python scripts/run_main_offline.py --out "$RUN" > "$LOG/offline.log" 2>&1
.venv/bin/python scripts/verify_main_offline.py --run "$RUN" > "$LOG/verify.log" 2>&1
for stage in pool train dev main decompose_main decompose; do
    env MAIN_RUN="$RUN" .venv/bin/python scripts/run_baseline_revision.py --out "$REV" --stage "$stage" > "$LOG/$stage.log" 2>&1
done
.venv/bin/python scripts/run_main_api.py prepare --run "$RUN" --baselines "$REV/primary_baselines.json" --out "${RUN}_api" > "$LOG/api_prepare.log" 2>&1
.venv/bin/python scripts/check_main_evaluation_mocks.py --run "$RUN" --baselines "$REV/primary_baselines.json" --out "${RUN}_mock_revision" > "$LOG/mock.log" 2>&1
printf '%s\n' OFFLINE_COMPLETE > "$LOG/STATUS"
