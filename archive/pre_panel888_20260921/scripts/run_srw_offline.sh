#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=results/main_experiment/cells10_srw_htime60_20260920
REV=results/baseline_revision_cells10_srw_htime60_20260920
LOG=results/srw_20260920_logs
mkdir -p "$LOG"
if [ ! -d "$RUN" ]; then .venv/bin/python scripts/prepare_srw_reuse.py > "$LOG/reuse.log" 2>&1; fi
.venv/bin/python scripts/run_main_offline.py --out "$RUN" > "$LOG/offline.log" 2>&1
.venv/bin/python scripts/verify_main_offline.py --run "$RUN" > "$LOG/verify.log" 2>&1
for stage in pool train dev main; do
    .venv/bin/python scripts/run_baseline_revision.py --out "$REV" --stage "$stage" > "$LOG/$stage.log" 2>&1
done
.venv/bin/python scripts/run_main_api.py prepare --run "$RUN" --baselines "$REV/primary_baselines.json" --out "${RUN}_api" > "$LOG/api_prepare.log" 2>&1
.venv/bin/python scripts/check_main_evaluation_mocks.py --run "$RUN" --baselines "$REV/primary_baselines.json" --out "${RUN}_mock" > "$LOG/mock.log" 2>&1
.venv/bin/python scripts/analyze_htime.py > "$LOG/h_sensitivity.log" 2>&1
