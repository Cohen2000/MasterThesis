#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
RUN=results/main_experiment/panel888_pwt_srw_20260921
REV=results/baseline_revision_panel888_pwt_srw_20260921
LOG=results/panel888_logs
mkdir -p "$LOG"
.venv/bin/python scripts/run_main_offline.py --out "$RUN" > "$LOG/offline.log" 2>&1 &
MAIN_PID=$!
# Pool graph generation is independent of main graph preparation. Only the
# immutable initial source binding is needed before starting it.
while [ ! -f "$RUN/preparation_inputs.json" ]; do
 kill -0 "$MAIN_PID" 2>/dev/null || { wait "$MAIN_PID"; exit 1; }
 sleep 1
done
.venv/bin/python scripts/run_baseline_revision.py --out "$REV" --stage pool > "$LOG/pool.log" 2>&1 &
POOL_PID=$!
wait "$MAIN_PID"
.venv/bin/python scripts/verify_main_offline.py --run "$RUN" > "$LOG/verify.log" 2>&1
wait "$POOL_PID"
for stage in train dev main decompose decompose_main; do
 .venv/bin/python scripts/run_baseline_revision.py --out "$REV" --stage "$stage" > "$LOG/$stage.log" 2>&1
done
.venv/bin/python scripts/analyze_htime.py > "$LOG/h_sensitivity.log" 2>&1
.venv/bin/python scripts/analyze_srw.py > "$LOG/srw_diagnostics.log" 2>&1
.venv/bin/python scripts/analyze_panel888_controls.py > "$LOG/control_diagnostics.log" 2>&1
.venv/bin/python scripts/audit_prompt_freeze.py > "$LOG/prompt_audit.log" 2>&1
.venv/bin/python scripts/check_main_evaluation_mocks.py --run "$RUN" --baselines "$REV/primary_baselines.json" --out "${RUN}_mock" > "$LOG/mock.log" 2>&1
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v > "$LOG/tests.log" 2>&1
