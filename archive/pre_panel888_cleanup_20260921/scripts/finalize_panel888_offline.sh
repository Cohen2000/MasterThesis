#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
RUN=results/main_experiment/panel888_pwt_srw_20260921
REV=results/baseline_revision_panel888_pwt_srw_20260921
LOG=results/panel888_logs
.venv/bin/python scripts/run_main_api.py prepare --run "$RUN" --baselines "$REV/primary_baselines.json" --out "${RUN}_api" > "$LOG/api_prepare.log" 2>&1
.venv/bin/python scripts/evaluate_panel888_pairs.py --out "$REV/paired_control" > "$LOG/paired_control.log" 2>&1
.venv/bin/python scripts/analyze_panel888_census.py > "$LOG/census_diagnostics.log" 2>&1
.venv/bin/python scripts/build_panel888_seed_manifest.py > "$LOG/seed_audit.log" 2>&1
.venv/bin/python scripts/audit_panel888.py > "$LOG/independent_audit.log" 2>&1
.venv/bin/python scripts/seal_panel888.py > "$LOG/seal.log" 2>&1
