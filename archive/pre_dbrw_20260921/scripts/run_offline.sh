#!/bin/bash
# The complete offline study, from raw data to the sealed pre-inference freeze.
# No model is called. Every stage writes into results/panel888/<stage>; the run
# refuses to start if results/panel888 already exists (recompute from scratch).
#   bash scripts/run_offline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=.venv/bin/python
OUT=results/panel888
LOG=$OUT/logs
if [ -e "$OUT" ]; then echo "$OUT exists; move it away to recompute" >&2; exit 1; fi
mkdir -p "$LOG"
step() { local name=$1; shift; echo "$(date -u +%FT%TZ) $name"; "$@" > "$LOG/$name.log" 2>&1; }

# 1-4: prepared study, training pool, ExtraTrees folds, reference predictions
step prepare     $PY scripts/prepare_study.py
step pool        $PY scripts/build_references.py pool
step train       $PY scripts/build_references.py train
step references  $PY scripts/build_references.py references
# 5: fixed offline diagnostics
step decomposition   $PY scripts/diagnose_decomposition.py
step history         $PY scripts/diagnose_history.py
step srw             $PY scripts/diagnose_srw.py
step null_model      $PY scripts/diagnose_null_model.py
step windows         $PY scripts/diagnose_windows.py
step mixture_bounds  $PY scripts/diagnose_mixture_bounds.py
# 6: Sol/DeepSeek ledger (prepared, dispatch disabled) and paired controls of the references
step api_prepare       $PY scripts/run_api.py prepare
step paired_references $PY scripts/evaluate_paired_controls.py --out "$OUT/paired_control_references"
# 7: audits and tests, then the seal
step mock_evaluation $PY scripts/check_evaluation_mocks.py
step audit           $PY scripts/audit_offline.py
step tests           env PYTHONPATH=src $PY -m unittest discover -s tests -v
step seal            $PY scripts/seal_offline.py
echo "$(date -u +%FT%TZ) offline study complete"
