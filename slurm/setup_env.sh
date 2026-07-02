#!/bin/bash
# One-time setup. Run on the LOGIN node (it has internet; compute nodes do NOT).
# This is only pip install, so it is fine on the login node.
set -e
cd "$(dirname "$0")"
# Default cluster python is 3.9.x and works. To use a module python instead,
# `module load devel/python/3.11` BEFORE running this script.
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || {
  echo "pip failed (likely xgboost). Retrying without xgboost..."
  grep -v xgboost requirements.txt > /tmp/req_noxgb.txt
  python -m pip install -r /tmp/req_noxgb.txt
}
echo ""
python - <<'PY'
import numpy,pandas,networkx,sklearn,matplotlib,yaml,tabulate
print("core deps OK")
try:
    import xgboost; print("xgboost OK")
except Exception:
    print("xgboost absent -> pilot_eval uses HistGradientBoosting (fine)")
PY
echo "ENV READY."
