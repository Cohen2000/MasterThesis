#!/bin/bash -l
# vLLM environment for the main experiment.
#
# The login shell matters: the modulefiles are Lmod .lua, and a non-interactive
# shell loads classic Tcl modules instead, which fails with "Magic cookie
# missing" and silently leaves the system Python 3.9 in place. A venv built that
# way pulls cp39 wheels.
set -e
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
export TMPDIR=$WS/tmp PIP_CACHE_DIR=$WS/pip_cache HF_HOME=$WS/hf_cache
mkdir -p "$TMPDIR"
# A killed pip can leave undeletable handles on Lustre; move the old tree aside
[ -d "$WS/venv_mainexp" ] && mv "$WS/venv_mainexp" "$WS/venv_mainexp.old.$$"
rm -rf "$WS"/venv_mainexp.old.* 2>/dev/null || true
python3 -m venv "$WS/venv_mainexp"
source "$WS/venv_mainexp/bin/activate"
echo "PYTHON $(python -V 2>&1)"
python -m pip install -q --upgrade pip setuptools wheel
python -m pip install "vllm==0.11.0"
python "$WS/mainexp/check_support.py"
echo INSTALL_COMPLETE
