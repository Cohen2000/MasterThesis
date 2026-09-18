#!/bin/bash -l
# The first install resolved against a Python 3.9 venv, where 0.11.0 was the
# newest compatible release. On Python 3.12 the current release is 0.29.0, and
# the model card asks for >= 0.19.0 for this architecture.
set -e
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_mainexp/bin/activate"
export TMPDIR=$WS/tmp PIP_CACHE_DIR=$WS/pip_cache HF_HOME=$WS/hf_cache
python -m pip install --upgrade "vllm==0.29.0"
python "$WS/mainexp/check_support.py"
python -m pip freeze > "$WS/mainexp/requirements.pinned.txt"
echo UPGRADE_COMPLETE
