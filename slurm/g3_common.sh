#!/bin/bash
# Shared environment for the G3 main-run cluster jobs. Sourced by the
# g3_qwen36_*.sbatch files; not a job by itself.
#
# Workspace layout (created by scripts/upload_g3_cluster.sh):
#   $WS/llm_g3/run_llm_v2.py
#   $WS/llm_g3/check_generation_noise.py
#   $WS/llm_g3/prompts_<model>_g<generation>.jsonl
# Answers are written next to them and copied back into
# results/final_run_g2/answers/ afterwards.
#
# Resume is per prompt_id *within one generation file*: generations live in
# separate answer files, so a rerun of generation 1 cannot be satisfied by
# generation 0's record. Only a record with a parseable final JSON carrying
# all nine keys counts as done.

if command -v ws_find >/dev/null 2>&1 && ws_find llm_pilot >/dev/null 2>&1; then
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
fi

# venv_v21 is built on the module Python 3.12; the system 3.9 is too old for
# current transformers, so the module has to be loaded before activating it.
V21_PYTHON_MODULE="devel/python/3.12.3-gnu-14.2"
if [ -d "$WS/venv_v21" ]; then
    if command -v module >/dev/null 2>&1; then
        module load "$V21_PYTHON_MODULE"
    fi
    source "$WS/venv_v21/bin/activate"
else
    source "$WS/venv/bin/activate"
fi
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
cd "$WS/llm_g3"

GEN="${GEN:?set GEN to the generation index}"
SHARDS="${SHARDS:-8}"
