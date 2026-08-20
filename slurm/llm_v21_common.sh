#!/bin/bash
# Shared environment for the V2.1 LLM cluster runs. Sourced by the
# llm_v21_*.sbatch files; not a job by itself.
#
# Expected workspace layout (upload once from the repo root, login node):
#   $WS/llm_v21/run_llm_v2.py    <- src/run_llm_v2.py
#   $WS/llm_v21/prompts.jsonl    <- results/llm_v2/prompts.jsonl (frozen, 420)
# Answers are written next to them as answers_*.jsonl and copied back into
# results/llm_v21/ afterwards. Reruns resume by prompt_id: only records with
# a complete final JSON (all nine keys, not error/length) count as done.
#
# SMOKE=1 sbatch <file>  runs only the three smoke prompts (one per access
# strategy) into a separate *_smoke.jsonl and skips array tasks > 0.

if command -v ws_find >/dev/null 2>&1 && ws_find llm_pilot >/dev/null 2>&1; then
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
fi
# prefer the newer V2.1 venv if it exists (Qwen3.6 needs a recent
# transformers; the pinned pilot venv stays untouched for old jobs).
# venv_v21 is built on the module Python 3.12 (the system 3.9 is too old for
# current transformers), so the job has to load the same module first --
# otherwise the venv's interpreter and its shared libs are not found.
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
cd "$WS/llm_v21"

PROMPTS=prompts.jsonl
# one frozen prompt per access strategy:
# time_agnostic_t, time_respecting, recent_history_k20
# Overridable from the submit environment so a smoke can be narrowed to a
# single prompt when it has to fit into a 30 min dev partition.
SMOKE_IDS="${SMOKE_IDS:-0b394cf2d923,3106eb7c74bb,90e26b753383}"
