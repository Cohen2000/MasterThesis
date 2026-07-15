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
source "$WS/venv/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
cd "$WS/llm_v21"

PROMPTS=prompts.jsonl
# one frozen prompt per access strategy:
# time_agnostic_t, time_respecting, recent_history_k20
SMOKE_IDS="0b394cf2d923,3106eb7c74bb,90e26b753383"
