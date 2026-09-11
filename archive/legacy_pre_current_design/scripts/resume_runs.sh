#!/usr/bin/env bash
# Restart every local LLM run where it stopped. Safe to call at any time:
# answer files are append-only and resume is by prompt_id, so a completed
# generation is never paid for twice and an interrupted one is simply retried.
#
#   bash scripts/resume_runs.sh            # start what is not already running
#   bash scripts/resume_runs.sh --status   # report only, start nothing
#
# Keys come from the environment (see ~/.config/keys and the loader in
# ~/.bashrc). Nothing here reads, prints, or writes a key.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROBE=results/llm_noise_probe
PILOT=results/llm_openai_pilot

running() { pgrep -f "run_llm_v2.py.*$1" >/dev/null 2>&1; }

echo "== current state =="
bash scripts/llm_noise_probe_status.sh 2>/dev/null | tail -6 || true
echo
if [ -f "$PILOT/answers_gpt56sol_notools.jsonl" ]; then
    echo "openai pilot records: $(wc -l < "$PILOT/answers_gpt56sol_notools.jsonl")"
fi
echo

[ "${1:-}" = "--status" ] && exit 0

# ---- DeepSeek noise probe, four shards ----
if [ -n "${NVIDIA_API_KEY:-}" ]; then
    for i in 0 1 2 3; do
        if running "shard-index $i"; then
            echo "dsv4 shard $i: already running"
        else
            echo "dsv4 shard $i: starting"
            NIM_MAXTOK=0 nohup bash scripts/run_llm_v21_nim.sh dsv4-flash \
                --prompts "$PROBE/prompts.jsonl" \
                --out "$PROBE/answers_dsv4flash.shard${i}.jsonl" \
                --shard-index "$i" --shard-count 4 \
                >> "$PROBE/dsv4_shard${i}.log" 2>&1 &
        fi
    done
else
    echo "NVIDIA_API_KEY not set -- skipping the DeepSeek shards"
fi

# ---- OpenAI fairness pilot (half set) ----
if [ -n "${OPENAI_API_KEY:-}" ]; then
    if running "gpt56sol_notools"; then
        echo "openai pilot: already running"
    else
        echo "openai pilot: starting"
        # 32768, not the runner default: every prompt still open after a pass
        # is one that ran out of budget, and retrying it at the same cap pays
        # for the same truncation twice.
        OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6-sol}" OPENAI_EFFORT="${OPENAI_EFFORT:-high}" \
        OPENAI_MAXTOK="${OPENAI_MAXTOK:-32768}" \
        nohup bash scripts/run_llm_openai.sh gpt56sol_notools \
            --ids "$(cat "$PILOT/pilot_prompt_ids_half.txt")" \
            >> "$PILOT/pilot_run.log" 2>&1 &
    fi
else
    echo "OPENAI_API_KEY not set -- skipping the pilot"
fi

# ---- OpenAI response-noise arm (4 graphs x 5 repeats of one prompt) ----
# Separate prompt and answer files from the pilot, so the two never contend
# for the same output and each resumes on its own prompt_id set.
if [ -n "${OPENAI_API_KEY:-}" ]; then
    if running "answers_gpt56sol\.jsonl"; then
        echo "openai noise arm: already running"
    else
        echo "openai noise arm: starting"
        OPENAI_MODEL="${OPENAI_MODEL:-gpt-5.6-sol}" OPENAI_EFFORT="${OPENAI_EFFORT:-high}" \
        OPENAI_MAXTOK="${OPENAI_MAXTOK:-32768}" \
        PROMPTS_FILE="$PROBE/prompts.jsonl" \
        OUT_FILE="$PROBE/answers_gpt56sol.jsonl" \
        nohup bash scripts/run_llm_openai.sh gpt56sol_noise \
            --ids "$(cat "$PILOT/noise_prompt_ids.txt")" \
            >> "$PROBE/gpt56sol_noise.log" 2>&1 &
    fi
fi

echo
echo "started in the background; follow with:"
echo "  tail -f $PROBE/dsv4_shard0.log"
echo "  tail -f $PILOT/pilot_run.log"
echo "  bash scripts/llm_noise_probe_status.sh"
