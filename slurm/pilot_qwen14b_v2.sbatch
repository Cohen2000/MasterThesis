#!/bin/bash
#SBATCH --job-name=qwen14b_v2
#SBATCH --partition=gpu_h100_short
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=qwen14b_v2_%j.out

set -e
if command -v ws_find >/dev/null 2>&1 && ws_find llm_pilot >/dev/null 2>&1; then
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
fi
source "$WS/venv/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
cd "$WS/pilot"

python3 run_llm_pilot_v3.py \
    --prompts prompts.jsonl \
    --out answers_qwen14b.jsonl \
    --model Qwen/Qwen2.5-14B-Instruct
echo "run complete: $(wc -l < answers_qwen14b.jsonl) answers"
