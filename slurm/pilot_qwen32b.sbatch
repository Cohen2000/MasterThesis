#!/bin/bash
#SBATCH --job-name=qwen32b
#SBATCH --partition=gpu_h100_short
#SBATCH --time=02:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=qwen32b_%j.out

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
    --out answers_qwen32b.jsonl \
    --model Qwen/Qwen2.5-32B-Instruct
echo "run complete: $(wc -l < answers_qwen32b.jsonl) answers"
