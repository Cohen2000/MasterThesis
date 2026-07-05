#!/bin/bash
#SBATCH --job-name=r1_14b
#SBATCH --partition=gpu_h100_short
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --output=r1_14b_%j.out

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

# Optional faster reasoning run (~45s/prompt -> ~2.3h). Requires prior download:
#   huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
python3 run_llm_pilot_v3.py \
    --prompts prompts.jsonl \
    --out answers_r1_14b.jsonl \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
    --temperature 0.6 --top-p 0.95 --seed 0 \
    --think-budget 2048
echo "run complete: $(wc -l < answers_r1_14b.jsonl) answers"
