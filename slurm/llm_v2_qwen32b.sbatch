#!/bin/bash
#SBATCH --job-name=llmv2_qwen32b
#SBATCH --partition=gpu_h100_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=llmv2_qwen32b_%j.out

set -e
MODEL="Qwen/Qwen2.5-32B-Instruct"   # ggf. auf neueres Qwen im HF-Cache umstellen
if command -v ws_find >/dev/null 2>&1 && ws_find llm_pilot >/dev/null 2>&1; then
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
fi
source "$WS/venv/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
cd "$WS/pilot"

python3 run_llm_v2.py --backend hf \
    --prompts prompts_v2.jsonl \
    --out answers_qwen32b_v2.jsonl \
    --model "$MODEL" \
    --max-new-tokens 3072
echo "run complete: $(wc -l < answers_qwen32b_v2.jsonl) answers"
