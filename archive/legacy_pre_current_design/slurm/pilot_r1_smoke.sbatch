#!/bin/bash
#SBATCH --job-name=r1smoke_v3
#SBATCH --partition=dev_gpu_h100
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --output=r1smoke_v3_%j.out

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

# 4 prompts: verifies forcing, per-prompt runtime (~100-110s expected) and parsing.
python3 run_llm_pilot_v3.py \
    --prompts prompts.jsonl \
    --out answers_r1_smoke_v3.jsonl \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --temperature 0.6 --top-p 0.95 --seed 0 \
    --think-budget 2048 \
    --limit 4
echo "smoke done:"
python3 -c "
import json
for l in open(\"answers_r1_smoke_v3.jsonl\"):
    r = json.loads(l)
    print(r[\"case_id\"], r[\"condition\"], \"pred=\", r[\"pred\"],
          \"forced=\", r[\"forced\"], f\"({r[\"gen_seconds\"]}s, {r[\"n_gen_tokens\"]} tok)\")"
