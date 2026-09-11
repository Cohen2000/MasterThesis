#!/bin/bash
#SBATCH --job-name=llmv2_r1_32b
#SBATCH --partition=gpu_h100_short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=llmv2_r1_32b_%A_%a.out

set -e
MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
if command -v ws_find >/dev/null 2>&1 && ws_find llm_pilot >/dev/null 2>&1; then
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
fi
source "$WS/venv/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
cd "$WS/pilot"

# Reasoning-Modell: langes Denken erlaubt (kein Budget-Cap), 4 Shards parallel;
# Resume: Job einfach erneut submitten, fertige prompt_ids werden uebersprungen.
python3 run_llm_v2.py --backend hf \
    --prompts prompts_v2.jsonl \
    --out "answers_r1_32b_v2.shard${SLURM_ARRAY_TASK_ID}.jsonl" \
    --model "$MODEL" \
    --max-new-tokens 12288 \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 4
echo "shard ${SLURM_ARRAY_TASK_ID} complete"
