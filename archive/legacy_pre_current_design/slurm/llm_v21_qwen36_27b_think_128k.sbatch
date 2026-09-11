#!/bin/bash
#SBATCH --job-name=llmv21_q36_t128k
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --output=llmv21_qwen36_27b_think_128k_%A_%a.out

# Adaptive third tier: rerun every case still incomplete after the 32k primary
# and the 64k second tier. All 88 remaining cases are finish_reason=length at
# exactly 65536, and the elasticity is still live -- 32k resolved 23 of 420,
# 64k resolved 309 of 397 -- so the required-token distribution has a tail
# that plausibly terminates above 64k (as it did for mistral-high in July).
#
# Budget is 126976 rather than 131072 on purpose: the generation has to fit
# together with the prompt inside the model's position limit, and the frozen
# prompts run to about 2.5k tokens. Verify the ceiling before submitting, see
# the preflight in docs/LLM_V21_RUNS.md. Override with MAX_NEW_TOKENS.
#
# Memory is the tightest in the suite here: 27B bf16 weights plus a 128k KV
# cache. An OOM surfaces in the first minutes of the log and corrupts nothing,
# because this tier writes its own files and never touches the earlier ones.
set -e
MODEL="Qwen/Qwen3.6-27B"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-126976}"
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

# Both earlier tiers are passed, otherwise the selector would return every case
# the 32k primary left open and the 64k tier has since resolved.
IDS=$(python3 select_llm_escalation.py --prompts "$PROMPTS" \
    --answers 'answers_qwen36-27b_think.shard*.jsonl' \
    --answers 'answers_qwen36-27b_think_64k.shard*.jsonl')
if [ -z "$IDS" ]; then echo "no incomplete cases after the 64k tier"; exit 0; fi
echo "escalating $(echo "$IDS" | tr ',' '\n' | wc -l) cases to $MAX_NEW_TOKENS tokens"

OUT="answers_qwen36-27b_think_128k.shard${SLURM_ARRAY_TASK_ID}.jsonl"
python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" --out "$OUT" --model "$MODEL" \
    --ids "$IDS" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 16 \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 0 \
    --max-new-tokens "$MAX_NEW_TOKENS" --max-length-attempts 1
echo "done: $(wc -l < "$OUT") records in $OUT"
