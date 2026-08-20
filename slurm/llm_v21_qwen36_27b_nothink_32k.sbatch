#!/bin/bash
#SBATCH --job-name=llmv21_q36_n32k
#SBATCH --partition=gpu_h100
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=llmv21_qwen36_27b_nothink_32k_%A_%a.out

# Adaptive third tier: rerun every case still incomplete after the 8k primary
# and the 16k second tier, with 32768 new tokens. The 12 remaining cases are
# all finish_reason=length at exactly 16384, i.e. the budget is the binding
# constraint, and 16384 is far below the 65536 the thinking arm already got.
# Sampling is identical to the earlier NoThink tiers, so the budget is the only
# variable; records carry max_tokens=32768 and stay conditionable. Results go
# to their own files, nothing earlier is rewritten.
set -e
MODEL="Qwen/Qwen3.6-27B"
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

# Both earlier tiers are passed, otherwise the selector would return every case
# the 8k primary left open and the 16k tier has since resolved.
IDS=$(python3 select_llm_escalation.py --prompts "$PROMPTS" \
    --answers 'answers_qwen36-27b_nothink.shard*.jsonl' \
    --answers 'answers_qwen36-27b_nothink_16k.shard*.jsonl')
if [ -z "$IDS" ]; then echo "no incomplete cases after the 16k tier"; exit 0; fi
echo "escalating $(echo "$IDS" | tr ',' '\n' | wc -l) cases to 32768 tokens"

OUT="answers_qwen36-27b_nothink_32k.shard${SLURM_ARRAY_TASK_ID}.jsonl"
python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" --out "$OUT" --model "$MODEL" \
    --ids "$IDS" --shard-index "$SLURM_ARRAY_TASK_ID" --shard-count 4 \
    --thinking off \
    --temperature 0.7 --top-p 0.8 --top-k 20 --seed 0 \
    --max-new-tokens 32768 --max-length-attempts 1
echo "done: $(wc -l < "$OUT") records in $OUT"
