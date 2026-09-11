#!/bin/bash
#SBATCH --job-name=llmv21_q36_think
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=llmv21_qwen36_27b_think_%A_%a.out

# Qwen3.6-27B, thinking mode (main comparison, replaces Qwen2.5).
# Official thinking-mode sampling (model card, general tasks):
#   temperature 1.0, top_p 0.95, top_k 20, min_p 0.
# Thinking output is long -> 8 shards; budget 16384 -> 32768 (2026-07-17,
# "no thinking cap" decision, matches the mistral-high API budget; the
# mistral-high run showed successful reasoning pressing against 32k).
# Truncated or incomplete answers do not count as done, so resubmitting
# the same job continues where it stopped.
# Partition 2026-07-19: gpu_h100_short caps at 00:30:00 (verified with
# scontrol), far too little for 52 GB of weights plus 32k-token thinking
# generations -- moved to gpu_h100 (MaxTime 3-00:00:00).
# Time limit 40:00:00 -> 08:00:00 (2026-07-20): after ~24 h nothing had
# started. `squeue --start` showed the 24 h nothink job with a concrete
# reservation but the 40 h and 48 h jobs with StartTime=N/A, at identical
# priority (12641) -- i.e. the wall time, not the priority, was blocking
# backfill. Short passes plus a longer --dependency=afterany chain is
# equivalent here because resume skips completed prompt_ids, so a pass
# that hits its limit costs only the model reload.
set -e
MODEL="Qwen/Qwen3.6-27B"
# sbatch spools the script away, so $0 does not point at slurm/ -- find
# llm_v21_common.sh relative to the submit directory instead
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/llm_v21_common.sh" ]; then COMMON="$d/llm_v21_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "llm_v21_common.sh not found" >&2; exit 1; fi
source "$COMMON"

OUT="answers_qwen36-27b_think.shard${SLURM_ARRAY_TASK_ID}.jsonl"
EXTRA=(--shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 8)
if [ -n "${SMOKE:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
        echo "smoke mode: array task ${SLURM_ARRAY_TASK_ID} skipped"; exit 0
    fi
    OUT="answers_qwen36-27b_think_smoke.jsonl"
    # SMOKE_MAXTOK trims the generation budget so the plumbing check fits
    # into a short dev-partition slot; the full run keeps the real budget.
    EXTRA=(--ids "$SMOKE_IDS" --max-new-tokens "${SMOKE_MAXTOK:-32768}")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 0 \
    --max-new-tokens 32768 \
    --max-length-attempts 1 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
