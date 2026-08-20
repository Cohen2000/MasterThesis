#!/bin/bash
#SBATCH --job-name=llmv21_q36_nothink
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=llmv21_qwen36_27b_nothink_%A_%a.out

# Qwen3.6-27B, non-thinking mode (enable_thinking=False via chat template).
# Official instruct-mode sampling (model card):
#   temperature 0.7, top_p 0.8, top_k 20, min_p 0
#   (presence_penalty 1.5 is also recommended, but HF transformers generate()
#   has no presence_penalty -- documented open point in docs/LLM_V21_RUNS.md)
# Budget 3072 -> 8192 (2026-07-17): the mistral-none run showed non-thinking
# models often enumerate edges/windows verbosely before the JSON; 8192
# matches dsv4-nothink.
# Partition 2026-07-19: moved from gpu_h100_short (00:30:00 cap) to
# gpu_h100.
# Sharded 2026-07-19 after the cluster smoke: the observed generation rate
# is only ~16.5 tok/s (Qwen3.6 falls back to the torch attention path --
# flash-linear-attention/causal-conv1d are not installed), and the smoke
# filled its whole 4096 budget, so a prompt costs up to ~8 min. 420
# prompts in a single job would be ~58 h; 4 shards bring that to ~15 h.
# Time limit 24:00:00 -> 08:00:00 (2026-07-20): see the backfill note in
# llm_v21_qwen36_27b_think.sbatch. A ~105-prompt shard needs ~2 passes.
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

OUT="answers_qwen36-27b_nothink.shard${SLURM_ARRAY_TASK_ID}.jsonl"
EXTRA=(--shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count 4)
if [ -n "${SMOKE:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
        echo "smoke mode: array task ${SLURM_ARRAY_TASK_ID} skipped"; exit 0
    fi
    OUT="answers_qwen36-27b_nothink_smoke.jsonl"
    # SMOKE_MAXTOK trims the generation budget so the plumbing check fits
    # into a short dev-partition slot; the full run keeps the real budget.
    EXTRA=(--ids "$SMOKE_IDS" --max-new-tokens "${SMOKE_MAXTOK:-8192}")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking off \
    --temperature 0.7 --top-p 0.8 --top-k 20 --seed 0 \
    --max-new-tokens 8192 \
    --max-length-attempts 1 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
