#!/bin/bash
#SBATCH --job-name=g3_q36_think
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=g3_qwen36_think_g%A_%a.out

# Qwen3.6-27B thinking, G3 main run. 976 prompts per generation, 3 generations.
#
# One generation per submission, selected by GEN, because the generations must
# not share an answers file: resume is per prompt_id, so a second generation
# written into the first one's file would be skipped as already done and the
# response-noise measurement would silently collapse.
#
#   GEN=0 sbatch slurm/g3_qwen36_think.sbatch
#
# Sampling per the model card (thinking, general tasks): temperature 1.0,
# top_p 0.95, top_k 20. The per-record gen_seed varies across generations and
# is what makes two generations of one prompt differ at all; --seed is only the
# fallback for records without one, and every G3 record has one.
#
# 8 shards and a short 8 h wall time: the V2.1 experience was that long wall
# times block backfill (40 h and 48 h jobs sat at StartTime=N/A behind a 24 h
# job at identical priority). Resume makes a truncated pass cheap -- it costs
# one model reload -- so several short passes beat one long reservation.
set -e
MODEL="Qwen/Qwen3.6-27B"
COMMON=""
for d in "${SLURM_SUBMIT_DIR:-.}" "${SLURM_SUBMIT_DIR:-.}/slurm" "$(dirname "$0")"; do
    if [ -f "$d/g3_common.sh" ]; then COMMON="$d/g3_common.sh"; break; fi
done
if [ -z "$COMMON" ]; then echo "g3_common.sh not found" >&2; exit 1; fi
source "$COMMON"

TAG="qwen36-27b_think"
PROMPTS="prompts_${TAG}_g${GEN}.jsonl"
OUT="answers_${TAG}_g${GEN}.shard${SLURM_ARRAY_TASK_ID}.jsonl"
EXTRA=(--shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "$SHARDS")
if [ -n "${SMOKE:-}" ]; then
    if [ "${SLURM_ARRAY_TASK_ID:-0}" -ne 0 ]; then
        echo "smoke mode: array task ${SLURM_ARRAY_TASK_ID} skipped"; exit 0
    fi
    OUT="answers_${TAG}_g${GEN}_smoke.jsonl"
    EXTRA=(--ids "$SMOKE_IDS" --max-new-tokens "${SMOKE_MAXTOK:-8192}")
fi

python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 --seed 0 \
    --max-new-tokens "${MAXTOK:-32768}" \
    --max-length-attempts 1 \
    "${EXTRA[@]}"
echo "done: $(wc -l < "$OUT") records in $OUT"
