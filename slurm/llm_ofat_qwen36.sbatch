#!/bin/bash
#SBATCH --job-name=ofat_q36
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --output=llm_ofat_qwen36_%x_%A_%a.out

# Qwen3.6-27B OFAT stages. Submit through submit_llm_ofat_qwen.sh; it builds
# the dependency chain and supplies STAGE and SHARDS.
#
# The settings intentionally match the frozen-suite Qwen run exactly:
# nothink: temp=.7, top_p=.8, top_k=20, seed 0, tiers 8k -> 16k -> 32k
# think:   temp=1,  top_p=.95, top_k=20, seed 0, tiers 32k -> 64k -> 128k
# No repetition penalty is used here.  The penalty belonged to the separate
# non-walk screen; adding it only to the two new OFAT cells would confound the
# factor comparison.
set -euo pipefail

MODEL="Qwen/Qwen3.6-27B"
STAGE="${STAGE:-primary}"
SHARDS="${SHARDS:-8}"
TASK=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}

COMMON=""
for directory in "${SLURM_SUBMIT_DIR:-.}" \
                 "${SLURM_SUBMIT_DIR:-.}/slurm" \
                 "$(dirname "$0")"; do
    if [[ -f "$directory/llm_v21_common.sh" ]]; then
        COMMON="$directory/llm_v21_common.sh"
        break
    fi
done
if [[ -z "$COMMON" ]]; then
    echo "llm_v21_common.sh not found" >&2
    exit 1
fi
source "$COMMON"

PROMPTS="prompts_ofat_qwen.jsonl"
if [[ ! -f "$PROMPTS" ]]; then
    echo "prompt file not found: $PWD/$PROMPTS" >&2
    exit 1
fi

select_open() {
    local mode="$1" previous=()
    case "$mode" in
        n16)
            previous=(--answers 'answers_ofat_qwen36_nothink_8k.shard*.jsonl') ;;
        n32)
            previous=(--answers 'answers_ofat_qwen36_nothink_8k.shard*.jsonl'
                      --answers 'answers_ofat_qwen36_nothink_16k.shard*.jsonl') ;;
        t64)
            previous=(--answers 'answers_ofat_qwen36_think_32k.shard*.jsonl') ;;
        t128)
            previous=(--answers 'answers_ofat_qwen36_think_32k.shard*.jsonl'
                      --answers 'answers_ofat_qwen36_think_64k.shard*.jsonl') ;;
        *) echo "unknown escalation mode: $mode" >&2; return 2 ;;
    esac
    python3 select_llm_escalation.py --prompts "$PROMPTS" "${previous[@]}"
}

MODE=""
THINKING=""
TEMP=""
TOP_P=""
MAXTOK=""
OUT=""
EXTRA=()

case "$STAGE" in
    primary)
        if [[ "$TASK" -lt "$SHARDS" ]]; then
            MODE=nothink
            THINKING=off
            TEMP=0.7
            TOP_P=0.8
            MAXTOK=8192
            SHARD=$TASK
            OUT="answers_ofat_qwen36_nothink_8k.shard${SHARD}.jsonl"
        else
            MODE=think
            THINKING=on
            TEMP=1.0
            TOP_P=0.95
            MAXTOK=32768
            SHARD=$((TASK - SHARDS))
            OUT="answers_ofat_qwen36_think_32k.shard${SHARD}.jsonl"
        fi
        EXTRA=(--shard-index "$SHARD" --shard-count "$SHARDS")
        ;;
    n16|n32|t64|t128)
        SHARD=$TASK
        IDS=$(select_open "$STAGE")
        if [[ -z "$IDS" ]]; then
            echo "$STAGE: no incomplete prompts"
            exit 0
        fi
        case "$STAGE" in
            n16)
                MODE=nothink; THINKING=off; TEMP=0.7; TOP_P=0.8; MAXTOK=16384
                OUT="answers_ofat_qwen36_nothink_16k.shard${SHARD}.jsonl" ;;
            n32)
                MODE=nothink; THINKING=off; TEMP=0.7; TOP_P=0.8; MAXTOK=32768
                OUT="answers_ofat_qwen36_nothink_32k.shard${SHARD}.jsonl" ;;
            t64)
                MODE=think; THINKING=on; TEMP=1.0; TOP_P=0.95; MAXTOK=65536
                OUT="answers_ofat_qwen36_think_64k.shard${SHARD}.jsonl" ;;
            t128)
                MODE=think; THINKING=on; TEMP=1.0; TOP_P=0.95; MAXTOK=126976
                OUT="answers_ofat_qwen36_think_128k.shard${SHARD}.jsonl" ;;
        esac
        EXTRA=(--ids "$IDS" --shard-index "$SHARD" --shard-count "$SHARDS")
        ;;
    *)
        echo "STAGE must be primary, n16, n32, t64 or t128; got: $STAGE" >&2
        exit 2
        ;;
esac

echo "stage=$STAGE mode=$MODE shard=$SHARD/$SHARDS max=$MAXTOK"
python3 run_llm_v2.py --backend hf \
    --prompts "$PROMPTS" \
    --out "$OUT" \
    --model "$MODEL" \
    --thinking "$THINKING" \
    --temperature "$TEMP" --top-p "$TOP_P" --top-k 20 --seed 0 \
    --max-new-tokens "$MAXTOK" \
    --max-length-attempts 1 \
    "${EXTRA[@]}"

echo "done: stage=$STAGE mode=$MODE shard=$SHARD; $(wc -l < "$OUT") records"
