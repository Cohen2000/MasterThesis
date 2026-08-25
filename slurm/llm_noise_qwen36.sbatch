#!/bin/bash
#SBATCH --job-name=noise_q36
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-15
#SBATCH --output=llm_noise_qwen36_%x_%A_%a.out

# Qwen3.6-27B arm of the reduced noise probe. Submit through
# submit_llm_noise_qwen.sh, which builds the dependency chain and sets
# STAGE and SHARDS.
#
# Hyperparameters match the frozen-suite Qwen run exactly, so this arm's error
# level stays comparable with the OFAT and the 420-prompt suite:
# nothink: temp=.7, top_p=.8, top_k=20, tiers 8k -> 16k -> 32k
# think:   temp=1,  top_p=.95, top_k=20, single pass at 126976 (STAGE=tfull)
#
# The think ladder from the OFAT is kept only for reproducing that run.  For
# new work use STAGE=tfull; the measured cost is 1.0 vs 1.9 GPU-hours per
# usable answer.
#
# --seed is the fallback only. Every prompt in the subset carries its own
# gen_seed, and run_llm_v2.py reseeds per generation from it. That is what
# makes the response arm measure anything: its three prompts share one prompt
# text, so a single fixed seed would decode them byte-identically and report
# zero response noise as an artifact.
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

PROMPTS="prompts_subset160.jsonl"
if [[ ! -f "$PROMPTS" ]]; then
    echo "prompt file not found: $PWD/$PROMPTS" >&2
    exit 1
fi

select_open() {
    local mode="$1" previous=()
    case "$mode" in
        n16)
            previous=(--answers 'answers_noise_qwen36_nothink_8k.shard*.jsonl') ;;
        n32)
            previous=(--answers 'answers_noise_qwen36_nothink_8k.shard*.jsonl'
                      --answers 'answers_noise_qwen36_nothink_16k.shard*.jsonl') ;;
        t64)
            previous=(--answers 'answers_noise_qwen36_think_32k.shard*.jsonl') ;;
        t128)
            previous=(--answers 'answers_noise_qwen36_think_32k.shard*.jsonl'
                      --answers 'answers_noise_qwen36_think_64k.shard*.jsonl') ;;
        *) echo "unknown escalation mode: $mode" >&2; return 2 ;;
    esac
    python3 select_llm_escalation.py --prompts "$PROMPTS" "${previous[@]}"
}

MODE=""; THINKING=""; TEMP=""; TOP_P=""; MAXTOK=""; OUT=""; EXTRA=()

case "$STAGE" in
    primary)
        if [[ "$TASK" -lt "$SHARDS" ]]; then
            MODE=nothink; THINKING=off; TEMP=0.7; TOP_P=0.8; MAXTOK=8192
            SHARD=$TASK
            OUT="answers_noise_qwen36_nothink_8k.shard${SHARD}.jsonl"
        else
            MODE=think; THINKING=on; TEMP=1.0; TOP_P=0.95; MAXTOK=32768
            SHARD=$((TASK - SHARDS))
            OUT="answers_noise_qwen36_think_32k.shard${SHARD}.jsonl"
        fi
        EXTRA=(--shard-index "$SHARD" --shard-count "$SHARDS")
        ;;
    tfull)
        # Ein Durchlauf am vollen Fenster, keine Leiter.  In der OFAT loeste
        # die 32k-Sprosse 1 von 72 Prompts und kostete dabei 38 GPU-Stunden:
        # der Median-Output von think liegt bei ~57k Tokens, also weit ueber
        # der Sprosse.  Da max_new_tokens eine Obergrenze und kein Ziel ist,
        # kostet ein Prompt, der bei 57k endet, unter einem 128k-Limit genau
        # dasselbe wie unter einem 64k-Limit -- die abgeschnittenen Versuche
        # sind vollstaendig weggeworfene Rechenzeit.
        SHARD=$TASK
        MODE=think; THINKING=on; TEMP=1.0; TOP_P=0.95; MAXTOK=126976
        OUT="answers_noise_qwen36_think_full.shard${SHARD}.jsonl"
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
                OUT="answers_noise_qwen36_nothink_16k.shard${SHARD}.jsonl" ;;
            n32)
                MODE=nothink; THINKING=off; TEMP=0.7; TOP_P=0.8; MAXTOK=32768
                OUT="answers_noise_qwen36_nothink_32k.shard${SHARD}.jsonl" ;;
            t64)
                MODE=think; THINKING=on; TEMP=1.0; TOP_P=0.95; MAXTOK=65536
                OUT="answers_noise_qwen36_think_64k.shard${SHARD}.jsonl" ;;
            t128)
                MODE=think; THINKING=on; TEMP=1.0; TOP_P=0.95; MAXTOK=126976
                OUT="answers_noise_qwen36_think_128k.shard${SHARD}.jsonl" ;;
        esac
        EXTRA=(--ids "$IDS" --shard-index "$SHARD" --shard-count "$SHARDS")
        ;;
    *)
        echo "STAGE must be primary, tfull, n16, n32, t64 or t128; got: $STAGE" >&2
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

records=0
[[ -f "$OUT" ]] && records=$(wc -l <"$OUT")
echo "done: stage=$STAGE mode=$MODE shard=$SHARD; $records records"
