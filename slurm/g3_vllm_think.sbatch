#!/bin/bash
#SBATCH --job-name=g3v_think
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=g3v_think_%A_%a.out

# Qwen3.6-27B thinking on vLLM. One generation per submission, chosen by GEN:
# generations must not share an answers file, because resume is per prompt_id
# and a second generation written into the first one's file would be skipped.
#
#   GEN=0 sbatch slurm/g3_vllm_think.sbatch
#
# Budget 98,304. The ladder: 32,768 historically gave 49.8% complete, 65,536
# gave 84.9%. 65k was still cutting into the distribution rather than sitting
# above it -- p75 58,361 and p90 exactly on the cap.
#
# The reason to raise it is not the rate but which cases it removes. Truncated
# thinking cases are the ones needing the largest correction: median delta_i
# +0.432 against +0.147 on time_agnostic_t, +0.427 against +0.124 on
# time_respecting, +0.300 against +0.150 on recent_history_k20. That is the
# worst possible selection -- the cases the direction test is decided on drop
# out preferentially -- and it is invisible pooled, where the arms' opposite
# signs cancel to +0.042 against +0.045.
#
# Truncation also rose with how much prose a condition carries (hidden 9.8%,
# mechanism 15.9%, mechanism_direction 18.4%, direction_only 21.6%), which is
# differential dropout across the experimental contrast itself.
#
# Raised uniformly, like the non-thinking cap: max_tokens is a ceiling and
# generation stops at EOS. The model's context is 262,144, so 98k plus a ~3.3k
# prompt is far inside it. A truncated generation costs full compute and
# returns nothing, so this is not a straight cost/completeness trade.
set -e
WS=$(ws_find llm_pilot)
module load devel/python/3.12.3-gnu-14.2
module load devel/cuda/12.8
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
source "$WS/venv_vllm/bin/activate"
export HF_HOME="$WS/hf_cache"
export HF_HUB_OFFLINE=1
export VLLM_LOGGING_LEVEL=WARNING
cd "$WS/llm_g3"

GEN="${GEN:?set GEN to the generation index}"
SHARDS="${SHARDS:-4}"
TAG="qwen36-27b_think"
OUT="answers_vllm_${TAG}_g${GEN}.shard${SLURM_ARRAY_TASK_ID}.jsonl"

python3 run_llm_vllm_g3.py \
    --prompts "prompts_${TAG}_g${GEN}.jsonl" \
    --out "$OUT" \
    --model "Qwen/Qwen3.6-27B" \
    --thinking on \
    --temperature 1.0 --top-p 0.95 --top-k 20 \
    --max-new-tokens "${MAXTOK:-98304}" \
    --max-num-seqs "${SEQS:-32}" \
    --chunk "${CHUNK:-64}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "$SHARDS" \
    ${LIMIT:+--limit $LIMIT}
echo "done: $(wc -l < "$OUT") records in $OUT"
