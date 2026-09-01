#!/bin/bash
#SBATCH --job-name=g3v_nothink
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0-3
#SBATCH --output=g3v_nothink_%A_%a.out

# Qwen3.6-27B non-thinking on vLLM. One generation per submission, chosen by GEN:
# generations must not share an answers file, because resume is per prompt_id
# and a second generation written into the first one's file would be skipped.
#
#   GEN=0 sbatch slurm/g3_vllm_think.sbatch
#
# Budget 24,576. The ladder so far: 8,192 historically gave 53% complete,
# 16,384 gave 93.4%. 16k was still not enough, and the residual truncation was
# not random -- it rose with how much prose the condition carried (hidden 3.9%,
# mechanism 7.7%, mechanism_direction 11.8%) and it hit the low-coverage cases
# on arm B hardest (truncated cases had median coverage 0.062 against 0.251 for
# complete ones). Differential dropout across the experimental conditions is a
# threat to the primary contrast, not a cosmetic rate.
#
# Raised uniformly rather than per arm: max_tokens is a ceiling and generation
# stops at EOS, so an arm that finishes well under it costs nothing extra,
# and a uniform cap avoids introducing another arm-dependent parameter.
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
TAG="qwen36-27b_nothink"
OUT="answers_vllm_${TAG}_g${GEN}.shard${SLURM_ARRAY_TASK_ID}.jsonl"

python3 run_llm_vllm_g3.py \
    --prompts "prompts_${TAG}_g${GEN}.jsonl" \
    --out "$OUT" \
    --model "Qwen/Qwen3.6-27B" \
    --thinking off \
    --temperature 0.7 --top-p 0.8 --top-k 20 \
    --max-new-tokens "${MAXTOK:-24576}" \
    --max-num-seqs "${SEQS:-32}" \
    --chunk "${CHUNK:-64}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" --shard-count "$SHARDS" \
    ${LIMIT:+--limit $LIMIT}
echo "done: $(wc -l < "$OUT") records in $OUT"
