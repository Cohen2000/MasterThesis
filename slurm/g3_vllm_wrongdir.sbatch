#!/bin/bash
#SBATCH --job-name=g3v_wrongdir
#SBATCH --partition=gpu_h100
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --array=0
#SBATCH --output=g3v_wrongdir_%A_%a.out

# Qwen3.6-27B thinking on vLLM. One generation per submission, chosen by GEN:
# generations must not share an answers file, because resume is per prompt_id
# and a second generation written into the first one's file would be skipped.
#
#   GEN=0 sbatch slurm/g3_vllm_think.sbatch
#
# The missing cell: correct process description, inverted direction claim.
# 64 prompts x 3 generations x 2 models = 384 calls, minutes on the vLLM path.
#
# Bottom of the cut ladder by instruction: it must not compete with the main
# run for GPU, so the keeper only launches it once every main generation is
# complete. Its own prompt file and its own hash; the frozen set is untouched.
#
# MODEL selects the roster row, since thinking and non-thinking differ in
# sampling and budget.
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
TAG="${TAG:?set TAG to qwen36-27b_think or qwen36-27b_nothink}"
case "$TAG" in
    *_think)   THINK=on;  TEMP=1.0; TOPP=0.95; MAXTOK_DEFAULT=98304 ;;
    *_nothink) THINK=off; TEMP=0.7; TOPP=0.8;  MAXTOK_DEFAULT=24576 ;;
    *) echo "unknown TAG: $TAG" >&2; exit 2 ;;
esac
OUT="answers_vllm_wrongdir_${TAG}_g${GEN}.jsonl"

python3 run_llm_vllm_g3.py \
    --prompts "prompts_wrongdir_${TAG}_g${GEN}.jsonl" \
    --out "$OUT" \
    --model "Qwen/Qwen3.6-27B" \
    --thinking "$THINK" \
    --temperature "$TEMP" --top-p "$TOPP" --top-k 20 \
    --max-new-tokens "${MAXTOK:-$MAXTOK_DEFAULT}" \
    --max-num-seqs "${SEQS:-32}" \
    --chunk "${CHUNK:-64}"
echo "done: $(wc -l < "$OUT") records in $OUT"
