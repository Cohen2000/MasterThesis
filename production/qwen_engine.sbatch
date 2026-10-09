#!/bin/bash -l
#SBATCH --job-name=qwen_engine
#SBATCH --partition=gpu_h100
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/engine_%x_%A_%a.out
#
# Generic Qwen generation job for one experiment directory $WS/<EXP>/mainexp.
#
#   cd $WS/<EXP>/mainexp && sbatch --array=0-5 qwen_engine.sbatch <EXP> 6 16 all
#   smoke test:  sbatch --partition=gpu_h100_short,gpu_h100 --time=00:30:00 --array=0 \
#                qwen_engine.sbatch <EXP> 6 16 all 8 answers_smoke 900 1500
#
# Positional: EXP SHARDS MAXSEQ ARMS(all|comma list) [LIMIT] [OUTDIR] [ADMIT_S] [STOP_S]
# Only gpu_h100 (94 GB): gpu_h100_il has 80 GB and cannot hold one request at the
# native 262144 context next to the weights. Requests are admitted during the first
# 80 minutes of the two hours; each finished answer is written at once, so a job
# killed by wall time records in-flight requests as technical failures on resume.
# Resubmitting admits only requests that have never been started; no retries.
# A login shell is mandatory (Lmod modules); positional arguments instead of --export.
set -euo pipefail

WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
EXP="${1:?experiment directory under \$WS}"
SHARDS="${2:-6}"
MAXSEQ="${3:-16}"
ARMS="${4:-all}"
LIMIT="${5:-0}"
OUTDIR="${6:-answers}"
ADMIT="${7:-4800}"
STOP="${8:-6600}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"
[ "$ARMS" = "all" ] && ARMS=""

module load devel/python/3.12.3-gnu-14.2
source "$WS/venv_mainexp/bin/activate"
export HF_HOME="$WS/hf_cache" HF_HUB_OFFLINE=1
export TMPDIR="$WS/tmp"
export CUDA_HOME="$WS/venv_mainexp/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_HAS_FLASHINFER_CUBIN=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
cd "$WS/$EXP/mainexp"
mkdir -p logs

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

srun python run_qwen_engine.py \
    --run  "$WS/$EXP/mainexp/run" \
    --out  "$WS/$EXP/mainexp/$OUTDIR" \
    --model "$WS/models/Qwen3.6-35B-A3B" \
    --arms "$ARMS" \
    --shard-index "$IDX" --shard-count "$SHARDS" \
    --max-num-seqs "$MAXSEQ" \
    --admit-seconds "$ADMIT" --stop-seconds "$STOP" \
    --limit "$LIMIT"
