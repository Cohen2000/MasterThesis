#!/bin/bash
# One-time setup on the bwUniCluster LOGIN node (downloads are IO, not compute).
# Creates a workspace, a pinned venv (works with the default Python 3.9),
# and downloads the model into the workspace HF cache.
#
#   bash pilot_env_setup.sh
#
# Pins chosen deliberately: torch 2.5.1 (cu124 wheel, runs on the H100 driver,
# supports Python 3.9) + transformers 4.46.3 (supports Qwen2.5) + accelerate.
set -e

# workspace (fallback to $HOME if the ws_* tools are unavailable)
if command -v ws_allocate >/dev/null 2>&1; then
    ws_find llm_pilot >/dev/null 2>&1 || ws_allocate llm_pilot 60
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
    mkdir -p "$WS"
fi
echo "workspace: $WS"
mkdir -p "$WS/pilot"

# venv with pinned packages
if [ ! -d "$WS/venv" ]; then
    python3 -m venv "$WS/venv"
fi
source "$WS/venv/bin/activate"
pip install --upgrade pip
pip install "torch==2.5.1" "transformers==4.46.3" "accelerate==1.1.1" "huggingface_hub"

# model download into the workspace cache (~28 GB; fallback: 7B, ~15 GB)
export HF_HOME="$WS/hf_cache"
MODEL=${1:-Qwen/Qwen2.5-14B-Instruct}
echo "downloading $MODEL into $HF_HOME ..."
huggingface-cli download "$MODEL"

echo ""
echo "setup done. Next:"
echo "  1) copy prompts.jsonl and run_llm_pilot.py into $WS/pilot/"
echo "  2) sbatch pilot_smoke.sbatch   (from the slurm/ directory)"
