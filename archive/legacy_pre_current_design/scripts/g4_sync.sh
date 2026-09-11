#!/usr/bin/env bash
# Pull the Qwen answer files from the cluster workspace.
#
# Copy direction is one way, cluster -> laptop: the cluster files are the
# originals and the local ones are copies, so nothing here can modify a
# generated answer. `--ignore-existing` is deliberately NOT used for the main
# shards, which grow as a generation is resumed; rsync updates them in place
# and the append-only property lives on the cluster side.
set -euo pipefail

cd "$(dirname "$0")/.."
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/llm_g3
DEST=results/final_run_g2/answers/qwen

mkdir -p "$DEST"
rsync -av \
    "uc3:$WS/answers_vllm_qwen36-27b_*.shard*.jsonl" \
    "uc3:$WS/answers_vllm_wrongdir_qwen36-27b_*.jsonl" \
    "$DEST/"

# The wrong-direction prompt file is its own frozen set with its own hashes;
# report_wrong_direction.py verifies every answer against it.
rsync -av --ignore-existing \
    "uc3:$WS/prompts_wrongdir_qwen36-27b_nothink_g0.jsonl" \
    results/final_run_g2/

echo
echo "synced. run scripts/g4_rebuild.sh to recompute."
