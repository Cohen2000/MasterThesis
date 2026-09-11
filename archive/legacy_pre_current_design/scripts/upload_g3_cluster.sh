#!/usr/bin/env bash
# Push the G3 run inputs to the bwUniCluster workspace.
#
# Only the runner, the checker and the prompt files go up; answers come back
# with scripts/fetch_g3_cluster.sh. Nothing in results/ is overwritten here.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"

HOST="${UC3_HOST:-uc3}"
REMOTE_WS=$(ssh "$HOST" 'ws_find llm_pilot')
DEST="$REMOTE_WS/llm_g3"
echo "workspace: $REMOTE_WS"

ssh "$HOST" "mkdir -p '$DEST'"
scp -q src/run_llm_v2.py src/check_generation_noise.py "$HOST:$DEST/"
scp -q results/final_run_g2/qwen/prompts_*.jsonl "$HOST:$DEST/"
scp -q slurm/g3_common.sh slurm/g3_qwen36_think.sbatch \
       slurm/g3_qwen36_nothink.sbatch scripts/count_g3_qwen_tokens.py \
       "$HOST:$REMOTE_WS/"
ssh "$HOST" "ls -la '$DEST' | head -20; echo; ls '$REMOTE_WS'/g3_*.sbatch"
echo "uploaded to $DEST"
