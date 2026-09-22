#!/bin/bash
# Build and submit the Qwen workspace of one H-ablation level from the finished
# build_h_ablation.py output, using a completed final-v9 H workspace as the
# code template (same bundle, SPEC_COMMIT, engine, sbatch; only mainexp/run differs).
#   bash cluster/make_h_ablation_workspace.sh h040
set -euo pipefail
LEVEL="$1"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
TEMPLATE=$WS/panel888_access_v9_hknown_b050
EXP=panel888_h_ablation_$LEVEL
SRC=$WS/panel888_access_v9_cpu/results/panel888_h_ablation/$LEVEL/qwen_run
DST=$WS/$EXP
test -s "$SRC/requests.jsonl"
test ! -e "$DST" || { echo "$DST exists; reconcile instead of rebuilding" >&2; exit 1; }
mkdir -p "$DST/mainexp"
cp -a "$TEMPLATE"/{BUNDLE_SHA256SUMS,config,docs,src,SPEC_COMMIT,WORKTREE_STATUS} "$DST/"
cp -a "$TEMPLATE"/mainexp/{run_qwen_engine.py,qwen_engine.sbatch,submit_production.sh,status.py,build_archive.py,SPEC_COMMIT} "$DST/mainexp/"
cp -a "$SRC" "$DST/mainexp/run"
mkdir -p "$DST/mainexp/answers"
cd "$DST"
grep -v 'mainexp/run/' BUNDLE_SHA256SUMS > BUNDLE_SHA256SUMS.new
find ./mainexp/run -type f | sort | xargs sha256sum >> BUNDLE_SHA256SUMS.new
mv BUNDLE_SHA256SUMS.new BUNDLE_SHA256SUMS
sha256sum -c --quiet BUNDLE_SHA256SUMS
cd mainexp && bash submit_production.sh "$EXP" 6 all "$(cat ../SPEC_COMMIT)"
