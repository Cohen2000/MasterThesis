#!/bin/bash
# Build the cluster bundle for one experiment from the verified working-tree package and the
# offline run, upload it to $WS/<EXP> and verify every file there.
#   bash scripts/cluster_bundle.sh panel888_pwt_srw_20260921 results/main_experiment/panel888_pwt_srw_20260921
# Needs an open ssh ControlMaster to uc3 (the login requires an interactive OTP).
# Refuses to overwrite an existing $WS/<EXP>/mainexp/run.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -n "$(git status --porcelain)" ]; then
    echo 'Refusing bundle: commit the reviewed offline freeze first.' >&2; exit 1
fi
EXP="${1:?experiment name}"; RUN="${2:?offline run directory}"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
[[ "$EXP" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid experiment name"; exit 2; }
.venv/bin/python scripts/check_panel888_seal.py
.venv/bin/python scripts/verify_main_offline.py --run "$RUN"
TMP=$(mktemp -d); B="$TMP/$EXP"
mkdir -p "$B/src" "$B/mainexp/run/observations/sample" "$B/mainexp/logs"
mkdir -p "$B/src/main_experiment"
cp src/main_experiment/*.py src/main_experiment/*.cpp "$B/src/main_experiment/"
# Both committed source identity and exact bundle bytes are verified.
cp scripts/run_qwen_engine.py cluster/qwen_engine.sbatch cluster/submit_production.sh \
   cluster/status.py cluster/build_archive.py "$B/mainexp/"
mkdir -p "$B/config" "$B/docs/results"
cp -r config/main_experiment "$B/config/"
cp config/study.yaml config/datasets.yaml "$B/config/"
cp -r docs/results/panel888_20260921 "$B/docs/results/"
cp docs/PROTOCOL_PANEL888_20260921.md "$B/docs/"
cp "$RUN/requests.jsonl" "$RUN/report.json" "$B/mainexp/run/"
cp "$RUN"/observations/sample/*.json "$B/mainexp/run/observations/sample/"
git status --porcelain > "$B/WORKTREE_STATUS"
git rev-parse HEAD | tee "$B/SPEC_COMMIT" > "$B/mainexp/SPEC_COMMIT"
(cd "$B" && find . -type f ! -name BUNDLE_SHA256SUMS | sort | xargs sha256sum > BUNDLE_SHA256SUMS)
(cd "$TMP" && tar czf "$EXP.tgz" "$EXP")
ssh -o BatchMode=yes uc3 "test ! -e $WS/$EXP/mainexp/run || { echo 'refusing: $WS/$EXP/mainexp/run exists'; exit 1; }"
scp -q "$TMP/$EXP.tgz" "uc3:$WS/${EXP}_bundle.tgz"
ssh -o BatchMode=yes uc3 "cd $WS && tar xzf ${EXP}_bundle.tgz && cd $EXP && sha256sum -c --quiet BUNDLE_SHA256SUMS && echo BUNDLE_OK \$(cat SPEC_COMMIT) \$(ls mainexp/run/observations/sample | wc -l) observations"
rm -rf "$TMP"
