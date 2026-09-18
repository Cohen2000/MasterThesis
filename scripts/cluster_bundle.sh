#!/bin/bash
# Build the cluster bundle for one experiment from the verified working-tree package and the
# offline run, upload it to $WS/<EXP> and verify every file there.
#   bash scripts/cluster_bundle.sh cells10_json_20260918 results/main_experiment/cells10_json_20260918
# Needs an open ssh ControlMaster to uc3 (the login requires an interactive OTP).
# Refuses to overwrite an existing $WS/<EXP>/mainexp/run.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="${1:?experiment name}"; RUN="${2:?offline run directory}"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
[[ "$EXP" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid experiment name"; exit 2; }
.venv/bin/python scripts/verify_main_offline.py --run "$RUN"
TMP=$(mktemp -d); B="$TMP/$EXP"
mkdir -p "$B/src" "$B/mainexp/run/observations/sample" "$B/mainexp/logs"
mkdir -p "$B/src/main_experiment"
cp src/main_experiment/*.py src/main_experiment/*.cpp "$B/src/main_experiment/"
# Bundle checksums include uncommitted source. SPEC_COMMIT alone is not its identity.
cp scripts/run_qwen_engine.py cluster/qwen_engine.sbatch cluster/submit_production.sh \
   cluster/status.py cluster/build_archive.py "$B/mainexp/"
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
