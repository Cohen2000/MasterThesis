#!/bin/bash
# Build the cluster bundle for one experiment from the committed package and the
# offline run, upload it to $WS/<EXP> and verify every file there.
#   bash scripts/cluster_bundle.sh cells10 results/main_experiment/cells10_20260917
# Needs an open ssh ControlMaster to uc3 (the login requires an interactive OTP).
# Refuses to overwrite an existing $WS/<EXP>/mainexp/run.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="${1:?experiment name}"; RUN="${2:?offline run directory}"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
TMP=$(mktemp -d); B="$TMP/$EXP"
mkdir -p "$B/src" "$B/mainexp/run/observations/sample" "$B/mainexp/logs"
git archive HEAD src/main_experiment | tar -x -C "$B"
cp scripts/run_qwen_engine.py cluster/qwen_engine.sbatch cluster/submit_production.sh \
   cluster/status.py cluster/build_archive.py "$B/mainexp/"
cp "$RUN/requests.jsonl" "$RUN/report.json" "$B/mainexp/run/"
cp "$RUN"/observations/sample/*.json "$B/mainexp/run/observations/sample/"
git rev-parse HEAD | tee "$B/SPEC_COMMIT" > "$B/mainexp/SPEC_COMMIT"
(cd "$B" && find . -type f ! -name BUNDLE_SHA256SUMS | sort | xargs sha256sum > BUNDLE_SHA256SUMS)
(cd "$TMP" && tar czf "$EXP.tgz" "$EXP")
ssh -o BatchMode=yes uc3 "test ! -e $WS/$EXP/mainexp/run || { echo 'refusing: $WS/$EXP/mainexp/run exists'; exit 1; }"
scp -q "$TMP/$EXP.tgz" "uc3:$WS/${EXP}_bundle.tgz"
ssh -o BatchMode=yes uc3 "cd $WS && tar xzf ${EXP}_bundle.tgz && cd $EXP && sha256sum -c --quiet BUNDLE_SHA256SUMS && echo BUNDLE_OK \$(cat SPEC_COMMIT) \$(ls mainexp/run/observations/sample | wc -l) observations"
rm -rf "$TMP"
