#!/bin/bash
# Upload the sealed, committed study to the cluster workspace $WS/<EXP> and verify it there.
#   bash scripts/cluster_bundle.sh panel888_pwt_srw_20260921                       (main study)
#   bash scripts/cluster_bundle.sh panel888_budget_sensitivity results/panel888_budget_sensitivity/run
# Requires a clean working tree whose HEAD contains the sealed sources, and an
# open ssh ControlMaster to uc3. Refuses to overwrite an existing $WS/<EXP>.
set -euo pipefail
cd "$(dirname "$0")/.."
EXP="${1:?experiment name}"
RUN="${2:-results/panel888/prepared}"
[[ "$EXP" =~ ^[a-zA-Z0-9_-]+$ ]] || { echo "invalid experiment name" >&2; exit 2; }
if [ -n "$(git status --porcelain)" ]; then echo 'Refusing bundle: commit the offline freeze first.' >&2; exit 1; fi
if [ "$RUN" = results/panel888/prepared ]; then
    .venv/bin/python scripts/seal_offline.py --verify
else
    grep -q '"verified": true' "$RUN/report.json" || { echo "Refusing bundle: $RUN is not audited" >&2; exit 1; }
fi
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
TMP=$(mktemp -d); B="$TMP/$EXP"
mkdir -p "$B/src/main_experiment" "$B/config" "$B/docs/results" "$B/mainexp/run/observations/sample" "$B/mainexp/logs"
cp src/main_experiment/*.py src/main_experiment/*.cpp "$B/src/main_experiment/"
cp -r config/main_experiment config/study.yaml config/datasets.yaml "$B/config/"
cp -r docs/results/panel888_offline "$B/docs/results/"
cp docs/PROTOCOL_PANEL888_20260921.md "$B/docs/"
cp scripts/run_qwen_engine.py cluster/qwen_engine.sbatch cluster/submit_production.sh cluster/status.py \
   cluster/build_archive.py "$B/mainexp/"
cp "$RUN/requests.jsonl" "$RUN/report.json" "$B/mainexp/run/"
cp "$RUN"/observations/sample/*.json "$B/mainexp/run/observations/sample/"
git status --porcelain > "$B/WORKTREE_STATUS"
git rev-parse HEAD | tee "$B/SPEC_COMMIT" > "$B/mainexp/SPEC_COMMIT"
(cd "$B" && find . -type f ! -name BUNDLE_SHA256SUMS | sort | xargs sha256sum > BUNDLE_SHA256SUMS)
(cd "$TMP" && tar czf "$EXP.tgz" "$EXP")
ssh -o BatchMode=yes uc3 "test ! -e $WS/$EXP || { echo 'refusing: $WS/$EXP exists'; exit 1; }"
scp -q "$TMP/$EXP.tgz" "uc3:$WS/${EXP}_bundle.tgz"
ssh -o BatchMode=yes uc3 "cd $WS && tar xzf ${EXP}_bundle.tgz && cd $EXP && sha256sum -c --quiet BUNDLE_SHA256SUMS && echo BUNDLE_OK \$(cat SPEC_COMMIT) \$(ls mainexp/run/observations/sample | wc -l) observations"
rm -rf "$TMP"
