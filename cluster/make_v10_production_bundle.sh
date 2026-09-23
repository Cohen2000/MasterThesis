#!/bin/bash
# Build the Qwen bundle from the sealed v10 preparation on uc3.
set -euo pipefail
COMMIT="${1:?committed source SHA}"
WS=/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot
SRC="$WS/panel888_v10_main"
EXP="$WS/panel888_access_v10_main"
test ! -e "$EXP" || { echo 'v10 production workspace already exists' >&2; exit 1; }
test -f "$SRC/results/panel888_v10/REQUEST_FREEZE.json"
python3 - "$SRC/results/panel888_v10/REQUEST_FREEZE.json" <<'PY'
import json,sys
x=json.load(open(sys.argv[1]))
assert x['verified'] and x['observations']==360 and x['requests']==4320
assert x['rhb_requests']==2592 and x['new_s_qwen_requests']==864
assert x['qwen_requests_to_generate']==2160
PY
mkdir -p "$EXP/src" "$EXP/config" "$EXP/docs/results" "$EXP/mainexp/run/observations" "$EXP/mainexp/logs"
cp -a "$SRC/src/main_experiment" "$EXP/src/"
cp -a "$SRC/config/main_experiment" "$EXP/config/"
cp -a "$SRC/config/study.yaml" "$SRC/config/datasets.yaml" "$EXP/config/"
cp -a "$SRC/results/panel888_v10/REQUEST_FREEZE.json" "$EXP/docs/results/"
cp -a "$SRC/scripts/run_qwen_engine.py" "$EXP/mainexp/"
cp -a "$SRC/cluster/qwen_engine.sbatch" "$SRC/cluster/submit_production.sh" \
      "$SRC/cluster/status.py" "$SRC/cluster/build_archive.py" "$EXP/mainexp/"
cp -a "$SRC/results/panel888_v10/prepared/requests.jsonl" \
      "$SRC/results/panel888_v10/prepared/report.json" "$EXP/mainexp/run/"
cp -a "$SRC/results/panel888_v10/prepared/observations/sample" "$EXP/mainexp/run/observations/"
printf '%s\n' "$COMMIT" > "$EXP/SPEC_COMMIT"
printf '%s\n' "$COMMIT" > "$EXP/mainexp/SPEC_COMMIT"
: > "$EXP/WORKTREE_STATUS"
(cd "$EXP" && find . -type f ! -name BUNDLE_SHA256SUMS | sort | xargs sha256sum > BUNDLE_SHA256SUMS)
(cd "$EXP" && sha256sum -c --quiet BUNDLE_SHA256SUMS)
echo BUNDLE_OK "$COMMIT" "$(find "$EXP/mainexp/run/observations/sample" -name '*.json' | wc -l)" observations
