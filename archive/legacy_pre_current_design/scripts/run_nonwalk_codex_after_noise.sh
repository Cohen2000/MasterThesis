#!/usr/bin/env bash
# Start the non-walk Codex arm once the noise-probe Codex arm is finished.
#
# Two Codex runs at once would share one quota and one CLI session, so they
# have to be serialised.  The wait is on *completeness*, not on the process
# exiting: a crashed noise run that stopped at 90/160 must not be treated as
# a green light, or the quota goes to the second experiment while the first
# stays unfinished.
set -uo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"
INTERVAL="${CODEX_CHAIN_INTERVAL:-300}"

complete() {
    PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import glob, json, sys
from pathlib import Path
from run_llm_v2 import is_complete_record

ids = {json.loads(l)["prompt_id"]
       for l in open("results/llm_noise_probe/prompts_subset160.jsonl")
       if l.strip()}
done = set()
for pattern in ("results/llm_noise_probe/answers_codex*.jsonl",
                str(Path.home() / "Dokumente/codex_noise/answers_codex*.jsonl")):
    for path in glob.glob(pattern):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("prompt_id") in ids and is_complete_record(rec):
                done.add(rec["prompt_id"])
print(len(done))
sys.exit(0 if len(done) >= len(ids) else 1)
PY
}

echo "$(date '+%H:%M:%S') warte auf die Codex-Rauschprobe (160/160)"
while ! complete >/dev/null 2>&1; do
    if ! pgrep -f 'run_codex_screen\.py.*noise' >/dev/null 2>&1; then
        n=$(complete || true)
        echo "$(date '+%H:%M:%S') Rauschprobe gestoppt bei ${n:-?}/160, nicht vollstaendig."
        echo "Non-Walk-Codex wird NICHT gestartet. Erst die Rauschprobe fortsetzen:"
        echo "  bash scripts/run_noise_codex.sh"
        exit 1
    fi
    sleep "$INTERVAL"
done

echo "$(date '+%H:%M:%S') Rauschprobe vollstaendig, starte Non-Walk-Codex (128 Prompts)"
mkdir -p results/nonwalk_llm_expansion/logs
exec bash scripts/run_nonwalk_expansion.sh codex
