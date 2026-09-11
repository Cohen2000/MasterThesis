#!/bin/bash
# Progress overview of the V2.1 full runs. Use live via:
#   watch -n 60 bash scripts/llm_v21_status.sh
cd "$(dirname "$0")/.."

python3 - <<'PY'
import glob
import sys
sys.path.insert(0, "src")
import run_llm_v2 as r

files = sorted(f for f in glob.glob("results/llm_v21/answers_*.jsonl")
               if "_smoke" not in f)
if not files:
    print("noch keine Vollrun-Dateien")
for f in files:
    done = len(r.done_ids(f))
    total = sum(1 for _ in open(f))
    name = f.split("answers_")[1].removesuffix(".jsonl")
    print(f"{done:4d} / 420 komplett   ({total:5d} Records)   {name}")
PY

echo "--- aktive Läufe ---"
pgrep -af "run_llm_v21_(nim|gemini)\.sh" || echo "(keine)"
