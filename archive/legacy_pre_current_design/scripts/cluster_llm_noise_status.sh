#!/usr/bin/env bash
# Read-only Qwen noise-probe status on the UC3 login node.
set -euo pipefail

if command -v ws_find >/dev/null 2>&1 && ws_find llm_pilot >/dev/null 2>&1; then
    WS=$(ws_find llm_pilot)
else
    WS=$HOME/llm_pilot_ws
fi
cd "$WS/llm_v21"

echo "== Queue =="
squeue --me
echo

python3 - <<'PY'
import collections
import glob
import json
import sys

sys.path.insert(0, ".")
from run_llm_v2 import is_complete_record

prompt_path = "prompts_subset160.jsonl"
prompts = {json.loads(line)["prompt_id"]: json.loads(line)
           for line in open(prompt_path) if line.strip()}

patterns = {
    "nothink": ["answers_noise_qwen36_nothink_*.shard*.jsonl"],
    "think": ["answers_noise_qwen36_think_*.shard*.jsonl"],
}
for mode, pats in patterns.items():
    latest, complete = {}, {}
    files = []
    for pat in pats:
        files.extend(glob.glob(pat))
    for path in sorted(files):
        for line in open(path):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            pid = row.get("prompt_id")
            if pid not in prompts:
                continue
            latest[pid] = row
            if is_complete_record(row):
                complete[pid] = row
    cells = collections.Counter(prompts[pid]["input_kind"] for pid in complete)
    reasons = collections.Counter(str(row.get("finish_reason"))
                                  for row in latest.values())
    print(f"== Qwen {mode} ==")
    print(f"  complete {len(complete)}/{len(prompts)}; "
          f"attempted {len(latest)}; files {len(files)}")
    if cells:
        print("  cells: " + ", ".join(
            f"{key}={cells[key]}" for key in sorted(cells)))
    if reasons:
        print("  latest reasons: " + ", ".join(
            f"{key}={reasons[key]}" for key in sorted(reasons)))
    print()
PY
