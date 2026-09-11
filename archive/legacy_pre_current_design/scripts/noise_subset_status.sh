#!/usr/bin/env bash
# Read-only coverage of the 160-prompt noise subset across all four arms.
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
PYTHON_BIN="${PYTHON_BIN:-$REPO/.venv/bin/python}"

PYTHONPATH=src "$PYTHON_BIN" - <<'PY'
import collections
import json
from pathlib import Path

from report_llm_noise import load_by_model
from run_llm_v2 import is_complete_record

probe = Path("results/llm_noise_probe")
subset = {}
with open(probe / "prompts_subset160.jsonl") as fh:
    for line in fh:
        if line.strip():
            r = json.loads(line)
            subset[r["prompt_id"]] = r

by_model = load_by_model(["results/llm_noise_probe/answers_*.jsonl"])
total = len(subset)
print(f"Noise subset: {total} prompts, "
      f"{len({r['instance_id'] for r in subset.values()})} graphs, "
      f"{len({r['group_id'] for r in subset.values()})} groups\n")

for model in sorted(by_model):
    recs = {pid: rec for pid, rec in by_model[model].items() if pid in subset}
    done = {pid for pid, rec in recs.items() if is_complete_record(rec)}
    if not recs:
        continue
    arms = collections.Counter()
    for pid in done:
        for arm in subset[pid]["arms"].split(","):
            arms[arm] += 1
    finish = collections.Counter(r.get("finish_reason") for r in recs.values())
    # A graph only contributes to a variance component once both of its arms
    # are complete, so graph counts are the honest progress measure.
    per_graph = collections.Counter(subset[pid]["instance_id"] for pid in done)
    full = sum(1 for inst, n in per_graph.items() if n == 5)
    print(f"{model}")
    print(f"  complete {len(done):3d}/{total}  ({100*len(done)/total:5.1f}%)"
          f" | attempted {len(recs):3d} | whole graphs {full}/32")
    print(f"  arms: response={arms['response']}/96 input={arms['input']}/96")
    print(f"  finish reasons: {dict(sorted(finish.items(), key=str))}\n")

missing = [m for m in ("codex-gpt-5.6-sol", "Qwen/Qwen3.6-27B")
           if not any(m.split("/")[-1].lower() in k.lower() for k in by_model)]
if missing:
    print("noch nicht gestartet: " + ", ".join(missing))
PY
