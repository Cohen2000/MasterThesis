#!/usr/bin/env bash
# Progress of the LLM noise probe. Read-only; safe to run while jobs are live.
#
#   bash scripts/llm_noise_probe_status.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PROMPTS="${PROMPTS:-results/llm_noise_probe/prompts.jsonl}"
ANSWERS="${ANSWERS:-results/llm_noise_probe/answers_*.jsonl}"
PYTHON="${PYTHON:-.venv/bin/python}"

PROMPTS="$PROMPTS" ANSWERS="$ANSWERS" PYTHONPATH=src "$PYTHON" - <<'PY'
import collections, glob, json, os, sys
from run_llm_v2 import is_complete_record

prompts = {json.loads(l)["prompt_id"]: json.loads(l)
           for l in open(os.environ["PROMPTS"]) if l.strip()}
total = len(prompts)
print(f"probe: {total} prompts\n")

overall = set()
for path in sorted(glob.glob(os.environ["ANSWERS"])):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    done = {r["prompt_id"] for r in rows if is_complete_record(r)} & set(prompts)
    if not rows:
        continue
    overall |= done
    reasons = collections.Counter(str(r.get("finish_reason"))[:26] for r in rows)
    arms = collections.Counter()
    for pid in done:
        for arm in prompts[pid]["arms"].split(","):
            arms[arm] += 1
    bar = int(40 * len(done) / total)
    print(f"{os.path.basename(path)}")
    print(f"  [{'#' * bar}{'.' * (40 - bar)}] {len(done)}/{total} "
          f"({len(done) / total:.0%}), {total - len(done)} open")
    print(f"  records {len(rows)} | arms {dict(arms)}")
    print(f"  finish_reason {dict(reasons)}")
    graphs = {prompts[p]["instance_id"] for p in done}
    print(f"  graphs touched {len(graphs)}/32")
    print()

if not overall:
    print("no complete probe answers yet")
    raise SystemExit

# Aggregate across every answer file: shards and models are separate files,
# but the probe is one experiment and the arms are what have to fill up.
arms = collections.Counter()
per_graph = collections.Counter()
for pid in overall:
    meta = prompts[pid]
    for arm in meta["arms"].split(","):
        arms[arm] += 1
    per_graph[meta["instance_id"]] += 1
bar = int(40 * len(overall) / total)
print("=" * 60)
print(f"TOTAL (unique prompts, any model)")
print(f"  [{'#' * bar}{'.' * (40 - bar)}] {len(overall)}/{total} "
      f"({len(overall) / total:.0%})")
print(f"  response arm {arms['response']}/160 | input arm {arms['input']}/160")
full = sum(1 for g in per_graph.values() if g >= 9)
print(f"  graphs complete (all 9 cells) {full}/32, "
      f"touched {len(per_graph)}/32")
if arms['response'] >= 2 and arms['input'] >= 2:
    print("  enough for a first variance read; more repeats tighten it")
PY
