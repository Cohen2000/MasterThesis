#!/usr/bin/env bash
# Live status of the two local LLM runs: DeepSeek noise probe and OpenAI
# fairness pilot. Read-only -- safe to run at any time, also while the runs
# are writing. Nothing here reads, prints, or writes an API key.
#
#   bash scripts/live_status.sh              # one snapshot
#   watch -n 30 -c bash scripts/live_status.sh   # live view, Ctrl-C to leave
#
# The cost figure uses USD_PER_MTOK_OUT, derived from the observed billing of
# this account, not from a published price list. Treat it as an estimate.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-.venv/bin/python}"

# "laeuft (seit HH:MM)" so a fresh restart is distinguishable from a stall.
alive() {
    local pid
    pid="$(pgrep -f "run_llm_v2\.py.*$1" | head -1)"
    if [ -z "$pid" ]; then echo "GESTOPPT"; return; fi
    echo "laeuft/$(ps -o etime= -p "$pid" | tr -d ' ')"
}

DS_ALIVE=""
for i in 0 1 2 3; do DS_ALIVE+="$i:$(alive "shard-index $i") "; done
PILOT_ALIVE="$(alive 'gpt56sol_notools')"
NOISE_ALIVE="$(alive 'answers_gpt56sol\.jsonl')"

DS_ALIVE="$DS_ALIVE" PILOT_ALIVE="$PILOT_ALIVE" NOISE_ALIVE="$NOISE_ALIVE" \
USD_PER_MTOK_OUT="${USD_PER_MTOK_OUT:-23.45}" \
PYTHONPATH=src "$PYTHON" - <<'PY'
import collections, datetime as dt, glob, json, os

from run_llm_v2 import is_complete_record

NOW = dt.datetime.now()


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:                       # the last line may be half-written
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def bar(done, total, width=32):
    n = int(width * done / total) if total else 0
    return f"[{'#' * n}{'.' * (width - n)}] {done}/{total} ({done / total:.0%})" \
        if total else "[keine Prompts]"


def rate_and_eta(rows, open_n):
    """Records per hour over the last 30 minutes, and the resulting ETA."""
    stamps = []
    for r in rows:
        try:
            stamps.append(dt.datetime.fromisoformat(r["ts"]))
        except Exception:
            pass
    if not stamps:
        return "", ""
    recent = [s for s in stamps if (NOW - s).total_seconds() <= 1800]
    last = max(stamps)
    age = (NOW - last).total_seconds() / 60
    if len(recent) < 2:
        return f"letzte Antwort vor {age:.0f} min", ""
    span = (max(recent) - min(recent)).total_seconds() / 3600
    if span <= 0:
        return f"letzte Antwort vor {age:.0f} min", ""
    per_h = (len(recent) - 1) / span
    eta = open_n / per_h if per_h > 0 else 0
    return (f"{per_h:.1f} Antworten/h, letzte vor {age:.0f} min",
            f"ETA {eta:.1f} h" if open_n else "fertig")


print(f"== {NOW:%H:%M:%S} ==\n")

# ---------------- DeepSeek noise probe ----------------
prompts = {json.loads(l)["prompt_id"]: json.loads(l)
           for l in open("results/llm_noise_probe/prompts.jsonl") if l.strip()}
total = len(prompts)

allrows, done = [], set()
for path in sorted(glob.glob("results/llm_noise_probe/answers_dsv4flash.shard*.jsonl")):
    rows = load(path)
    allrows += rows
    done |= {r["prompt_id"] for r in rows if is_complete_record(r)} & set(prompts)

arms = collections.Counter()
for pid in done:
    for arm in prompts[pid]["arms"].split(","):
        arms[arm] += 1
reasons = collections.Counter(str(r.get("finish_reason")) for r in allrows)
speed, eta = rate_and_eta(allrows, total - len(done))

print("DeepSeek V4 Flash -- Rausch-Probe")
print(f"  {bar(len(done), total)}   {eta}")
print(f"  Antwort-Arm {arms['response']}/160 | Input-Arm {arms['input']}/160")
print(f"  finish_reason {dict(reasons)}")
print(f"  {speed}")
print(f"  Shards {os.environ['DS_ALIVE'].strip()}")

# ---------------- OpenAI fairness pilot ----------------
raw = open("results/llm_openai_pilot/pilot_prompt_ids_half.txt").read()
ids = [x.strip() for x in raw.replace("\n", ",").split(",") if x.strip()]
rows = load("results/llm_openai_pilot/answers_gpt56sol_notools.jsonl")
ok = {r["prompt_id"] for r in rows if is_complete_record(r)} & set(ids)
reasons = collections.Counter(str(r.get("finish_reason")) for r in rows)

out_tok = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
in_tok = sum((r.get("usage") or {}).get("prompt_tokens", 0) for r in rows)
usd_out = float(os.environ["USD_PER_MTOK_OUT"])
spent = out_tok / 1e6 * usd_out
per_gen = spent / len(rows) if rows else 0.0
speed, eta = rate_and_eta(rows, len(ids) - len(ok))

print("\nOpenAI gpt-5.6-sol -- Fairness-Pilot (Halb-Set)")
print(f"  {bar(len(ok), len(ids))}   {eta}")
print(f"  finish_reason {dict(reasons)}")
print(f"  Tokens: {in_tok:,} in / {out_tok:,} out"
      f"  ~${spent:.2f} bisher, ~${per_gen:.2f}/Generierung")
print(f"  noch offen {len(ids) - len(ok)} -> ~${per_gen * (len(ids) - len(ok)):.2f}")
print(f"  {speed}")
print(f"  Prozess: {os.environ['PILOT_ALIVE']}")

# ---------------- OpenAI response-noise arm ----------------
# 4 graphs x 5 repeats of one identical prompt: the strong-model check on the
# response-noise finding. Same pricing basis as the pilot.
npath = "results/llm_noise_probe/answers_gpt56sol.jsonl"
nids = [x.strip() for x in
        open("results/llm_openai_pilot/noise_prompt_ids.txt").read()
        .replace("\n", ",").split(",") if x.strip()]
rows = load(npath) if os.path.exists(npath) else []
ok = {r["prompt_id"] for r in rows if is_complete_record(r)} & set(nids)
reasons = collections.Counter(str(r.get("finish_reason")) for r in rows)
out_tok = sum((r.get("usage") or {}).get("completion_tokens", 0) for r in rows)
spent = out_tok / 1e6 * usd_out
speed, eta = rate_and_eta(rows, len(nids) - len(ok))
graphs = len({r.get("case_id") for r in rows if is_complete_record(r)})

print("\nOpenAI gpt-5.6-sol -- Antwortrausch-Arm (4 Graphen x 5)")
print(f"  {bar(len(ok), len(nids))}   {eta}")
print(f"  finish_reason {dict(reasons)}  | Faelle beantwortet {graphs}/4")
print(f"  {out_tok:,} out  ~${spent:.2f} bisher"
      f"  -> Rest ~${(spent / len(rows) if rows else 0.25) * (len(nids) - len(ok)):.2f}")
print(f"  {speed}")
print(f"  Prozess: {os.environ['NOISE_ALIVE']}")
PY
