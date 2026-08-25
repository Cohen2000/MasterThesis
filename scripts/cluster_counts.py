# Laeuft auf dem Login-Knoten. Gibt je Lauf eine Zeile: KEY done seen total
import glob, json, sys, collections
sys.path.insert(0, ".")
from run_llm_v2 import is_complete_record

def ids(path):
    try:
        return {json.loads(l)["prompt_id"] for l in open(path) if l.strip()}
    except OSError:
        return set()

def count(patterns, wanted):
    done, seen = set(), set()
    for pat in patterns:
        for p in glob.glob(pat):
            if "smoke" in p:
                continue
            for line in open(p):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                pid = r.get("prompt_id")
                if pid not in wanted:
                    continue
                seen.add(pid)
                if is_complete_record(r):
                    done.add(pid)
    return len(done), len(seen)

noise = ids("prompts_subset160.jsonl")
nwexp = ids("prompts_nonwalk_expansion_qwen.jsonl")
for key, pats, wanted in (
        ("NOISE_NOTHINK", ["answers_noise_qwen36_nothink_*.shard*.jsonl"], noise),
        ("NOISE_THINK", ["answers_noise_qwen36_think_*.shard*.jsonl"], noise),
        ("NWEXP_NOTHINK", ["answers_nwexp_qwen36_nothink*.shard*.jsonl"], nwexp)):
    d, s = count(pats, wanted)
    print(key, d, s, len(wanted))
