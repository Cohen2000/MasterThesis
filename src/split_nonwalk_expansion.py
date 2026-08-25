#!/usr/bin/env python3
"""Split the expanded non-walk prompt set into per-model and per-shard files.

The expansion is deliberately *nested*: the 64 prompts of the original Qwen
screen are contained in it with identical text, so the 128 answers already on
disk stay valid and only the remainder has to be generated. This script
verifies that containment rather than assuming it.

Sharding is done by writing separate files, not by passing a shard index. The
Gemini runner's completion loop counts what is left in the file it was handed,
so a shard expressed as an index into a larger file would never reach zero.
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write(path, rows):
    with open(path, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return len(rows)


def shard(rows, count):
    """Round-robin, so every shard spans the whole strategy/condition mix."""
    return [rows[i::count] for i in range(count)]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", required=True, help="full 512-prompt file")
    ap.add_argument("--codex", required=True, help="nested 16-graph subset")
    ap.add_argument("--previous", default=str(
        ROOT / "results/nonwalk_llm_qwen36_screen/prompts.jsonl"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gemini-shards", type=int, default=4)
    ap.add_argument("--deepseek-shards", type=int, default=8)
    args = ap.parse_args()

    rows = load(args.all)
    codex = load(args.codex)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Containment check. A silent break here would not fail loudly later: the
    # runs would simply re-answer 64 prompts nobody needed and the old answers
    # would drop out of the evaluation for want of a matching prompt_id.
    index = {r["prompt_id"]: r for r in rows}
    previous = load(args.previous) if Path(args.previous).exists() else []
    missing = [p for p in previous if p["prompt_id"] not in index]
    drifted = [p for p in previous
               if p["prompt_id"] in index
               and p["prompt"] != index[p["prompt_id"]]["prompt"]]
    if missing or drifted:
        raise SystemExit(
            f"expansion is not nested: {len(missing)} of {len(previous)} old "
            f"prompts absent, {len(drifted)} changed text")
    done = {p["prompt_id"] for p in previous}

    # Qwen already holds the original 64; only the remainder goes to the cluster.
    qwen = [r for r in rows if r["prompt_id"] not in done]
    write(out / "prompts_qwen.jsonl", qwen)
    write(out / "prompts_all.jsonl", rows)
    write(out / "prompts_codex.jsonl", codex)

    for name, source, count in (("gemini", rows, args.gemini_shards),
                                ("deepseek", rows, args.deepseek_shards)):
        write(out / f"prompts_{name}.jsonl", source)
        for i, part in enumerate(shard(source, count)):
            write(out / f"prompts_{name}.shard{i}.jsonl", part)

    strategies = len({r["strategy"] for r in rows})
    graphs = len({r["case_id"].split("|")[0] for r in rows})
    per = collections.Counter((r["strategy"], r["condition"]) for r in rows)
    print(f"{len(rows)} prompts -> {out}/prompts_all.jsonl")
    print(f"  {graphs} graphs x {strategies} strategies x 2 conditions")
    print(f"  cases per strategy/condition: {sorted(set(per.values()))}")
    print(f"  old screen prompts contained: {len(previous)}/{len(previous)}, "
          f"0 text deviations")
    print()
    print(f"  gemini    {len(rows):4d} prompts  ({args.gemini_shards} shards)")
    print(f"  deepseek  {len(rows):4d} prompts  ({args.deepseek_shards} shards)")
    print(f"  qwen      {len(qwen):4d} prompts  ({len(done)} reused)")
    codex_graphs = len({r["case_id"].split("|")[0] for r in codex})
    print(f"  codex     {len(codex):4d} prompts  ({codex_graphs} graphs, nested)")


if __name__ == "__main__":
    main()
