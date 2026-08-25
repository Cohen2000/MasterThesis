#!/usr/bin/env python3
"""Select the reduced noise-probe design from the frozen 288-prompt file.

Nothing is generated here. The frozen probe already contains, per graph, five
generations of one prompt text (response arm, walk seed 0, reps 1-5) and four
prompts from redrawn walks (input arm, seeds 1-4), with the (rep 1, seed 0)
cell shared by both arms. Dropping reps and seeds above `--max-draw` keeps that
structure intact at a smaller size:

    max_draw=3 -> response arm (1,0) (2,0) (3,0)   3 draws, identical text
                  input arm    (1,0) (2,1) (3,2)   3 draws, three texts
                  union        5 prompts per graph

Both arms keep the same number of draws and the same shared cell, so the
comparison between them still isolates the walk seed rather than the number of
generations. Prompt text, prompt_id and prompt_sha256 are copied verbatim, so
answers land in the same identifier space as the completed Gemini and DeepSeek
arms and can be scored against them without a translation step.

Prompts are emitted graph-blocked with graphs cycled across their groups, so a
run that stops early leaves whole graphs spread over all 12 groups rather than
a fragment of one group.
"""
import argparse
import collections
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROBE = ROOT / "results/llm_noise_probe"
FROZEN = PROBE / "prompts.jsonl"


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def group_interleaved(instances_by_group):
    """Graph order that cycles through groups before repeating one."""
    queues = {g: list(v) for g, v in instances_by_group.items()}
    order = []
    while any(queues.values()):
        for group in sorted(queues):
            if queues[group]:
                order.append(queues[group].pop(0))
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frozen", default=str(FROZEN))
    ap.add_argument("--out-dir", default=str(PROBE))
    ap.add_argument("--max-draw", type=int, default=3,
                    help="keep reps and walk seeds below this rank (default 3)")
    ap.add_argument("--graphs", type=int, default=0,
                    help="keep only the first N graphs of the interleaved "
                         "order; 0 keeps all")
    args = ap.parse_args()

    rows = load(args.frozen)
    keep = [r for r in rows
            if r["rep"] <= args.max_draw and r["walk_seed"] < args.max_draw]

    by_group = collections.defaultdict(list)
    seen = set()
    for r in rows:
        if r["instance_id"] not in seen:
            seen.add(r["instance_id"])
            by_group[r["group_id"]].append(r["instance_id"])
    order = group_interleaved(by_group)
    if args.graphs:
        order = order[:args.graphs]
    rank = {inst: i for i, inst in enumerate(order)}

    keep = [r for r in keep if r["instance_id"] in rank]
    keep.sort(key=lambda r: (rank[r["instance_id"]], r["rep"], r["walk_seed"]))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"subset{len(keep)}"
    prompt_path = out_dir / f"prompts_{tag}.jsonl"
    with open(prompt_path, "w") as fh:
        for r in keep:
            fh.write(json.dumps(r) + "\n")
    id_path = out_dir / f"ids_{tag}.txt"
    id_path.write_text(",".join(r["prompt_id"] for r in keep) + "\n")

    frozen_by_id = {r["prompt_id"]: r for r in rows}
    deviations = sum(1 for r in keep
                     if r["prompt"] != frozen_by_id[r["prompt_id"]]["prompt"])

    arms = collections.Counter(r["arms"] for r in keep)
    texts = len({r["prompt_sha256"] for r in keep})
    print(f"{len(keep)} prompts -> {prompt_path}")
    print(f"  graphs {len(rank)}  groups {len(by_group)}  distinct texts {texts}")
    print(f"  arms: {dict(sorted(arms.items()))}")
    print(f"  draws per graph: response {args.max_draw}, input {args.max_draw} "
          f"(shared cell rep1/seed0)")
    print(f"  text deviations against frozen: {deviations}")
    print(f"  ids -> {id_path}")


if __name__ == "__main__":
    main()
