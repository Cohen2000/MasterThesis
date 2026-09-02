"""Select the Codex prompts that the primary claim actually needs.

Codex is quota-bound (~5M tokens per 5-hour window, ~56k tokens per call), so
the 736-prompt set is not free. This script splits it into

  core  -- the 2x2 factorial {process described} x {direction stated}; this is
           the design the thesis claim rests on and it is run in full
  hold  -- ``metadata_only`` and ``mismatched``; both are auxiliary cells that
           qualify the claim rather than establish it, and both are deferred
           until quota allows

``--instances`` narrows the core further, to a stratified subsample of the
32 instances. The rule is fixed in advance and reads nothing but the case
metadata: every group keeps at least one instance, the remaining slots go to
the smallest groups first, and where a group has more instances than slots the
choice is drawn by an RNG seeded with the frozen master seed. Groups are the
resampling unit of the cluster bootstrap, so keeping all twelve matters more
than keeping every variant within one.

It also drops from the core anything that already has a usable Codex answer in
the Step 1 files. Step 1 used the same frozen prompts, the same CLI binary,
the same model and the same effort, so its generation-0 record *is* the Step 2
generation-0 record for those prompt_ids; re-running them would buy nothing.

The reduction is by cell, never by outcome: no answer from the current run is
read here, only prompt_ids.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "codex_screen"))

from run_codex_screen import usable_record  # noqa: E402

CORE_CONDITIONS = ("hidden", "mechanism", "direction_only", "mechanism_direction")
MASTER_SEED = 20260901


def stratified_instances(rows, target):
    """``target`` instances, every group represented, drawn reproducibly.

    Slots are handed out one per group first, so no group can fall out of the
    cluster bootstrap; the leftovers go to the smallest groups, which are the
    ones a single instance would represent worst. Within a group that has more
    instances than slots the pick is random but seeded, never by position --
    prompt_id order tracks the graph family, so taking the first would select
    on the covariate the analysis is about.
    """
    import random

    groups = {}
    for r in rows:
        groups.setdefault(r["group_id"], set()).add(r["instance_id"])
    groups = {g: sorted(v) for g, v in sorted(groups.items())}
    if target >= sum(len(v) for v in groups.values()):
        return {i for v in groups.values() for i in v}
    if target < len(groups):
        raise SystemExit(f"--instances {target} cannot cover {len(groups)} groups")

    slots = {g: 1 for g in groups}
    order = sorted(groups, key=lambda g: (len(groups[g]), g))
    left = target - len(groups)
    while left:
        progressed = False
        for g in order:
            if not left:
                break
            if slots[g] < len(groups[g]):
                slots[g] += 1
                left -= 1
                progressed = True
        if not progressed:
            break

    rng = random.Random(MASTER_SEED)
    keep = set()
    for g in sorted(groups):
        keep.update(rng.sample(groups[g], slots[g]))
    return keep


def usable_ids(paths, arm="notools"):
    ids = set()
    for path in paths:
        if not Path(path).exists():
            continue
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if usable_record(rec, arm):
                    ids.add(rec.get("prompt_id"))
    return ids


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(REPO / "results/final_run_g2/prompts_codex.jsonl"))
    ap.add_argument("--reuse", nargs="*", default=[
        str(REPO / "results/final_run_g2/answers/step1_codex_gen0.jsonl")])
    ap.add_argument("--out-core", default=str(REPO / "results/final_run_g2/prompts_codex_core.jsonl"))
    ap.add_argument("--out-hold", default=str(REPO / "results/final_run_g2/prompts_codex_hold.jsonl"))
    ap.add_argument("--arm", default="notools")
    ap.add_argument("--instances", type=int, default=0,
                    help="keep only this many of the 32 instances in the core, "
                         "stratified over groups (0 = all)")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.prompts) if l.strip()]
    rows.sort(key=lambda r: r["prompt_id"])
    reuse = usable_ids(args.reuse, args.arm)

    core = [r for r in rows if r["condition"] in CORE_CONDITIONS]
    hold = [r for r in rows if r["condition"] not in CORE_CONDITIONS]
    if args.instances:
        keep = stratified_instances(core, args.instances)
        hold += [r for r in core if r["instance_id"] not in keep]
        core = [r for r in core if r["instance_id"] in keep]
        print(f"instances     {len(keep)} of "
              f"{len({r['instance_id'] for r in rows})}: "
              + ", ".join(sorted(keep)))
    hold.sort(key=lambda r: r["prompt_id"])
    reused = [r for r in core if r["prompt_id"] in reuse]
    core = [r for r in core if r["prompt_id"] not in reuse]

    for path, sel in ((args.out_core, core), (args.out_hold, hold)):
        with open(path, "w") as fh:
            for r in sel:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"prompts      {len(rows)}")
    print(f"core         {len(core)}  -> {args.out_core}")
    print(f"reused       {len(reused)}  (usable Step 1 answers, not re-run)")
    print(f"on hold      {len(hold)}  -> {args.out_hold}")
    from collections import Counter
    for name, sel in (("core", core), ("hold", hold)):
        c = Counter(r["condition"] for r in sel)
        print(f"  {name}: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))


if __name__ == "__main__":
    main()
