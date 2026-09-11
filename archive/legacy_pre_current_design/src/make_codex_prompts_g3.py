#!/usr/bin/env python3
"""Codex prompt set for G3, with the cut ladder applied and recorded.

The cut order is prespecified in the G3 brief -- irrelevant_context,
disclosed_historical, mismatched, the third walk arm, direction_only -- and
`hidden`, `mechanism`, `node_panel_full_history` and
`event_sample_then_full_history` are never cut because they carry the primary
metric.  This applies the first two rungs plus seed_replication, and records
what was removed and what carries it instead.

seed_replication is not on the brief's ladder. Cutting it from Codex is
defensible only because it does not disappear: the G3 roster also gives it to
`qwen3.6-27b_think`, which runs three generations against Codex's two and is
now nearly free on the vLLM path. A seed-by-condition interaction is a variance
question, so more generations is the better home for it anyway. It is recorded
as a deviation from the ladder rather than folded in silently.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

# Ordered as the brief orders them, so a later cut extends this list rather
# than choosing freshly.
CUT_LADDER = ("irrelevant_context", "disclosed_historical", "mismatched",
              "third_walk_arm", "direction_only")
PROTECTED_CONDITIONS = ("hidden", "mechanism")
PROTECTED_ARMS = ("node_panel_full_history", "event_sample_then_full_history")


def apply_cuts(records: list[dict], cut_subsets: set[str],
               cut_arm: str | None) -> tuple[list[dict], pd.DataFrame]:
    keep, removed = [], []
    for record in records:
        if record["subset"] in cut_subsets:
            removed.append((record, f"subset {record['subset']} cut"))
        elif cut_arm and record["strategy"] == cut_arm:
            removed.append((record, f"arm {cut_arm} cut"))
        else:
            keep.append(record)

    for condition in PROTECTED_CONDITIONS:
        if not any(r["condition"] == condition for r in keep):
            raise RuntimeError(f"cut removed the protected condition {condition}")
    for arm in PROTECTED_ARMS:
        if not any(r["strategy"] == arm for r in keep):
            raise RuntimeError(f"cut removed the protected arm {arm}")

    counts = collections.Counter(reason for _, reason in removed)
    by_subset = collections.Counter(r["subset"] for r, _ in removed)
    log = pd.DataFrame([
        {"removed": subset, "prompts": n, "reason": counts_for(counts, subset),
         "carried_by": CARRIED_BY.get(subset, "nothing -- dropped")}
        for subset, n in sorted(by_subset.items())])
    return keep, log


def counts_for(counts, subset):
    for reason in counts:
        if subset in reason:
            return reason
    return "cut"


CARRIED_BY = {
    "seed_replication": "qwen3.6-27b_think, 3 generations",
    "irrelevant_context": "nothing -- dropped, ladder rung 1",
    "disclosed_historical": "nothing -- dropped, ladder rung 2",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts",
                        default="results/final_run_g2/prompts.jsonl")
    parser.add_argument("--cut", nargs="*", default=[
        "irrelevant_context", "disclosed_historical", "seed_replication"])
    parser.add_argument("--cut-arm", default=None)
    parser.add_argument("--out",
                        default="results/final_run_g2/prompts_codex.jsonl")
    parser.add_argument("--summary-dir", default="results_summary/g3")
    args = parser.parse_args()

    records = [json.loads(l) for l in
               Path(args.prompts).read_text().splitlines() if l.strip()]
    keep, log = apply_cuts(records, set(args.cut), args.cut_arm)

    out = Path(args.out)
    with out.open("w") as handle:
        for record in sorted(keep, key=lambda r: r["prompt_id"]):
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    summary = Path(args.summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    log.to_csv(summary / "codex_cut_log.csv", index=False)
    inventory = pd.DataFrame(
        [{"subset": s, "prompts": n} for s, n in
         sorted(collections.Counter(r["subset"] for r in keep).items())])
    inventory.to_csv(summary / "codex_scope.csv", index=False)

    print(f"wrote {out}: {len(keep)} prompts (from {len(records)})")
    print()
    print(log.to_markdown(index=False))
    print()
    print(inventory.to_markdown(index=False))
    print(f"\ncalls at 2 generations: {len(keep) * 2} "
          f"(was {len(records) * 2})")


if __name__ == "__main__":
    main()
