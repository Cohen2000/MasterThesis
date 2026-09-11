#!/usr/bin/env python3
"""Does any prompt leak the identity of the arm that produced its sample?

If an arm slug, sampler name or config key appears in a `hidden` prompt then
`hidden` is not hidden and the factorial collapses: the mechanism condition is
supposed to be the only source of that information.  Two arms are especially
self-describing -- `event_sample_then_full_history` and
`node_panel_full_history` state their own mechanism in the name.

Only the `prompt` field is searched.  The record also carries `strategy`,
`case_id` and other bookkeeping, but none of that is sent to the model, and
searching it would produce a false alarm on every single prompt.

Budget values are searched as standalone numbers. `800`, `2500` and `9600`
each identify an arm family, but they also occur inside the observed-data
histogram as ordinary counts, so a bare substring match is useless; the check
distinguishes the two.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ARM_SLUGS = ("time_agnostic_t", "time_respecting", "recent_history_k20",
             "node_panel_full_history", "event_sample_then_full_history")

# Every sampler entry point and helper that could name a mechanism.
SAMPLER_NAMES = (
    "uniform_event_reservoir", "time_prefix_events",
    "time_random_window_events", "node_panel_full_history",
    "activity_proportional_dyad_full_history",
    "event_sample_then_full_history", "ego_recent_k_snowball",
    "neighbourhood_crawl", "prepare_dyad_histories", "prepare_events",
    "node_panel_size", "oracle_reservoir_ht", "run_walk", "build_index",
)

# Keys from config/final_run_g2.yaml and the case tables.
CONFIG_KEYS = (
    "target_budget", "node_panel_min_budget", "sample_seed", "walk_seed",
    "sample_rng_seed", "walk_rng_seed", "master_seed", "seed_slot",
    "base_strategy", "strategy", "instance_id", "case_id", "group_id",
    "recent_events_json_limit", "stationarity_bins", "budgets",
    "strategies", "final_run", "g2_arm_a", "g2_arm_b", "sampling_design",
    "task_class", "phase1_design", "phase2_design",
)

# A budget that identifies an arm family, matched as a standalone number so a
# histogram count of 800 does not raise a false alarm.
ARM_BUDGETS = {"800": "walks", "2500": "node_panel_full_history",
               "9600": "event_sample_then_full_history"}


def _standalone(value: str) -> re.Pattern:
    return re.compile(rf"(?<![\d.]){re.escape(value)}(?![\d.])")


def scan(prompt: str) -> dict[str, list[str]]:
    """Every leak class found in one prompt's text."""
    low = prompt.lower()
    found = defaultdict(list)
    for slug in ARM_SLUGS:
        if slug in low:
            found["arm_slug"].append(slug)
    for name in SAMPLER_NAMES:
        if name in low and name not in ARM_SLUGS:
            found["sampler_name"].append(name)
    for key in CONFIG_KEYS:
        if key in low and key not in ARM_SLUGS:
            found["config_key"].append(key)
    for value, arm in ARM_BUDGETS.items():
        if _standalone(value).search(prompt):
            found["arm_budget"].append(f"{value} ({arm})")
    return dict(found)


def sample_block(prompt: str) -> str | None:
    """The observed-data block, or None for a prompt that shows no sample."""
    if "OBSERVED DATA" not in prompt:
        return None
    start = prompt.index("OBSERVED DATA")
    end = prompt.index("TASK\n") if "TASK\n" in prompt else len(prompt)
    return prompt[start:end]


def structural_signature(block: str | None) -> str:
    """Shape of the sample block, ignoring the values inside it.

    This is not a leak in the same sense as a slug -- the shape of the data is
    a genuine property of the sample. It is reported so it can sit in the
    limitations beside the 0.858 detectability number rather than be
    discovered later.
    """
    if block is None:
        return "no sample"
    header = block.split("\n{", 1)[0]
    try:
        payload = json.loads(block[block.index("{"):].strip())
        keys = sorted(payload)
        shape = (f"json object, {len(keys)} keys, "
                 f"key format {'n,hexmask' if ',' in keys[0] else 'other'}")
    except (ValueError, json.JSONDecodeError, IndexError):
        shape = "unparsed"
    return f"{header.splitlines()[0]} | {shape}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts",
                        default="results/final_run_g2/prompts.jsonl")
    parser.add_argument("--out-dir", default="results_summary/g3")
    args = parser.parse_args()

    records = [json.loads(l) for l in
               Path(args.prompts).read_text().splitlines() if l.strip()]

    rows, leaks = [], []
    struct = defaultdict(set)
    for record in records:
        prompt = record["prompt"]
        found = scan(prompt)
        rows.append({"condition": record["condition"],
                     "arm": record["strategy"],
                     "leak_classes": len(found)})
        if found:
            leaks.append({"prompt_id": record["prompt_id"],
                          "condition": record["condition"],
                          "arm": record["strategy"],
                          **{k: "; ".join(sorted(set(v)))
                             for k, v in found.items()}})
        block = sample_block(prompt)
        struct[record["strategy"]].add(structural_signature(block))

    frame = pd.DataFrame(rows)
    by_condition = (frame.groupby("condition")
                    .agg(prompts=("leak_classes", "size"),
                         prompts_with_a_leak=("leak_classes",
                                              lambda s: int((s > 0).sum())))
                    .reset_index())

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    by_condition.to_csv(out / "prompt_leakage_by_condition.csv", index=False)
    leak_frame = pd.DataFrame(leaks)
    leak_frame.to_csv(out / "prompt_leakage_hits.csv", index=False)
    struct_frame = pd.DataFrame(
        [{"arm": arm, "distinct_block_shapes": len(sigs),
          "shape": " || ".join(sorted(sigs))} for arm, sigs in
         sorted(struct.items())])
    struct_frame.to_csv(out / "sample_block_structure.csv", index=False)

    print(f"scanned {len(records)} prompts, prompt text only\n")
    print(by_condition.to_markdown(index=False))
    print()
    if leaks:
        print(f"!! {len(leaks)} prompts carry at least one leak class:")
        print(leak_frame.head(20).to_markdown(index=False))
    else:
        print("no arm slug, sampler name, config key or arm-identifying "
              "budget value appears in any prompt")
    print()
    print("sample block structure by arm:")
    print(struct_frame.to_markdown(index=False))
    return 1 if leaks else 0


if __name__ == "__main__":
    raise SystemExit(main())
