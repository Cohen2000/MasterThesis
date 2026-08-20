#!/usr/bin/env python3
"""Build a paired 18-case, three-contract target-isolation prompt set."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import make_llm_prompts_v2 as base


VARIANTS = {
    "profile_only": ["rho_k2", "rho_k3", "rho_k4", "rho_k5"],
    "profile_plus_C": ["rho_k2", "rho_k3", "rho_k4", "rho_k5", "C_one_step"],
    "full9": base.SCHEMA_KEYS,
}


def choose(cases, per_strategy=6):
    picked = []
    for _, g in cases.groupby("strategy", sort=True):
        g = g.sort_values(["rho_W5_k2", "coverage", "case_id"])
        idx = np.linspace(0, len(g) - 1, per_strategy).round().astype(int)
        picked.append(g.iloc[idx])
    return pd.concat(picked, ignore_index=True)


def task(keys):
    descriptions = {
        "rho_k2": "rho_k2", "rho_k3": "rho_k3",
        "rho_k4": "rho_k4", "rho_k5": "rho_k5",
        "C_one_step": "C_one_step",
    }
    schema = {k: "number" for k in keys}
    return (
        "TASK\nEstimate for the FULL network: " +
        ", ".join(descriptions[k] for k in keys) +
        ". Each value must be in [0,1], and rho_k2 >= rho_k3 >= rho_k4 "
        ">= rho_k5 whenever those keys are requested. Reason as much as you "
        "need. The LAST line must be exactly one JSON object with precisely "
        "these keys and nothing after it:\n" +
        json.dumps(schema, separators=(",", ":"))
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="results/llm_v2/llm_cases.csv")
    ap.add_argument("--out", default="results/target_diagnostics/prompt_isolation/prompts.jsonl")
    ap.add_argument("--per-strategy", type=int, default=6)
    args = ap.parse_args()
    cases = pd.read_csv(args.cases)
    selected = choose(cases, args.per_strategy)
    records = []
    for _, row in selected.iterrows():
        prefix = "\n\n".join([
            base.DEFINITIONS,
            base.MECHANISM[row.strategy],
            "NOW THE ACTUAL OBSERVATION TO EVALUATE:",
            base.render_input(row, "mask"),
        ])
        for variant, keys in VARIANTS.items():
            prompt = prefix + "\n\n" + (base.TASK if variant == "full9" else task(keys))
            pid = hashlib.sha1(f"{row.case_id}|{variant}".encode()).hexdigest()[:12]
            records.append({
                "id": pid, "prompt_id": pid, "case_id": row.case_id,
                "condition": "target_isolation", "input_kind": "mask",
                "strategy": row.strategy, "block_group": row.block_group,
                "coverage_band": row.coverage_band, "budget": int(row.budget),
                "prompt_variant": variant, "required_keys": keys,
                "prompt": prompt,
            })
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    selected.to_csv(out.with_name("selected_cases.csv"), index=False)
    print(pd.DataFrame(records).groupby(["strategy", "prompt_variant"]).size())
    print(f"\nwrote {len(records)} paired prompts to {out}")


if __name__ == "__main__":
    main()
