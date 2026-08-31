#!/usr/bin/env python3
"""Render the frozen G2 prompt set from the G1 contract.

Every prompt is built by `prompt_contract_g1.build_prompt` from one case row,
so the sample a case shows is a property of (graph, arm, seed slot) and the
condition only changes the prose block in front of it.  Nothing here can
regenerate a sample per condition even by accident.

One record per prompt, not per call.  Generations are a runner-level repeat of
the same frozen text into a separate answers file, so the prompt set stays the
1,136 counted in `docs/FREEZE_2026-09.md` while the call total varies by model.
Putting the generation index in `prompt_id` instead would have made the frozen
prompt set depend on the model roster, which is exactly backwards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

import prompt_contract_g1 as C

# The conditions every arm and every model runs.
FACTORIAL = ("hidden", "direction_only", "mechanism", "mechanism_direction")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _record(row: dict, condition: str, stated_arm: str | None,
            subset: str) -> dict:
    text = C.build_prompt(row, condition, stated_arm)
    arm = str(row["strategy"])
    return {
        "prompt_id": f"{row['case_id']}|{condition}",
        "case_id": str(row["case_id"]),
        "instance_id": str(row["instance_id"]),
        "group_id": str(row["group_id"]),
        "strategy": arm,
        "condition": condition,
        "stated_arm": stated_arm or arm,
        "seed_slot": int(row.get("seed_slot", 0)),
        "subset": subset,
        # The runner filters on this; every prompt uses the one frozen input.
        "input_kind": "mask",
        "budget": int(row["budget"]),
        "coverage": float(row["coverage"]),
        "prompt": text,
        "prompt_sha256": _sha(text),
    }


def build(primary: pd.DataFrame, accepted: pd.DataFrame, config: dict,
          replication_graphs: list[str], run_historical: bool) -> list[dict]:
    fr = config["final_run"]
    out = []
    for row in primary.to_dict("records"):
        arm = row["strategy"]
        for condition in FACTORIAL:
            out.append(_record(row, condition, None, "factorial"))
        if arm in C.MISMATCH_PAIR:
            other = [a for a in C.MISMATCH_PAIR if a != arm][0]
            out.append(_record(row, "mismatched", other, "mismatched"))
        # Every arm renders the same metadata_only text, so it is emitted once
        # per graph rather than once per arm.
        if arm == list(fr["arms"])[0]:
            out.append(_record(row, "metadata_only", None, "metadata_only"))
        if arm in fr["subsets"]["irrelevant_context"]["arms"]:
            out.append(_record(row, "irrelevant_context", None,
                               "irrelevant_context"))
        if run_historical and arm in C.WALK_ARMS:
            out.append(_record(row, "disclosed_historical", None,
                               "disclosed_historical"))

    replication = accepted[
        accepted.instance_id.isin(replication_graphs) &
        (accepted.seed_slot > 0)]
    for row in replication.to_dict("records"):
        for condition in fr["subsets"]["seed_replication"]["conditions"]:
            record = _record(row, condition, None, "seed_replication")
            record["prompt_id"] = (f"{row['case_id']}|slot{row['seed_slot']}|"
                                   f"{condition}")
            out.append(record)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/final_run_g2.yaml")
    parser.add_argument("--primary",
                        default="results/final_run_g2/primary_cases.csv.gz")
    parser.add_argument("--accepted",
                        default="results/final_run_g2/final_cases.csv.gz")
    parser.add_argument("--replication", default=(
        "results_summary/g2/subset_seed_replication.csv"))
    parser.add_argument("--out", default="results/final_run_g2/prompts.jsonl")
    parser.add_argument("--summary-dir", default="results_summary/g3")
    parser.add_argument("--run-historical", action="store_true",
                        help="include the disclosed_historical bridge")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    primary = pd.read_csv(args.primary)
    accepted = pd.read_csv(args.accepted)
    replication = pd.read_csv(args.replication).instance_id.tolist()

    records = build(primary, accepted, config, replication,
                    args.run_historical)
    ids = [r["prompt_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise RuntimeError("duplicate prompt_id")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as handle:
        for record in sorted(records, key=lambda r: r["prompt_id"]):
            handle.write(json.dumps(record, separators=(",", ":")) + "\n")

    frame = pd.DataFrame([{k: v for k, v in r.items() if k != "prompt"}
                          for r in records])
    summary = (frame.groupby(["subset", "condition", "strategy"])
               .agg(prompts=("prompt_id", "size"),
                    graphs=("instance_id", "nunique")).reset_index())
    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_dir / "prompt_inventory.csv", index=False)
    print(f"wrote {out}: {len(records)} prompts")
    print(f"prompt file sha256: {_sha(out.read_text())}")
    print(summary.groupby("subset").prompts.sum().to_string())


if __name__ == "__main__":
    main()
