#!/usr/bin/env python3
"""Emit paired sample/full and metadata-only prompts for selected cases."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from nonwalk_prompt_contract import (
    assert_prompt_metadata_parity,
    render_nonwalk_prompt,
)


def _stable_order(value: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def select_cases(cases: pd.DataFrame, strategies: list[str], budget: int,
                 sample_seed: int | None = None,
                 instances_per_block: int = 0,
                 selection_seed: int = 0) -> pd.DataFrame:
    """Select a reproducible, paired screen without looking at targets.

    When ``instances_per_block`` is positive, the same instance ids are used
    for every strategy. Selection is stratified by ``data_block`` and ordered
    by a stable hash of the instance id, never by truth or baseline quality.
    """
    wanted = set(strategies)
    selected = cases[(cases.strategy.isin(wanted)) &
                     (cases.target_budget == budget)].copy()
    if sample_seed is not None:
        selected = selected[selected.sample_seed == sample_seed].copy()
    missing = wanted - set(selected.strategy)
    if missing:
        raise ValueError(f"selected strategies absent from cases: {sorted(missing)}")

    if instances_per_block:
        if instances_per_block < 1:
            raise ValueError("instances_per_block must be non-negative")
        if "data_block" not in selected or "instance_id" not in selected:
            raise ValueError("paired block selection needs data_block and instance_id")
        coverage = (selected[["instance_id", "strategy"]].drop_duplicates()
                    .groupby("instance_id").strategy.nunique())
        common = set(coverage[coverage == len(wanted)].index)
        candidates = (selected[selected.instance_id.isin(common)]
                      [["instance_id", "data_block"]].drop_duplicates())
        chosen = []
        for block, group in candidates.groupby("data_block", sort=True):
            ids = list(group.instance_id)
            ids.sort(key=lambda x: _stable_order(str(x), selection_seed))
            if len(ids) < instances_per_block:
                raise ValueError(
                    f"data_block {block!r} has only {len(ids)} common instances, "
                    f"need {instances_per_block}")
            chosen.extend(ids[:instances_per_block])
        selected = selected[selected.instance_id.isin(chosen)].copy()

    duplicates = selected.duplicated(["instance_id", "strategy"])
    if duplicates.any():
        raise ValueError(
            "selection has multiple rows per instance/strategy; set --sample-seed")
    order = {strategy: i for i, strategy in enumerate(strategies)}
    selected["_strategy_order"] = selected.strategy.map(order)
    return (selected.sort_values(
        ["_strategy_order", "data_block", "instance_id", "case_id"])
        .drop(columns="_strategy_order").reset_index(drop=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--strategies", required=True,
                        help="comma-separated preregistered selected arms")
    parser.add_argument("--budget", type=int, default=800,
                        help="requested event budget")
    parser.add_argument("--input-kind", default="window_counts_crawl_temporal")
    parser.add_argument("--conditions",
                        default="sample,metadata_only_no_sample",
                        help="comma-separated prompt conditions")
    parser.add_argument("--sample-seed", type=int, default=None,
                        help="optional common sample seed for paired arms")
    parser.add_argument("--instances-per-block", type=int, default=0,
                        help="if >0, select this many common instances from "
                             "each data_block using only stable hashes")
    parser.add_argument("--selection-seed", type=int, default=0)
    parser.add_argument("--selected-cases-out", default=None,
                        help="optional CSV manifest of exactly selected cases")
    args = parser.parse_args()
    wanted = [x.strip() for x in args.strategies.split(",") if x.strip()]
    conditions = [x.strip() for x in args.conditions.split(",") if x.strip()]
    valid_conditions = {"sample", "metadata_only_no_sample"}
    unknown = set(conditions) - valid_conditions
    if unknown or not conditions:
        raise ValueError(f"invalid conditions: {sorted(unknown)}")
    cases = pd.read_csv(args.cases)
    cases = select_cases(
        cases, wanted, args.budget, sample_seed=args.sample_seed,
        instances_per_block=args.instances_per_block,
        selection_seed=args.selection_seed)
    if args.selected_cases_out:
        selected_out = Path(args.selected_cases_out)
        selected_out.parent.mkdir(parents=True, exist_ok=True)
        cases.to_csv(selected_out, index=False)
        print(f"wrote {selected_out}: {len(cases)} selected cases")

    records = []
    for _, row in cases.iterrows():
        full = render_nonwalk_prompt(row, True, args.input_kind)
        control = render_nonwalk_prompt(row, False, args.input_kind)
        assert_prompt_metadata_parity(full, control, row)
        prompts = {"sample": full, "metadata_only_no_sample": control}
        for condition in conditions:
            prompt = prompts[condition]
            key = f"{row.case_id}|{condition}|{args.input_kind}"
            pid = hashlib.sha1(key.encode()).hexdigest()[:12]
            records.append({
                "id": pid, "prompt_id": pid, "case_id": row.case_id,
                "condition": condition, "input_kind": args.input_kind,
                "strategy": row.strategy, "target_budget": args.budget,
                "prompt": prompt,
            })
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for record in records:
            fh.write(json.dumps(record) + "\n")
    print(f"wrote {out}: {len(records)} prompts from {len(cases)} cases")


if __name__ == "__main__":
    main()
