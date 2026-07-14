#!/usr/bin/env python3
"""Report exact instance/group/case scale implied by a built manifest and preset."""

import argparse
from pathlib import Path

import pandas as pd

from build_benchmark_data import load_plan
from run_benchmark_walks import _sentinel_selected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/benchmark.yaml")
    ap.add_argument("--preset", default="v2")
    ap.add_argument("--manifest")
    args = ap.parse_args()
    _, plan = load_plan(args.config, args.preset)
    manifest = Path(args.manifest or f"data/benchmark_{args.preset}/manifest.csv")
    d = pd.read_csv(manifest)
    walks = plan["walks"]
    strategies = list(walks["strategies"])
    budgets = len(walks["budgets"])
    seeds = int(walks["seeds"])
    per_strategy = {}
    for strategy in strategies:
        if strategy == "time_agnostic":
            frac = float(walks.get("time_agnostic_sentinel_fraction", 0.0))
            n = sum(_sentinel_selected(x, frac, int(plan["seed"]))
                    for x in d["instance_id"].astype(str))
        else:
            n = len(d)
        per_strategy[strategy] = int(n * budgets * seeds)
    print(f"Instances: {len(d):,}")
    print(f"Independent groups: {d['group_id'].nunique():,}")
    print(f"Expected total cases: {sum(per_strategy.values()):,}")
    print("Cases by strategy:")
    for strategy, n in per_strategy.items():
        print(f"  {strategy:34s} {n:>10,}")


if __name__ == "__main__":
    main()
