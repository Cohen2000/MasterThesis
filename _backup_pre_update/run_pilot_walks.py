#!/usr/bin/env python3
"""Run all walk strategies over the pilot instances and write summaries.

For every instance x strategy x walk seed, one walk log is produced up to the
largest budget; sweep points are prefixes of that log (checkpoint trick).

Usage:
  python run_pilot_walks.py --manifest data_pilot/manifest.csv --out summaries.csv
"""

import argparse
import zlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from walks import build_index, run_walk, summaries_at_checkpoints

STRATEGIES = ["time_agnostic", "time_agnostic_t", "time_respecting", "recency_biased"]
BUDGETS = [50, 100, 200, 400, 800, 1600, 3200]
META = ["substrate", "family", "rho_target", "rep", "hub_bias", "rho_headline"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_pilot/manifest.csv")
    ap.add_argument("--out", default="summaries.csv")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--walk-seeds", type=int, default=2)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0, help="only first N instances (testing)")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    strategies = args.strategies.split(",")
    mf = pd.read_csv(args.manifest)
    if args.limit:
        mf = mf.head(args.limit)

    rows = []
    t0 = time.time()
    for i, r in mf.iterrows():
        events = pd.read_csv(r["path"])
        idx = build_index(events, T=args.T, W=args.W)
        for strat in strategies:
            for ws in range(args.walk_seeds):
                seed = zlib.crc32(f"{r['family']}|{r['rho_target']}|{r['rep']}|{r['hub_bias']}|{strat}|{ws}".encode()) % 1_000_000
                log = run_walk(idx, strat, max_budget=max(budgets), seed=seed)
                for feats in summaries_at_checkpoints(log, idx, budgets):
                    rows.append({**{k: r[k] for k in META},
                                 "strategy": strat, "walk_seed": ws, **feats})
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(mf)}] {time.time()-t0:.0f}s elapsed", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} summary rows, "
          f"{time.time()-t0:.0f}s total)")

    # sanity pivot: the thesis signal preview at the largest budget
    top = out[out["budget"] == max(budgets)]
    if "walk_rho_plugin" in top.columns:
        piv = top.pivot_table(index="rho_target", columns="strategy",
                              values="walk_rho_plugin", aggfunc="mean")
        print("\nmean walk_rho_plugin at top budget "
              "(time_agnostic is blank by design):")
        print(piv.round(3).to_string())
    piv2 = top.pivot_table(index="rho_target", columns="strategy",
                           values="n_restarts", aggfunc="mean")
    print("\nmean restarts at top budget (dead-end pressure):")
    print(piv2.round(1).to_string())


if __name__ == "__main__":
    main()
