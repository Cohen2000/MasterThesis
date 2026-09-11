#!/usr/bin/env python3
"""Run access models and build estimator/ML/LLM-ready benchmark cases."""

import argparse
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from benchmark_features import build_case_features
from build_benchmark_data import load_plan, stable_seed
from walks import build_index, run_walk


TARGET_COLUMNS = [
    "rho_headline", "rho_W5_k1", "rho_W5_k2", "rho_W5_k3",
    "rho_W5_k4", "rho_W5_k5", "rho_event_weighted", "mean_span_frac",
    "C_one_step", "lifetime_mean_over_T", "burstiness_pooled",
    "share_single_event_pairs",
]


def _sentinel_selected(instance_id, fraction, seed):
    value = stable_seed(seed, "time_agnostic_sentinel", instance_id) / 2**32
    return value < fraction


def _event_path(manifest_path, rel):
    p = Path(str(rel))
    return p if p.is_absolute() else manifest_path.parent / p


def _strategy_spec(name, default_history_k):
    m = re.fullmatch(r"recent_history_k(\d+)", name)
    if m:
        return {"base": "recent_history", "history_k": int(m.group(1)), "starts": 1}
    m = re.fullmatch(r"(time_respecting|recency_biased)_multistart(\d+)", name)
    if m:
        return {"base": m.group(1), "history_k": default_history_k,
                "starts": int(m.group(2))}
    return {"base": name, "history_k": default_history_k, "starts": 1}


def _multistart_log(idx, base_strategy, budget, seed, starts,
                    decay_scale, history_k):
    """Pool several independently restarted walks at fixed total budget."""
    if starts < 2:
        return run_walk(idx, base_strategy, max_budget=budget, seed=seed,
                        decay_scale=decay_scale, history_k=history_k)
    sizes = [budget // starts] * starts
    for i in range(budget % starts):
        sizes[i] += 1
    pieces = []
    for i, size in enumerate(sizes):
        if size <= 0:
            continue
        sub_seed = stable_seed(seed, "multistart", i)
        pieces.append(run_walk(idx, base_strategy, max_budget=size, seed=sub_seed,
                               decay_scale=decay_scale, history_k=history_k))
    return pd.concat(pieces, ignore_index=True).iloc[:budget].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/benchmark.yaml")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--manifest", help="default: data/benchmark_<preset>/manifest.csv")
    ap.add_argument("--out", help="default: results/benchmark_<preset>/cases.csv.gz")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    _, plan = load_plan(args.config, args.preset)
    manifest_path = Path(args.manifest or f"data/benchmark_{args.preset}/manifest.csv")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0,num-shards)")
    mf = pd.read_csv(manifest_path)
    mf = mf.iloc[[i % args.num_shards == args.shard_id for i in range(len(mf))]].copy()
    if args.limit:
        mf = mf.head(args.limit)
    walks = plan["walks"]
    strategies = list(walks["strategies"])
    budgets = sorted(map(int, walks["budgets"]))
    walk_seeds = int(walks["seeds"])
    W = int(plan["W"]); base_seed = int(plan["seed"])
    sentinel_fraction = float(walks.get("time_agnostic_sentinel_fraction", 0.0))
    default_history_k = int(walks.get("recent_history_k", 20))
    decay_scale = float(walks.get("recency_decay_scale", 0.10))
    recent_limit = int(walks.get("recent_events_json_limit", 100))

    rows = []
    t0 = time.time()
    for pos, (_, r) in enumerate(mf.iterrows(), start=1):
        path = _event_path(manifest_path, r["path"])
        events = pd.read_csv(path)
        idx = build_index(events, T=1.0, W=W)
        total_edges = len(idx.edge_times)
        for strategy in strategies:
            if strategy == "time_agnostic" and not _sentinel_selected(
                    r["instance_id"], sentinel_fraction, base_seed):
                continue
            spec = _strategy_spec(strategy, default_history_k)
            for ws in range(walk_seeds):
                seed = stable_seed(base_seed, r["instance_id"], strategy, ws)
                # Standard strategies share one max-budget trajectory exactly as
                # v1 did. Multi-start is defined independently at each budget.
                max_log = None
                if spec["starts"] == 1:
                    max_log = run_walk(
                        idx, spec["base"], max_budget=max(budgets), seed=seed,
                        decay_scale=decay_scale, history_k=spec["history_k"])
                for budget in budgets:
                    log = (max_log if max_log is not None else
                           _multistart_log(idx, spec["base"], budget, seed,
                                           spec["starts"], decay_scale,
                                           spec["history_k"]))
                    features = build_case_features(
                        log, budget=budget, W=W, idx=idx, recent_limit=recent_limit)
                    case_id = f"{r['instance_id']}|{strategy}|ws{ws}|b{budget}"
                    row = {
                        "case_id": case_id, "instance_id": r["instance_id"],
                        "group_id": r["group_id"], "family": r["family"],
                        "data_block": r["data_block"], "source": r["source"],
                        "domain": r["domain"], "substrate": r["substrate"],
                        "generator": r["generator"], "strategy": strategy,
                        "base_strategy": spec["base"], "history_k": spec["history_k"],
                        "n_starts": spec["starts"],
                        "walk_seed": ws, "walk_rng_seed": seed,
                        "budget": budget, "total_edges": total_edges,
                        "coverage": features["observed_walk_edges"] / max(1, total_edges),
                        "n_nodes_true": r["n_nodes"], "n_events_true": r["n_events"],
                        "rho_target": r.get("rho_target", np.nan),
                        "alpha": r.get("alpha", np.nan), "chi": r.get("chi", np.nan),
                    }
                    for col in TARGET_COLUMNS:
                        row[col] = r.get(col, np.nan)
                    row.update(features)
                    rows.append(row)
        if pos % 5 == 0 or pos == len(mf):
            print(f"[walks shard {args.shard_id}] {pos}/{len(mf)} instances, "
                  f"{len(rows)} cases, {time.time()-t0:.1f}s", flush=True)

    out_path = Path(args.out or f"results/benchmark_{args.preset}/cases.csv.gz")
    if args.num_shards > 1 and args.out is None:
        out_path = out_path.with_name(f"cases_shard_{args.shard_id:03d}.csv.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, compression="gzip")
    print(f"wrote {out_path}: {len(rows)} cases in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
