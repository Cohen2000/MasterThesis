#!/usr/bin/env python3
"""Build non-walk screening cases without touching frozen walk outputs."""

import argparse
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd
import yaml

from benchmark_features import build_case_features
from build_benchmark_data import stable_seed
from nonwalk_samplers import (
    activity_proportional_dyad_full_history,
    ego_recent_k_snowball,
    event_sample_then_full_history,
    neighbourhood_crawl,
    node_panel_full_history,
    node_selection_diagnostics,
    oracle_reservoir_ht,
    prepare_dyad_histories,
    prepare_events,
    temporal_nonstationarity_diagnostics,
    time_prefix_events,
    time_random_window_events,
    uniform_event_reservoir,
)
from walks import build_index


TARGET_COLUMNS = [
    "rho_headline", "rho_W5_k1", "rho_W5_k2", "rho_W5_k3",
    "rho_W5_k4", "rho_W5_k5", "rho_event_weighted", "mean_span_frac",
    "C_one_step", "lifetime_mean_over_T", "burstiness_pooled",
    "share_single_event_pairs",
]


def _event_path(manifest_path, rel):
    path = Path(str(rel))
    return path if path.is_absolute() else manifest_path.parent / path


def _load_config(path, preset):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    if preset not in cfg["presets"]:
        raise KeyError(f"unknown preset {preset!r}")
    return cfg["presets"][preset]


def _parse_strategy(name):
    match = re.fullmatch(r"(ego_recent|bfs_crawl|forest_fire)_k(\d+|all)", name)
    if match:
        base, value = match.group(1), match.group(2)
        return base, None if value == "all" else int(value)
    if name in {"uniform_event_reservoir", "time_prefix_events",
                "time_random_window_events", "node_panel_full_history",
                "activity_proportional_dyad_full_history",
                "event_sample_then_full_history"}:
        return name, None
    raise ValueError(f"unknown non-walk strategy {name!r}")


def _strategy_seed_key(name):
    # A k sweep must share the same restart order within its own crawl family.
    # Otherwise a k effect would be confounded with a different random path.
    # The families do not share a key with each other: their frontier rules
    # consume the generator differently, so a shared key would buy nothing.
    for family in ("ego_recent_k", "bfs_crawl_k", "forest_fire_k"):
        if name.startswith(family):
            return f"{family}_sweep"
    return name


def _sample(events, strategy, budget, seed, burn_prob=0.35):
    base, value = _parse_strategy(strategy)
    if base == "uniform_event_reservoir":
        return uniform_event_reservoir(events, budget, seed)
    if base == "time_prefix_events":
        return time_prefix_events(events, budget)
    if base == "time_random_window_events":
        return time_random_window_events(events, budget, seed)
    if base == "node_panel_full_history":
        return node_panel_full_history(events, budget, seed)
    if base == "activity_proportional_dyad_full_history":
        return activity_proportional_dyad_full_history(events, budget, seed)
    if base == "event_sample_then_full_history":
        return event_sample_then_full_history(events, budget, seed)
    if base == "bfs_crawl":
        return neighbourhood_crawl(events, budget, seed, value,
                                   expansion="bfs")
    if base == "forest_fire":
        return neighbourhood_crawl(events, budget, seed, value,
                                   expansion="forest_fire",
                                   burn_prob=burn_prob)
    return ego_recent_k_snowball(events, budget, seed, value)


def _csv_value(value):
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, separators=(",", ":"))
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/nonwalk_screen.yaml")
    parser.add_argument("--preset", default="smoke")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    args = parser.parse_args()

    plan = _load_config(args.config, args.preset)
    manifest_path = Path(args.manifest or plan["manifest"])
    manifest = pd.read_csv(manifest_path)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0,num-shards)")
    manifest = manifest.iloc[
        [i % args.num_shards == args.shard_id for i in range(len(manifest))]
    ].copy()
    if args.limit:
        manifest = manifest.head(args.limit)
    W = int(plan.get("W", 5))
    base_seed = int(plan["seed"])
    budgets = sorted(map(int, plan["budgets"]))
    seeds = int(plan["seeds"])
    strategies = list(plan["strategies"])
    node_panel_min_budget = int(plan.get("node_panel_min_budget", 400))
    recent_limit = int(plan.get("recent_events_json_limit", 100))
    stationarity_bins = int(plan.get("stationarity_bins", 10))
    burn_prob = float(plan.get("forest_fire_burn_prob", 0.35))

    rows = []
    started = time.time()
    for pos, (_, meta) in enumerate(manifest.iterrows(), 1):
        events = pd.read_csv(_event_path(manifest_path, meta["path"]))
        prepared = prepare_events(events)
        prepared_dyads = (prepare_dyad_histories(prepared)
                          if ({"activity_proportional_dyad_full_history",
                               "event_sample_then_full_history"} & set(strategies))
                          else None)
        idx = build_index(events, T=1.0, W=W)
        total_edges = len(idx.edge_times)
        full_degrees = {int(n): int(idx.coll_deg[n]) for n in idx.active_nodes}
        stationary = temporal_nonstationarity_diagnostics(
            prepared, bins=stationarity_bins, T=idx.T)
        for strategy in strategies:
            # A chronological prefix is deterministic. Repeating it under four
            # nominal seeds would create pseudo-replication, not robustness.
            strategy_seeds = 1 if strategy == "time_prefix_events" else seeds
            for sample_seed_index in range(strategy_seeds):
                seed = stable_seed(
                    base_seed, meta["instance_id"], _strategy_seed_key(strategy),
                    sample_seed_index)
                for target_budget in budgets:
                    if (strategy == "node_panel_full_history" and
                            target_budget < node_panel_min_budget):
                        continue
                    sample_input = (prepared_dyads
                                    if strategy in {
                                        "activity_proportional_dyad_full_history",
                                        "event_sample_then_full_history"}
                                    else prepared)
                    result = _sample(sample_input, strategy, target_budget,
                                     seed, burn_prob=burn_prob)
                    actual_budget = len(result.log)
                    features = build_case_features(
                        result.log, budget=actual_budget, W=W, idx=idx,
                        recent_limit=recent_limit, include_beta=False)
                    selected_nodes = None
                    if "query_node_order" in result.diagnostics:
                        selected_nodes = result.diagnostics["query_node_order"]
                    elif strategy == "node_panel_full_history":
                        selected_nodes = result.diagnostics["panel_node_order"]
                    selection = node_selection_diagnostics(
                        result.log, full_degrees, selected_nodes=selected_nodes)
                    oracle = {}
                    if strategy == "uniform_event_reservoir":
                        oracle = oracle_reservoir_ht(
                            result.log, idx.edge_times, len(events),
                            actual_budget, W=W, T=idx.T)
                    case_id = (f"{meta['instance_id']}|{strategy}|"
                               f"ss{sample_seed_index}|b{target_budget}")
                    row = {
                        "case_id": case_id,
                        "instance_id": meta["instance_id"],
                        "group_id": meta["group_id"],
                        "family": meta["family"],
                        "data_block": meta["data_block"],
                        "source": meta["source"],
                        "domain": meta["domain"],
                        "substrate": meta["substrate"],
                        "generator": meta["generator"],
                        "strategy": strategy,
                        "base_strategy": _parse_strategy(strategy)[0],
                        "sample_seed": sample_seed_index,
                        # Compatibility with the existing evaluator; these are
                        # sample seeds, not walk seeds.
                        "walk_seed": sample_seed_index,
                        "sample_rng_seed": seed,
                        "target_budget": target_budget,
                        "budget": actual_budget,
                        "total_edges": total_edges,
                        "coverage": features["observed_walk_edges"] / max(1, total_edges),
                        "n_nodes_true": meta["n_nodes"],
                        "n_events_true": meta["n_events"],
                        "rho_target": meta.get("rho_target", np.nan),
                        "alpha": meta.get("alpha", np.nan),
                        "chi": meta.get("chi", np.nan),
                    }
                    for column in TARGET_COLUMNS:
                        row[column] = meta.get(column, np.nan)
                    row.update(features)
                    row.update(stationary)
                    row.update(selection)
                    row.update(oracle)
                    row.update({f"sample__{key}": _csv_value(value)
                                for key, value in result.diagnostics.items()})
                    rows.append(row)
        print(f"[nonwalk] {pos}/{len(manifest)} instances, {len(rows)} cases, "
              f"{time.time() - started:.1f}s", flush=True)

    out_path = Path(args.out or plan["out"])
    if args.num_shards > 1:
        out_path = out_path.with_name(
            f"{out_path.name.removesuffix('.csv.gz')}_shard_"
            f"{args.shard_id:03d}.csv.gz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, compression="gzip")
    print(f"wrote {out_path}: {len(rows)} cases")


if __name__ == "__main__":
    main()
