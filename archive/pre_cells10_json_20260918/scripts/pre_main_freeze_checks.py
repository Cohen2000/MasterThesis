#!/usr/bin/env python3
"""Deterministic pre-main-experiment freeze checks; diagnostics only.

Six audits of the intended empirical panel: data integrity, timing-variant
feasibility at rho_2 = 0.15/0.35, sampler severity at 25%/50% visible events,
recency purity, simple-random-walk behaviour, and W robustness with a grouped
leave-one-backbone-out constant reference. Only local files are read. There are
no LLM/API calls, downloads, config changes or writes outside the output
directory. The observation samplers below implement the intended semantics for
these diagnostics; they are not the main-experiment pipeline.
"""

import argparse
from dataclasses import dataclass
from fractions import Fraction
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import zlib

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import dataset_census
from benchmark_generators import family_from_events, normalize_event_stream
from census import count_nodes, load_registry, window_index
from dataset_census import (occupancy_counts, parse_audited, prepare_complete,
                            sha256_file, survival)
from generator import make_instance
from nonwalk_samplers import uniform_event_reservoir
from persistence_evaluation import profile_mae
from walks import build_index, run_walk

# Intended final empirical panel: registry key and the name used in the design.
PANEL = (
    ("sp_hospital", "SocioPatterns Hospital"),
    ("sp_highschool2013", "SocioPatterns High School 2013"),
    ("copenhagen_bluetooth", "Copenhagen Bluetooth"),
    ("snap_collegemsg", "CollegeMsg"),
    ("snap_email_eu", "Email-Eu"),
    ("snap_mathoverflow", "MathOverflow"),
    ("nr_radoslaw_email", "Radoslaw Email"),
    ("jodie_reddit", "JODIE Reddit"),
)
W_MAIN = 5
W_GRID = (4, 5, 6, 8, 10)
RHO = tuple(f"rho_{k}" for k in range(2, W_MAIN + 1))
TIMING_TARGETS = ("0.15", "0.35")
# Seed rule of dataset_census.realized_feasibility; rho2_015 repeats the census attempt.
TIMING_SEED_BASE = 20260911
EVENT_FRACTIONS = ("0.25", "0.50")
SEED_SLOTS = (0, 1, 2, 3)
MECHANISMS = {
    "node_panel_full_history": "reference",
    "simple_random_walk_full_history": "selection",
    "recency_truncation": "history_loss",
    "sampled_event_stream": "both",
}
PARAMETER_NAMES = {
    "node_panel_full_history": "node_sample_size",
    "simple_random_walk_full_history": "walk_steps",
    "recency_truncation": "cutoff_normalized_time",
    "sampled_event_stream": "event_sample_size",
}
WALK_START_STEPS_PER_DYAD = 4
WALK_MAX_STEPS_PER_DYAD = 64
START_NODE_RULE = ("uniform over all nodes (walks.run_walk time_agnostic placement); "
                   "no restarts, so the walk stays in the start component")
# Published JODIE Reddit size (Kumar et al., KDD 2019); compared, not downloaded.
JODIE_REDDIT_REFERENCE = {"users": 10000, "items": 984, "interactions": 672447}
OUTPUTS = {
    "panel": "panel_integrity.csv",
    "timing": "timing_variant_check.csv",
    "invariants": "timing_variant_invariants.csv",
    "severity_raw": "severity_diagnostics_raw.csv",
    "severity_summary": "severity_diagnostics_summary.csv",
    "recency": "recency_purity.csv",
    "rw": "rw_diagnostics.csv",
    "w_raw": "w_sensitivity_raw.csv",
    "w_summary": "w_sensitivity_summary.csv",
    "loso": "constant_baseline_grouped_loso.csv",
}
INTEGER_COLUMNS = ("seed_slot", "seed", "n_nodes", "n_edges_full", "n_events",
                   "duplicate_same_dyad_same_timestamp_count", "self_events_removed",
                   "invalid_rows_removed", "allocation_deviations", "start_node",
                   "users_full", "items_full", "visited_users", "visited_items",
                   "rank_lowest_error_first", "n_cells")


def variant_name(target):
    return "rho2_" + target.replace(".", "")


def diagnostic_seed(dataset, mechanism, slot):
    """Fixed diagnostic seed (config master_seed is null); not a main-experiment seed."""
    text = f"pre_main_freeze_checks|{dataset}|{mechanism}|slot{slot}"
    return zlib.crc32(text.encode()) & 0xFFFFFFFF


# ----------------------------------------------------------------------------
# Complete backbone in census order
# ----------------------------------------------------------------------------

@dataclass
class Backbone:
    """Retained complete stream sorted by (dyad, time), exactly as in the census."""
    key: str
    frame: pd.DataFrame        # prepare_complete events: string labels, source time
    quality: dict
    pair: np.ndarray           # dyad id per event
    t_norm: np.ndarray         # complete-axis normalized time; t_max -> 1.0
    t_source: np.ndarray
    node_labels: np.ndarray    # sorted labels; node id = position
    dyad_u: np.ndarray         # lower node id per dyad
    dyad_v: np.ndarray
    m: np.ndarray              # events per dyad

    @property
    def n_nodes(self):
        return len(self.node_labels)

    @property
    def n_dyads(self):
        return len(self.m)

    @property
    def n_events(self):
        return len(self.pair)


def prepare_backbone(key, raw):
    """Census preprocessing plus dyad endpoints aligned to its event order."""
    frame, pair, t_norm, m, quality = prepare_complete(raw)
    # census.normalize keeps only (pair, t); rebuild its dyad keys for the endpoints.
    lower = np.minimum(frame["u"].astype(str), frame["v"].astype(str))
    upper = np.maximum(frame["u"].astype(str), frame["v"].astype(str))
    codes = pd.factorize(lower + "\x1f" + upper)[0]
    source = frame["t"].to_numpy(float)
    order = np.lexsort((source, codes))
    t_source = source[order]
    lo, hi = float(t_source.min()), float(t_source.max())
    if not (np.array_equal(codes[order], pair)
            and np.array_equal((t_source - lo) / (hi - lo), t_norm)):
        raise AssertionError(f"{key}: endpoints do not align with the census event order")
    starts = np.flatnonzero(np.r_[True, pair[1:] != pair[:-1]])
    if not np.array_equal(pair[starts], np.arange(len(m))):
        raise AssertionError(f"{key}: dyad ids are not contiguous in census order")
    first = order[starts]
    ends = np.concatenate([lower.to_numpy(object)[first],
                           upper.to_numpy(object)[first]]).astype(str)
    node_labels, node_ids = np.unique(ends, return_inverse=True)
    if len(node_labels) != count_nodes(frame):
        raise AssertionError(f"{key}: node count differs from census count_nodes")
    n = len(m)
    return Backbone(key, frame, quality, pair.astype(np.int64), t_norm, t_source,
                    node_labels, node_ids[:n].astype(np.int64),
                    node_ids[n:].astype(np.int64), m.astype(np.int64))


def load_backbone(key, spec, raw_dir):
    raw, parser_quality = parse_audited(Path(raw_dir) / spec["file"], spec["format"],
                                        spec.get("census_audit", {}))
    return prepare_backbone(key, raw), parser_quality


def complete_windows(bb):
    """Window ids on the complete axis for every W; never recomputed per observation."""
    return {W: window_index(bb.t_norm, 0.0, 1.0 / W, W) for W in W_GRID}


# ----------------------------------------------------------------------------
# Persistence profiles and errors
# ----------------------------------------------------------------------------

def occupancy(pair, windows, n_dyads, W):
    """K_e for every complete-stream dyad from visible events; unseen dyads get 0."""
    distinct = np.unique(np.asarray(pair, np.int64) * W + np.asarray(windows, np.int64))
    return np.bincount(distinct // W, minlength=n_dyads)


def survival_profile(K, W, denominator):
    """rho_k = |{e : K_e >= k}| / denominator for k = 2..W; NaN without dyads."""
    if denominator == 0:
        return np.full(W - 1, np.nan)
    tail = np.cumsum(np.bincount(K, minlength=W + 1)[::-1])[::-1]
    return tail[2:W + 1] / denominator


def plugin_profile(K, W):
    """Naive plug-in: the observed dyads (K > 0) form the denominator."""
    return survival_profile(K, W, int(np.count_nonzero(K)))


def roster_profile(K, W):
    """Known-roster counterfactual: every complete-stream dyad is in the denominator."""
    return survival_profile(K, W, len(K))


def profile_mae_w5(estimate, truth):
    """ProfileMAE via the repository scorer; NaN if the plug-in is undefined."""
    if not np.all(np.isfinite(estimate)):
        return float("nan")
    as_dict = lambda values: {name: float(v) for name, v in zip(RHO, values)}
    return profile_mae(as_dict(estimate), as_dict(truth))


def normalized_profile_error(estimate, truth, W):
    """(1/W) * sum_{k=2..W} |rho_full(W,k) - rho_hat(W,k)|, the L1 gap of K/W survival."""
    estimate, truth = np.asarray(estimate, float), np.asarray(truth, float)
    if estimate.shape != (W - 1,) or truth.shape != (W - 1,):
        raise ValueError("profiles must contain rho_2..rho_W")
    return float(np.abs(truth - estimate).sum() / W)


def observation_summary(bb, windows, rho_full, event_mask):
    """Coverage and plug-in errors of one observation, on the complete time axis."""
    idx = np.flatnonzero(event_mask)
    K = {W: occupancy(bb.pair[idx], windows[W][idx], bb.n_dyads, W) for W in W_GRID}
    seen = K[W_MAIN] > 0
    nodes = np.union1d(bb.dyad_u[seen], bb.dyad_v[seen])
    estimate = plugin_profile(K[W_MAIN], W_MAIN)
    row = {"node_coverage": len(nodes) / bb.n_nodes,
           "dyad_coverage": int(seen.sum()) / bb.n_dyads,
           "event_coverage": len(idx) / bb.n_events,
           "n_observed_nodes": len(nodes), "n_observed_dyads": int(seen.sum()),
           "n_observed_events": len(idx), "n_full_nodes": bb.n_nodes,
           "n_full_dyads": bb.n_dyads, "n_full_events": bb.n_events,
           **{f"rho_hat_{k}": float(estimate[k - 2]) for k in range(2, W_MAIN + 1)},
           "profile_mae_w5": profile_mae_w5(estimate, rho_full[W_MAIN])}
    errors = {W: normalized_profile_error(plugin_profile(K[W], W), rho_full[W], W)
              for W in W_GRID}
    return row, errors, K, seen


# ----------------------------------------------------------------------------
# Diagnostic observation mechanisms
# ----------------------------------------------------------------------------

def closest_median_parameter(candidates, coverage, target):
    """Smallest candidate whose median event coverage over seeds is closest to target."""
    median = np.median(np.asarray(coverage, float), axis=0)
    best = int(np.argmin(np.abs(median - float(target))))
    return int(candidates[best]), float(median[best])


def uniform_node_order(n_nodes, seed):
    """Uniform random node order; its length-n prefix is a fixed-size uniform sample."""
    return np.random.default_rng(seed).permutation(n_nodes)


def induced_dyads(bb, sample):
    """Dyads with both endpoints in the node sample (induced subgraph)."""
    chosen = np.zeros(bb.n_nodes, dtype=bool)
    chosen[np.asarray(sample, np.int64)] = True
    return chosen[bb.dyad_u] & chosen[bb.dyad_v]


def node_prefix_events(bb, order):
    """Visible full-history events of the induced subgraph on order[:n], n = 0..N."""
    rank = np.empty(bb.n_nodes, dtype=np.int64)
    rank[order] = np.arange(bb.n_nodes)
    added = np.zeros(bb.n_nodes + 1, dtype=np.int64)
    np.add.at(added, np.maximum(rank[bb.dyad_u], rank[bb.dyad_v]) + 1, bb.m)
    return np.cumsum(added)


def walk_index(bb):
    """walks.build_index on one row per dyad; SRW transitions ignore multiplicity."""
    return build_index(pd.DataFrame({"u": bb.dyad_u, "v": bb.dyad_v,
                                     "t": np.zeros(bb.n_dyads)}))


def dyad_lookup(bb, u, v):
    keys = bb.dyad_u * bb.n_nodes + bb.dyad_v
    sorter = np.argsort(keys, kind="stable")
    query = np.minimum(u, v).astype(np.int64) * bb.n_nodes + np.maximum(u, v)
    position = np.minimum(np.searchsorted(keys[sorter], query), len(keys) - 1)
    if not np.array_equal(keys[sorter][position], query):
        raise AssertionError("walk traversed a pair that is not a complete-stream dyad")
    return sorter[position]


def simple_random_walk(bb, index, steps, seed):
    """Visited nodes (steps + 1 positions) and traversed dyad ids (steps)."""
    log = run_walk(index, "time_agnostic", steps + 1, seed)
    kind = log["kind"].to_numpy()
    if kind[0] != 0 or np.any(kind[1:] != 1):
        raise AssertionError("a simple random walk must not restart")
    return (log["node"].to_numpy(np.int64),
            dyad_lookup(bb, log["u"].to_numpy(np.int64)[1:], log["v"].to_numpy(np.int64)[1:]))


def full_history_mask(bb, dyads):
    """Every event of each discovered dyad exactly once, however often it was traversed."""
    discovered = np.zeros(bb.n_dyads, dtype=bool)
    discovered[np.asarray(dyads, np.int64)] = True
    return discovered[bb.pair]


def discovery_curve(dyads, m):
    """Step at which each dyad is first traversed and cumulative full-history events."""
    unique, first = np.unique(dyads, return_index=True)
    order = np.argsort(first, kind="stable")
    return first[order] + 1, np.cumsum(m[unique[order]])


def events_by_step(curve, steps):
    at, cumulative = curve
    count = np.searchsorted(at, steps, side="right")
    return np.where(count > 0, cumulative[np.maximum(count - 1, 0)], 0)


def calibrate_walks(bb, index, seeds, targets, structure):
    """Lengthen all walks until median coverage passes the largest target.

    Stops early if every walk exhausted its start component or the step cap is hit.
    """
    steps = WALK_START_STEPS_PER_DYAD * bb.n_dyads
    cap = WALK_MAX_STEPS_PER_DYAD * bb.n_dyads
    while True:
        walks = [simple_random_walk(bb, index, steps, seed) for seed in seeds]
        curves = [discovery_curve(dyads, bb.m) for _, dyads in walks]
        candidates = np.unique(np.concatenate([at for at, _ in curves]))
        coverage = np.vstack([events_by_step(c, candidates) for c in curves]) / bb.n_events
        saturated = all(len(c[0]) == structure["component_dyads"][structure["component"][nodes[0]]]
                        for c, (nodes, _) in zip(curves, walks))
        if (np.median(coverage[:, -1]) >= max(map(float, targets)) or saturated
                or steps >= cap):
            break
        steps = min(2 * steps, cap)
    return walks, {t: closest_median_parameter(candidates, coverage, t) for t in targets}, steps


def recency_cutoff(t_norm, target):
    """Distinct complete-axis time c whose suffix {t >= c} has the event share closest to target.

    Exact integer comparison; ties go to the later cutoff.
    """
    times = np.sort(np.asarray(t_norm, float))
    distinct = np.unique(times)
    retained = len(times) - np.searchsorted(times, distinct, side="left")
    share = Fraction(target)
    gap = np.abs(retained * share.denominator - share.numerator * len(times))
    best = len(distinct) - 1 - int(np.argmin(gap[::-1]))
    return float(distinct[best]), int(retained[best])


def event_sample_size(n_events, target):
    """round(target * M) with halves up, in exact arithmetic."""
    return int(math.floor(Fraction(target) * n_events + Fraction(1, 2)))


def sampled_event_mask(bb, events, size, seed):
    """Uniform event records without replacement via nonwalk_samplers.uniform_event_reservoir."""
    log = uniform_event_reservoir(events, size, seed).log
    ids = log["event_id"].to_numpy(np.int64)
    if not (np.array_equal(log["t"].to_numpy(float), bb.t_norm[ids])
            and np.array_equal(log["u"].to_numpy(np.int64), bb.dyad_u[bb.pair[ids]])):
        raise AssertionError("event sample does not index the complete stream")
    mask = np.zeros(bb.n_events, dtype=bool)
    mask[ids] = True
    return mask


def graph_structure(bb):
    adjacency = coo_matrix((np.ones(bb.n_dyads), (bb.dyad_u, bb.dyad_v)),
                           shape=(bb.n_nodes, bb.n_nodes))
    n_components, component = connected_components(adjacency, directed=False)
    dyad_component = component[bb.dyad_u]
    sizes = np.bincount(component, minlength=n_components)
    largest = int(np.argmax(sizes))
    return {"n_components": int(n_components), "component": component,
            "component_nodes": sizes,
            "component_dyads": np.bincount(dyad_component, minlength=n_components),
            "largest": largest,
            "largest_node_fraction": sizes[largest] / bb.n_nodes,
            "largest_dyad_fraction": float(np.mean(dyad_component == largest)),
            "largest_event_fraction": int(bb.m[dyad_component == largest].sum()) / bb.n_events,
            "degree": (np.bincount(bb.dyad_u, minlength=bb.n_nodes)
                       + np.bincount(bb.dyad_v, minlength=bb.n_nodes))}


# ----------------------------------------------------------------------------
# Task 1: panel integrity
# ----------------------------------------------------------------------------

def same_value(a, b):
    if isinstance(a, (int, np.integer)):
        return not pd.isna(b) and int(a) == int(b)
    return not pd.isna(b) and float(a) == float(b)


def bipartite_sides(bb):
    users = np.char.startswith(bb.node_labels, "u:")
    items = np.char.startswith(bb.node_labels, "i:")
    return users, items


def reddit_identity_check(bb, sha_ok):
    """Structural consistency with the published JODIE Reddit interaction dataset."""
    users, items = bipartite_sides(bb)
    user_ids = np.sort([int(label[2:]) for label in bb.node_labels[users]])
    item_ids = np.sort([int(label[2:]) for label in bb.node_labels[items]])
    file_order_sorted = bool(np.all(np.diff(bb.frame["t"].to_numpy(float)) >= 0))
    checks = {
        "sha256 equals census": sha_ok,
        "interactions = 672447": bb.n_events == JODIE_REDDIT_REFERENCE["interactions"],
        "users = 10000 (ids 0..9999)": np.array_equal(
            user_ids, np.arange(JODIE_REDDIT_REFERENCE["users"])),
        "items = 984 (ids 0..983)": np.array_equal(
            item_ids, np.arange(JODIE_REDDIT_REFERENCE["items"])),
        "every node is a user or item": bool(np.all(users ^ items)),
        "timestamps non-decreasing in file order": file_order_sorted,
    }
    span = bb.t_source.max() - bb.t_source.min()
    passed = all(checks.values())
    text = ("PASS" if passed else "FAIL") + ": " + "; ".join(
        f"{name}={'yes' if ok else 'NO'}" for name, ok in checks.items())
    text += (f"; header user_id,item_id,timestamp (three-column projection, no state label or "
             f"feature columns); time range {bb.t_source.min():.3f}..{bb.t_source.max():.3f} "
             f"({span / 86400:.2f} days if seconds; unit undocumented); matches published JODIE "
             "Reddit user-subreddit posting counts (10000 users, 984 subreddits, 672447 interactions), "
             "checked structurally without download")
    return passed, text


def integrity_row(key, name, spec, census, bb, parser_quality, rho_full, window_census):
    base = {"dataset": key, "panel_name": name, "label": spec["label"],
            "domain": spec.get("domain", "unknown"),
            "bipartite": "yes" if spec.get("format", {}).get("bipartite") else "no",
            "file": spec["file"]}
    if bb is None:
        return {**base, "status": "absent", "source_status": "registered file not found locally; "
                "not downloaded"}, ["dataset unavailable locally"]
    failures = []
    recomputed = {
        "n_nodes": bb.n_nodes, "n_edges_full": bb.n_dyads, "n_events": bb.n_events,
        **{f"rho_{k}": float(rho_full[W_MAIN][k - 2]) for k in range(2, W_MAIN + 1)},
        "fraction_m_eq_1": float(np.mean(bb.m == 1)), "p_m_ge_2": float(np.mean(bb.m >= 2)),
        "duplicate_dyad_timestamp_events": bb.quality["duplicate_dyad_timestamp_events"],
        "self_events_removed": bb.quality["self_events_removed"],
        "invalid_rows_removed": (bb.quality["invalid_rows_removed"]
                                 + parser_quality["invalid_rows_removed"]),
        "observation_start": bb.quality["observation_start"],
        "observation_end": bb.quality["observation_end"],
    }
    mismatches = [field for field, value in recomputed.items()
                  if not same_value(value, census.get(field))]
    if mismatches:
        failures.append(f"recomputed census values differ: {', '.join(mismatches)}")
    grid = window_census[(window_census.dataset == key) & (window_census.statistic == "rho")]
    grid_ok = all(np.array_equal(grid[grid.W == W].sort_values("k").value.to_numpy(float),
                                 rho_full[W]) for W in W_GRID)
    if not grid_ok:
        failures.append("W-grid rho differs from census window_sensitivity.csv")
    sha = sha256_file(ROOT / "data/raw" / spec["file"])
    sha_ok = sha == census.get("raw_sha256")
    if not sha_ok:
        failures.append("raw file SHA-256 differs from the census")
    sides, identity = "NA", "NA"
    if base["bipartite"] == "yes":
        users, items = bipartite_sides(bb)
        sides = f"users={int(users.sum())}; items={int(items.sum())}"
    if key == "jodie_reddit":
        passed, identity = reddit_identity_check(bb, sha_ok)
        if not passed:
            failures.append("JODIE Reddit identity check failed")
    row = {**base, "status": census.get("status"),
           "n_nodes": bb.n_nodes, "n_edges_full": bb.n_dyads, "n_events": bb.n_events,
           "observation_start_source_units": census.get("observation_start"),
           "observation_end_source_units": census.get("observation_end"),
           "observation_span_source_units": census.get("observation_span_source_units"),
           "timestamp_unit": census.get("timestamp_unit"),
           "physical_observation_span_seconds": census.get("physical_observation_span_seconds"),
           "w5_window_duration_source_units": census.get("window_duration_source_units"),
           "physical_w5_window_seconds": census.get("physical_W5_window_seconds"),
           **{name: census.get(name) for name in RHO},
           "fraction_m_eq_1": census.get("fraction_m_eq_1"), "p_m_ge_2": census.get("p_m_ge_2"),
           "duplicate_same_dyad_same_timestamp_count": census.get("duplicate_dyad_timestamp_events"),
           "self_events_removed": census.get("self_events_removed"),
           "invalid_rows_removed": census.get("invalid_rows_removed"),
           "bipartite_sides": sides,
           "raw_sha256": sha, "sha256_matches_census": sha_ok,
           "recomputed_matches_census": not mismatches,
           "recompute_mismatches": "; ".join(mismatches) or "none",
           "w_grid_rho_matches_census": grid_ok,
           "dataset_identity_check": identity,
           "source_status": (f"census status={census.get('status')}; unit evidence: "
                             f"{census.get('time_unit_evidence')}; source: {census.get('metadata_source')}"),
           "census_warnings": census.get("warnings") if not pd.isna(census.get("warnings")) else ""}
    return row, failures


# ----------------------------------------------------------------------------
# Task 2: timing variants
# ----------------------------------------------------------------------------

def timing_variants(bb, targets=TIMING_TARGETS, seed_base=TIMING_SEED_BASE):
    """One attempt per target with the census generator settings; yields label-mapped variants."""
    original = normalize_event_stream(bb.frame)
    both = pd.concat([bb.frame["u"].astype(str), bb.frame["v"].astype(str)], ignore_index=True)
    labels = np.asarray(pd.factorize(both, sort=True)[1], dtype=object)
    family = family_from_events(original, name=bb.key, W=W_MAIN, T=1.0)
    family.timestamps = "empirical"
    for target in targets:
        seed = zlib.crc32(f"{seed_base}|census_twin|{bb.key}|{target}".encode()) & 0xFFFFFFFF
        try:
            instance = make_instance(family, float(target), seed=seed, hub_bias=False,
                                     span_layout="contiguous")
        except (ValueError, RuntimeError, AssertionError) as exc:
            yield target, seed, None, labels, str(exc)
            continue
        yield target, seed, instance, labels, ""


def timing_invariants(bb, variant_u, variant_v, variant_t):
    """Exact comparison of a variant (endpoint labels, normalized times) with the backbone."""
    u = pd.Series(np.asarray(variant_u, dtype=object)).astype(str)
    v = pd.Series(np.asarray(variant_v, dtype=object)).astype(str)
    dyads = (np.minimum(u, v) + "\x1f" + np.maximum(u, v)).value_counts()
    full = pd.Series(bb.m, index=pd.Index(bb.node_labels[bb.dyad_u].astype(object))
                     + "\x1f" + bb.node_labels[bb.dyad_v].astype(object))
    variant_nodes = set(u.unique()) | set(v.unique())
    same_dyads = len(dyads) == len(full) and set(dyads.index) == set(full.index)
    times = np.sort(np.asarray(variant_t, float))
    result = {
        "same_node_set": variant_nodes == set(bb.node_labels.tolist()),
        "same_collapsed_dyad_set": same_dyads,
        "same_total_events": len(times) == bb.n_events,
        "same_per_dyad_count_multiset": np.array_equal(np.sort(dyads.to_numpy()), np.sort(bb.m)),
        "same_exact_per_dyad_counts": same_dyads and np.array_equal(
            dyads.reindex(full.index).to_numpy(np.int64), full.to_numpy(np.int64)),
        "same_global_timestamp_multiset": (len(times) == bb.n_events
                                           and np.array_equal(times, np.sort(bb.t_norm))),
        "timestamp_normalization_injective": (len(np.unique(bb.t_norm))
                                              == len(np.unique(bb.t_source))),
        "n_nodes_full": bb.n_nodes, "n_nodes_variant": len(variant_nodes),
        "n_dyads_full": bb.n_dyads, "n_dyads_variant": len(dyads),
        "n_events_full": bb.n_events, "n_events_variant": len(times),
    }
    # Distinct source times map one-to-one to normalized times, so equality carries over.
    result["same_global_timestamp_multiset_source_units"] = (
        result["same_global_timestamp_multiset"] and result["timestamp_normalization_injective"])
    checks = ("same_node_set", "same_collapsed_dyad_set", "same_total_events",
              "same_per_dyad_count_multiset", "same_exact_per_dyad_counts",
              "same_global_timestamp_multiset", "same_global_timestamp_multiset_source_units")
    result["hard_failure"] = not all(result[name] for name in checks)
    return result


def task_timing(bb, natural_rho2, census_twins):
    checks, invariants, profiles, failures = [], [], [], []
    upper = np.mean(bb.m >= 2)
    n_multi = int(np.sum(bb.m >= 2))
    for target, seed, instance, labels, error in timing_variants(bb):
        exact = Fraction(target)
        common = {"dataset": bb.key, "variant": variant_name(target),
                  "target_rho2": float(target), "seed": seed}
        row = {**common, "status": "error" if instance is None else "ok",
               "natural_rho2": natural_rho2, "p_m_ge_2_count_upper_bound": float(upper),
               "count_bound_feasible": n_multi * exact.denominator >= bb.n_dyads * exact.numerator,
               "count_bound_margin": float(upper) - float(target), "error": error}
        if instance is None:
            failures.append(f"{bb.key} {variant_name(target)}: generator error: {error}")
            checks.append(row)
            continue
        events = instance.events
        _, pairs, times, _, _ = prepare_complete(events)
        rho = survival(occupancy_counts(pairs, times, W_MAIN), W_MAIN)[2:W_MAIN + 1]
        invariant = timing_invariants(bb, labels[events["u"].to_numpy(np.int64)],
                                      labels[events["v"].to_numpy(np.int64)],
                                      events["t"].to_numpy(float))
        census = census_twins[(census_twins.dataset == bb.key)
                              & (census_twins.requested_rho_2 == float(target))]
        reproduced = "NA"
        if len(census):
            reproduced = bool(int(census.seed.iloc[0]) == seed
                              and float(census.achieved_rho_2.iloc[0]) == float(rho[0]))
            if not reproduced:
                failures.append(f"{bb.key} {variant_name(target)}: census attempt not reproduced")
        row.update(achieved_rho2=float(rho[0]), absolute_target_error=abs(float(rho[0]) - float(target)),
                   achieved_rho3=float(rho[1]), achieved_rho4=float(rho[2]),
                   achieved_rho5=float(rho[3]), allocation_deviations=int(instance.deviations),
                   invariants_all_hold=not invariant["hard_failure"],
                   census_attempt_reproduced=reproduced)
        checks.append(row)
        invariants.append({**common, **invariant})
        if invariant["hard_failure"]:
            failures.append(f"{bb.key} {variant_name(target)}: timing-variant invariant violated")
        profiles.append({"dataset": bb.key, "graph_kind": variant_name(target),
                         "valid": not invariant["hard_failure"],
                         **{name: float(value) for name, value in zip(RHO, rho)}})
    return checks, invariants, profiles, failures


# ----------------------------------------------------------------------------
# Tasks 3-5 and 6A: observations
# ----------------------------------------------------------------------------

def walk_row(bb, structure, bipartite, nodes, dyads, steps):
    visited = np.unique(nodes[:steps + 1])
    traversed = np.unique(dyads[:steps])
    degree = structure["degree"]
    start = int(nodes[0])
    start_component = structure["component"][start]
    row = {"start_node_rule": START_NODE_RULE, "start_node": start,
           "start_in_largest_component": bool(start_component == structure["largest"]),
           "start_component_node_fraction": structure["component_nodes"][start_component] / bb.n_nodes,
           "walk_steps": steps, "steps_per_full_dyad": steps / bb.n_dyads,
           "unique_visited_nodes": len(visited), "unique_traversed_dyads": len(traversed),
           "node_coverage": len(visited) / bb.n_nodes,
           "dyad_coverage": len(traversed) / bb.n_dyads,
           "event_coverage": int(bb.m[traversed].sum()) / bb.n_events,
           # Steps arriving at an already visited node / traversing a known dyad.
           "revisit_fraction": (steps - (len(visited) - 1)) / steps,
           "repeated_dyad_step_fraction": (steps - len(traversed)) / steps,
           "visited_degree_mean": float(np.mean(degree[visited])),
           "visited_degree_median": float(np.median(degree[visited])),
           "full_degree_mean": float(np.mean(degree)), "full_degree_median": float(np.median(degree)),
           "visited_mean_degree_ratio": float(np.mean(degree[visited]) / np.mean(degree)),
           "largest_component_node_fraction": structure["largest_node_fraction"],
           "largest_component_dyad_fraction": structure["largest_dyad_fraction"],
           "largest_component_event_fraction": structure["largest_event_fraction"],
           "n_components": structure["n_components"], "bipartite": "yes" if bipartite else "no"}
    sides = dict.fromkeys(("users_full", "items_full", "visited_users", "visited_items",
                           "user_side_coverage", "item_side_coverage",
                           "visited_user_proportion", "visited_item_proportion"))
    row.update(side_labels="not bipartite", **sides)
    if bipartite:
        users, items = bipartite_sides(bb)
        n_users, n_items = int(np.sum(users[visited])), int(np.sum(items[visited]))
        row.update(side_labels="registry bipartite namespaces u:/i: (user/item)",
                   users_full=int(users.sum()), items_full=int(items.sum()),
                   visited_users=n_users, visited_items=n_items,
                   user_side_coverage=n_users / users.sum(), item_side_coverage=n_items / items.sum(),
                   visited_user_proportion=n_users / len(visited),
                   visited_item_proportion=n_items / len(visited))
    return row


def complete_targets(bb):
    """Census K and rho_2..rho_W for every W of the grid."""
    K_full = {W: occupancy_counts(bb.pair, bb.t_norm, W) for W in W_GRID}
    return K_full, {W: survival(K_full[W], W)[2:W + 1] for W in W_GRID}


def recency_purity(bb, windows, rho_full, K_full, target):
    """Suffix observation plus its roster and survivor decompositions."""
    cutoff, retained = recency_cutoff(bb.t_norm, target)
    mask = bb.t_norm >= cutoff
    if int(mask.sum()) != retained:
        raise AssertionError("recency cutoff accounting does not close")
    summary = observation_summary(bb, windows, rho_full, mask)
    row, _, K, seen = summary
    roster = roster_profile(K[W_MAIN], W_MAIN)
    survivors = survival_profile(np.where(seen, K_full[W_MAIN], 0), W_MAIN, int(seen.sum()))
    window = int(window_index(np.array([cutoff]), 0.0, 1.0 / W_MAIN, W_MAIN)[0])
    purity = {
        "dataset": bb.key, "target_event_fraction": float(target),
        "cutoff_normalized": cutoff,
        "cutoff_source_units": float(bb.t_source[bb.t_norm == cutoff][0]),
        "cutoff_w5_window": window, "w5_windows_in_suffix": W_MAIN - window,
        "event_coverage": row["event_coverage"], "dyad_coverage": row["dyad_coverage"],
        "fraction_full_dyads_with_recent_event": row["dyad_coverage"],
        "vanished_dyad_fraction": (bb.n_dyads - row["n_observed_dyads"]) / bb.n_dyads,
        "node_coverage": row["node_coverage"],
        **{f"rho_full_{k}": float(rho_full[W_MAIN][k - 2]) for k in range(2, W_MAIN + 1)},
        # History loss plus disappearance: only dyads with suffix events are seen.
        **{f"plugin_rho_{k}": row[f"rho_hat_{k}"] for k in range(2, W_MAIN + 1)},
        "plugin_profile_mae_w5": row["profile_mae_w5"],
        # History loss only: known roster, zero visible windows allowed.
        **{f"roster_rho_{k}": float(roster[k - 2]) for k in range(2, W_MAIN + 1)},
        "roster_profile_mae_w5": profile_mae_w5(roster, rho_full[W_MAIN]),
        # Selection only: the surviving dyads with their complete histories.
        **{f"survivor_full_history_rho_{k}": float(survivors[k - 2])
           for k in range(2, W_MAIN + 1)},
        "survivor_full_history_profile_mae_w5": profile_mae_w5(survivors, rho_full[W_MAIN]),
    }
    return cutoff, summary, purity


def task_observations(bb, spec):
    windows = complete_windows(bb)
    K_full, rho_full = complete_targets(bb)
    for W in W_GRID:
        if not np.array_equal(occupancy(bb.pair, windows[W], bb.n_dyads, W), K_full[W]):
            raise AssertionError(f"{bb.key}: subset occupancy differs from census at W={W}")
    structure = graph_structure(bb)
    bipartite = bool(spec.get("format", {}).get("bipartite"))
    severity, w_rows, recency, rw = [], [], [], []

    def record(mechanism, target, parameter, slot, seed, summary):
        identity = {"dataset": bb.key, "mechanism": mechanism, "role": MECHANISMS[mechanism],
                    "target_event_fraction": float(target),
                    "sampler_parameter_name": PARAMETER_NAMES[mechanism],
                    "sampler_parameter_or_budget": parameter, "seed_slot": slot, "seed": seed}
        row, errors = summary[0], summary[1]
        severity.append({**identity, **row})
        w_rows.extend({key: identity[key] for key in ("dataset", "mechanism", "role",
                                                      "target_event_fraction", "seed_slot", "seed")}
                      | {"W": W, "normalized_profile_error": errors[W]} for W in W_GRID)
        return row

    def observe(mask):
        return observation_summary(bb, windows, rho_full, mask)

    # Reference: fixed-size uniform node sample, induced subgraph, full histories.
    mechanism = "node_panel_full_history"
    seeds = [diagnostic_seed(bb.key, mechanism, slot) for slot in SEED_SLOTS]
    orders = [uniform_node_order(bb.n_nodes, seed) for seed in seeds]
    prefix = np.vstack([node_prefix_events(bb, order) for order in orders])
    sizes = np.arange(2, bb.n_nodes + 1)
    for target in EVENT_FRACTIONS:
        size, _ = closest_median_parameter(sizes, prefix[:, sizes] / bb.n_events, target)
        for slot, seed, order in zip(SEED_SLOTS, seeds, orders):
            mask = induced_dyads(bb, order[:size])[bb.pair]
            record(mechanism, target, size, slot, seed, observe(mask))

    # Selection: one simple random walk; full histories of traversed dyads.
    mechanism = "simple_random_walk_full_history"
    seeds = [diagnostic_seed(bb.key, mechanism, slot) for slot in SEED_SLOTS]
    walks, choices, simulated = calibrate_walks(bb, walk_index(bb), seeds, EVENT_FRACTIONS,
                                                structure)
    for target in EVENT_FRACTIONS:
        steps, _ = choices[target]
        for slot, seed, (nodes, dyads) in zip(SEED_SLOTS, seeds, walks):
            row = record(mechanism, target, steps, slot, seed,
                         observe(full_history_mask(bb, dyads[:steps])))
            diagnostics = walk_row(bb, structure, bipartite, nodes, dyads, steps)
            if diagnostics["event_coverage"] != row["event_coverage"]:
                raise AssertionError("walk event coverage differs from its observation")
            rw.append({"dataset": bb.key, "target_event_fraction": float(target),
                       "seed_slot": slot, "seed": seed, "walk_steps_simulated": simulated,
                       **diagnostics})
    del walks

    # History loss: recent suffix of the complete time axis.
    mechanism = "recency_truncation"
    for target in EVENT_FRACTIONS:
        cutoff, summary, purity = recency_purity(bb, windows, rho_full, K_full, target)
        record(mechanism, target, cutoff, None, None, summary)
        recency.append(purity)

    # Both: uniform event sample without replacement; only sampled records visible.
    mechanism = "sampled_event_stream"
    events = pd.DataFrame({"u": bb.dyad_u[bb.pair], "v": bb.dyad_v[bb.pair], "t": bb.t_norm})
    for target in EVENT_FRACTIONS:
        size = event_sample_size(bb.n_events, target)
        for slot in SEED_SLOTS:
            seed = diagnostic_seed(bb.key, mechanism, slot)
            record(mechanism, target, size, slot, seed,
                   observe(sampled_event_mask(bb, events, size, seed)))
    return rho_full, severity, w_rows, recency, rw


# ----------------------------------------------------------------------------
# Summaries and Task 6B
# ----------------------------------------------------------------------------

def summarize_severity(raw):
    keys = ["dataset", "mechanism", "role", "target_event_fraction", "sampler_parameter_name",
            "sampler_parameter_or_budget"]
    rows = []
    for values, group in raw.groupby(keys, sort=False):
        row = dict(zip(keys, values), n_realizations=len(group))
        for metric in ("event_coverage", "dyad_coverage", "node_coverage", "profile_mae_w5"):
            row.update({f"{metric}_median": group[metric].median(),
                        f"{metric}_min": group[metric].min(), f"{metric}_max": group[metric].max()})
        rows.append(row)
    return pd.DataFrame(rows)


def mechanism_order(errors):
    """Mechanisms by ascending median error; the name only breaks exact ties."""
    ranked = sorted(errors.items(), key=lambda item: (item[1], item[0]))
    return tuple(name for name, _ in ranked)


def summarize_w_sensitivity(raw):
    """Seed-median per dataset x mechanism x severity cell, then medians over cells."""
    cells = (raw.groupby(["W", "dataset", "mechanism", "target_event_fraction"], as_index=False)
             ["normalized_profile_error"].median())
    orders, rows = {}, []
    for W in W_GRID:
        at_w = cells[cells.W == W]
        orders[W, "all"] = mechanism_order(at_w.groupby("mechanism").normalized_profile_error
                                           .median().to_dict())
        for target, group in at_w.groupby("target_event_fraction"):
            orders[W, target] = mechanism_order(group.groupby("mechanism")
                                                .normalized_profile_error.median().to_dict())
        for (dataset, target), group in at_w.groupby(["dataset", "target_event_fraction"]):
            orders[W, dataset, target] = mechanism_order(
                dict(zip(group.mechanism, group.normalized_profile_error)))
    for W in W_GRID:
        at_w = cells[cells.W == W]
        for mechanism, group in at_w.groupby("mechanism"):
            rows.append({"summary_type": "mechanism", "W": W, "mechanism": mechanism,
                         "target_event_fraction": "all", "n_cells": len(group),
                         "median_normalized_profile_error": group.normalized_profile_error.median(),
                         "rank_lowest_error_first": orders[W, "all"].index(mechanism) + 1,
                         "mechanism_ordering": " < ".join(orders[W, "all"]),
                         "same_ordering_as_W5": orders[W, "all"] == orders[W_MAIN, "all"]})
        for target, group in at_w.groupby("target_event_fraction"):
            rows.append({"summary_type": "severity", "W": W, "mechanism": "all",
                         "target_event_fraction": target, "n_cells": len(group),
                         "median_normalized_profile_error": group.normalized_profile_error.median()})
            for mechanism, sub in group.groupby("mechanism"):
                rows.append({"summary_type": "mechanism_within_severity", "W": W,
                             "mechanism": mechanism, "target_event_fraction": target,
                             "n_cells": len(sub),
                             "median_normalized_profile_error": sub.normalized_profile_error.median(),
                             "rank_lowest_error_first": orders[W, target].index(mechanism) + 1,
                             "mechanism_ordering": " < ".join(orders[W, target]),
                             "same_ordering_as_W5": orders[W, target] == orders[W_MAIN, target]})
        pairs = at_w[["dataset", "target_event_fraction"]].drop_duplicates().itertuples(index=False)
        same = [orders[W, d, t] == orders[W_MAIN, d, t] for d, t in pairs]
        rows.append({"summary_type": "dataset_severity_ordering_agreement", "W": W,
                     "mechanism": "all", "target_event_fraction": "all", "n_cells": len(same),
                     "share_dataset_severity_cells_same_ordering_as_W5": float(np.mean(same))})
    return pd.DataFrame(rows)


def grouped_loso_constant(profiles):
    """Held-out prediction = mean profile of the other backbones' graphs of the same kind."""
    rows = []
    for kind, group in profiles.groupby("graph_kind", sort=False):
        for held_out in group.dataset:
            reference = group[(group.dataset != held_out) & group.valid]
            if held_out in set(reference.dataset) or reference.empty:
                raise AssertionError("grouped LOSO reference must exclude the held-out backbone")
            prediction = reference[list(RHO)].to_numpy(float).mean(axis=0)
            truth = group.loc[group.dataset == held_out, list(RHO)].to_numpy(float)[0]
            rows.append({"held_out_dataset": held_out, "graph_kind": kind,
                         "held_out_valid": bool(group.loc[group.dataset == held_out, "valid"].iloc[0]),
                         "reference_datasets": ";".join(reference.dataset),
                         "n_reference_datasets": len(reference),
                         "held_out_in_reference": False,
                         **{f"pred_{name}": float(p) for name, p in zip(RHO, prediction)},
                         **{f"true_{name}": float(t) for name, t in zip(RHO, truth)},
                         "profile_mae_w5": profile_mae_w5(prediction, truth)})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Documentation checks and README
# ----------------------------------------------------------------------------

def design_mismatches():
    study = yaml.safe_load((ROOT / "config/study.yaml").read_text(encoding="utf-8"))
    timing = study.get("controlled_timing", {})
    notes = []
    if [float(t) for t in timing.get("candidate_rho_2_targets", [])] != [0.15, 0.35]:
        notes.append(
            f"Timing targets: `config/study.yaml` lists {timing.get('candidate_rho_2_targets')}, "
            f"`dataset_census.TARGETS` is {list(dataset_census.TARGETS)}, and README.md, "
            "docs/STUDY_DESIGN.md and docs/DATASETS.md say 0.15/0.55. These checks use the "
            "intended 0.15/0.35.")
    if "global_timestamp_multiset" not in " ".join(timing.get("preserve", [])):
        notes.append(
            f"`config/study.yaml` controlled_timing.preserve = {timing.get('preserve')} omits the "
            "global timestamp multiset, which the empirical-pool generator preserves and which is "
            "checked here.")
    if study.get("datasets", {}).get("panel_status") != "final":
        notes.append(
            f"`config/study.yaml` datasets.panel_status = "
            f"{study.get('datasets', {}).get('panel_status')!r}; the intended eight-backbone "
            "panel is not recorded in the config (left unchanged).")
    if study.get("replication", {}).get("master_seed") is None:
        notes.append("`config/study.yaml` master_seed is null and no numeric sampler seeds "
                     "exist; the diagnostic seeds in section 3 were defined for these checks only.")
    notes += [
        "Reference mechanism: docs/SAMPLING.md gives full histories of relations *incident* to "
        "selected participants (README.md and docs/STUDY_DESIGN.md say only \"uniform random "
        "node/participant panel\"). `nonwalk_samplers.node_panel_full_history` returns incident "
        "dyads with an adaptive whole-node event-budget stop. The intended mechanism is a "
        "fixed-size uniform node sample with the *induced* subgraph, which is what is "
        "implemented here. `nonwalk_samplers.node_panel_size` already uses the induced "
        "inclusion probability n(n-1)/(N(N-1)).",
        "Selection mechanism: no full-history random walk exists in src/. `walks.run_walk` "
        "(time_agnostic) records no timestamps. These checks reuse its transitions and add the "
        "full-history lookup. The start-node and component policy is not documented; the rule "
        "used is stated in the seeds section.",
        "History-loss mechanism: not implemented in src/. docs/SAMPLING.md says it keeps the "
        "\"same conceptual population access as the reference\" and leaves the roster and the "
        "zero-event dyads unspecified. Here it is a suffix of the complete stream, so dyads without "
        "suffix events vanish; the known-roster counterfactual is reported separately.",
        "Time axis: the design text says [0,1), but the census/generator map t_max to 1.0 in "
        "the last window with a 1e-9 guard (documented in docs/REPRODUCIBILITY.md). That "
        "convention is used unchanged, and observations are never rescaled.",
    ]
    return notes


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "NA"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "NO"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return f"{value:.{digits}f}"


def table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    return lines + ["| " + " | ".join(map(str, row)) + " |" for row in rows]


def git_state():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout.strip()
        return head + (" (working tree has uncommitted changes)" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def write_readme(path, command, out, analyzed, absent, failures, seconds):
    panel, timing, severity = out["panel"], out["timing"], out["severity_summary"]
    recency, rw, w_summary, loso = out["recency"], out["rw"], out["w_summary"], out["loso"]
    names = dict(PANEL)
    lines = [
        "# Pre-main-experiment freeze checks", "",
        "Deterministic diagnostics for the intended design (W=5, rho_2..rho_5, ProfileMAE). "
        "There are no LLM/API calls, downloads or config changes, and the census outputs are "
        "read only. These tables contain numbers only; no design decision is made here.", "",
        "## 1. Command", "", "```bash", command, "```", "",
        f"Git HEAD `{git_state()}`; Python {platform.python_version()}, NumPy {np.__version__}, "
        f"pandas {pd.__version__}; runtime {seconds / 60:.1f} min. Outputs are written only to "
        "this directory, and existing outputs are not overwritten unless `--overwrite` is given. "
        "To reproduce elsewhere, add `--out-dir <new-dir>`. `/results/**` is git-ignored by the "
        "existing `.gitignore`.", "",
        "## 2. Datasets analyzed", ""]
    lines += table(["Design name", "Registry key", "Status"],
                   [[names[key], f"`{key}`", "analyzed" if key in analyzed else "ABSENT"]
                    for key, _ in PANEL if key in analyzed or key in absent])
    lines += ["", "## 3. Seeds and parameter rules", "",
              "- Sampler seeds (diagnostic only; no final seeds exist): four slots 0-3. Each is "
              "`numpy.random.default_rng(crc32(\"pre_main_freeze_checks|<dataset>|<mechanism>|slot<j>\"))`. "
              "The numbers are in every raw CSV row and carry no scientific meaning. The 25% and "
              "50% observations of a seed are nested (prefix designs).",
              f"- Timing variants: one attempt per backbone x target, seed `crc32(\"{TIMING_SEED_BASE}|census_twin|<dataset>|<target>\")`. "
              "This is the census rule, so rho2_015 repeats the census attempt. The generator is "
              "`family_from_events` + `make_instance(hub_bias=False, span_layout=\"contiguous\")` "
              "with the empirical timestamp pool.",
              "- Reference: uniform node order (`default_rng(seed).permutation(N)`); panel size n "
              "= smallest n whose median event coverage over the 4 seeds is closest to the target; "
              "induced dyads with full histories.",
              f"- Selection: `walks.run_walk(time_agnostic)` on the collapsed graph. Start rule: {START_NODE_RULE}. "
              "Walk steps = smallest step count whose median event coverage over the 4 seeds is "
              "closest to the target. Walks are simulated for 4|E| steps, doubled up to 64|E| "
              "until the median passes 50% (`walk_steps_simulated`). Each traversed dyad exposes "
              "its full history once.",
              "- History loss: cutoff c = the distinct complete-axis time whose suffix {t >= c} "
              "has the event share closest to the target (ties go to the later c). Suffix events "
              "only.",
              "- Both: `nonwalk_samplers.uniform_event_reservoir` with round(target x M) records.",
              "- Plug-in: K from visible events on the complete-axis windows, with observed dyads "
              "(K>0) as the denominator. normalized_profile_error(W) = (1/W) sum_{k=2..W} "
              "|rho_full(W,k) - rho_hat(W,k)|, which is 0.8 x ProfileMAE at W=5.",
              "- RW revisit_fraction = steps that arrive at an already visited node / steps. "
              "repeated_dyad_step_fraction = steps along an already discovered dyad / steps. "
              "Degrees are collapsed full-graph degrees.",
              "- W summaries take the seed median within each dataset x mechanism x severity "
              "cell, then the median over cells.", ""]

    lines += ["## 4. Key results", "", "**Panel and timing variants (W=5).** Achieved rho_2 per "
              "target; `inv` = all exact invariants hold.", ""]
    rows = []
    for key in analyzed:
        p = panel[panel.dataset == key].iloc[0]
        t = timing[timing.dataset == key].set_index("variant")
        cell = lambda v: (f"{fmt(t.loc[v, 'achieved_rho2'])} (inv {fmt(bool(t.loc[v, 'invariants_all_hold']))})"
                          if v in t.index and t.loc[v, "status"] == "ok" else "error")
        rows.append([key, fmt(int(p.n_nodes)), fmt(int(p.n_edges_full)), fmt(int(p.n_events)),
                     " / ".join(fmt(float(p[name])) for name in RHO), fmt(float(p.p_m_ge_2)),
                     cell("rho2_015"), cell("rho2_035")])
    lines += table(["dataset", "nodes", "dyads", "events", "rho_2..rho_5", "P(m>=2)",
                    "rho2_015", "rho2_035"], rows)
    analyzed_panel = panel[panel.dataset.isin(analyzed)]
    cross = ["sha256_matches_census", "recomputed_matches_census", "w_grid_rho_matches_census"]
    lines += ["", f"Census cross-checks (raw SHA-256; recomputed counts, rho, m_e shares, duplicates, "
              f"exclusions and horizon; W-grid rho) hold for "
              f"{int(analyzed_panel[cross].all(axis=1).sum())} of {len(analyzed_panel)} backbones."]
    reddit = panel[panel.dataset == "jodie_reddit"]
    if len(reddit) and reddit.status.iloc[0] == "ok":
        r = reddit.iloc[0]
        lines += [f"JODIE Reddit identity check: {r.dataset_identity_check.split(':')[0]} "
                  f"({r.bipartite_sides}; {int(r.n_events)} interactions, matching the published "
                  "JODIE Reddit counts; header, id ranges, SHA-256 and file time order also "
                  "checked; see `dataset_identity_check`)."]

    lines += ["", "**Severity (ProfileMAE at W=5, median over seeds; median event coverage in "
              "brackets).**", ""]
    rows = []
    for key in analyzed:
        for target in map(float, EVENT_FRACTIONS):
            s = severity[(severity.dataset == key) & (severity.target_event_fraction == target)]
            s = s.set_index("mechanism")
            rows.append([key, fmt(target, 2)] + [
                f"{fmt(s.loc[m, 'profile_mae_w5_median'])} [{fmt(s.loc[m, 'event_coverage_median'])}]"
                for m in MECHANISMS])
    lines += table(["dataset", "target", "reference (node)", "selection (RW)",
                    "history loss (recency)", "both (events)"], rows)

    lines += ["", "**Recency purity (W=5).** plug-in = observed suffix. roster = full dyad roster "
              "with truncated histories. survivor = surviving dyads with full histories.", ""]
    rows = [[r.dataset, fmt(r.target_event_fraction, 2), fmt(r.cutoff_normalized),
             fmt(int(r.w5_windows_in_suffix)), fmt(r.vanished_dyad_fraction),
             fmt(r.plugin_profile_mae_w5), fmt(r.roster_profile_mae_w5),
             fmt(r.survivor_full_history_profile_mae_w5)] for r in recency.itertuples()]
    lines += table(["dataset", "target", "cutoff", "W5 windows in suffix", "vanished dyads",
                    "MAE plug-in", "MAE roster", "MAE survivor"], rows)

    lines += ["", "**Simple random walk at 50% target (median over seeds).**", ""]
    rows = []
    for key in analyzed:
        r = rw[(rw.dataset == key) & (rw.target_event_fraction == 0.5)]
        if r.empty:
            continue
        rows.append([key, fmt(int(r.walk_steps.iloc[0])), fmt(r.steps_per_full_dyad.iloc[0], 2),
                     fmt(r.node_coverage.median()), fmt(r.dyad_coverage.median()),
                     fmt(r.event_coverage.median()), fmt(r.revisit_fraction.median()),
                     fmt(r.repeated_dyad_step_fraction.median()),
                     fmt(r.visited_mean_degree_ratio.median(), 2),
                     fmt(r.largest_component_node_fraction.iloc[0])])
    lines += table(["dataset", "steps", "steps/dyads", "node cov", "dyad cov", "event cov",
                    "revisit", "repeat-dyad steps", "visited/full mean degree", "LCC nodes"], rows)
    reddit = rw[(rw.dataset == "jodie_reddit")]
    if len(reddit):
        lines += [""] + [
            f"Reddit sides at {fmt(target, 2)} (median): user coverage "
            f"{fmt(g.user_side_coverage.median())}, item coverage {fmt(g.item_side_coverage.median())}, "
            f"visited nodes that are items {fmt(g.visited_item_proportion.median())} "
            f"(items are {fmt(g.items_full.iloc[0] / (g.users_full.iloc[0] + g.items_full.iloc[0]))} of all nodes)."
            for target, g in reddit.groupby("target_event_fraction")]

    lines += ["", "**W robustness.** Mechanisms by median normalized error over cells, lowest "
              "first.", ""]
    rows = []
    for W in W_GRID:
        m = w_summary[(w_summary.summary_type == "mechanism") & (w_summary.W == W)].iloc[0]
        within = w_summary[(w_summary.summary_type == "mechanism_within_severity")
                           & (w_summary.W == W)].drop_duplicates("target_event_fraction")
        agree = w_summary[(w_summary.summary_type == "dataset_severity_ordering_agreement")
                          & (w_summary.W == W)].iloc[0]
        rows.append([W, m.mechanism_ordering.replace("_full_history", "").replace("simple_random_walk", "srw"),
                     fmt(bool(m.same_ordering_as_W5)),
                     " / ".join(fmt(bool(x)) for x in within.same_ordering_as_W5),
                     fmt(agree.share_dataset_severity_cells_same_ordering_as_W5, 2)])
    lines += table(["W", "ordering (all cells)", "same as W=5", "same within 0.25 / 0.50",
                    "dataset x severity cells same"], rows)

    lines += ["", "**Grouped leave-one-backbone-out constant reference (ProfileMAE, W=5).**", ""]
    rows = []
    for key in analyzed:
        g = loso[loso.held_out_dataset == key].set_index("graph_kind")
        rows.append([key] + [fmt(g.loc[k, "profile_mae_w5"]) if k in g.index else "NA"
                             for k in ("natural", "rho2_015", "rho2_035")])
    lines += table(["held-out backbone", "natural", "rho2_015", "rho2_035"], rows)

    lines += ["", "## 5. Hard failures and mismatches", "", "**Hard failures:** "
              + ("none." if not failures else "")]
    lines += [f"- {failure}" for failure in failures]
    lines += ["", "**Implementation/documentation mismatches (recorded, not fixed):**", ""]
    lines += [f"- {note}" for note in design_mismatches()]
    lines += ["", "## 6. Outputs", ""]
    folder = path.parent.relative_to(ROOT) if ROOT in path.parent.parents else path.parent
    lines += [f"- `{folder / name}`" for name in list(OUTPUTS.values()) + ["README.md"]]
    lines += ["", "Code: `scripts/pre_main_freeze_checks.py`; tests: "
              "`tests/diagnostics/test_pre_main_freeze_checks.py`.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def write_csv(frame, path, overwrite):
    # Nullable integers keep counts and seeds integral next to NA cells.
    for column in INTEGER_COLUMNS:
        if column in frame:
            frame[column] = frame[column].astype("Int64")
    if "sampler_parameter_or_budget" in frame:
        # Integer budgets and the fractional recency cutoff share one column.
        frame["sampler_parameter_or_budget"] = pd.Series(
            [value if isinstance(value, float) and not value.is_integer() else int(value)
             for value in frame["sampler_parameter_or_budget"]], index=frame.index, dtype=object)
    with path.open("w" if overwrite else "x", encoding="utf-8", newline="") as stream:
        frame.to_csv(stream, index=False, na_rep="NA")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/pre_main_freeze_checks")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace existing outputs in --out-dir (never census outputs)")
    parser.add_argument("--only", nargs="+", choices=[key for key, _ in PANEL],
                        help="subset of the panel, for quick checks only")
    args = parser.parse_args()
    destination = args.out_dir.resolve()
    for protected in (ROOT / "data", ROOT / "archive", ROOT / "config",
                      ROOT / "results/dataset_census"):
        if destination == protected or protected in destination.parents:
            parser.error("output must be outside data/, archive/, config/ and the census results")
    paths = [destination / name for name in list(OUTPUTS.values()) + ["README.md"]]
    if not args.overwrite and any(path.exists() for path in paths):
        parser.error("outputs already exist; use --overwrite or a new --out-dir")
    started = time.time()
    registry = load_registry(ROOT / "config/datasets.yaml")
    # The census wrote shortest round-trip floats; the default fast parser can be off by one ulp.
    read = lambda name: pd.read_csv(ROOT / "results/dataset_census" / name,
                                    float_precision="round_trip")
    census = read("empirical_census.csv").set_index("dataset")
    census_windows = read("window_sensitivity.csv")
    census_twins = read("twin_realized_feasibility.csv")
    selected = [(key, name) for key, name in PANEL if not args.only or key in args.only]
    tables = {name: [] for name in OUTPUTS if name not in ("severity_summary", "w_summary")}
    profiles, analyzed, absent, failures = [], [], [], []

    def progress(key, step):
        print(f"[{time.time() - started:7.1f}s] {key}: {step}", flush=True)

    for key, name in selected:
        spec = registry[key]
        if not (ROOT / "data/raw" / spec["file"]).is_file():
            row, problems = integrity_row(key, name, spec, {}, None, None, None, None)
            tables["panel"].append(row)
            absent.append(key)
            failures.append(f"{key}: {problems[0]}")
            progress(key, "ABSENT")
            continue
        progress(key, "load")
        bb, parser_quality = load_backbone(key, spec, ROOT / "data/raw")
        progress(key, "observations (tasks 3-5, 6A)")
        rho_full, severity, w_rows, recency, rw = task_observations(bb, spec)
        row, problems = integrity_row(key, name, spec, census.loc[key].to_dict(), bb,
                                      parser_quality, rho_full, census_windows)
        tables["panel"].append(row)
        failures += [f"{key}: {problem}" for problem in problems]
        progress(key, "timing variants (task 2)")
        checks, invariants, variant_profiles, problems = task_timing(bb, float(rho_full[W_MAIN][0]),
                                                                     census_twins)
        failures += problems
        tables["timing"] += checks
        tables["invariants"] += invariants
        tables["severity_raw"] += severity
        tables["w_raw"] += w_rows
        tables["recency"] += recency
        tables["rw"] += rw
        profiles.append({"dataset": key, "graph_kind": "natural", "valid": True,
                         **{n: float(v) for n, v in zip(RHO, rho_full[W_MAIN])}})
        profiles += variant_profiles
        analyzed.append(key)
        del bb

    out = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    out["severity_summary"] = summarize_severity(out["severity_raw"])
    out["w_summary"] = summarize_w_sensitivity(out["w_raw"])
    ordered = pd.DataFrame(profiles)
    ordered["graph_kind"] = pd.Categorical(ordered.graph_kind,
                                           ["natural"] + [variant_name(t) for t in TIMING_TARGETS])
    ordered = ordered.sort_values("graph_kind", kind="mergesort")
    ordered["graph_kind"] = ordered.graph_kind.astype(str)
    out["loso"] = (grouped_loso_constant(ordered) if len(analyzed) >= 2
                   else pd.DataFrame(columns=["held_out_dataset"]))
    destination.mkdir(parents=True, exist_ok=True)
    for name, filename in OUTPUTS.items():
        write_csv(out[name], destination / filename, args.overwrite)
    python = Path(sys.executable)
    python = python.relative_to(ROOT) if ROOT in python.parents else python
    command = " ".join(
        (["PYTHONDONTWRITEBYTECODE=1"] if os.environ.get("PYTHONDONTWRITEBYTECODE") else [])
        + [str(python), str(Path(sys.argv[0]).resolve().relative_to(ROOT))] + sys.argv[1:])
    write_readme(destination / "README.md", command, out, analyzed, absent, failures,
                 time.time() - started)
    progress("all", f"wrote {len(OUTPUTS) + 1} files to {destination}")
    if failures:
        print("HARD FAILURES:\n" + "\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(1)
    print("No hard failures.")


if __name__ == "__main__":
    main()
