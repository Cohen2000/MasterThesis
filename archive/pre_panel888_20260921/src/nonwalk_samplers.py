#!/usr/bin/env python3
"""Non-walk access models for the low-budget temporal-network screen.

All returned logs use the same core schema as :mod:`walks`, so the existing
observable-only feature builder can be reused.  A budget unit is one unique
event record.  Query counts are reported separately for ego retrieval.

The functions never mutate the input frame.  Event ids are stable row numbers
in the supplied complete stream and are used only to deduplicate retrievals;
they are not exposed as model features.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd


EXTRA_LOG_COLS = [
    "event_id", "query_id", "query_node", "partial_response",
    "response_size_total", "response_size_kept",
]


@dataclass(frozen=True)
class SampleResult:
    log: pd.DataFrame
    diagnostics: dict


@dataclass(frozen=True)
class PreparedEvents:
    events: pd.DataFrame
    active_nodes: np.ndarray
    incident_event_ids: dict
    chronological_order: np.ndarray


@dataclass(frozen=True)
class PreparedDyadHistories:
    """Reusable complete-dyad index for repeated PPS samples of one graph."""
    prepared: PreparedEvents
    edges: tuple
    event_ids: tuple
    weights: np.ndarray


def _events_with_ids(events) -> pd.DataFrame:
    if isinstance(events, PreparedEvents):
        return events.events
    required = {"u", "v", "t"}
    missing = required - set(events.columns)
    if missing:
        raise ValueError(f"events missing columns: {sorted(missing)}")
    if events.empty:
        raise ValueError("temporal graph contains no events")
    x = events[["u", "v", "t"]].copy().reset_index(drop=True)
    x["u"] = x["u"].astype(np.int64)
    x["v"] = x["v"].astype(np.int64)
    x["t"] = x["t"].astype(float)
    if not np.isfinite(x["t"]).all():
        raise ValueError("event timestamps must be finite")
    swap = x["u"] > x["v"]
    if swap.any():
        old_u = x.loc[swap, "u"].copy()
        x.loc[swap, "u"] = x.loc[swap, "v"].to_numpy()
        x.loc[swap, "v"] = old_u.to_numpy()
    x["event_id"] = np.arange(len(x), dtype=np.int64)
    return x


def prepare_events(events: pd.DataFrame) -> PreparedEvents:
    """Build immutable reusable indices once per complete graph."""
    x = _events_with_ids(events)
    incident = {}
    for row in x.itertuples(index=False):
        incident.setdefault(int(row.u), []).append(int(row.event_id))
        incident.setdefault(int(row.v), []).append(int(row.event_id))
    for ids in incident.values():
        ids.sort(key=lambda i: (float(x.at[i, "t"]), i), reverse=True)
    chronological = np.lexsort((x["event_id"].to_numpy(), x["t"].to_numpy()))
    return PreparedEvents(
        events=x,
        active_nodes=np.array(sorted(incident), dtype=np.int64),
        incident_event_ids=incident,
        chronological_order=chronological.astype(np.int64),
    )


def prepare_dyad_histories(events) -> PreparedDyadHistories:
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    edges = []
    event_ids = []
    for edge, frame in prepared.events.groupby(["u", "v"], sort=True):
        edges.append(tuple(map(int, edge)))
        event_ids.append(frame["event_id"].to_numpy(np.int64))
    weights = np.asarray([len(ids) for ids in event_ids], dtype=float)
    return PreparedDyadHistories(
        prepared=prepared, edges=tuple(edges), event_ids=tuple(event_ids),
        weights=weights)


def _as_log(selected: pd.DataFrame, **extra) -> pd.DataFrame:
    """Convert selected unique events to the common observation-log schema."""
    n = len(selected)
    out = pd.DataFrame({
        "kind": np.ones(n, dtype=np.int8),
        # For non-walk logs the endpoint union, not this convenience column,
        # defines observed nodes in benchmark_features.py.
        "node": selected["v"].to_numpy(np.int64),
        "u": selected["u"].to_numpy(np.int64),
        "v": selected["v"].to_numpy(np.int64),
        "t": selected["t"].to_numpy(float),
        "dt": np.full(n, np.nan),
        "event_id": selected["event_id"].to_numpy(np.int64),
        "query_id": np.full(n, -1, dtype=np.int64),
        "query_node": np.full(n, -1, dtype=np.int64),
        "partial_response": np.zeros(n, dtype=bool),
        "response_size_total": np.ones(n, dtype=np.int64),
        "response_size_kept": np.ones(n, dtype=np.int64),
    })
    for key, value in extra.items():
        out[key] = value
    return out


def uniform_event_reservoir(events: pd.DataFrame, budget: int,
                            seed: int) -> SampleResult:
    """Uniform fixed-size sample without replacement (random-priority prefix)."""
    if budget < 1:
        raise ValueError("budget must be positive")
    x = _events_with_ids(events)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))[:min(int(budget), len(x))]
    selected = x.iloc[order].reset_index(drop=True)
    return SampleResult(_as_log(selected), {
        "sampling_design": "srswor_events",
        "target_budget": int(budget),
        "realized_event_budget": int(len(selected)),
        "sampling_fraction_events": float(len(selected) / len(x)),
    })


def time_prefix_events(events: pd.DataFrame, budget: int) -> SampleResult:
    """The first ``budget`` events in chronological order (Rocha-TS analogue)."""
    if budget < 1:
        raise ValueError("budget must be positive")
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    x = prepared.events.iloc[prepared.chronological_order].reset_index(drop=True)
    selected = x.head(min(int(budget), len(x))).copy()
    return SampleResult(_as_log(selected), _window_diagnostics(
        selected, len(x), budget, placement="prefix"))


def time_random_window_events(events: pd.DataFrame, budget: int,
                              seed: int) -> SampleResult:
    """A uniformly anchored contiguous block containing ``budget`` events.

    The random anchor is seed-stable.  Increasing budgets therefore gives
    nested event sets, although their chronologically sorted logs are not
    literal row prefixes.  This is an event-count-matched window; its realized
    temporal width is reported and must not be confused with fixed-duration TS.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    x = prepared.events.iloc[prepared.chronological_order].reset_index(drop=True)
    take = min(int(budget), len(x))
    rng = np.random.default_rng(seed)
    anchor = int(rng.integers(len(x)))
    # Nearest event ranks to a fixed anchor form a nested contiguous block.
    priority = np.lexsort((np.arange(len(x)), np.abs(np.arange(len(x)) - anchor)))
    ranks = np.sort(priority[:take])
    selected = x.iloc[ranks].reset_index(drop=True)
    diag = _window_diagnostics(selected, len(x), budget, placement="random_anchor")
    diag["window_anchor_event_rank"] = anchor
    return SampleResult(_as_log(selected), diag)


def _window_diagnostics(selected, total_events, budget, placement):
    if selected.empty:
        t0 = t1 = width = float("nan")
    else:
        t0 = float(selected["t"].min())
        t1 = float(selected["t"].max())
        width = t1 - t0
    return {
        "sampling_design": f"time_window_{placement}_event_count",
        "target_budget": int(budget),
        "realized_event_budget": int(len(selected)),
        "sampling_fraction_events": float(len(selected) / total_events),
        "observed_time_start": t0,
        "observed_time_end": t1,
        "observed_time_width": width,
        "task_class": "full_horizon_extrapolation",
    }


def node_panel_size(n_active_nodes: int, n_events: int,
                    target_budget: int) -> int:
    """Panel size whose expected retained-event count is closest to target."""
    if n_active_nodes < 2 or n_events < 1 or target_budget < 1:
        raise ValueError("need >=2 active nodes, events, and a positive budget")
    n = np.arange(2, n_active_nodes + 1, dtype=float)
    expected = n_events * n * (n - 1) / (n_active_nodes * (n_active_nodes - 1))
    return int(n[np.argmin(np.abs(expected - target_budget))])


def node_panel_full_history(events: pd.DataFrame, target_budget: int,
                            seed: int) -> SampleResult:
    """Uniform node-order panel returning complete incident dyad histories.

    Nodes are drawn without replacement from every node represented in the
    event stream.  Selecting either endpoint reveals the dyad's complete event
    record.  Nodes are consumed in random-priority order and the sampler stops
    *before* the first node whose previously unseen incident events would
    exceed ``target_budget``.  Consequently no node or dyad history is ever
    truncated, although the realized budget can have slack.

    With a fixed panel size, every dyad would have the same inclusion
    probability.  Here the whole-node budget stop makes panel size a stopping
    time; diagnostics expose the realized size and slack rather than claiming
    exact fixed-size inclusion probabilities.
    """
    if target_budget < 1:
        raise ValueError("target_budget must be positive")
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    x = prepared.events
    active = prepared.active_nodes
    rng = np.random.default_rng(seed)
    node_order = rng.permutation(active)
    selected_ids = set()
    panel = []
    blocked_node = None
    blocked_response_size = 0
    for raw_node in node_order:
        node = int(raw_node)
        new_ids = [int(eid) for eid in prepared.incident_event_ids[node]
                   if int(eid) not in selected_ids]
        if len(selected_ids) + len(new_ids) > int(target_budget):
            blocked_node = node
            blocked_response_size = len(new_ids)
            break
        panel.append(node)
        selected_ids.update(new_ids)
    selected = x.loc[sorted(selected_ids)].sort_values(
        ["t", "event_id"], kind="mergesort").reset_index(drop=True)
    n_panel = len(panel)
    if len(active) >= 2:
        exact_fixed_size_inclusion = 1.0 - (
            (len(active) - n_panel) * (len(active) - n_panel - 1)
            / (len(active) * (len(active) - 1)))
    else:
        exact_fixed_size_inclusion = float(n_panel > 0)
    return SampleResult(_as_log(selected), {
        "sampling_design": "uniform_node_incident_full_history_whole_node_stop",
        "target_budget": int(target_budget),
        "realized_event_budget": int(len(selected)),
        "budget_slack": int(target_budget - len(selected)),
        "panel_nodes": int(n_panel),
        "panel_node_order": [int(node) for node in panel],
        "blocked_node": blocked_node,
        "blocked_new_event_count": int(blocked_response_size),
        "active_nodes_total": int(len(active)),
        "panel_node_fraction": float(n_panel / len(active)),
        "fixed_size_dyad_inclusion_probability": float(exact_fixed_size_inclusion),
        "adaptive_whole_node_stop": True,
        "partial_response_count": 0,
        "sampling_fraction_events": float(len(selected) / len(x)),
        "task_class": "node_panel_oracle_reference",
    })


def activity_proportional_dyad_full_history(events: pd.DataFrame, budget: int,
                                            seed: int) -> SampleResult:
    """PPS-without-replacement dyad sample with complete histories.

    Dyad selection weight is its event count.  An exponential-race weighted
    permutation implements sequential probability-proportional-to-size
    sampling without replacement.  Complete dyad histories are appended in
    that order.  A dyad
    that does not fit in the remaining unique-event budget is skipped, so one
    large history cannot turn an otherwise feasible sample into an empty one.
    No selected dyad is censored; unused budget is reported as slack.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    dyads = (events if isinstance(events, PreparedDyadHistories)
             else prepare_dyad_histories(events))
    prepared = dyads.prepared
    x = prepared.events
    grouped = tuple(zip(dyads.edges, dyads.event_ids))
    weights = dyads.weights
    rng = np.random.default_rng(seed)
    # Independent exponential clocks generate the exact size-biased order:
    # the next clock is selected with probability w_i / sum_j(w_j), and the
    # memoryless property repeats this rule after every removal.  This avoids
    # NumPy's much slower all-items weighted ``choice(..., replace=False)``.
    clocks = rng.exponential(scale=1.0 / weights)
    order = np.argsort(clocks, kind="stable")
    selected_ids = []
    selected_dyads = []
    skipped_dyads = []
    for position in order:
        edge, ids = grouped[int(position)]
        if len(selected_ids) + len(ids) > int(budget):
            skipped_dyads.append((edge, int(len(ids))))
            continue
        selected_dyads.append(edge)
        selected_ids.extend(map(int, ids))
    selected = x.loc[sorted(selected_ids)].sort_values(
        ["t", "event_id"], kind="mergesort").reset_index(drop=True)
    return SampleResult(_as_log(selected), {
        "sampling_design": "pps_event_count_dyad_full_history_whole_dyad_skip",
        "selection_unit": "dyad",
        "selection_weight": "full_event_count",
        "without_replacement": True,
        "target_budget": int(budget),
        "realized_event_budget": int(len(selected)),
        "budget_slack": int(budget - len(selected)),
        "selected_dyad_count": int(len(selected_dyads)),
        "selected_dyads": [[int(a), int(b)] for a, b in selected_dyads],
        "population_dyad_count": int(len(grouped)),
        "skipped_oversize_dyad_count": int(len(skipped_dyads)),
        "first_skipped_dyad": (list(skipped_dyads[0][0])
                                if skipped_dyads else None),
        "first_skipped_dyad_event_count": (int(skipped_dyads[0][1])
                                            if skipped_dyads else 0),
        "partial_response_count": 0,
        "sampling_fraction_events": float(len(selected) / len(x)),
        "task_class": "activity_size_biased_full_history",
    })


def event_sample_then_full_history(events: pd.DataFrame, budget: int,
                                   seed: int) -> SampleResult:
    """Uniform event discovery followed by complete-dyad lookup.

    Phase 1 exposes a random-priority prefix of the full event stream, i.e. a
    simple random sample without replacement at every fixed prefix length.
    When that prefix first names a dyad, phase 2 retrieves the dyad's complete
    event history.  The emitted log is the union of those complete histories.

    The prefix stops *before* the first newly discovered dyad whose complete
    history would make the union exceed ``budget``.  This whole-lookup stop is
    necessary for the two stated invariants to hold simultaneously: every
    discovered dyad in the emitted sample has a complete record, and at most
    ``budget`` unique event records are observed.  It can leave budget slack
    and, if the first random event belongs to a dyad with more than ``budget``
    events, can emit an empty sample.  Both outcomes are exposed explicitly.

    No full-stream dyad count is used to set a selection weight.  Activity
    proportionality arises because a dyad with more events has more chances to
    be the first newly named dyad in the uniform event ordering.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    dyads = (events if isinstance(events, PreparedDyadHistories)
             else prepare_dyad_histories(events))
    x = dyads.prepared.events
    history_by_edge = {
        tuple(map(int, edge)): np.asarray(ids, dtype=np.int64)
        for edge, ids in zip(dyads.edges, dyads.event_ids)
    }
    edge_by_event = list(zip(x["u"].astype(int), x["v"].astype(int)))
    order = np.random.default_rng(seed).permutation(len(x))

    phase1_ids = []
    selected_ids = set()
    selected_dyads = set()
    blocked_edge = None
    blocked_size = 0
    for raw_event_id in order:
        event_id = int(raw_event_id)
        edge = tuple(map(int, edge_by_event[event_id]))
        if edge in selected_dyads:
            # The event is a later member of the same uniform event prefix.
            # It was already returned by phase 2, so it costs no additional
            # unique-event budget, but remains part of the phase-1 split.
            phase1_ids.append(event_id)
            continue
        history = history_by_edge[edge]
        if len(selected_ids) + len(history) > int(budget):
            blocked_edge = edge
            blocked_size = int(len(history))
            break
        phase1_ids.append(event_id)
        selected_dyads.add(edge)
        selected_ids.update(map(int, history))

    selected = x.loc[sorted(selected_ids)].sort_values(
        ["t", "event_id"], kind="mergesort").reset_index(drop=True)
    phase1_set = set(phase1_ids)
    phase2_added = len(selected_ids - phase1_set)
    if len(phase1_ids) + phase2_added != len(selected_ids):
        raise AssertionError("two-phase unique-event accounting does not close")
    return SampleResult(_as_log(selected), {
        "sampling_design": (
            "uniform_event_random_priority_then_full_history_whole_lookup_stop"),
        "selection_unit_phase1": "event",
        "phase1_design": "srswor_random_priority_prefix",
        "phase2_design": "complete_history_lookup_for_discovered_dyads",
        "target_budget": int(budget),
        "realized_event_budget": int(len(selected_ids)),
        "budget_slack": int(budget - len(selected_ids)),
        "phase1_unique_event_count": int(len(phase1_ids)),
        "phase1_discovery_event_count": int(len(selected_dyads)),
        "phase2_additional_unique_event_count": int(phase2_added),
        "phase2_complete_history_event_count": int(len(selected_ids)),
        "selected_dyad_count": int(len(selected_dyads)),
        "population_dyad_count": int(len(history_by_edge)),
        "blocked_dyad": (list(blocked_edge) if blocked_edge is not None else None),
        "blocked_dyad_event_count": int(blocked_size),
        "stopped_before_nonfitting_lookup": bool(blocked_edge is not None),
        "empty_due_to_first_nonfitting_lookup": bool(
            blocked_edge is not None and not selected_dyads),
        "partial_response_count": 0,
        "sampling_fraction_events": float(len(selected_ids) / len(x)),
        "phase1_sampling_fraction_events": float(len(phase1_ids) / len(x)),
        "task_class": "sampled_event_log_plus_complete_dyad_lookup",
    })


def ego_recent_k_snowball(events: pd.DataFrame, budget: int, seed: int,
                          k: int | None) -> SampleResult:
    """End-time ego retrieval with recent-history depth ``k`` or all events.

    A queried ego returns its newest unique incident records first.  Only its
    single newest incident event expands the FIFO frontier.  Consequently, the
    query-node order is identical across k at a fixed query count; the extra
    history changes information depth but not frontier construction.  At a
    fixed event budget, larger k naturally permits fewer ego queries, so both
    event and query budgets are reported.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if k is not None and k < 1:
        raise ValueError("k must be positive or None for all history")
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    x = prepared.events
    incident = prepared.incident_event_ids
    active = prepared.active_nodes
    rng = np.random.default_rng(seed)
    restart_order = rng.permutation(active).tolist()
    restart_pos = 0
    frontier = deque()
    queued = set()
    queried = set()
    query_order = []
    seen_events = set()
    records = []
    query_id = 0
    n_restarts = 0
    partial_queries = 0

    def restart():
        nonlocal restart_pos, n_restarts
        while restart_pos < len(restart_order):
            node = int(restart_order[restart_pos]); restart_pos += 1
            if node not in queried and node not in queued:
                frontier.append(node); queued.add(node); n_restarts += 1
                return True
        return False

    restart()
    while frontier and len(records) < budget:
        ego = int(frontier.popleft()); queued.discard(ego)
        if ego in queried:
            if not frontier:
                restart()
            continue
        queried.add(ego)
        query_order.append(ego)
        ids_all = incident[ego]
        response = ids_all if k is None else ids_all[:int(k)]
        new_ids = [i for i in response if i not in seen_events]
        remaining = int(budget) - len(records)
        kept = new_ids[:remaining]
        is_partial = len(kept) < len(new_ids)
        partial_queries += int(is_partial)
        for eid in kept:
            seen_events.add(eid)
            row = x.iloc[eid]
            records.append({
                "u": int(row.u), "v": int(row.v), "t": float(row.t),
                "event_id": int(eid), "query_id": int(query_id),
                "query_node": ego, "partial_response": bool(is_partial),
                "response_size_total": int(len(new_ids)),
                "response_size_kept": int(len(kept)),
            })

        # k-independent expansion: the newest event is available for every
        # k>=1 and is the sole source of the next frontier node.
        newest = int(ids_all[0])
        row = x.iloc[newest]
        other = int(row.v) if int(row.u) == ego else int(row.u)
        if other not in queried and other not in queued:
            frontier.append(other); queued.add(other)
        query_id += 1
        if not frontier and len(records) < budget:
            restart()

    selected = pd.DataFrame(records)
    if selected.empty:
        # This should not occur on a nonempty graph but keeps the schema total.
        selected = pd.DataFrame(columns=[
            "u", "v", "t", "event_id", "query_id", "query_node",
            "partial_response", "response_size_total", "response_size_kept",
        ])
    log = _as_log(selected[["u", "v", "t", "event_id"]])
    for col in ("query_id", "query_node", "partial_response",
                "response_size_total", "response_size_kept"):
        log[col] = selected[col].to_numpy() if len(selected) else log[col]
    return SampleResult(log, {
        "sampling_design": "ego_recent_k_snowball_latest_frontier",
        "history_k": "all" if k is None else int(k),
        "query_time": "end",
        "deduplicate_events": True,
        "frontier_expansion": "newest_event_only",
        "target_budget": int(budget),
        "realized_event_budget": int(len(log)),
        "query_budget_realized": int(len(queried)),
        "query_node_order": query_order,
        "restart_count": int(n_restarts),
        "partial_query_count": int(partial_queries),
        "sampling_fraction_events": float(len(log) / len(x)),
        "task_class": "end_time_ego_retrieval",
    })


def neighbourhood_crawl(events: pd.DataFrame, budget: int, seed: int,
                        k: int | None, expansion: str = "bfs",
                        burn_prob: float = 0.35) -> SampleResult:
    """Snowball crawl that builds its frontier from the whole query response.

    The access primitive is the one used by :func:`ego_recent_k_snowball`: a
    queried node returns its ``k`` newest unique incident records at ``T_end``
    and only unique events are charged against the budget.  What differs is the
    frontier rule, so a contrast against ego retrieval at equal ``k`` isolates
    crawl geometry rather than information depth.

    ``bfs`` queues every neighbour appearing in the response, which spreads the
    crawl breadth-first.  ``forest_fire`` queues each of them independently
    with probability ``burn_prob``, the classic forward-burning design.  Ego
    retrieval expands only along the endpoint of the single newest event and
    therefore crawls a chain; that is why its query order is k-invariant while
    the query order here is not.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if k is not None and k < 1:
        raise ValueError("k must be positive or None for all history")
    if expansion not in {"bfs", "forest_fire"}:
        raise ValueError(f"unknown expansion {expansion!r}")
    if not 0.0 < float(burn_prob) <= 1.0:
        raise ValueError("burn_prob must lie in (0,1]")
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    x = prepared.events
    incident = prepared.incident_event_ids
    active = prepared.active_nodes
    rng = np.random.default_rng(seed)
    restart_order = rng.permutation(active).tolist()
    restart_pos = 0
    frontier = deque()
    queued = set()
    queried = set()
    query_order = []
    seen_events = set()
    records = []
    query_id = 0
    n_restarts = 0
    partial_queries = 0
    n_discovered = 0
    n_queued = 0

    def restart():
        nonlocal restart_pos, n_restarts
        while restart_pos < len(restart_order):
            node = int(restart_order[restart_pos]); restart_pos += 1
            if node not in queried and node not in queued:
                frontier.append(node); queued.add(node); n_restarts += 1
                return True
        return False

    restart()
    while frontier and len(records) < budget:
        ego = int(frontier.popleft()); queued.discard(ego)
        if ego in queried:
            if not frontier:
                restart()
            continue
        queried.add(ego)
        query_order.append(ego)
        ids_all = incident[ego]
        response = ids_all if k is None else ids_all[:int(k)]
        new_ids = [i for i in response if i not in seen_events]
        remaining = int(budget) - len(records)
        kept = new_ids[:remaining]
        is_partial = len(kept) < len(new_ids)
        partial_queries += int(is_partial)
        for eid in kept:
            seen_events.add(eid)
            row = x.iloc[eid]
            records.append({
                "u": int(row.u), "v": int(row.v), "t": float(row.t),
                "event_id": int(eid), "query_id": int(query_id),
                "query_node": ego, "partial_response": bool(is_partial),
                "response_size_total": int(len(new_ids)),
                "response_size_kept": int(len(kept)),
            })

        # The frontier is built from the response the query actually returned,
        # not from the events the budget happened to pay for: a repeated event
        # still names its endpoints.  This matches the ego convention, which
        # also expands from the response rather than from the charged events.
        candidates = []
        seen_candidates = set()
        for eid in response:
            row = x.iloc[eid]
            other = int(row.v) if int(row.u) == ego else int(row.u)
            if other == ego or other in queried or other in queued:
                continue
            if other not in seen_candidates:
                seen_candidates.add(other); candidates.append(other)
        n_discovered += len(candidates)
        if expansion == "forest_fire" and candidates:
            burn = rng.random(len(candidates)) < float(burn_prob)
            candidates = [node for node, hit in zip(candidates, burn) if hit]
        for node in candidates:
            frontier.append(node); queued.add(node)
        n_queued += len(candidates)
        query_id += 1
        if not frontier and len(records) < budget:
            restart()

    selected = pd.DataFrame(records)
    if selected.empty:
        selected = pd.DataFrame(columns=[
            "u", "v", "t", "event_id", "query_id", "query_node",
            "partial_response", "response_size_total", "response_size_kept",
        ])
    log = _as_log(selected[["u", "v", "t", "event_id"]])
    for col in ("query_id", "query_node", "partial_response",
                "response_size_total", "response_size_kept"):
        log[col] = selected[col].to_numpy() if len(selected) else log[col]
    return SampleResult(log, {
        "sampling_design": f"neighbourhood_crawl_{expansion}",
        "history_k": "all" if k is None else int(k),
        "query_time": "end",
        "deduplicate_events": True,
        "frontier_expansion": expansion,
        "burn_probability": (float(burn_prob) if expansion == "forest_fire"
                             else np.nan),
        "target_budget": int(budget),
        "realized_event_budget": int(len(log)),
        "query_budget_realized": int(len(queried)),
        "query_node_order": query_order,
        "restart_count": int(n_restarts),
        "partial_query_count": int(partial_queries),
        "discovered_node_count": int(n_discovered),
        "queued_node_count": int(n_queued),
        "sampling_fraction_events": float(len(log) / len(x)),
        "task_class": "end_time_neighbourhood_crawl",
    })


def temporal_nonstationarity_diagnostics(events: pd.DataFrame,
                                         bins: int = 10,
                                         T: float = 1.0) -> dict:
    """Truth-only diagnostics for prefix/window interpretation."""
    if bins < 2:
        raise ValueError("bins must be >=2")
    x = _events_with_ids(events)
    if (x["t"].min() < -1e-9) or (x["t"].max() > T + 1e-9):
        raise ValueError(f"event timestamps must lie in [0,{T}]")
    # Fixed full-horizon bins preserve empty startup/shutdown periods; using
    # the observed min/max would hide exactly the nonstationarity at issue.
    edges = np.linspace(0.0, T + np.finfo(float).eps, bins + 1)
    counts, _ = np.histogram(x["t"].to_numpy(float), bins=edges)
    counts = counts.astype(float)
    mean = float(counts.mean())
    slope = float(np.polyfit(np.arange(bins), counts, 1)[0] / mean) if mean else np.nan
    half = bins // 2
    early = float(counts[:half].mean())
    late = float(counts[-half:].mean())
    first = x[x["t"] < T / 2.0]
    all_nodes = set(x["u"]).union(x["v"])
    first_nodes = set(first["u"]).union(first["v"])
    all_dyads = set(zip(x["u"], x["v"]))
    first_dyads = set(zip(first["u"], first["v"]))
    return {
        "diag__event_rate_cv": float(np.std(counts) / mean) if mean else np.nan,
        "diag__event_rate_linear_slope_per_bin_mean": slope,
        "diag__late_early_event_rate_ratio": late / early if early else np.nan,
        "diag__nodes_seen_first_half_share": len(first_nodes) / max(1, len(all_nodes)),
        "diag__dyads_seen_first_half_share": len(first_dyads) / max(1, len(all_dyads)),
    }


def node_selection_diagnostics(log: pd.DataFrame, full_degrees,
                               selected_nodes=None) -> dict:
    """Truth-only realized degree-selection diagnostics.

    ``full_degrees`` maps active node id to its collapsed full-graph degree.
    When explicit query/panel nodes are unavailable, unique observed event
    endpoints define the selected set.
    """
    from scipy.stats import ks_2samp

    truth = np.asarray(list(full_degrees.values()), dtype=float)
    if selected_nodes is None:
        steps = log[log["kind"] == 1]
        selected_nodes = set(steps["u"].astype(int)).union(
            steps["v"].astype(int))
    chosen = np.asarray([
        full_degrees[int(node)] for node in sorted(set(selected_nodes))
        if int(node) in full_degrees
    ], dtype=float)
    if not len(chosen) or not len(truth):
        return {
            "diag__selected_node_count": int(len(chosen)),
            "diag__selected_true_degree_mean": np.nan,
            "diag__selected_degree_mean_ratio": np.nan,
            "diag__selected_degree_ks": np.nan,
        }
    return {
        "diag__selected_node_count": int(len(chosen)),
        "diag__selected_true_degree_mean": float(chosen.mean()),
        "diag__selected_degree_mean_ratio": float(chosen.mean() / truth.mean()),
        "diag__selected_degree_ks": float(ks_2samp(chosen, truth).statistic),
    }


def oracle_reservoir_ht(log: pd.DataFrame, full_edge_times: dict,
                        total_events: int, sample_size: int, W: int = 5,
                        T: float = 1.0) -> dict:
    """Truth-label Hájek/HT diagnostic for fixed-size event sampling.

    This deliberately uses each seen dyad's full multiplicity and full label.
    It is therefore an oracle decomposition, never an observable estimator.
    Inclusion probabilities are exact for simple random sampling without
    replacement: 1-C(M-m_e,B)/C(M,B).
    """
    from scipy.stats import hypergeom

    out = {f"oracle__reservoir_ht_true_label_rho_k{k}": np.nan
           for k in range(2, W + 1)}
    steps = log[log["kind"] == 1]
    seen = sorted(set(zip(steps["u"].astype(int), steps["v"].astype(int))))
    B = min(int(sample_size), int(total_events))
    if not seen or B <= 0:
        return out
    weights, occupancies = [], []
    for edge in seen:
        times = np.asarray(full_edge_times[edge], dtype=float)
        m = int(len(times))
        pi = float(hypergeom.sf(0, int(total_events), m, B))
        if pi <= 0:
            continue
        wi = np.floor(np.clip(times, 0.0, T) / (T / W)).astype(int)
        wi = np.clip(wi, 0, W - 1)
        occupancies.append(int(len(np.unique(wi))))
        weights.append(1.0 / pi)
    if not weights:
        return out
    weights = np.asarray(weights, dtype=float)
    occupancies = np.asarray(occupancies, dtype=int)
    for k in range(2, W + 1):
        out[f"oracle__reservoir_ht_true_label_rho_k{k}"] = float(
            np.average(occupancies >= k, weights=weights))
    return out
