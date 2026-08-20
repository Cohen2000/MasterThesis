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
    """Uniform node-induced panel retaining complete internal dyad histories.

    The target budget calibrates panel *size in expectation*.  No panel is
    chosen or resized using its realized event count, which would leak outcome
    information into the design.  The realized count is deliberately allowed
    to vary and is the actual information budget.
    """
    prepared = events if isinstance(events, PreparedEvents) else prepare_events(events)
    x = prepared.events
    active = prepared.active_nodes
    n_panel = node_panel_size(len(active), len(x), int(target_budget))
    rng = np.random.default_rng(seed)
    panel = rng.permutation(active)[:n_panel]
    keep = x["u"].isin(panel) & x["v"].isin(panel)
    selected = x.loc[keep].sort_values(
        ["t", "event_id"], kind="mergesort").reset_index(drop=True)
    expected = len(x) * n_panel * (n_panel - 1) / (len(active) * (len(active) - 1))
    return SampleResult(_as_log(selected), {
        "sampling_design": "uniform_node_induced_full_history",
        "target_budget": int(target_budget),
        "realized_event_budget": int(len(selected)),
        "expected_event_budget": float(expected),
        "panel_nodes": int(n_panel),
        "panel_node_order": [int(node) for node in panel],
        "active_nodes_total": int(len(active)),
        "panel_node_fraction": float(n_panel / len(active)),
        "sampling_fraction_events": float(len(selected) / len(x)),
        "task_class": "node_panel_oracle_reference",
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
