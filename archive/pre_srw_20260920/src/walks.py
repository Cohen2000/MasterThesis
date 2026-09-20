#!/usr/bin/env python3
"""Temporal walk strategies and walk-summary features for the pilot.

Strategies
  time_agnostic    random walk on the collapsed graph; sees no timestamps at
                   all. This is the TRUE negative control: identical across
                   twins by construction, so any signal it shows is leakage.
  time_agnostic_t  same (time-agnostic) transitions, but each traversal also
                   observes one uniformly sampled event timestamp of that edge.
                   This is NOT a negative control but a static-transition /
                   temporal-observation ablation: it isolates whether merely
                   observing edge times (without time-respecting ordering)
                   already suffices to recover rho. (Related to "non-monotonic
                   temporal walks", Ma et al. 2026; avoid calling it an oracle,
                   it only ever sees real historical timestamps.)
  time_respecting  CTDNE-style: from (node x, time tau) move along a uniformly
                   chosen incident event with t > tau; dead end -> restart.
  recency_biased   like time_respecting, but among future events the weight is
                   exp(-(t - tau) / decay_scale): sooner events preferred.
  recent_history   reverse-time temporal walk.  At each node it retrieves the
                   k most recent incident events strictly before the current
                   query time and samples one uniformly.  This mirrors the
                   recent-neighbour access used by temporal-graph systems; it
                   is a different access model from the forward short-gap walk.

Budget accounting (fairness rule from the design):
  every log entry costs exactly 1 budget unit, both observed events ('step')
  and (re)placements ('restart', incl. the initial placement). All strategies
  are compared at identical budgets; sweep points are prefixes of one log.

Feature provenance:
  the structural summary features (unique nodes/edges, degree stats, discovery
  rate, edge-revisit, return times) adapt the EstGraph recipe (Maurya & Liu
  2026) from top-k lists / binned curves down to compact scalars; the temporal
  features (window coverage, walk_rho_*, step_dt) are this work's addition.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from census import window_index   # shared windowing convention (boundary guard)

LOG_COLS = ["kind", "node", "u", "v", "t", "dt"]  # kind: 0=restart, 1=step


# ----------------------------------------------------------------------------
# Index
# ----------------------------------------------------------------------------

@dataclass
class TemporalGraphIndex:
    n_nodes: int
    active_nodes: np.ndarray
    T: float
    W: int
    # per node: sorted event times, the other endpoint, canonical edge id parts
    nbr_times: list
    nbr_other: list
    # collapsed graph
    coll_adj: list
    coll_deg: np.ndarray
    # per canonical edge: sorted event times (for time_agnostic_t sampling)
    edge_times: dict


def build_index(events: pd.DataFrame, T: float = 1.0, W: int = 5) -> TemporalGraphIndex:
    u = events["u"].to_numpy(np.int64)
    v = events["v"].to_numpy(np.int64)
    t = events["t"].to_numpy(np.float64)
    # Timestamps must already be normalized to [0, T]. Synthetic pilot data is
    # in [0, 1] with T=1; real datasets are rescaled at load time (see
    # load_events in run_pilot_walks.py). The window math in run_walk/summarize
    # assumes this range, so fail loudly here instead of clipping silently.
    assert t.size == 0 or (t.min() >= -1e-9 and t.max() <= T + 1e-9), (
        f"event timestamps must lie in [0, T={T}]; got "
        f"[{float(t.min()):.4g}, {float(t.max()):.4g}] -- rescale at load time"
    )
    n = int(max(u.max(), v.max())) + 1

    nbr_t = [[] for _ in range(n)]
    nbr_o = [[] for _ in range(n)]
    edge_times = {}
    for ui, vi, ti in zip(u, v, t):
        a, b = (ui, vi) if ui < vi else (vi, ui)
        nbr_t[ui].append(ti); nbr_o[ui].append(vi)
        nbr_t[vi].append(ti); nbr_o[vi].append(ui)
        edge_times.setdefault((a, b), []).append(ti)

    nbr_times, nbr_other = [], []
    for x in range(n):
        ts = np.array(nbr_t[x]); os_ = np.array(nbr_o[x], dtype=np.int64)
        order = np.argsort(ts, kind="mergesort")
        nbr_times.append(ts[order]); nbr_other.append(os_[order])

    coll = [np.unique(nbr_other[x]) for x in range(n)]
    deg = np.array([len(c) for c in coll], dtype=np.int64)
    edge_times = {e: np.sort(np.array(ts)) for e, ts in edge_times.items()}
    active_nodes = np.flatnonzero(deg > 0).astype(np.int64)
    if len(active_nodes) == 0:
        raise ValueError("temporal graph contains no traversable edge")
    return TemporalGraphIndex(n_nodes=n, active_nodes=active_nodes, T=T, W=W, nbr_times=nbr_times,
                              nbr_other=nbr_other, coll_adj=coll,
                              coll_deg=deg, edge_times=edge_times)


# ----------------------------------------------------------------------------
# Walks
# ----------------------------------------------------------------------------

def run_walk(idx: TemporalGraphIndex, strategy: str, max_budget: int,
             seed: int, decay_scale: float = None,
             history_k: int = 20) -> pd.DataFrame:
    """One walk process under the given strategy until the budget is spent.
    Returns the log as a DataFrame with one row per budget unit."""
    rng = np.random.default_rng(seed)
    rng_aux = np.random.default_rng(seed ^ 0x9E3779B9)  # only for _t time sampling
    if decay_scale is None:
        decay_scale = idx.T / 10.0
    valid = {"time_agnostic", "time_agnostic_t", "time_respecting",
             "recency_biased", "recent_history"}
    if strategy not in valid:
        raise ValueError(f"unknown walk strategy {strategy!r}; choose from {sorted(valid)}")
    if history_k < 1:
        raise ValueError("history_k must be >= 1")
    forward = strategy in ("time_respecting", "recency_biased")
    backward = strategy == "recent_history"
    temporal = forward or backward
    record_time = temporal or strategy == "time_agnostic_t"

    kinds = np.empty(max_budget, np.int8)
    nodes = np.empty(max_budget, np.int64)
    us = np.full(max_budget, -1, np.int64)
    vs = np.full(max_budget, -1, np.int64)
    ts = np.full(max_budget, np.nan)
    dts = np.full(max_budget, np.nan)

    def place(i):
        # Walk start. CTDNE (Nguyen et al. 2018) samples the initial edge from a
        # global time-sorted distribution; under walk-limited access we instead
        # start from a random node at a uniform time. Access-faithful deviation.
        x = int(idx.active_nodes[rng.integers(len(idx.active_nodes))])
        if forward:
            tau = float(rng.uniform(0.0, idx.T))
        elif backward:
            # Query from "now" and move strictly backwards.  Adding a tiny
            # epsilon includes an event exactly at T in the first candidate set.
            tau = float(idx.T + np.finfo(float).eps * max(1.0, idx.T))
        else:
            tau = 0.0
        kinds[i] = 0; nodes[i] = x
        return x, tau

    i = 0
    x, tau = place(i); i += 1
    while i < max_budget:
        if forward:
            times = idx.nbr_times[x]
            j0 = int(np.searchsorted(times, tau, side="right"))
            if j0 >= len(times):                      # temporal dead end
                # Saramaki & Holme (2015) greedy walks terminate here; we
                # restart so every strategy spends an identical budget.
                x, tau = place(i); i += 1
                continue
            if strategy == "time_respecting":
                # CTDNE time-respecting step: next event strictly later than the
                # arrival time tau (Nguyen et al. 2018 use strictly increasing
                # times); searchsorted(..., "right") already enforces t > tau.
                j = int(rng.integers(j0, len(times)))
            else:                                      # recency_biased
                # CTDNE Eq. (6) as printed uses exp(+(t - tau)), which *grows*
                # with the waiting time and contradicts their stated preference
                # for a small in-between time; we use exp(-(t - tau)/scale),
                # matching that intent (sign-corrected).
                d = times[j0:] - tau
                # Subtract the minimum for numerical stability; it cancels in
                # the normalized probabilities.
                w = np.exp(-(d - d.min()) / decay_scale)
                w /= w.sum()
                j = j0 + int(rng.choice(len(w), p=w))
            y = int(idx.nbr_other[x][j]); te = float(times[j])
            a, b = (x, y) if x < y else (y, x)
            kinds[i] = 1; nodes[i] = y; us[i] = a; vs[i] = b
            ts[i] = te; dts[i] = te - tau
            x, tau = y, te
            i += 1
        elif backward:
            times = idx.nbr_times[x]
            j1 = int(np.searchsorted(times, tau, side="left"))
            if j1 <= 0:
                x, tau = place(i); i += 1
                continue
            j0 = max(0, j1 - history_k)
            j = int(rng.integers(j0, j1))
            y = int(idx.nbr_other[x][j]); te = float(times[j])
            a, b = (x, y) if x < y else (y, x)
            kinds[i] = 1; nodes[i] = y; us[i] = a; vs[i] = b
            ts[i] = te; dts[i] = tau - te
            x, tau = y, te
            i += 1
        else:
            nbrs = idx.coll_adj[x]
            y = int(nbrs[rng.integers(len(nbrs))])
            a, b = (x, y) if x < y else (y, x)
            kinds[i] = 1; nodes[i] = y; us[i] = a; vs[i] = b
            if record_time:                            # time_agnostic_t baseline
                et = idx.edge_times[(a, b)]
                ts[i] = float(et[rng_aux.integers(len(et))])
            x = y
            i += 1
    return pd.DataFrame({"kind": kinds, "node": nodes, "u": us, "v": vs,
                         "t": ts, "dt": dts})


# ----------------------------------------------------------------------------
# Summaries
# ----------------------------------------------------------------------------

def summarize(log: pd.DataFrame, idx: TemporalGraphIndex, budget: int) -> dict:
    """Feature vector from the first `budget` log entries."""
    L = log.iloc[:budget]
    steps = L[L["kind"] == 1]
    n_steps = len(steps)
    feats = {
        "budget": budget,
        "n_observed_events": n_steps,
        "n_restarts": int((L["kind"] == 0).sum()),
    }
    if n_steps == 0:
        return feats

    edges = list(zip(steps["u"].to_numpy(), steps["v"].to_numpy()))
    uniq_edges = set(edges)
    visited = set(L["node"].to_numpy().tolist())
    observed_degree = {}
    for a, b in uniq_edges:
        observed_degree[a] = observed_degree.get(a, 0) + 1
        observed_degree[b] = observed_degree.get(b, 0) + 1
    observed_degree_values = list(observed_degree.values())
    feats.update({
        "unique_nodes": len(visited),
        "unique_edges": len(uniq_edges),
        "edge_revisit_rate": 1.0 - len(uniq_edges) / n_steps,
        # Only degrees in the observed walk subgraph are legal partial-access
        # features.  The old implementation read idx.coll_deg, i.e. the true
        # full-graph degree, which is retained nowhere in the new benchmark.
        "observed_deg_mean": float(np.mean(observed_degree_values)),
        "observed_deg_max": float(np.max(observed_degree_values)),
    })

    # discovery slope: new-edge rate in first vs second half of the steps
    half = n_steps // 2
    if half >= 1:
        first, second = set(edges[:half]), set()
        seen = set(edges[:half])
        new_second = 0
        for e in edges[half:]:
            if e not in seen:
                new_second += 1
                seen.add(e)
        feats["discovery_rate_first_half"] = len(first) / half
        feats["discovery_rate_second_half"] = new_second / max(1, n_steps - half)

    # return times in steps (edges observed more than once)
    last_pos, gaps = {}, []
    for pos, e in enumerate(edges):
        if e in last_pos:
            gaps.append(pos - last_pos[e])
        last_pos[e] = pos
    feats["return_time_steps_mean"] = float(np.mean(gaps)) if gaps else np.nan

    # temporal features (only if timestamps were observed)
    tvals = steps["t"].to_numpy()
    if np.isfinite(tvals).any():
        win = idx.T / idx.W
        widx = window_index(tvals, 0.0, win, idx.W)   # shared convention
        ew = {}
        obs_count = {}
        for e, w in zip(edges, widx):
            ew.setdefault(e, set()).add(int(w))
            obs_count[e] = obs_count.get(e, 0) + 1
        multi = [e for e, ws in ew.items() if len(ws) >= 2]
        feats["walk_rho_plugin"] = len(multi) / len(ew)          # the analog of rho
        # bias-corrected variant: only edges with >= 2 observations had a
        # chance to reveal a cross-window revisit at all
        ge2 = [e for e, c in obs_count.items() if c >= 2]
        feats["share_edges_multi_observed"] = len(ge2) / len(ew)
        # walk_rho_conditional conditions on edges seen >= 2x (a recurrence can
        # only show up on a revisit; collision principle of Hardiman-Katzir 2013
        # / Ben-Hamou 2018). CAVEAT: no inverse-degree reweighting, so it is
        # heuristic conditioning, not a provably unbiased estimator.
        feats["walk_rho_conditional"] = (
            len([e for e in ge2 if len(ew[e]) >= 2]) / len(ge2) if ge2 else np.nan
        )
        feats["mean_windows_per_observed_edge"] = float(np.mean([len(w) for w in ew.values()]))
        spans = [max(ws) - min(ws) for ws in ew.values() if len(ws) >= 2]
        feats["mean_window_gap_revisits"] = float(np.mean(spans)) if spans else np.nan
        feats["windows_covered_frac"] = len(set(widx.tolist())) / idx.W
        dt = steps["dt"].to_numpy()
        dt = dt[np.isfinite(dt)]
        if len(dt):
            feats["step_dt_mean"] = float(np.mean(dt))
            feats["step_dt_median"] = float(np.median(dt))
    return feats


def summaries_at_checkpoints(log: pd.DataFrame, idx: TemporalGraphIndex,
                             checkpoints: list) -> list:
    return [summarize(log, idx, b) for b in checkpoints if b <= len(log)]
