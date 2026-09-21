#!/usr/bin/env python3
"""Literature-grounded event-stream generators for the estimator benchmark.

The module deliberately separates three roles:

* empirical streams: unchanged real observations;
* microcanonical P[w,t] timestamp shuffles and controlled timing twins built
  on real substrates;
* mechanistic models: DAR(1) dynamics on a DCSBM/LFR candidate topology and
  an activity-driven-with-memory event process.

Every public function returns the same ``u,v,t`` event-list interface consumed
by walks.py.  Snapshot models therefore declare their event-observation layer
explicitly instead of silently treating an active snapshot as one event.
"""

from collections import defaultdict
from typing import Optional

import networkx as nx
import numpy as np
import pandas as pd

import census
from generator import Family


def normalize_event_stream(raw: pd.DataFrame, T: float = 1.0) -> pd.DataFrame:
    """Convert arbitrary endpoint ids to dense undirected ids and t in [0,T]."""
    need = {"u", "v", "t"}
    if not need.issubset(raw.columns):
        raise ValueError(f"event table needs columns {sorted(need)}")
    d = raw[["u", "v", "t"]].copy()
    d["t"] = pd.to_numeric(d["t"], errors="coerce")
    d = d.dropna(subset=["t"])
    d = d[d["u"].astype(str) != d["v"].astype(str)].copy()
    if d.empty:
        raise ValueError("event stream is empty after dropping invalid/self events")
    both = pd.concat([d["u"].astype(str), d["v"].astype(str)], ignore_index=True)
    codes, _ = pd.factorize(both, sort=True)
    m = len(d)
    u, v = codes[:m].astype(np.int64), codes[m:].astype(np.int64)
    d["u"] = np.minimum(u, v)
    d["v"] = np.maximum(u, v)
    t = d["t"].to_numpy(dtype=float)
    lo, hi = float(t.min()), float(t.max())
    if hi <= lo:
        raise ValueError("event stream has a degenerate time horizon")
    d["t"] = (t - lo) / (hi - lo) * T
    return d.sort_values("t", kind="mergesort").reset_index(drop=True)


def truth_for_events(events: pd.DataFrame, label: str, W: int = 5) -> dict:
    """Ground-truth targets, using census.py as the single definition source."""
    if W != census.HEADLINE_W:
        raise ValueError("current census target columns are defined for W=5")
    raw = events.astype({"u": str, "v": str})
    row = census.census_row(census.normalize(raw), label=label)
    keep = [
        "n_pairs", "n_events", "rho_headline", "rho_event_weighted",
        "mean_span_frac", "C_one_step", "lifetime_mean_over_T",
        "burstiness_pooled", "share_single_event_pairs",
    ] + [f"rho_W{W}_k{k}" for k in range(1, W + 1)]
    return {k: row[k] for k in keep}


def timestamp_shuffle(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Uniform P[w,t] timestamp shuffle.

    The collapsed topology, each edge's event count, and the exact global
    timestamp multiset are invariant.  Permuting timestamps over labelled event
    slots samples uniformly; because every edge has a fixed number of slots,
    quotienting identical slots introduces the same factorial constant for all
    admissible edge assignments.
    """
    rng = np.random.default_rng(seed)
    out = events[["u", "v", "t"]].copy()
    out["t"] = rng.permutation(out["t"].to_numpy(dtype=float))
    return out.sort_values("t", kind="mergesort").reset_index(drop=True)



def within_window_timestamp_shuffle(events: pd.DataFrame, seed: int,
                                    W: int = 5, T: float = 1.0) -> pd.DataFrame:
    """Shuffle timestamps only among event slots in the same analysis window.

    This preserves every edge's event count *and* its exact five-window mask,
    while destroying sub-window ordering/burst structure.  It is therefore a
    stricter reference than the global P[w,t] shuffle.
    """
    rng = np.random.default_rng(seed)
    out = events[["u", "v", "t"]].copy()
    wi = np.minimum((out["t"].to_numpy(float) / (T / W)).astype(int), W - 1)
    tt = out["t"].to_numpy(float).copy()
    for w in range(W):
        idx = np.flatnonzero(wi == w)
        if len(idx) > 1:
            tt[idx] = rng.permutation(tt[idx])
    out["t"] = tt
    return out.sort_values("t", kind="mergesort").reset_index(drop=True)


def lifetime_resample(events: pd.DataFrame, seed: int, T: float = 1.0) -> pd.DataFrame:
    """Resample each edge's internal event times inside its observed lifetime.

    Endpoints, per-edge counts, and each edge's first/last timestamp are kept.
    Intermediate events are uniform within that lifetime.  Singletons are
    unchanged.  This isolates lifetime support from within-lifetime timing.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for (u, v), g in events.groupby(["u", "v"], sort=True):
        t = np.sort(g["t"].to_numpy(float))
        if len(t) > 2 and t[-1] > t[0]:
            mid = np.sort(rng.uniform(t[0], t[-1], len(t) - 2))
            t = np.concatenate([[t[0]], mid, [t[-1]]])
        rows.extend((int(u), int(v), float(x)) for x in t)
    return pd.DataFrame(rows, columns=["u", "v", "t"]).sort_values(
        "t", kind="mergesort").reset_index(drop=True)


def edge_rewire_surrogate(events: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Degree-preserving endpoint rewiring with per-edge time series intact.

    A double-edge-swap rewires the collapsed graph.  Complete timestamp series
    are then assigned one-to-one to the rewired edges, preserving the multiset
    of edge-level temporal profiles and all global timestamps while changing
    their topological placement.
    """
    series = []
    for edge, g in events.groupby(["u", "v"], sort=True):
        series.append((tuple(map(int, edge)), np.sort(g["t"].to_numpy(float))))
    g = nx.Graph()
    g.add_edges_from(edge for edge, _ in series)
    if g.number_of_edges() >= 4:
        try:
            nx.double_edge_swap(g, nswap=max(1, 3 * g.number_of_edges()),
                                max_tries=max(100, 60 * g.number_of_edges()),
                                seed=int(seed))
        except (nx.NetworkXAlgorithmError, nx.NetworkXError):
            # Some tiny/star-like graphs have no valid swap.  Keeping the graph
            # unchanged is preferable to silently breaking degree invariants.
            pass
    new_edges = sorted((min(int(u), int(v)), max(int(u), int(v))) for u, v in g.edges())
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(series))
    rows = []
    for new_edge, old_i in zip(new_edges, perm):
        for t in series[int(old_i)][1]:
            rows.append((new_edge[0], new_edge[1], float(t)))
    return pd.DataFrame(rows, columns=["u", "v", "t"]).sort_values(
        "t", kind="mergesort").reset_index(drop=True)


def temporal_chunks(events: pd.DataFrame, n_chunks: int,
                    min_events: int = 500) -> list[pd.DataFrame]:
    """Split a stream into disjoint equal-duration panels and renormalize each.

    Empty/tiny panels are skipped.  Downstream grouping must retain the parent
    source id, so chunks increase within-source replication without leakage.
    """
    if n_chunks < 2:
        return []
    out = []
    t = events["t"].to_numpy(float)
    cuts = np.linspace(float(t.min()), float(t.max()), n_chunks + 1)
    for i in range(n_chunks):
        left, right = cuts[i], cuts[i + 1]
        mask = (t >= left) & ((t <= right) if i == n_chunks - 1 else (t < right))
        d = events.loc[mask, ["u", "v", "t"]].copy()
        if len(d) < min_events or d["t"].nunique() < 2:
            continue
        out.append(normalize_event_stream(d))
    return out


def family_from_events(events: pd.DataFrame, name: str, W: int = 5,
                       T: float = 1.0) -> Family:
    """Turn a real event stream into the invariant part of a timing twin."""
    e = events[["u", "v", "t"]].copy()
    a = np.minimum(e["u"].to_numpy(np.int64), e["v"].to_numpy(np.int64))
    b = np.maximum(e["u"].to_numpy(np.int64), e["v"].to_numpy(np.int64))
    counts = (pd.DataFrame({"u": a, "v": b})
              .value_counts(sort=False).rename("m").reset_index()
              .sort_values(["u", "v"], kind="mergesort"))
    edges = counts[["u", "v"]].to_numpy(np.int64)
    m_e = counts["m"].to_numpy(np.int64)
    n = int(edges.max()) + 1
    deg = np.zeros(n, dtype=np.int64)
    for u, v in edges:
        deg[u] += 1; deg[v] += 1
    return Family(name=name, kind="real_substrate", edges=edges, m_e=m_e,
                  t_all=np.sort(e["t"].to_numpy(float)), T=T, W=W, deg=deg,
                  timestamps="empirical")


def dar_event_stream(graph: nx.Graph, alpha: float, chi: float, seed: int,
                     W: int = 5, event_rate: float = 1.0,
                     T: float = 1.0, alpha_concentration: Optional[float] = None,
                     alpha_within: Optional[float] = None,
                     alpha_between: Optional[float] = None) -> pd.DataFrame:
    """DAR(1) snapshots plus an explicit active-edge event layer.

    For each candidate edge e and window t,

        A[e,t] = V[e,t] A[e,t-1] + (1-V[e,t]) Y[e,t],
        V ~ Bernoulli(alpha), Y ~ Bernoulli(chi).

    Every active edge/window emits ``1 + Poisson(event_rate)`` timestamped
    events uniformly inside that window.  Thus ``alpha`` controls link-state
    copying while ``chi`` independently controls marginal occupancy.
    """
    if not 0 <= alpha <= 1 or not 0 <= chi <= 1:
        raise ValueError("alpha and chi must lie in [0,1]")
    if W < 2 or event_rate < 0:
        raise ValueError("W must be >=2 and event_rate non-negative")
    rng = np.random.default_rng(seed)
    g = nx.convert_node_labels_to_integers(nx.Graph(graph))
    g.remove_edges_from(nx.selfloop_edges(g))
    edges = np.array([(min(u, v), max(u, v)) for u, v in g.edges()], dtype=np.int64)
    if len(edges) == 0:
        raise ValueError("DAR candidate graph has no edges")
    if alpha_within is not None or alpha_between is not None:
        if alpha_within is None or alpha_between is None:
            raise ValueError("alpha_within and alpha_between must be supplied together")
        blocks = nx.get_node_attributes(g, "block")
        if not blocks:
            # LFR stores community sets; canonicalize them to stable integer ids.
            communities = nx.get_node_attributes(g, "community")
            labels = {}
            lut = {}
            for node, comm in communities.items():
                key = tuple(sorted(map(int, comm))) if isinstance(comm, (set, frozenset)) else str(comm)
                lut.setdefault(key, len(lut))
                labels[int(node)] = lut[key]
            blocks = labels
        if not blocks:
            raise ValueError("community-correlated DAR requires node block/community labels")
        alpha_vec = np.array([alpha_within if blocks.get(int(u)) == blocks.get(int(v))
                              else alpha_between for u, v in edges], dtype=float)
    elif alpha_concentration is not None:
        c = float(alpha_concentration)
        if c <= 0 or not 0 < alpha < 1:
            raise ValueError("heterogeneous alpha needs 0<alpha<1 and concentration>0")
        alpha_vec = rng.beta(alpha * c, (1.0 - alpha) * c, size=len(edges))
    else:
        alpha_vec = np.full(len(edges), float(alpha), dtype=float)
    if np.any((alpha_vec < 0) | (alpha_vec > 1)):
        raise ValueError("edge-level alpha values must lie in [0,1]")
    state = rng.random(len(edges)) < chi
    us, vs, ts = [], [], []
    width = T / W
    for w in range(W):
        if w > 0:
            copy = rng.random(len(edges)) < alpha_vec
            fresh = rng.random(len(edges)) < chi
            state = np.where(copy, state, fresh)
        active = np.flatnonzero(state)
        counts = 1 + rng.poisson(event_rate, size=len(active))
        for ei, c in zip(active, counts):
            u, v = edges[ei]
            times = (w + rng.random(int(c))) * width
            us.extend([int(u)] * int(c)); vs.extend([int(v)] * int(c))
            ts.extend(times.tolist())
    if not ts:
        # Extremely sparse parameter combinations should not crash a large
        # grid.  Activate one random candidate in one random window, while the
        # manifest records that the realized stream is degenerate/small.
        u, v = edges[int(rng.integers(len(edges)))]
        us, vs, ts = [int(u)], [int(v)], [float(rng.uniform(0, T))]
    out = pd.DataFrame({"u": us, "v": vs, "t": ts})
    # census requires a non-degenerate horizon.  A second event is practically
    # always present, but add one deterministically for a one-event corner case.
    if len(out) == 1:
        extra = out.iloc[0].copy(); extra["t"] = min(T, float(extra["t"]) + T / (10 * W))
        out = pd.concat([out, extra.to_frame().T], ignore_index=True)
    return out.astype({"u": "int64", "v": "int64", "t": "float64"}).sort_values(
        "t", kind="mergesort").reset_index(drop=True)



def renewal_event_stream(graph: nx.Graph, lifetime_mean_windows: float,
                         iet_shape: float, seed: int, W: int = 5,
                         events_per_active_window: float = 1.0,
                         T: float = 1.0) -> pd.DataFrame:
    """Edge-lifetime process with Pareto renewal events inside the lifetime.

    ``lifetime_mean_windows`` controls how many consecutive windows an edge is
    alive; ``iet_shape`` controls burstiness of extra inter-event gaps.  One
    anchor event per alive window guarantees that lifetime and burstiness are
    separately manipulable rather than one accidentally erasing the other.
    """
    if lifetime_mean_windows <= 1 or iet_shape <= 1:
        raise ValueError("lifetime_mean_windows and iet_shape must exceed 1")
    rng = np.random.default_rng(seed)
    g = nx.convert_node_labels_to_integers(nx.Graph(graph))
    g.remove_edges_from(nx.selfloop_edges(g))
    width = T / W
    rows = []
    p = min(1.0, 1.0 / float(lifetime_mean_windows))
    for u, v in g.edges():
        span = min(W, int(rng.geometric(p)))
        start = int(rng.integers(0, W - span + 1))
        # Guaranteed one event per active window.
        times = [(start + j + rng.random()) * width for j in range(span)]
        extra_n = int(rng.poisson(max(0.0, events_per_active_window) * span))
        if extra_n:
            gaps = 1.0 + rng.pareto(float(iet_shape), size=extra_n)
            c = np.cumsum(gaps)
            lo, hi = start * width, (start + span) * width
            extra = lo + (c / (c[-1] + 1e-12)) * (hi - lo) * (1.0 - 1e-9)
            times.extend(extra.tolist())
        a, b = (int(u), int(v)) if u < v else (int(v), int(u))
        rows.extend((a, b, float(t)) for t in times)
    out = pd.DataFrame(rows, columns=["u", "v", "t"])
    if len(out) < 2:
        raise ValueError("renewal generator produced too few events")
    return out.sort_values("t", kind="mergesort").reset_index(drop=True)


def activity_memory_event_stream(n: int, seed: int, W: int = 5,
                                 slots_per_window: int = 40,
                                 mean_activity: float = 0.04,
                                 m: int = 1,
                                 memory_beta: Optional[float] = 1.0,
                                 memory_c: float = 1.0,
                                 T: float = 1.0) -> pd.DataFrame:
    """Activity-driven temporal network with optional tie reinforcement.

    Node activities are heavy-tailed.  In the memoryless mode a partner is
    uniform over all other nodes.  In the reinforced mode, a node with n_i old
    partners explores a new partner with

        p_new(n_i) = (1 + n_i / c)^(-beta),

    otherwise reusing an old partner proportional to past interaction count.
    This is the standard exploration/reinforcement form used in
    activity-driven-with-memory models.  It produces events directly and is a
    mechanism-shift check against the edge-state DAR family.
    """
    if n < 3 or slots_per_window < 2 or m < 1:
        raise ValueError("invalid activity-driven dimensions")
    if memory_beta is not None and memory_beta < 0:
        raise ValueError("memory_beta must be non-negative or None")
    rng = np.random.default_rng(seed)
    raw = 1.0 + rng.pareto(2.5, size=n)
    activity = raw / raw.mean() * mean_activity
    activity = np.clip(activity, 1e-4, 0.35)
    partners = [set() for _ in range(n)]
    strength = [defaultdict(int) for _ in range(n)]
    us, vs, ts = [], [], []
    n_slots = W * slots_per_window

    def random_other(i, exclude_old=False):
        if exclude_old and len(partners[i]) < n - 1:
            # Rejection is efficient until the graph is nearly saturated; in
            # that rare case the bounded fallback enumerates candidates.
            for _ in range(20):
                j = int(rng.integers(n - 1)); j += int(j >= i)
                if j not in partners[i]:
                    return j
            candidates = [j for j in range(n) if j != i and j not in partners[i]]
            if candidates:
                return int(rng.choice(candidates))
        j = int(rng.integers(n - 1))
        return j + int(j >= i)

    for slot in range(n_slots):
        active = np.flatnonzero(rng.random(n) < activity)
        for i in active:
            for _ in range(m):
                if memory_beta is None or not partners[i]:
                    j = random_other(int(i), exclude_old=False)
                else:
                    p_new = (1.0 + len(partners[i]) / memory_c) ** (-memory_beta)
                    if rng.random() < p_new and len(partners[i]) < n - 1:
                        j = random_other(int(i), exclude_old=True)
                    else:
                        old = np.array(sorted(partners[i]), dtype=np.int64)
                        weights = np.array([strength[i][int(x)] for x in old], dtype=float)
                        j = int(rng.choice(old, p=weights / weights.sum()))
                a, b = (int(i), j) if i < j else (j, int(i))
                partners[i].add(j); partners[j].add(int(i))
                strength[i][j] += 1; strength[j][int(i)] += 1
                us.append(a); vs.append(b)
                ts.append((slot + float(rng.random())) / n_slots * T)
    if len(ts) < 2:
        raise RuntimeError("activity-driven stream produced fewer than two events")
    return pd.DataFrame({"u": us, "v": vs, "t": ts}).sort_values(
        "t", kind="mergesort").reset_index(drop=True)
