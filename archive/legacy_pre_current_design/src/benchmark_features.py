#!/usr/bin/env python3
"""Observable-only feature extraction for benchmark walk prefixes.

Feature prefixes define the input ablations used by evaluate_benchmark.py:

``occ__``
    count-only occupancy information derived from exact per-edge (n,w);
``pat__``
    position-aware temporal masks, first/last activity and inter-event gaps;
``crawl__``
    walk/retrieval diagnostics such as restarts, discovery and revisits.

No feature reads the true edge count, true coverage, full degrees, generator
parameters, or unobserved neighbours.  Those quantities may appear as metadata
in a case row for stratified evaluation, but never carry one of these prefixes.
"""

from collections import Counter, defaultdict
from functools import lru_cache
import json

import numpy as np
import pandas as pd

from census import window_index
from corrected_estimator import rho_mle
from mask_estimator import mask_mle


def _q(values, probs=(0.25, 0.5, 0.75, 0.9)):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return [float("nan")] * len(probs)
    return [float(x) for x in np.quantile(a, probs)]


def edge_observations(log: pd.DataFrame, budget: int, W: int = 5,
                      T: float = 1.0):
    """Return per-edge n, w, bit mask and sampled timestamps for a prefix."""
    s = log.iloc[:budget]
    s = s[(s["kind"] == 1) & np.isfinite(s["t"].to_numpy(dtype=float))]
    if s.empty:
        return []
    wi = window_index(s["t"].to_numpy(float), 0.0, T / W, W)
    rec = {}
    for a, b, t, w in zip(s["u"], s["v"], s["t"], wi):
        edge = (int(a), int(b))
        d = rec.setdefault(edge, {"n": 0, "mask": 0, "times": []})
        d["n"] += 1
        d["mask"] |= 1 << int(w)
        d["times"].append(float(t))
    out = []
    for edge in sorted(rec):
        d = rec[edge]
        d["times"].sort()
        d["w"] = int(d["mask"]).bit_count()
        d["edge"] = edge
        out.append(d)
    return out


def sparse_histograms(obs):
    """Exact, compact JSON forms retained for the later LLM stage."""
    nw = Counter((int(d["n"]), int(d["w"])) for d in obs)
    nm = Counter((int(d["n"]), int(d["mask"])) for d in obs)
    nw_json = json.dumps({f"{n},{w}": c for (n, w), c in sorted(nw.items())},
                         separators=(",", ":"))
    nm_json = json.dumps({f"{n},{m:02x}": c for (n, m), c in sorted(nm.items())},
                         separators=(",", ":"))
    return nw_json, nm_json


def window_count_histogram(obs, W: int = 5) -> str:
    """Exact frequency-of-frequencies for per-dyad window count vectors."""
    hist = Counter()
    for d in obs:
        counts = np.zeros(W, dtype=np.int64)
        if d["times"]:
            wi = window_index(np.asarray(d["times"], dtype=float),
                              0.0, 1.0 / W, W)
            counts += np.bincount(wi, minlength=W)[:W]
        hist[tuple(map(int, counts))] += 1
    return json.dumps({",".join(map(str, key)): int(value)
                       for key, value in sorted(hist.items())},
                      separators=(",", ":"))


def window_count_features(obs, W: int = 5, ncap: int = 5):
    """Fixed prompt-derivable summaries of exact window-count vectors."""
    out = {}
    denom = max(1, len(obs))
    matrix = np.zeros((len(obs), W), dtype=float)
    for i, d in enumerate(obs):
        wi = window_index(np.asarray(d["times"], dtype=float),
                          0.0, 1.0 / W, W)
        matrix[i] = np.bincount(wi, minlength=W)[:W]
    for w in range(W):
        for count in range(ncap + 1):
            label = f"{ncap}p" if count == ncap else str(count)
            if len(matrix):
                value = np.mean(np.minimum(matrix[:, w], ncap) == count)
            else:
                value = 0.0
            out[f"wcnt__w{w}_n{label}"] = float(value)
        values = matrix[:, w] if len(matrix) else np.array([], dtype=float)
        out[f"wcnt__w{w}_mean"] = float(values.mean()) if len(values) else 0.0
    for a in range(W):
        for b in range(a + 1, W):
            if len(matrix) >= 2 and matrix[:, a].std() > 0 and matrix[:, b].std() > 0:
                corr = float(np.corrcoef(matrix[:, a], matrix[:, b])[0, 1])
            else:
                corr = float("nan")
            out[f"wcnt__corr_w{a}_w{b}"] = corr
            out[f"wcnt__both_positive_w{a}_w{b}"] = (
                float(np.mean((matrix[:, a] > 0) & (matrix[:, b] > 0)))
                if len(matrix) else 0.0)
    out["wcnt__log_observed_dyads"] = float(np.log1p(len(obs)))
    out["wcnt__normalization_denom"] = float(denom)
    return out



def recent_events_json(log: pd.DataFrame, budget: int, limit: int = 100) -> str:
    """Last observed timed tuples with node ids anonymized by first appearance."""
    s = log.iloc[:budget]
    s = s[(s["kind"] == 1) & np.isfinite(s["t"].to_numpy(float))].tail(limit)
    mapping = {}
    next_id = 0
    rows = []
    for u, v, t in zip(s["u"], s["v"], s["t"]):
        pair = []
        for x in (int(u), int(v)):
            if x not in mapping:
                mapping[x] = next_id
                next_id += 1
            pair.append(mapping[x])
        rows.append([pair[0], pair[1], round(float(t), 8)])
    return json.dumps(rows, separators=(",", ":"))


def _rank_auc(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    pos = y == 1; neg = y == 0
    if not pos.any() or not neg.any():
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy(float)
    n1, n0 = int(pos.sum()), int(neg.sum())
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def edgebank_diagnostics(obs, idx, W: int = 5):
    """Frequency/recency AUC against true edge persistence for observed edges.

    These ``diag__`` columns are analysis metadata, never learned-model inputs.
    """
    if idx is None or not obs:
        return {"diag__edgebank_frequency_auc": float("nan"),
                "diag__edgebank_recency_auc": float("nan")}
    labels, freq, rec = [], [], []
    for d in obs:
        true_t = np.asarray(idx.edge_times.get(tuple(d["edge"]), []), dtype=float)
        if not len(true_t):
            continue
        wi = window_index(true_t, 0.0, idx.T / W, W)
        labels.append(int(len(np.unique(wi)) >= 2))
        freq.append(float(d["n"]))
        rec.append(float(max(d["times"])))
    return {"diag__edgebank_frequency_auc": _rank_auc(labels, freq),
            "diag__edgebank_recency_auc": _rank_auc(labels, rec)}


def oracle_label_diagnostics(obs, idx, W: int = 5):
    """Truth-label decomposition for simulation diagnostics only.

    ``seen`` replaces the inferred label but leaves unique-dyad selection bias.
    ``hh`` additionally weights by traversal multiplicity.  The latter has a
    design-based interpretation only for stationary ``time_agnostic_t`` and is
    therefore gated by the evaluator, not exposed as an observable feature.
    """
    out = {}
    for k in range(2, W + 1):
        out[f"oracle__seen_label_rho_k{k}"] = float("nan")
        out[f"oracle__hh_label_rho_k{k}"] = float("nan")
    if idx is None or not obs:
        return out
    true_k, weights = [], []
    for d in obs:
        times = np.asarray(idx.edge_times.get(tuple(d["edge"]), []), dtype=float)
        if not len(times):
            continue
        wi = window_index(times, 0.0, idx.T / W, W)
        true_k.append(len(np.unique(wi)))
        weights.append(int(d["n"]))
    true_k = np.asarray(true_k, dtype=int)
    weights = np.asarray(weights, dtype=float)
    if not len(true_k):
        return out
    for k in range(2, W + 1):
        label = (true_k >= k).astype(float)
        out[f"oracle__seen_label_rho_k{k}"] = float(label.mean())
        out[f"oracle__hh_label_rho_k{k}"] = float(np.average(label, weights=weights))
    return out


def occupancy_features(obs, W: int = 5, ncap: int = 20):
    f = {}
    E = len(obs)
    denom = max(1, E)
    # Fixed ML representation.  The exact uncapped histogram remains in JSON,
    # and analytical MLEs below receive the exact n values.
    for n in range(1, ncap + 1):
        nlab = f"{ncap}p" if n == ncap else str(n)
        for w in range(1, W + 1):
            f[f"occ__n{nlab}_w{w}"] = 0.0
    for d in obs:
        nbin = min(int(d["n"]), ncap)
        nlab = f"{ncap}p" if nbin == ncap else str(nbin)
        f[f"occ__n{nlab}_w{int(d['w'])}"] += 1.0 / denom
    ns = [d["n"] for d in obs]
    f["occ__log_unique_edges"] = float(np.log1p(E))
    f["occ__observations_per_edge"] = float(np.mean(ns)) if ns else 0.0
    q = _q(ns)
    for lab, val in zip(("q25", "q50", "q75", "q90"), q):
        f[f"occ__n_{lab}"] = val
    return f


def pattern_features(obs, W: int = 5, ncap: int = 10):
    f = {}
    E = len(obs)
    denom = max(1, E)
    max_mask = (1 << W) - 1
    for m in range(1, max_mask + 1):
        f[f"pat__mask_{m:02x}"] = 0.0
    for n in range(1, ncap + 1):
        nlab = f"{ncap}p" if n == ncap else str(n)
        for m in range(1, max_mask + 1):
            f[f"pat__n{nlab}_mask{m:02x}"] = 0.0
    first_counts = np.zeros(W, dtype=float)
    last_counts = np.zeros(W, dtype=float)
    event_window_counts = np.zeros(W, dtype=float)
    gaps, lifetimes, widths = [], [], []
    adjacent_num = adjacent_den = 0
    noncontiguous = 0
    for d in obs:
        m = int(d["mask"]); nbin = min(int(d["n"]), ncap)
        nlab = f"{ncap}p" if nbin == ncap else str(nbin)
        f[f"pat__mask_{m:02x}"] += 1.0 / denom
        f[f"pat__n{nlab}_mask{m:02x}"] += 1.0 / denom
        active = [w for w in range(W) if m & (1 << w)]
        first_counts[active[0]] += 1; last_counts[active[-1]] += 1
        for t in d["times"]:
            w = int(window_index(np.array([t]), 0.0, 1.0 / W, W)[0])
            event_window_counts[w] += 1
        widths.append(active[-1] - active[0])
        if len(active) > 1 and active[-1] - active[0] + 1 > len(active):
            noncontiguous += 1
        for w in range(W - 1):
            if m & (1 << w):
                adjacent_den += 1
                adjacent_num += int(bool(m & (1 << (w + 1))))
        times = np.asarray(d["times"], dtype=float)
        if len(times) >= 2:
            gaps.extend(np.diff(times).tolist())
            lifetimes.append(float(times[-1] - times[0]))
    for w in range(W):
        f[f"pat__first_w{w}"] = float(first_counts[w] / denom)
        f[f"pat__last_w{w}"] = float(last_counts[w] / denom)
        f[f"pat__event_share_w{w}"] = float(
            event_window_counts[w] / max(1.0, event_window_counts.sum()))
    f["pat__adjacent_observed_C"] = (
        float(adjacent_num / adjacent_den) if adjacent_den else float("nan"))
    f["pat__noncontiguous_edge_share"] = float(noncontiguous / denom)
    f["pat__mean_mask_width"] = float(np.mean(widths)) if widths else float("nan")
    for stem, values in (("iet", gaps), ("lifetime", lifetimes)):
        q25, q50, q75, q90 = _q(values)
        f[f"pat__{stem}_q25"] = q25; f[f"pat__{stem}_q50"] = q50
        f[f"pat__{stem}_q75"] = q75; f[f"pat__{stem}_q90"] = q90
        f[f"pat__{stem}_mean"] = float(np.mean(values)) if values else float("nan")
    return f


def crawl_features(log: pd.DataFrame, budget: int):
    L = log.iloc[:budget]
    steps = L[L["kind"] == 1]
    edges = [(int(a), int(b)) for a, b in zip(steps["u"], steps["v"])]
    if "event_id" in L.columns:
        # Non-walk access logs contain event records rather than a one-node
        # trajectory.  Both endpoints are observed and must contribute to
        # node-discovery diagnostics; using only canonical ``v`` would create
        # an artificial id-order bias.
        nodes = steps[["u", "v"]].to_numpy(dtype=np.int64).ravel().tolist()
    else:
        nodes = L["node"].to_numpy(dtype=np.int64).tolist()
    unique_edges = set(edges)
    n_steps = len(edges)
    f = {
        "crawl__log_budget": float(np.log1p(budget)),
        "crawl__step_fraction": float(n_steps / max(1, budget)),
        "crawl__restart_fraction": float((L["kind"] == 0).sum() / max(1, budget)),
        "crawl__log_unique_nodes": float(np.log1p(len(set(nodes)))),
        "crawl__log_unique_edges": float(np.log1p(len(unique_edges))),
        "crawl__edge_revisit_rate": float(1 - len(unique_edges) / max(1, n_steps)),
    }
    edge_hits = list(Counter(edges).values())
    node_hits = list(Counter(nodes).values())
    for stem, vals in (("edge_hits", edge_hits), ("node_hits", node_hits)):
        q25, q50, q75, q90 = _q(vals)
        f[f"crawl__{stem}_q25"] = q25; f[f"crawl__{stem}_q50"] = q50
        f[f"crawl__{stem}_q75"] = q75; f[f"crawl__{stem}_q90"] = q90

    observed_degree = Counter()
    for a, b in unique_edges:
        observed_degree[a] += 1; observed_degree[b] += 1
    deg = list(observed_degree.values())
    f["crawl__observed_degree_mean"] = float(np.mean(deg)) if deg else 0.0
    f["crawl__observed_degree_max"] = float(np.max(deg)) if deg else 0.0

    for frac in (0.10, 0.25, 0.50, 0.75, 1.00):
        stop = max(1, int(np.ceil(frac * n_steps))) if n_steps else 0
        discovered = len(set(edges[:stop])) if stop else 0
        f[f"crawl__discovery_{int(frac * 100):03d}"] = (
            float(discovered / max(1, stop)))

    last_edge, edge_gaps = {}, []
    for i, e in enumerate(edges):
        if e in last_edge:
            edge_gaps.append(i - last_edge[e])
        last_edge[e] = i
    q25, q50, q75, q90 = _q(edge_gaps)
    f["crawl__return_gap_q25"] = q25; f["crawl__return_gap_q50"] = q50
    f["crawl__return_gap_q75"] = q75; f["crawl__return_gap_q90"] = q90

    first_seen = {}
    first_collision = float("nan")
    for i, x in enumerate(nodes):
        if x in first_seen:
            first_collision = float(i)
            break
        first_seen[x] = i
    f["crawl__first_node_collision_frac"] = (
        first_collision / max(1, len(nodes)) if np.isfinite(first_collision) else 1.0)

    dt = steps["dt"].to_numpy(dtype=float)
    dt = dt[np.isfinite(dt)]
    q25, q50, q75, q90 = _q(dt)
    f["crawl__dt_q25"] = q25; f["crawl__dt_q50"] = q50
    f["crawl__dt_q75"] = q75; f["crawl__dt_q90"] = q90
    t = steps["t"].to_numpy(dtype=float)
    t = t[np.isfinite(t)]
    f["crawl__observed_time_span"] = float(t.max() - t.min()) if len(t) else float("nan")
    return f


@lru_cache(maxsize=1)
def _beta_tables():
    from bias_identifiability import _dense
    betas = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
    return betas, {b: _dense(b) for b in betas}


def _rho_at_beta_aggregated(n, w, dense, W: int = 5, ncap: int = 60,
                            iters: int = 120):
    patterns, counts = np.unique(np.column_stack([n, w]), axis=0,
                                 return_counts=True)
    nc = np.minimum(patterns[:, 0], ncap - 1).astype(int)
    wc = patterns[:, 1].astype(int)
    likelihood = dense[nc, wc, :].copy()
    big = patterns[:, 0] >= ncap
    if big.any():
        likelihood[big] = 0.0
        likelihood[np.flatnonzero(big), wc[big] - 1] = 1.0
    pi = np.ones(W) / W
    for _ in range(iters):
        weighted = likelihood * pi[None, :]
        den = weighted.sum(axis=1, keepdims=True)
        den[den <= 0] = 1.0
        new = ((weighted / den) * counts[:, None]).sum(axis=0) / counts.sum()
        if np.max(np.abs(new - pi)) < 1e-10:
            pi = new
            break
        pi = new
    return float(1.0 - pi[0])


def estimator_readouts(obs, W: int = 5, include_beta: bool = True):
    """Training-free estimator columns; exact n and masks are used."""
    out = {}
    if not obs:
        for k in range(2, W + 1):
            out[f"est__plugin_rho_k{k}"] = float("nan")
            out[f"est__occ_mle_rho_k{k}"] = float("nan")
            out[f"est__mask_mle_rho_k{k}"] = float("nan")
        out["est__plugin_C_one_step"] = float("nan")
        out["est__mask_mle_C_one_step"] = float("nan")
        out["est__plugin_mean_occupancy"] = float("nan")
        out["est__plugin_rho_event_weighted"] = float("nan")
        out["est__occ_mle_mean_occupancy"] = float("nan")
        out["est__mask_mle_mean_occupancy"] = float("nan")
        for k in range(2, W + 1):
            out[f"est__conditional_rho_k{k}"] = float("nan")
        if include_beta:
            for b in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
                out[f"est__beta_rho_b{b:.1f}"] = float("nan")
        return out

    n = np.array([d["n"] for d in obs], dtype=np.int64)
    w = np.array([d["w"] for d in obs], dtype=np.int64)
    masks = np.array([d["mask"] for d in obs], dtype=np.int64)
    for k in range(2, W + 1):
        out[f"est__plugin_rho_k{k}"] = float(np.mean(w >= k))
        eligible = n >= k
        out[f"est__conditional_rho_k{k}"] = (
            float(np.mean(w[eligible] >= k)) if eligible.any() else float("nan"))
    out["est__plugin_mean_occupancy"] = float(np.mean(w) / W)
    out["est__plugin_rho_event_weighted"] = float(n[w >= 2].sum() / max(1, n.sum()))
    _, pi = rho_mle(n, w, W=W)
    if pi is None:
        for k in range(2, W + 1):
            out[f"est__occ_mle_rho_k{k}"] = float("nan")
        out["est__occ_mle_mean_occupancy"] = float("nan")
    else:
        for k in range(2, W + 1):
            out[f"est__occ_mle_rho_k{k}"] = float(pi[k - 1:].sum())
        out["est__occ_mle_mean_occupancy"] = float(
            np.dot(np.arange(1, W + 1), pi) / W)

    mask_out, _ = mask_mle(n, masks, W=W)
    for k in range(2, W + 1):
        out[f"est__mask_mle_rho_k{k}"] = mask_out.get(f"rho_k{k}", float("nan"))
    out["est__mask_mle_mean_occupancy"] = mask_out.get(
        "mean_occupancy", float("nan"))
    out["est__mask_mle_C_one_step"] = mask_out.get("C_one_step", float("nan"))

    adjacent_num = adjacent_den = 0
    for m in masks:
        for j in range(W - 1):
            if int(m) & (1 << j):
                adjacent_den += 1
                adjacent_num += int(bool(int(m) & (1 << (j + 1))))
    out["est__plugin_C_one_step"] = (
        float(adjacent_num / adjacent_den) if adjacent_den else float("nan"))

    if include_beta:
        betas, tables = _beta_tables()
        for b in betas:
            out[f"est__beta_rho_b{b:.1f}"] = _rho_at_beta_aggregated(
                n, w, tables[b], W=W)
    return out


def build_case_features(log: pd.DataFrame, budget: int, W: int = 5,
                        T: float = 1.0, idx=None, recent_limit: int = 100,
                        include_beta: bool = True):
    obs = edge_observations(log, budget=budget, W=W, T=T)
    prefix_steps = log.iloc[:budget]
    prefix_steps = prefix_steps[prefix_steps["kind"] == 1]
    walk_edges = set(zip(prefix_steps["u"].astype(int), prefix_steps["v"].astype(int)))
    if "event_id" in log.columns:
        walk_nodes = set(prefix_steps["u"].astype(int)).union(
            prefix_steps["v"].astype(int))
    else:
        walk_nodes = set(log.iloc[:budget]["node"].astype(int).tolist())
    nw_json, nm_json = sparse_histograms(obs)
    out = {"input__nw_exact_json": nw_json,
           "input__nmask_exact_json": nm_json,
           "input__window_counts_exact_json": window_count_histogram(obs, W=W),
           "input__recent_events_json": recent_events_json(log, budget, recent_limit),
           "observed_timed_edges": len(obs),
           "observed_walk_edges": len(walk_edges),
           "observed_walk_nodes": len(walk_nodes)}
    out.update(occupancy_features(obs, W=W))
    out.update(pattern_features(obs, W=W))
    out.update(window_count_features(obs, W=W))
    out.update(crawl_features(log, budget=budget))
    out.update(estimator_readouts(obs, W=W, include_beta=include_beta))
    out.update(edgebank_diagnostics(obs, idx=idx, W=W))
    out.update(oracle_label_diagnostics(obs, idx=idx, W=W))
    return out
