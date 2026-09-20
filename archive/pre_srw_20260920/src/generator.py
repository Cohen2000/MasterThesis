#!/usr/bin/env python3
"""Timing-only twin generator for the persistence-estimation pilot.

Core idea (Definition Note v3 + thesis design):
A *family* fixes everything static and everything count-based:
  - the static graph (substrate: ER / BA / LFR / degree-corrected SBM,
    largest connected component)
  - the per-edge event counts m_e (heavy-tailed, with a single-event mass,
    calibrated to the dataset census)
  - the global timestamp multiset t_all (drawn once, sorted)
An *instance* of a family only chooses WHICH timestamps go to WHICH edge.
Hence, across instances of one family:
  - collapsed graphs are identical,
  - per-edge event counts are identical,
  - the timestamp multiset is identical,
  - only the assignment (and therefore rho) differs.
Anything that ignores time sees identical inputs -> negative control by
construction.

The family invariants {topology, per-edge counts, timestamp multiset} match the
timestamp-shuffling reference model (P[w,t] in Gauvin et al. 2022; the RP/DCW
null of Holme-Saramaki 2012 / Karsai et al. 2011). Because we *steer* rho rather
than sample uniformly, this is a biased/steered surrogate, NOT a maximum-entropy
null -- rho is the experimental knob, not a p-value against a null. The "same
topology, different timing changes the dynamics" rationale is Scholtes (2014).

rho control: a target share of edges is marked persistent (span >= 2 windows,
small spans preferred, matching the census profiles); the rest are bursty
(span = 1). Capacity bookkeeping keeps the per-window timestamp counts exactly
equal to the family's multiset. Small target deviations are possible (integer
constraints, capacity repair); the *achieved* rho is computed exactly via the
census code (single source of truth) and is what downstream experiments use as
ground truth.
"""

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import pandas as pd

from census import HEADLINE_K, HEADLINE_W, census_row, normalize, window_index

# census-calibrated defaults
P_SINGLE = 0.35          # mass of single-event edges (census bulk: 0.26-0.52)
LOGN_MU, LOGN_SIGMA = np.log(2.0), 0.9   # heavy tail for extra events
SPAN_GEOM_P = 0.6        # persistent spans mostly 2-3, decaying (census-like)


# ----------------------------------------------------------------------------
# Substrates
# ----------------------------------------------------------------------------

def make_dcsbm_graph(n: int, seed: int, average_degree: float = 12.0,
                     n_blocks: int = 4, mixing: float = 0.20,
                     theta_sigma: float = 0.8) -> nx.Graph:
    """Sparse degree-corrected planted-partition graph.

    Nodes receive balanced block labels and log-normal degree propensities
    ``theta``.  Conditional on those latent variables, dyads are independent
    with

        P(A_ij=1) = 1 - exp(-d/(n-1) * theta_i theta_j * omega[z_i,z_j]).

    ``omega`` is normalized so that ``mixing`` is the expected between-block
    edge share when propensities are homogeneous.  This is a Bernoulli DCSBM
    (rather than an ad-hoc BA graph) and supplies degree heterogeneity plus
    community structure, two ubiquitous real-network properties.  The direct
    vectorized sampler is O(n^2), so the benchmark keeps this substrate at
    moderate sizes and obtains its large-size coverage cases from real data.
    """
    if n < 8:
        raise ValueError("DCSBM needs at least 8 nodes")
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("mixing must be in [0,1]")
    rng = np.random.default_rng(seed)
    k = max(2, min(int(n_blocks), n // 4))
    z = np.arange(n, dtype=np.int64) % k
    rng.shuffle(z)
    theta = rng.lognormal(mean=-0.5 * theta_sigma ** 2,
                          sigma=theta_sigma, size=n)
    # Equalize the mean propensity per block so omega retains its mixing
    # interpretation while individual degrees remain heterogeneous.
    for b in range(k):
        theta[z == b] /= theta[z == b].mean()
    theta = np.clip(theta, 0.15, 5.0)

    same_mass = 1.0 / k
    same_factor = (1.0 - mixing) / same_mass
    diff_factor = mixing / (1.0 - same_mass)
    base = float(average_degree) / max(1, n - 1)
    edges = []
    for i in range(n - 1):
        js = np.arange(i + 1, n, dtype=np.int64)
        factor = np.where(z[js] == z[i], same_factor, diff_factor)
        lam = base * theta[i] * theta[js] * factor
        p = -np.expm1(-np.clip(lam, 0.0, 30.0))
        keep = rng.random(len(js)) < p
        edges.extend((i, int(j)) for j in js[keep])
    g = nx.Graph()
    g.add_nodes_from(range(n))
    nx.set_node_attributes(g, {int(i): int(z[i]) for i in range(n)}, "block")
    g.add_edges_from(edges)
    if g.number_of_edges() == 0:
        raise RuntimeError("DCSBM generated no edges")
    return g

def make_substrate(kind: str, n: int, seed: int) -> nx.Graph:
    """ER / BA / LFR / DCSBM graph, reduced to the largest connected component
    and relabeled to 0..n-1."""
    if kind == "er":
        g = nx.gnm_random_graph(n, 4 * n, seed=seed)
    elif kind == "ba":
        g = nx.barabasi_albert_graph(n, 4, seed=seed)
    elif kind == "lfr":
        g = None
        for attempt in range(8):
            try:
                g = nx.LFR_benchmark_graph(
                    n, tau1=2.5, tau2=1.5, mu=0.2, average_degree=8,
                    min_community=max(20, n // 15), seed=seed + attempt,
                )
                g = nx.Graph(g)  # drop multi-edges/attrs
                g.remove_edges_from(nx.selfloop_edges(g))
                break
            except Exception:
                g = None
        if g is None:
            raise RuntimeError("LFR generation failed after retries")
    elif kind == "dcsbm":
        g = make_dcsbm_graph(n=n, seed=seed)
    else:
        raise ValueError(f"unknown substrate kind: {kind}")
    lcc = max(nx.connected_components(g), key=len)
    g = g.subgraph(lcc).copy()
    return nx.convert_node_labels_to_integers(g)


# ----------------------------------------------------------------------------
# Family (the invariant part)
# ----------------------------------------------------------------------------

@dataclass
class Family:
    name: str
    kind: str
    edges: np.ndarray          # (E, 2) canonical u < v
    m_e: np.ndarray            # (E,) events per edge, >= 1
    t_all: np.ndarray          # (M,) sorted timestamps in [0, T)
    T: float
    W: int
    deg: np.ndarray            # node degrees (for hub-biased selection)
    timestamps: str = "uniform"  # "uniform" | "bursty" (clumped + burst hand-out)

    @property
    def window_caps(self) -> np.ndarray:
        win = self.T / self.W
        idx = window_index(self.t_all, 0.0, win, self.W)  # shared convention
        return np.bincount(idx, minlength=self.W)

    @property
    def max_rho(self) -> float:
        return float(np.mean(self.m_e >= 2))


def sample_event_counts(n_edges: int, rng: np.random.Generator,
                        p_single: float = P_SINGLE) -> np.ndarray:
    m = np.ones(n_edges, dtype=np.int64)
    multi = rng.random(n_edges) >= p_single
    extra = np.maximum(
        1, np.round(rng.lognormal(LOGN_MU, LOGN_SIGMA, multi.sum())).astype(np.int64)
    )
    m[multi] = 1 + extra
    return m


def make_family(name: str, kind: str, n: int, seed: int,
                T: float = 1.0, W: int = HEADLINE_W,
                timestamps: str = "uniform", burst_sigma: float = 1.6) -> Family:
    g = make_substrate(kind, n, seed)
    edges = np.array([(min(u, v), max(u, v)) for u, v in g.edges()], dtype=np.int64)
    edges = edges[np.lexsort((edges[:, 1], edges[:, 0]))]
    rng = np.random.default_rng(seed)
    m_e = sample_event_counts(len(edges), rng)
    M = int(m_e.sum())
    if timestamps == "uniform":
        t_all = np.sort(rng.uniform(0.0, T, M))
    elif timestamps == "bursty":
        # Bursty activity on a fixed topology (Sheng et al. 2023). lognormal
        # gaps -> clumped activity; the multiset is still drawn once per family,
        # so the twin logic is untouched. We calibrate to the burstiness
        # *statistic*, not a distributional family, so lognormal is just a
        # tunable heavy tail (empirical IET laws are debated), not a claim about
        # the true inter-event-time form.
        gaps = rng.lognormal(0.0, burst_sigma, M)
        t = np.cumsum(gaps)
        t_all = (t - t[0]) / (t[-1] - t[0]) * T * (1.0 - 1e-9) if M > 1 \
            else np.array([T / 2.0])
    else:
        raise ValueError(f"unknown timestamps mode: {timestamps}")
    deg = np.zeros(g.number_of_nodes(), dtype=np.int64)
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1
    return Family(name=name, kind=kind, edges=edges, m_e=m_e, t_all=t_all,
                  T=T, W=W, deg=deg, timestamps=timestamps)


# ----------------------------------------------------------------------------
# Instance (the varying part: timestamp -> edge assignment)
# ----------------------------------------------------------------------------

@dataclass
class Instance:
    family: str
    rho_target: float
    seed: int
    hub_bias: bool
    events: pd.DataFrame       # columns u, v, t (sorted by t)
    achieved: dict = field(default_factory=dict)  # exact GT via census
    deviations: int = 0        # edges whose intended span had to be repaired


def _pick_windows(rng, caps, n_windows_wanted, need_total):
    """Pick distinct windows (preferring high remaining capacity) such that
    their combined capacity covers `need_total`. Returns list of window ids."""
    order = np.argsort(-(caps + rng.random(len(caps))))  # capacity desc, random ties
    chosen = []
    cap_sum = 0
    for w in order:
        if caps[w] <= 0:
            continue
        chosen.append(int(w))
        cap_sum += caps[w]
        if len(chosen) >= n_windows_wanted and cap_sum >= need_total:
            break
    return chosen


def _pick_contiguous_windows(rng, caps, n_windows_wanted, need_total):
    """Prefer one contiguous interval of windows with enough capacity.

    This is the v2 realistic-span layout.  It never changes v1 because the
    caller must request ``span_layout="contiguous"`` explicitly.  If no
    interval can hold all events, the highest-capacity interval is returned and
    the allocator may append fallback windows; that repair is counted as a
    deviation by ``make_instance``.
    """
    width = max(1, min(int(n_windows_wanted), len(caps)))
    candidates = []
    for start in range(0, len(caps) - width + 1):
        ws = list(range(start, start + width))
        positive = all(caps[w] > 0 for w in ws)
        total = int(sum(caps[w] for w in ws))
        candidates.append((positive and total >= need_total, total, rng.random(), ws))
    feasible = [x for x in candidates if x[0]]
    if feasible:
        return max(feasible, key=lambda x: (x[1], x[2]))[3]
    return max(candidates, key=lambda x: (x[1], x[2]))[3] if candidates else []


def _allocate(rng, caps, m, span_wanted, span_layout="legacy"):
    """Distribute m events over `span_wanted` distinct windows (>=1 event each),
    respecting remaining capacities. Falls back to more/fewer windows if needed.
    Returns dict window -> count. Capacities are decremented in place."""
    span_wanted = max(1, min(span_wanted, m, len(caps)))
    if span_layout == "legacy":
        chosen = _pick_windows(rng, caps, span_wanted, m)
    elif span_layout == "contiguous":
        chosen = _pick_contiguous_windows(rng, caps, span_wanted, m)
    else:
        raise ValueError(f"unknown span_layout: {span_layout}")
    if not chosen:
        raise RuntimeError("no window capacity left")
    counts = {w: 0 for w in chosen}
    # one event per chosen window first (as far as capacity allows)
    for w in chosen:
        if m == 0:
            break
        if caps[w] > 0:
            counts[w] += 1
            caps[w] -= 1
            m -= 1
    # remaining events: weighted by remaining capacity within chosen windows
    while m > 0:
        avail = [w for w in chosen if caps[w] > 0]
        if not avail:
            extra = _pick_windows(rng, caps, 1, m)
            if not extra:
                raise RuntimeError("global capacity exhausted (should not happen)")
            chosen.extend(extra)
            counts.update({w: counts.get(w, 0) for w in extra})
            continue
        weights = np.array([caps[w] for w in avail], dtype=float)
        w = int(rng.choice(avail, p=weights / weights.sum()))
        take = int(min(caps[w], m if len(avail) == 1 else max(1, m // len(avail))))
        counts[w] += take
        caps[w] -= take
        m -= take
    return {w: c for w, c in counts.items() if c > 0}


def make_instance(fam: Family, rho_target: float, seed: int,
                  hub_bias: bool = False, span_layout: str = "legacy") -> Instance:
    rng = np.random.default_rng(seed)
    E = len(fam.edges)
    caps = fam.window_caps.astype(np.int64).copy()

    # which edges are persistent
    n_persist = int(round(rho_target * E))
    eligible = np.where(fam.m_e >= 2)[0]
    n_persist = min(n_persist, len(eligible))
    if hub_bias:
        w = (fam.deg[fam.edges[eligible, 0]] * fam.deg[fam.edges[eligible, 1]]).astype(float)
        p = w / w.sum()
    else:
        p = None
    persistent = set(rng.choice(eligible, size=n_persist, replace=False, p=p).tolist())

    # intended spans: bursty = 1; persistent = 2 + geometric tail, capped
    span = np.ones(E, dtype=np.int64)
    for e in persistent:
        s = 2 + rng.geometric(SPAN_GEOM_P) - 1
        span[e] = min(s, fam.m_e[e], fam.W)

    # allocation order: bursty edges with many events first (hard single-window
    # constraint), then everything else shuffled
    bursty = np.array([e for e in range(E) if e not in persistent], dtype=np.int64)
    bursty = bursty[np.argsort(-fam.m_e[bursty])]
    pers = rng.permutation(np.array(sorted(persistent), dtype=np.int64))
    order = np.concatenate([bursty, pers]) if len(pers) else bursty

    win_of_slots = [None] * E
    deviations = 0
    for e in order:
        alloc = _allocate(rng, caps, int(fam.m_e[e]), int(span[e]),
                          span_layout=span_layout)
        if len(alloc) != span[e]:
            deviations += 1
        win_of_slots[e] = alloc

    # hand out actual timestamps per window (shuffled within window)
    win = fam.T / fam.W
    widx_all = window_index(fam.t_all, 0.0, win, fam.W)  # shared convention
    if fam.timestamps == "bursty":
        # time-sorted index pools; each (edge, window) allocation takes a
        # CONTIGUOUS run of remaining timestamps -> tight per-edge bursts,
        # while the family multiset stays exactly preserved.
        pools = {w: np.where(widx_all == w)[0] for w in range(fam.W)}
    else:
        pools = {w: list(rng.permutation(np.where(widx_all == w)[0])) for w in range(fam.W)}

    us, vs, ts = [], [], []
    for e in range(E):
        u, v = fam.edges[e]
        for w, c in win_of_slots[e].items():
            if fam.timestamps == "bursty":
                pool = pools[w]
                start = int(rng.integers(0, len(pool) - c + 1)) if len(pool) > c else 0
                for idx_t in pool[start:start + c]:
                    ts.append(fam.t_all[idx_t]); us.append(u); vs.append(v)
                pools[w] = np.delete(pool, np.s_[start:start + c])
            else:
                for _ in range(c):
                    ts.append(fam.t_all[pools[w].pop()])
                    us.append(u)
                    vs.append(v)
    assert all(len(p) == 0 for p in pools.values()), "capacity bookkeeping broken"

    df = pd.DataFrame({"u": us, "v": vs, "t": ts}).sort_values("t").reset_index(drop=True)

    gt = census_row(normalize(df.astype({"u": str, "v": str})), label=fam.name)
    achieved = {k: gt[k] for k in
                ["rho_headline", "rho_event_weighted", "mean_span_frac",
                 "C_one_step", "lifetime_mean_over_T", "burstiness_pooled",
                 "share_single_event_pairs"]}
    achieved.update({f"rho_W{fam.W}_k{k}": gt[f"rho_W{fam.W}_k{k}"]
                     for k in range(1, fam.W + 1)})

    return Instance(family=fam.name, rho_target=rho_target, seed=seed,
                    hub_bias=hub_bias, events=df, achieved=achieved,
                    deviations=deviations)
