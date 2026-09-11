#!/usr/bin/env python3
"""Per-window-count representation: does it make the bias correctable label-free?

Instead of storing per edge only (n, w) = (# observations, # distinct windows),
store the full per-window count vector, e.g. (0,0,0,0,4) for an edge seen 4 times
all in window 5. This is a strictly richer input than (n, w). Question: does it
let a label-free estimator recover rho under forward-time sampling, where the
(n,w) occupancy MLE cannot?

Answer (see results/bias_identifiability/RESULTS.md), three parts:

  (1) The generator's TRUE activity is positionally uniform (P(window j active |
      edge active) is flat), so any late-skew in observations is pure bias.

  (2) A per-window bias-aware MLE (latent = active-window SET, random k-subset;
      sampling weights exp(beta*j) over absolute window index j) DOES detect the
      bias: its profile likelihood peaks at beta>0 for forward walks and beta~0
      for clean walks, unlike the (n,w) MLE whose likelihood peaks at beta=0.
      BUT the free fit OVERCORRECTS rho (jumps to ~1.0), because it must guess
      the unobserved early-window activity. Detection != correction.

  (3) The mechanism: P(window j observed | window j truly active) under a forward
      walk is ~0.4% for the first window vs ~30% for the last (clean: flat ~14%).
      A forward crawl STRUCTURALLY never samples early windows, so the per-edge
      early activity is simply absent from the data. No representation of the
      observed data can recover what was never sampled.

Conclusion: per-window counts make the bias VISIBLE but not CORRECTABLE from a
crawl. The obstruction is the sampling, not the representation. This is the
impossibility result that complements Phase 2 (correction needs external bias
strength / disclosure).

Run:  python per_window_bias_test.py
Requires: census.py, generator.py, walks.py alongside (corrected_estimator.py
optional, used only for the (n,w) reference print).
"""
import warnings
from itertools import combinations
from math import comb as C

import numpy as np

warnings.filterwarnings("ignore")
import census
import generator
import walks

try:
    import corrected_estimator
    _HAVE_CE = True
except Exception:
    _HAVE_CE = False

W = 5
_SUBSETS_BY_K = {k: [sum(1 << j for j in s) for s in combinations(range(W), k)]
                 for k in range(1, W + 1)}


def _Zmask(beta):
    z = {}
    for k in _SUBSETS_BY_K:
        for sm in _SUBSETS_BY_K[k]:
            z[sm] = sum(np.exp(beta * (j + 1)) for j in range(W) if (sm >> j) & 1)
    return z


def _em(Lik, iters=150):
    pi = np.ones(W) / W
    for _ in range(iters):
        num = Lik * pi[None, :]
        den = num.sum(1, keepdims=True)
        den[den == 0] = 1.0
        pi = (num / den).mean(0)
        s = pi.sum()
        pi = pi / s if s > 0 else np.ones(W) / W
    return pi


def obs_counts(log, budget):
    """Per-edge per-window observation-count vectors (E x W)."""
    Lg = log.iloc[:budget]
    s = Lg[Lg["kind"] == 1]
    tv = s["t"].to_numpy(float)
    if len(s) == 0 or not np.isfinite(tv).any():
        return None
    win = census.window_index(tv, 0.0, 1.0 / W, W)
    nw = {}
    for a, b, w in zip(s["u"], s["v"], win):
        c = nw.setdefault((int(a), int(b)), np.zeros(W, int))
        c[int(w)] += 1
    return np.array(list(nw.values()))


def perwin_fit(counts, beta, Zc):
    """Per-window bias-aware MLE at fixed beta: returns (rho_hat, loglik)."""
    E = len(counts)
    Lik = np.zeros((E, W))
    n_arr = counts.sum(1)
    Tm = np.array([sum(1 << int(j) for j in np.nonzero(c)[0]) for c in counts])
    for e in range(E):
        T = int(Tm[e]); n = int(n_arr[e]); w = bin(T).count("1")
        for k in range(w, W + 1):
            ss = 0.0
            for sm in _SUBSETS_BY_K[k]:
                if (T & sm) == T:
                    ss += 1.0 / (Zc[sm] ** n)
            Lik[e, k - 1] = ss / C(W, k)
    pi = _em(Lik)
    posw = float((np.arange(1, W + 1) * counts).sum())
    ll = beta * posw + float(np.log(np.clip((Lik * pi[None, :]).sum(1), 1e-300, None)).sum())
    return float(1 - pi[0]), ll


def true_active(ev):
    t = ev["t"].to_numpy(float)
    win = census.window_index(t, 0.0, 1.0 / W, W)
    nw = {}
    for a, b, wi in zip(ev["u"], ev["v"], win):
        nw.setdefault((int(a), int(b)), set()).add(int(wi))
    return nw


def obs_active(log, budget):
    Lg = log.iloc[:budget]
    s = Lg[Lg["kind"] == 1]
    tv = s["t"].to_numpy(float)
    if len(s) == 0 or not np.isfinite(tv).any():
        return {}
    win = census.window_index(tv, 0.0, 1.0 / W, W)
    nw = {}
    for a, b, wi in zip(s["u"], s["v"], win):
        nw.setdefault((int(a), int(b)), set()).add(int(wi))
    return nw


def main():
    betas = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # (1) true activity positional marginal
    print("=== (1) true activity marginal over windows (ground truth) ===")
    acc = np.zeros(W); ntot = 0
    for kind in ["ba", "er"]:
        for sd in range(3):
            fam = generator.make_family(f"{kind}{sd}", kind, n=3000, seed=sd)
            inst = generator.make_instance(fam, 0.40, seed=sd + 10)
            for wins in true_active(inst.events).values():
                for wi in wins:
                    acc[wi] += 1
                ntot += 1
    print("  P(window active | edge active), windows 1..5:", np.round(acc / ntot, 3))
    print("  (flat => positionally neutral; the observation skew below is pure bias)\n")

    # (2) per-window MLE beta-profile: detects bias, but overcorrects
    print("=== (2) forward instance: does per-window identify beta? ===")
    fam = generator.make_family("demo", "ba", n=2500, seed=7)
    inst = generator.make_instance(fam, 0.40, seed=11)
    idx = walks.build_index(inst.events)
    log = walks.run_walk(idx, "time_respecting", max_budget=3200, seed=3)
    counts = obs_counts(log, 3200)
    rt = inst.achieved["rho_headline"]
    print(f"  rho_true={rt:.3f}, observed edges={len(counts)}")
    if _HAVE_CE:
        na = counts.sum(1); wa = (counts > 0).sum(1)
        print(f"  (n,w)-MLE uniform (beta=0): rho_hat="
              f"{corrected_estimator.rho_mle(na, wa)[0]:.3f}  <- Phase-2 failure")
    best = None
    for b in betas:
        Zc = _Zmask(b)
        rho, ll = perwin_fit(counts, b, Zc)
        if best is None or ll > best[1]:
            best = (b, ll, rho)
        print(f"    beta={b:+.2f}  loglik={ll:11.1f}  rho_hat={rho:.3f}")
    print(f"  --> per-window argmax beta={best[0]:.2f}, rho_hat={best[2]:.3f} "
          f"(true {rt:.3f}): detects bias, OVERCORRECTS rho\n")

    # (3) the mechanism: early windows are structurally unobserved under forward
    print("=== (3) P(window j observed | window j truly active), by walk ===")
    for strat in ["time_agnostic_t", "recency_biased", "time_respecting"]:
        seen = np.zeros(W); have = np.zeros(W)
        for kind in ["ba", "er"]:
            for sd in range(3):
                fam = generator.make_family(f"{kind}{sd}", kind, n=3000, seed=sd)
                inst = generator.make_instance(fam, 0.45, seed=sd + 10)
                tr = true_active(inst.events)
                idx = walks.build_index(inst.events)
                log = walks.run_walk(idx, strat, max_budget=3200, seed=7)
                ob = obs_active(log, 3200)
                for e, wins in tr.items():
                    for j in wins:
                        have[j] += 1
                        if e in ob and j in ob[e]:
                            seen[j] += 1
        print(f"  {strat:16s} windows 1..5: {np.round(seen / np.maximum(have, 1), 3)}")
    print("  forward misses early windows (~0.4% for W1 vs ~30% for W5): the")
    print("  per-edge early activity is never sampled, so it cannot be recovered.")


if __name__ == "__main__":
    main()
