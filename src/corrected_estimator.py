"""Label-free occupancy MLE for window persistence under uniform-occupancy.

Per observed edge we have (n, w): n = times observed, w = distinct windows
observed (1..W). Under the assumption that the n observations land uniformly at
random among the edge's k TRUE active windows, P(w | n, k) is a classical
occupancy (surjection) probability. We EM-fit the population distribution
pi = (pi_1..pi_W) over the true window-count k, then read off:
    rho_mle = 1 - pi_1            (window persistence = P(K >= 2))
    occupancy = (sum_k k*pi_k)/W  (mean occupancy, same fit)

This estimator is CORRECT under unbiased (time_agnostic_t) sampling and becomes
MISSPECIFIED under forward-time sampling (time_respecting), where the uniform-
occupancy assumption is violated. That misspecification is the object of study.
"""
import numpy as np
from math import comb

W = 5
NCAP = 60  # n >= NCAP is treated as "fully resolved" (w windows are the true k)


def occupancy_table(W=W, ncap=NCAP):
    """tab[(k, n)][w] = P(observe exactly w distinct windows | n draws, k true windows)."""
    tab = {}
    for k in range(1, W + 1):
        for n in range(1, ncap):
            p = np.zeros(k + 1)
            for w in range(1, k + 1):
                surj = sum((-1) ** i * comb(w, i) * (w - i) ** n for i in range(w + 1))
                p[w] = comb(k, w) * surj / (k ** n)
            tab[(k, n)] = p
    return tab


_TAB = occupancy_table()


def rho_mle(n_arr, w_arr, W=W, iters=300):
    """EM fit of pi over true window counts; returns (rho_mle, pi) or (nan, None)."""
    n_arr = np.asarray(n_arr)
    w_arr = np.asarray(w_arr)
    E = len(n_arr)
    if E == 0:
        return np.nan, None
    # Edges sharing (n,w) have identical likelihood rows.  Aggregating them is
    # exactly equivalent to the per-edge EM and makes the large benchmark
    # practical (usually tens of rows instead of thousands per case).
    patterns, counts = np.unique(
        np.column_stack([n_arr.astype(np.int64), w_arr.astype(np.int64)]),
        axis=0, return_counts=True)
    Lik = np.zeros((len(patterns), W))
    for e, (n0, w0) in enumerate(patterns):
        n = int(n0); w = int(w0)
        if n >= NCAP:
            Lik[e, w - 1] = 1.0
        else:
            for k in range(max(1, w), W + 1):
                Lik[e, k - 1] = _TAB[(k, n)][w] if w <= k else 0.0
    pi = np.ones(W) / W
    for _ in range(iters):
        num = Lik * pi[None, :]
        den = num.sum(1, keepdims=True); den[den == 0] = 1.0
        resp = num / den
        pi = (resp * counts[:, None]).sum(0) / counts.sum()
        s = pi.sum()
        pi = pi / s if s > 0 else np.ones(W) / W
    return float(1.0 - pi[0]), pi
