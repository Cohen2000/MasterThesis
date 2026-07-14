#!/usr/bin/env python3
"""Position-aware occupancy MLE for temporal window masks.

The existing occupancy estimator retains only ``(n, w)`` per sampled edge:
the number of observations and the number of distinct observed windows.  This
module retains the actual observed bit mask as well.  For W=5, the latent state
space contains only 31 non-empty true masks, so a small EM fit can estimate the
population distribution over temporal activity patterns without labels.

Under uniform sampling within an edge's true active windows S, the likelihood
of observing exactly mask O after n draws is

    1[O subset S] * surj(n, |O|) / |S|**n,

where ``surj`` is the number of onto maps from n labelled draws to the
observed windows.  This is the position-aware analogue of corrected_estimator.
It is still misspecified for biased walks; comparing it with the count-only MLE
tests whether preserving temporal position is enough to absorb that bias.
"""

from math import comb

import numpy as np


def _bitcount(x: int) -> int:
    return int(x).bit_count()


def _surjection_probability(n: int, observed_k: int, true_k: int) -> float:
    """P(draws use every one of a fixed observed_k-set and nothing else)."""
    if n < observed_k or observed_k < 1 or true_k < observed_k:
        return 0.0
    # Ratio form avoids enormous integer powers for heavily revisited edges.
    out = 0.0
    for i in range(observed_k + 1):
        out += ((-1.0) ** i * comb(observed_k, i)
                * ((observed_k - i) / true_k) ** n)
    return float(max(0.0, min(1.0, out)))


def mask_mle(n_arr, mask_arr, W: int = 5, iters: int = 300,
             prior: float = 1e-3):
    """Fit latent probabilities for all non-empty W-bit masks.

    Returns ``(readouts, probs)``.  ``readouts`` contains the full persistence
    profile ``rho_k=P(K>=k)``, mean occupancy, and adjacent-window persistence
    C.  Returns ``({}, None)`` for empty input.
    """
    n_arr = np.asarray(n_arr, dtype=np.int64)
    mask_arr = np.asarray(mask_arr, dtype=np.int64)
    if len(n_arr) == 0:
        return {}, None
    if len(n_arr) != len(mask_arr):
        raise ValueError("n_arr and mask_arr must have equal length")
    max_mask = (1 << W) - 1
    if np.any(mask_arr < 1) or np.any(mask_arr > max_mask):
        raise ValueError(f"observed masks must be in [1,{max_mask}]")

    patterns, counts = np.unique(np.column_stack([n_arr, mask_arr]), axis=0,
                                 return_counts=True)
    states = np.arange(1, max_mask + 1, dtype=np.int64)
    state_k = np.array([_bitcount(s) for s in states], dtype=np.int64)
    likelihood = np.zeros((len(patterns), len(states)), dtype=float)
    cache = {}
    for e, (n, obs) in enumerate(patterns):
        ko = _bitcount(int(obs))
        for j, (true, kt) in enumerate(zip(states, state_k)):
            if int(obs) & int(true) != int(obs):
                continue
            key = (int(n), ko, int(kt))
            if key not in cache:
                cache[key] = _surjection_probability(*key)
            likelihood[e, j] = cache[key]

    probs = np.ones(len(states), dtype=float) / len(states)
    for _ in range(iters):
        weighted = likelihood * probs[None, :]
        denom = weighted.sum(axis=1, keepdims=True)
        denom[denom <= 0] = 1.0
        new = ((weighted / denom) * counts[:, None]).sum(axis=0) + prior
        new /= new.sum()
        if np.max(np.abs(new - probs)) < 1e-10:
            probs = new
            break
        probs = new

    readouts = {}
    for k in range(1, W + 1):
        readouts[f"rho_k{k}"] = float(probs[state_k >= k].sum())
    readouts["mean_occupancy"] = float(np.dot(probs, state_k) / W)

    adjacent = np.array([
        sum(bool(s & (1 << w)) and bool(s & (1 << (w + 1)))
            for w in range(W - 1)) for s in states
    ], dtype=float)
    eligible = np.array([
        sum(bool(s & (1 << w)) for w in range(W - 1)) for s in states
    ], dtype=float)
    den = float(np.dot(probs, eligible))
    readouts["C_one_step"] = (
        float(np.dot(probs, adjacent) / den) if den > 0 else float("nan")
    )
    return readouts, probs


if __name__ == "__main__":
    # Minimal executable self-check: fully observed masks must reproduce the
    # empirical population profile up to the tiny Dirichlet smoothing prior.
    n = np.array([20, 20, 20, 20])
    masks = np.array([0b00001, 0b00011, 0b00111, 0b11111])
    out, _ = mask_mle(n, masks)
    assert abs(out["rho_k2"] - 0.75) < 0.01
    print("mask_estimator self-check OK")
