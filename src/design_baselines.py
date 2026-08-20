#!/usr/bin/env python3
"""Design-aware estimators and shape constraints for persistence profiles.

The Hansen--Hurwitz estimator here is deliberately narrow.  It is valid only
for ``time_agnostic_t`` after the simple random walk has mixed: stationary
directed-edge traversals are uniform, so traversal multiplicity is the design
weight.  The latent edge label is replaced by its posterior probability under
the existing position-aware mask model (a model-assisted HH estimator).

No such claim is made for time-respecting, recency-biased, or recent-history
walks: their edge probabilities depend on the clock and path history.
"""

import json

import numpy as np

from mask_estimator import mask_mle, mask_pattern_likelihood


def parse_nmask_hist(raw):
    """Return arrays (n, mask, count) from compact ``input__nmask`` JSON."""
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        raise ValueError("n-mask histogram must be a JSON object")
    rows = []
    for key, count in obj.items():
        n_text, mask_text = str(key).split(",", 1)
        rows.append((int(n_text), int(mask_text, 16), int(count)))
    rows.sort()
    if any(n < 1 or mask < 1 or count < 1 for n, mask, count in rows):
        raise ValueError("n, mask and count must all be positive")
    if not rows:
        return (np.array([], dtype=np.int64),) * 3
    return tuple(np.asarray(x, dtype=np.int64) for x in zip(*rows))


def model_assisted_hh(n, masks, counts, W=5):
    """Posterior-label Hansen--Hurwitz readouts for stationary edge draws.

    ``counts`` is the number of distinct sampled dyads with each (n, mask)
    pattern.  Each dyad contributes ``n`` traversal draws to the HH mean.
    """
    n = np.asarray(n, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.int64)
    counts = np.asarray(counts, dtype=float)
    if not (len(n) == len(masks) == len(counts)):
        raise ValueError("n, masks and counts must have equal length")
    if not len(n):
        return {}
    _, probs = mask_mle(n, masks, W=W, weights=counts)
    likelihood, states, state_k = mask_pattern_likelihood(n, masks, W=W)
    post = likelihood * probs[None, :]
    den = post.sum(axis=1, keepdims=True)
    if np.any(den <= 0):
        raise ValueError("mask model assigned zero probability to an observation")
    post /= den
    traversal_weight = counts * n
    total = float(traversal_weight.sum())

    out = {}
    for k in range(1, W + 1):
        q = post[:, state_k >= k].sum(axis=1)
        out[f"rho_k{k}"] = float(np.dot(traversal_weight, q) / total)
    expected_k = post @ state_k.astype(float)
    out["mean_occupancy"] = float(
        np.dot(traversal_weight, expected_k) / (total * W))

    adjacent = np.array([
        sum(bool(s & (1 << w)) and bool(s & (1 << (w + 1)))
            for w in range(W - 1)) for s in states
    ], dtype=float)
    eligible = np.array([
        sum(bool(s & (1 << w)) for w in range(W - 1)) for s in states
    ], dtype=float)
    num = float(np.dot(traversal_weight, post @ adjacent))
    denom = float(np.dot(traversal_weight, post @ eligible))
    out["C_one_step"] = num / denom if denom > 0 else float("nan")
    return out


def model_assisted_hh_from_json(raw, W=5):
    n, masks, counts = parse_nmask_hist(raw)
    return model_assisted_hh(n, masks, counts, W=W)


def discovery_diagnostics(n, counts):
    """Chao1/Good--Turing diagnostics; these do not estimate persistence."""
    n = np.asarray(n, dtype=np.int64)
    counts = np.asarray(counts, dtype=float)
    observed = float(counts.sum())
    traversals = float(np.dot(n, counts))
    f1 = float(counts[n == 1].sum())
    f2 = float(counts[n == 2].sum())
    # Bias-corrected Chao1 remains finite when f2=0.
    chao1 = observed + f1 * max(0.0, f1 - 1.0) / (2.0 * (f2 + 1.0))
    return {
        "observed_dyads": observed,
        "chao1_dyads": chao1,
        "chao1_observed_fraction": observed / chao1 if chao1 > 0 else np.nan,
        "good_turing_sample_coverage": (
            max(0.0, 1.0 - f1 / traversals) if traversals > 0 else np.nan),
    }


def project_profile_decreasing(values):
    """Euclidean projection onto ``1 >= x1 >= ... >= xp >= 0`` via PAVA."""
    y = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    if y.ndim != 1 or not len(y) or not np.isfinite(y).all():
        raise ValueError("profile must be a finite non-empty vector")
    # Increasing PAVA on -y, storing [mean, weight, start, end].
    blocks = []
    for i, value in enumerate(-y):
        blocks.append([float(value), 1, i, i + 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            b = blocks.pop()
            a = blocks.pop()
            weight = a[1] + b[1]
            blocks.append([
                (a[0] * a[1] + b[0] * b[1]) / weight,
                weight, a[2], b[3],
            ])
    out = np.empty_like(y)
    for mean, _, start, end in blocks:
        out[start:end] = -mean
    return np.clip(out, 0.0, 1.0)

