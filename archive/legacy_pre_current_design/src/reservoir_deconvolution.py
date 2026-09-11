#!/usr/bin/env python3
"""Observable temporal empirical-Bayes baseline for event reservoirs.

The model is deliberately richer than a one-dimensional species-richness
mixture: every latent component consists of an active-window mask and a common
per-active-window event rate.  Full counts in an active window follow a
zero-truncated Poisson distribution, so the component has exactly the stated
active mask.  Independent Bernoulli thinning approximates a low-fraction
fixed-size reservoir.  The EM fit is performed on detected dyads and then
corrected back to population component weights using component detection
probabilities.

The approximation and equal-rate-within-mask assumption must be reported.  It
is an observable persistence estimator, not an oracle and not merely Chao1.
"""

import json

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize


def parse_window_count_histogram(raw, W: int = 5):
    obj = json.loads(raw) if isinstance(raw, str) else raw
    if not isinstance(obj, dict):
        raise ValueError("window-count histogram must be a JSON object")
    vectors, frequencies = [], []
    for key, value in obj.items():
        vector = tuple(int(x) for x in str(key).split(","))
        if len(vector) != W or min(vector) < 0 or sum(vector) < 1:
            raise ValueError(f"invalid observed count vector {vector}")
        count = int(value)
        if count < 1:
            raise ValueError("histogram frequencies must be positive")
        vectors.append(vector); frequencies.append(count)
    if not vectors:
        return np.empty((0, W), dtype=np.int64), np.empty(0, dtype=float)
    order = np.lexsort(np.asarray(vectors).T[::-1])
    return (np.asarray(vectors, dtype=np.int64)[order],
            np.asarray(frequencies, dtype=float)[order])


def default_mu_grid(observed_vectors, sampling_fraction, size: int = 36):
    """Deterministic full-stream rate grid using observable sample counts."""
    p = float(sampling_fraction)
    if not 0 < p <= 1:
        raise ValueError("sampling_fraction must lie in (0,1]")
    x = np.asarray(observed_vectors, dtype=float)
    xmax = float(x.max()) if x.size else 1.0
    upper = min(1e5, max(20.0, 2.0 * (xmax + 1.0) / p))
    return np.geomspace(0.02, upper, int(size))


def _component_log_kernel(vectors, p, masks, mus):
    """Log P(X=x | component, X_total>0) and detection probabilities."""
    x = np.asarray(vectors, dtype=np.int64)
    W = x.shape[1]
    n_components = len(masks)
    log_kernel = np.full((len(x), n_components), -np.inf, dtype=float)
    detection = np.empty(n_components, dtype=float)
    for j, (mask, mu) in enumerate(zip(masks, mus)):
        active = np.array([bool(mask & (1 << w)) for w in range(W)])
        # For N~Poisson(mu)|N>=1 and Bernoulli thinning at rate p:
        # P(X=0|N>=1) = (exp(-p*mu)-exp(-mu))/(1-exp(-mu)).
        log_ztp_den = np.log(-np.expm1(-mu))
        log_p0 = (-p * mu + np.log(-np.expm1(-(1.0 - p) * mu))
                  - log_ztp_den) if p < 1.0 else -np.inf
        p0 = 0.0 if p == 1.0 else float(np.exp(log_p0))
        q0 = p0 ** int(active.sum())
        detection[j] = max(np.finfo(float).tiny, 1.0 - q0)
        compatible = ~(x[:, ~active] > 0).any(axis=1)
        if not compatible.any():
            continue
        rows = np.flatnonzero(compatible)
        values = np.zeros(len(rows), dtype=float)
        for w in np.flatnonzero(active):
            counts = x[rows, w]
            zero = counts == 0
            if zero.any():
                values[zero] += log_p0
            positive = ~zero
            if positive.any():
                xp = counts[positive].astype(float)
                values[positive] += (-p * mu + xp * np.log(p * mu)
                                     - gammaln(xp + 1.0) - log_ztp_den)
        log_kernel[rows, j] = values - np.log(detection[j])
    return log_kernel, detection


def temporal_mask_rate_mixture_eb(raw, sampling_fraction: float, W: int = 5,
                                  grid_size: int = 36, max_iter: int = 500,
                                  tol: float = 1e-7):
    """Fit the temporal mask/rate mixture and return rho-profile estimates."""
    vectors, frequencies = parse_window_count_histogram(raw, W=W)
    if not len(vectors):
        return {}, {"converged": False, "iterations": 0}
    p = float(sampling_fraction)
    grid = default_mu_grid(vectors, p, size=grid_size)
    component_masks = np.repeat(np.arange(1, 1 << W, dtype=np.int64), len(grid))
    component_mus = np.tile(grid, (1 << W) - 1)
    log_kernel, detection = _component_log_kernel(
        vectors, p, component_masks, component_mus)
    possible = np.isfinite(log_kernel).any(axis=0)
    log_kernel = log_kernel[:, possible]
    detection = detection[possible]
    component_masks = component_masks[possible]
    component_mus = component_mus[possible]
    if np.any(~np.isfinite(logsumexp(log_kernel, axis=1))):
        raise ValueError("mixture grid cannot explain an observed pattern")

    alpha = np.full(len(component_masks), 1.0 / len(component_masks))
    total = float(frequencies.sum())
    converged = False
    loglik = -np.inf
    previous_profile = None
    profile_delta = np.inf
    alpha_delta = np.inf
    stable_profile_iterations = 0
    for iteration in range(1, max_iter + 1):
        log_joint = log_kernel + np.log(np.maximum(alpha, 1e-300))[None, :]
        row_norm = logsumexp(log_joint, axis=1)
        responsibilities = np.exp(log_joint - row_norm[:, None])
        new_alpha = (frequencies[:, None] * responsibilities).sum(axis=0) / total
        new_alpha /= new_alpha.sum()
        new_loglik = float(np.dot(frequencies, row_norm))
        alpha_delta = float(np.max(np.abs(new_alpha - alpha)))
        alpha = new_alpha
        loglik = new_loglik
        # Mixture weights over nearby rate-grid points need not be uniquely
        # stable. The estimand only depends on aggregate active-mask weights,
        # so convergence is defined on the rho profile itself.
        current_population = alpha / detection
        current_population /= current_population.sum()
        occupancy_now = np.array([
            int(mask).bit_count() for mask in component_masks])
        profile = np.array([
            current_population[occupancy_now >= k].sum()
            for k in range(2, W + 1)
        ])
        if previous_profile is not None:
            profile_delta = float(np.max(np.abs(profile - previous_profile)))
            stable_profile_iterations = (
                stable_profile_iterations + 1 if profile_delta < tol else 0)
        previous_profile = profile
        if iteration >= 20 and stable_profile_iterations >= 5:
            converged = True
            break

    population = alpha / detection
    population /= population.sum()
    occupancy = np.array([int(mask).bit_count() for mask in component_masks])
    result = {}
    for k in range(2, W + 1):
        result[f"rho_k{k}"] = float(population[occupancy >= k].sum())
    result["mean_occupancy"] = float(np.dot(population, occupancy) / W)
    adjacent = np.array([
        sum(bool(mask & (1 << w)) and bool(mask & (1 << (w + 1)))
            for w in range(W - 1)) for mask in component_masks
    ], dtype=float)
    eligible = np.array([
        sum(bool(mask & (1 << w)) for w in range(W - 1))
        for mask in component_masks
    ], dtype=float)
    numerator = float(np.dot(population, adjacent))
    denominator = float(np.dot(population, eligible))
    result["C_one_step"] = numerator / denominator if denominator else np.nan
    diagnostics = {
        "converged": bool(converged), "iterations": int(iteration),
        "final_profile_delta": float(profile_delta),
        "final_alpha_delta": float(alpha_delta),
        "stable_profile_iterations": int(stable_profile_iterations),
        "log_likelihood_detected": loglik,
        "effective_components_detected": float(1.0 / np.sum(alpha ** 2)),
        "effective_components_population": float(1.0 / np.sum(population ** 2)),
        "mu_grid_min": float(grid.min()), "mu_grid_max": float(grid.max()),
        "bernoulli_sampling_fraction": p,
    }
    return result, diagnostics


def temporal_mask_rate_factorized_eb(raw, sampling_fraction: float, W: int = 5,
                                     grid_size: int = 28,
                                     max_iter: int = 500):
    """Stable semiparametric EB fit with independent mask/rate mixing.

    Population component weights factor as ``pi_mask * g_rate``.  This removes
    the severe non-identifiability of a free weight for every mask-rate pair
    while retaining a nonparametric grid distribution for activity rates.
    Parameters maximize the detected-dyad conditional likelihood directly.
    """
    vectors, frequencies = parse_window_count_histogram(raw, W=W)
    if not len(vectors):
        return {}, {"converged": False, "iterations": 0}
    p = float(sampling_fraction)
    grid = default_mu_grid(vectors, p, size=grid_size)
    masks_unique = np.arange(1, 1 << W, dtype=np.int64)
    n_masks, n_rates = len(masks_unique), len(grid)
    masks = np.repeat(masks_unique, n_rates)
    mus = np.tile(grid, n_masks)
    log_conditional, detection = _component_log_kernel(
        vectors, p, masks, mus)
    log_unconditional = log_conditional + np.log(detection)[None, :]
    row_scale = np.max(log_unconditional, axis=1)
    if np.any(~np.isfinite(row_scale)):
        raise ValueError("factorized mixture cannot explain an observed pattern")
    kernel = np.exp(log_unconditional - row_scale[:, None]).reshape(
        len(vectors), n_masks, n_rates)
    detection = detection.reshape(n_masks, n_rates)
    total = float(frequencies.sum())

    def softmax(values):
        shifted = values - np.max(values)
        out = np.exp(shifted)
        return out / out.sum()

    def objective_and_gradient(params):
        pi = softmax(params[:n_masks])
        rates = softmax(params[n_masks:])
        joint = pi[:, None] * rates[None, :]
        observed_prob = np.einsum("imr,mr->i", kernel, joint)
        detected_prob = float(np.sum(detection * joint))
        if np.any(observed_prob <= 0) or detected_prob <= 0:
            return np.inf, np.zeros_like(params)
        loglik = (float(np.dot(frequencies, np.log(observed_prob)))
                  - total * np.log(detected_prob))
        grad_pi = (np.einsum(
            "i,imr,r,i->m", frequencies, kernel, rates,
            1.0 / observed_prob)
            - total * (detection @ rates) / detected_prob)
        grad_rate = (np.einsum(
            "i,imr,m,i->r", frequencies, kernel, pi,
            1.0 / observed_prob)
            - total * (pi @ detection) / detected_prob)
        grad_pi_logits = pi * (grad_pi - np.dot(pi, grad_pi))
        grad_rate_logits = rates * (grad_rate - np.dot(rates, grad_rate))
        return -loglik, -np.concatenate([grad_pi_logits, grad_rate_logits])

    fit = minimize(
        objective_and_gradient, np.zeros(n_masks + n_rates), jac=True,
        method="L-BFGS-B",
        options={"maxiter": int(max_iter), "ftol": 1e-12,
                 "gtol": 1e-8, "maxcor": 30})
    pi = softmax(fit.x[:n_masks])
    rates = softmax(fit.x[n_masks:])
    occupancy = np.array([int(mask).bit_count() for mask in masks_unique])
    result = {
        f"rho_k{k}": float(pi[occupancy >= k].sum())
        for k in range(2, W + 1)
    }
    result["mean_occupancy"] = float(np.dot(pi, occupancy) / W)
    adjacent = np.array([
        sum(bool(mask & (1 << w)) and bool(mask & (1 << (w + 1)))
            for w in range(W - 1)) for mask in masks_unique
    ], dtype=float)
    eligible = np.array([
        sum(bool(mask & (1 << w)) for w in range(W - 1))
        for mask in masks_unique
    ], dtype=float)
    denominator = float(np.dot(pi, eligible))
    result["C_one_step"] = (
        float(np.dot(pi, adjacent) / denominator) if denominator else np.nan)
    diagnostics = {
        "converged": bool(fit.success), "iterations": int(fit.nit),
        "optimizer_status": int(fit.status),
        "optimizer_message": str(fit.message),
        "final_gradient_max": float(np.max(np.abs(fit.jac))),
        "negative_log_likelihood_scaled": float(fit.fun),
        "effective_mask_components": float(1.0 / np.sum(pi ** 2)),
        "effective_rate_components": float(1.0 / np.sum(rates ** 2)),
        "mu_grid_min": float(grid.min()), "mu_grid_max": float(grid.max()),
        "bernoulli_sampling_fraction": p,
    }
    return result, diagnostics
