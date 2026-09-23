"""Shared zero-truncated Beta-Binomial observation-model MLE.

A single working-model baseline used across every arm: a common latent
persistence model

    q_e ~ Beta(alpha, beta),   X_e,w | q_e ~ Bernoulli(q_e),  w = 1..5,

so K_e = sum_w X_e,w is Beta-Binomial(5, alpha, beta), and the target is

    rho_r = P(K >= r | K >= 1),  r = 2..5.

Differences between arms enter only through which released information can be
used to fit (alpha, beta) (and, for B, the extra event-detection nuisance
parameter lambda). This is a common-model-family competitor, not a
design-unbiased corrector or an oracle: informative S1 walk selection is
intentionally left uncorrected, and H's extrapolation from the visible windows
to W=5 relies on an exchangeability assumption that is not implied by the
access mechanism.

  R, S1   both observe complete 5-window histories of every observed dyad;
          fit is the zero-truncated Beta-Binomial likelihood over J=1..5.
  H       observes only the windows Temporal_access marks accessible (m of
          them); fit is the zero-truncated Beta-Binomial likelihood over
          J=1..m, and the SAME (alpha, beta) is then plugged into the W=5
          model. m=5 makes this identical to the R/S1 fit (test-checked).
  B       reuses mixtures.fit_events, which already implements exactly the
          Beta-activity / ZTP(lambda) event-detection / Binomial(p) thinning
          model of this module's docstring (cell_probs(a,b,d,n) with
          d=(1-exp(-p*lambda))/(1-exp(-lambda)) is delta(lambda,p); its event
          term is the same positive-count log-likelihood up to sign). No
          second copy of that algebra is written here.

Fallback (used only on a genuine optimizer failure or a starts-disagreement,
never as a manual truth-informed repair): the homogeneous-binomial limit of
the same family, i.e. one shared activity probability q (reusing
baselines.activity/profile), with B keeping its event-detection nuisance
parameter (baselines.corrector already implements exactly that homogeneous
B corrector).
"""
import math
from dataclasses import dataclass, field
import numpy as np
from . import baselines, mixtures
from .mixtures import LOG_KAPPA_BOUNDS, LOGIT_BOUND, PENALTY, START_KAPPAS, _pack, _solve, _unpack, cell_probs, diagnose, is_unreliable, predict_profile

BOUNDS = [(-LOGIT_BOUND, LOGIT_BOUND), LOG_KAPPA_BOUNDS]


@dataclass
class Result:
    rho: list
    alpha: float
    beta: float
    lam: float = float('nan')
    fit_status: str = 'ok'
    fallback_used: bool = False
    objective: float = float('nan')
    flags: dict = field(default_factory=dict)


def visible_counts(o):
    """(counts, m): counts[j] = observed dyads with j active windows among the
    m released-visible windows (j=1..m); derived only from o['table'], so an
    arm that never releases some window (H) never contributes a count for it.
    """
    m = len(o['table'][0][0].replace('?', ''))
    counts = [0.]*(m+1)
    for row in o['table']:
        weight = row[5] if o['arm'] == 'S' else row[3] if o['arm'] == 'S_obs' else row[1]
        counts[row[0].count('1')] += weight
    return counts, m


def zt_bb_nll(counts, n):
    """-log P(J=j | J>=1, alpha,beta) multinomial log-likelihood, j=1..n."""
    def nll(z):
        a, b = _unpack(z)
        if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)): return PENALTY
        pr = cell_probs(a, b, 1., n)
        tr = math.fsum(pr[1:])
        if tr <= 0: return PENALTY
        s = 0.
        for j in range(1, n+1):
            if counts[j]:
                if pr[j] <= 0: return PENALTY
                s += counts[j]*(math.log(pr[j])-math.log(tr))
        return -s
    return nll


def fit_zt_bb(counts, n):
    """Fit alpha,beta by the zero-truncated Beta-Binomial(n) likelihood of `counts`.

    Deterministic: a small fixed grid of starting kappas at the homogeneous-MLE
    mu (itself computed from `counts`, never from ground truth).
    """
    D = sum(counts[1:])
    if D <= 0: raise ValueError('empty sample requires fold median')
    mean = sum(j*counts[j] for j in range(1, n+1))/D
    mu0 = baselines.activity(mean, n)
    mu0 = min(max(mu0, 1e-6), 1-1e-6)
    starts = [_pack(mu0, k) for k in (0.2, *START_KAPPAS, 2000., 20000.)]
    best, values, _ = _solve(zt_bb_nll(counts, n), starts, BOUNDS)
    if best is None:
        return None, None, 'not_converged', {'starts_disagree': False}, float('nan')
    a, b = _unpack(best.x)
    spread = max(values)-min(values)
    status, flags = diagnose(best.x, BOUNDS, spread, float('nan'))
    return a, b, status, flags, float(best.fun)


def fit_profile_from_counts(counts, n):
    """Fit + W=5 target profile from a visible-window histogram, with the
    homogeneous-binomial fallback on a genuine optimizer failure or a
    starts-disagreement. Shared by fit_rsh and the offline adequacy diagnostics
    so both apply exactly the same fallback rule.
    """
    a, b, status, flags, objective = fit_zt_bb(counts, n)
    if a is None or is_unreliable(status, flags):
        D = sum(counts[1:])
        mean = sum(j*counts[j] for j in range(1, n+1))/D
        q = baselines.activity(mean, n)
        return Result(baselines.profile(q), float('nan'), float('nan'), fit_status=status or 'not_converged',
                      fallback_used=True, objective=float('nan'), flags=flags or {})
    return Result(predict_profile(a, b), a, b, fit_status=status, fallback_used=False,
                  objective=objective, flags=flags)


def fit_rsh(o):
    """R, S, S_obs, H: fit the visible histogram, with S pseudo-weights."""
    counts, m = visible_counts(o)
    return fit_profile_from_counts(counts, m)


def fit_b(o):
    """B: reuse mixtures.fit_events (same activity/ZTP/thinning model); fall back
    to the homogeneous B corrector (baselines.corrector) on genuine failure.
    """
    if o['D_obs'] == 0: raise ValueError('empty sample requires fold median')
    mu0, lam0 = baselines.mixture_start(o)
    fit = mixtures.fit_events(o, mu0, lam0)
    if is_unreliable(fit.status, fit.flags):
        return Result(baselines.corrector(o), float('nan'), float('nan'), lam=float('nan'),
                      fit_status=fit.status, fallback_used=True, objective=float('nan'), flags=dict(fit.flags))
    return Result(fit.prediction, fit.a, fit.b, lam=fit.lam, fit_status=fit.status,
                  fallback_used=False, objective=fit.nll, flags=dict(fit.flags))


def fit(o):
    """Common entrypoint: dispatch on the observation's arm."""
    if o['arm'] == 'B': return fit_b(o)
    if o['arm'] in ('R', 'S', 'S_obs', 'H'): return fit_rsh(o)
    raise ValueError(f'unsupported arm {o["arm"]!r}')
