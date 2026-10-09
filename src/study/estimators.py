"""Simple estimators computed from an observation block alone (plug-in, design, working models)."""
import math
from . import thinning_model
from .observation import validate


# In plain words: the simple estimators that need no training.
# - plugin: count, among the observed pairs, the share active in >= k windows. Under the
#   node panel R every pair is equally likely to be observed (no selection bias; as a ratio
#   it is not exactly unbiased); the other arms favour some pairs and bias it.
# - design_estimate: the walk (S) visits busy pairs more often; re-weighting each pair by
#   1/(its number of events) undoes that preference (a ratio estimator).
# - h_extrapolator / corrector: simple working models for H and B.
# The MLE lives in mle.py; ExtraTrees in production/extratrees.py.
#
# Starting point ('anchor') for ExtraTrees: the forest learns a correction to this value.
def anchor_profile(o):
    """Same-information residual-learning anchor for ExtraTrees."""
    if o['arm'] == 'R':
        return plugin(o)
    if o['arm'] in ('S', 'S_obs'): return design_estimate(o)
    if o['arm'] == 'H':
        from .mle import fit
        return fit(o).rho
    try:
        return mixture_reference(o, corrector(o))['prediction']
    except (ArithmeticError, FloatingPointError, OverflowError, ValueError):
        return corrector(o)


def bisect(fn, target, lo, hi):
    for _ in range(200):
        mid = (lo+hi)/2
        if hi-lo <= 1e-12+1e-10*abs(mid): return mid
        if fn(mid) < target: lo = mid
        else: hi = mid
    raise ArithmeticError('bisection did not converge')


# Helpers for the Binomial working model: a pair is active in each window with probability q.
def zbin_mean(q, n):
    """Mean of a zero-truncated Binomial(n, q)."""
    if q == 0: return 1.
    if q == 1: return float(n)
    return n*q/-math.expm1(n*math.log1p(-q))


def activity(mean, n):
    """q whose zero-truncated Binomial(n, q) mean equals `mean` (the MLE)."""
    if not 1 <= mean <= n: raise ValueError('inconsistent S/D')
    if mean == 1: return 0.
    if mean == n: return 1.
    return bisect(lambda q: zbin_mean(q, n), mean, 0., 1.)


# Convert an activity probability q into rho_2..rho_5 for pairs seen at least once.
def profile(q):
    """rho_2..rho_5 of K ~ Binomial(5, q) conditioned on K >= 1."""
    if q == 0: return [0.]*4
    if q == 1: return [1.]*4
    den = -math.expm1(5*math.log1p(-q))
    return [sum(math.comb(5, j)*q**j*(1-q)**(5-j) for j in range(k, 6))/den for k in range(2, 6)]


def active_windows(o):
    """S = sum over observed dyads of their observed active windows."""
    return sum(r[0].count('1')*r[1] for r in o['table'])


# Plug-in (naive) estimate: share of observed pairs active in >= k observed windows.
def plugin(o):
    if o['D_obs'] == 0: raise ValueError('empty sample requires fold median')
    return [sum(r[1] for r in o['table'] if r[0].count('1') >= k)/o['D_obs'] for k in range(2, 6)]


def h_extrapolator(o):
    """Homogeneous zero-truncated Binomial working model of arm H (not unbiased).

    J | J>0 ~ Binomial(n_visible, q) truncated at zero; the MLE solves
    n q / (1-(1-q)^n) = mean(J). The full-archive profile then uses K ~ Binomial(5, q).
    """
    validate(o)
    if o['arm'] != 'H' or not o['D_obs']: raise ValueError('nonempty H input required')
    n = sum(o['Temporal_access'])
    q = activity(active_windows(o)/o['D_obs'], n)
    return {'prediction': profile(q), 'q': q, 'visible_windows': n,
            'status': 'boundary_zero' if q == 0 else 'boundary_one' if q == 1 else 'ok',
            'model': 'homogeneous_zero_truncated_binomial', 'working_model': True}


def _event_rate(o):
    """Poisson rate r of a zero-truncated Poisson with mean M_obs/S (0 if every window has one event)."""
    mean = o['M_obs']/active_windows(o)
    if mean < 1: raise ValueError('inconsistent M/S')
    if mean == 1: return 0.
    return bisect(lambda r: 1. if r == 0 else r/-math.expm1(-r), mean, 0., mean)


# Simple arm-specific correction: plugin for R, design weights for S, the Binomial
# extrapolation for H, and for B a thinning correction (a window with events can lose all
# of them when events are dropped, so observed activity is scaled back up).
def corrector(o):
    validate(o)
    D = o['D_obs']
    if D == 0: raise ValueError('empty sample requires fold median')
    if o['arm'] == 'R': return plugin(o)
    if o['arm'] in ('S', 'S_obs'): return design_estimate(o)
    if o['arm'] == 'H': return h_extrapolator(o)['prediction']
    # B: observed-window activity theta = q * d, with d the probability that an
    # active window keeps at least one of its ZTP(lambda) events at retention p.
    theta = activity(active_windows(o)/D, 5)
    r = _event_rate(o); p = o['parameter']
    d = p if r == 0 else -math.expm1(-r)/-math.expm1(-r/p)
    q = theta/d
    return profile(q) if q <= 1 else [1.]*4


# Walk arm: weighted share, using the inverse-event weights released in the observation.
# Reference: inverse-probability weighting, Hansen & Hurwitz (1943), with walk visit weights
# (Ribeiro & Towsley 2010). Descriptive only; not exactly unbiased for a finite walk.
def design_estimate(o):
    """S-arm ratio estimate from released per-pattern weights."""
    if o['arm'] not in ('S', 'S_obs'): raise ValueError('S arm required')
    col = 5 if o['arm'] == 'S' else 3
    total = sum(row[col] for row in o['table'])
    if total <= 0: raise ValueError('empty observation')
    return [sum(row[col] for row in o['table'] if row[0].count('1') >= k)/total for k in range(2, 6)]


def mixture_start(o):
    """Homogeneous activity and event rate, used as starting values of the B mixture fit."""
    mu = activity(active_windows(o)/o['D_obs'], 5)
    p = o['parameter']
    mean = o['M_obs']/active_windows(o)
    r = 0. if mean <= 1 else bisect(lambda x: 1. if x == 0 else x/-math.expm1(-x), mean, 0., mean)
    return mu, (r/p if p > 0 else float('nan'))


# B arm: fit the event-thinning mixture model; fall back to the simple corrector if the
# optimiser does not converge.
def mixture_reference(o, corrector_prediction):
    fit = thinning_model.fit_events(o, *mixture_start(o))
    prediction = list(fit.prediction); fallback = ''
    if fit.status == 'not_converged':
        prediction = list(corrector_prediction); fallback = 'homogeneous_corrector'
    return {'prediction': prediction, 'status': fit.status, 'fallback': fallback,
            'flags': dict(fit.flags),
            'seconds': fit.seconds, 'kappa': fit.kappa, 'mu': fit.mu, 'lam': fit.lam,
            'lam_event_only': fit.lam_event_only, 'flat_per_decade': fit.flat_per_decade,
            'nll_spread': fit.nll_spread}
