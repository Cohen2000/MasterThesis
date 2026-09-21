"""Non-LLM references for the persistence profile.

  plugin     profile of the observed dyads' observed patterns.
  corrector  arm-specific working model:
               R  plugin (a node panel keeps complete histories),
               S  raw traversal frequency sum_{j>=k} A_j / L (stationary within
                  the visited component; not unbiased at finite L),
               H  homogeneous zero-truncated Binomial on the retrievable windows,
               B  homogeneous hurdle corrector for event thinning.
  mixture    arm B only: Beta-mixed activity with a ZTP event layer (mixtures.py);
             falls back to the B corrector when the fit is unreliable.
  median     median training-source profile of the fold.
  extratrees_pooled / extratrees_real_only   learned references (training.py).

The primary reference of each arm is PRIMARY_REFERENCE[arm].
"""
import math
from . import mixtures
from .observation import validate

PRIMARY_REFERENCE = {'R': 'corrector', 'S': 'corrector', 'H': 'corrector', 'B': 'mixture'}
PRIMARY_REFERENCE_NAME = {'R': 'plugin_equivalent_corrector', 'S': 'srw_traversal_frequency',
                          'H': 'homogeneous_zero_truncated_binomial', 'B': 'beta_ztp_mixture'}
METHODS = ('plugin', 'corrector', 'median', 'extratrees_pooled', 'extratrees_real_only', 'mixture')


def bisect(fn, target, lo, hi):
    for _ in range(200):
        mid = (lo+hi)/2
        if hi-lo <= 1e-12+1e-10*abs(mid): return mid
        if fn(mid) < target: lo = mid
        else: hi = mid
    raise ArithmeticError('bisection did not converge')


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


def profile(q):
    """rho_2..rho_5 of K ~ Binomial(5, q) conditioned on K >= 1."""
    if q == 0: return [0.]*4
    if q == 1: return [1.]*4
    den = -math.expm1(5*math.log1p(-q))
    return [sum(math.comb(5, j)*q**j*(1-q)**(5-j) for j in range(k, 6))/den for k in range(2, 6)]


def active_windows(o):
    """S = sum over observed dyads of their observed active windows."""
    return sum(r[0].count('1')*r[1] for r in o['table'])


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


def corrector(o):
    validate(o)
    D = o['D_obs']
    if D == 0: raise ValueError('empty sample requires fold median')
    if o['arm'] == 'R': return plugin(o)
    if o['arm'] == 'S':
        A = o['Walk_A']; L = sum(A)
        return [sum(A[k-1:])/L for k in range(2, 6)]
    if o['arm'] == 'H': return h_extrapolator(o)['prediction']
    # B: observed-window activity theta = q * d, with d the probability that an
    # active window keeps at least one of its ZTP(lambda) events at retention p.
    theta = activity(active_windows(o)/D, 5)
    r = _event_rate(o); p = o['parameter']
    d = p if r == 0 else -math.expm1(-r)/-math.expm1(-r/p)
    q = theta/d
    return profile(q) if q <= 1 else [1.]*4


def mixture_start(o):
    """Homogeneous activity and event rate, used as starting values of the B mixture fit."""
    mu = activity(active_windows(o)/o['D_obs'], 5)
    p = o['parameter']
    mean = o['M_obs']/active_windows(o)
    r = 0. if mean <= 1 else bisect(lambda x: 1. if x == 0 else x/-math.expm1(-x), mean, 0., mean)
    return mu, (r/p if p > 0 else float('nan'))


def mixture_reference(o, corrector_prediction):
    fit = mixtures.fit_events(o, *mixture_start(o))
    prediction = list(fit.prediction); fallback = ''
    if mixtures.is_unreliable(fit.status, fit.flags):
        # Fixed rule: an unreliable fit contributes no mixture prediction.
        prediction = list(corrector_prediction); fallback = 'homogeneous_corrector'
    return {'prediction': prediction, 'status': fit.status, 'fallback': fallback, 'flags': dict(fit.flags),
            'seconds': fit.seconds, 'kappa': fit.kappa, 'mu': fit.mu, 'lam': fit.lam,
            'lam_event_only': fit.lam_event_only, 'flat_per_decade': fit.flat_per_decade,
            'nll_spread': fit.nll_spread}


def all_references(o, models, median):
    """Every reference prediction for one observation.

    models = {'pooled': fitted forest, 'real_only': fitted forest} of the
    observation's fold; median = that fold's median training profile. An empty
    observation has no observed dyad, so every reference is the fold median.
    """
    if o['D_obs'] == 0:
        return {m: {'prediction': list(median), 'status': 'empty_sample'} for m in METHODS}
    out = {'plugin': {'prediction': plugin(o), 'status': 'ok'}}
    if o['arm'] == 'H':
        out['corrector'] = h_extrapolator(o)
    else:
        try: out['corrector'] = {'prediction': corrector(o), 'status': 'ok'}
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError) as e:
            out['corrector'] = {'prediction': plugin(o), 'status': f'fallback:{type(e).__name__}'}
        if o['arm'] == 'S':
            out['corrector'].update(model='srw_traversal_frequency', working_model=True)
    out['median'] = {'prediction': list(median), 'status': 'ok'}
    from .observation import features
    x = features(o).reshape(1, -1)
    for name in ('pooled', 'real_only'):
        out['extratrees_'+name] = {'prediction': list(map(float, models[name].predict(x)[0])), 'status': 'ok'}
    if o['arm'] == 'B':
        out['mixture'] = mixture_reference(o, out['corrector']['prediction'])
    return out
