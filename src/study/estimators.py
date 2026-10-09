"""The estimators that need no training: observed share, re-weighted share and the statistical model (MLE).

All of them read only the parsed sample text `o` and return rho_2..rho_5.

The MLE assumes that every pair has its own probability q of being active in a window, with q
drawn from a Beta(a, b) distribution. It fits a and b to the numbers of pairs seen with 1, 2, ...
active windows and reads rho_k off the fitted distribution: rho_k = P(K >= k | K >= 1) for
K ~ BetaBinomial(5, a, b). Pairs that were never seen are handled by leaving out K = 0.
The arms differ only in what enters the fit:
  R  the pairs as seen;
  S  each pair weighted by visits / events, which undoes the walk's preference for busy pairs;
  H  the three visible windows; the same a, b are then used for all five;
  B  a window with events can lose all of them, so the model adds the number of events per
     active window (at least one, Poisson) and the known keep probability p.
Beta mixture: Dorazio & Royle (2003). Weights: Hansen & Hurwitz (1943), Pfeffermann (1993).
"""
import math
import numpy as np
from scipy.optimize import minimize

MU_BOUND = (-12., 12.)                              # logit of the mean a/(a+b)
SIZE_BOUND = (math.log(1e-3), math.log(1e6))        # log of a+b; 1e6 means: all pairs alike
RATE_BOUND = (math.log(1e-6), math.log(1e3))        # log of the event rate (arm B)
INVALID = 1e18                                      # value of the objective outside the model
COMB = [[math.comb(n, k) for k in range(n+1)] for n in range(6)]


# ---------------------------------------------------------------- simple estimators
def plugin(o):
    """Observed share: the share of seen pairs that are active in at least k seen windows."""
    return [sum(r[1] for r in o['table'] if r[0].count('1') >= k)/o['D_obs'] for k in range(2, 6)]


def ratio(o):
    """Arm S, shown for comparison only: the share with each pair weighted by visits / events."""
    total = sum(r[5] for r in o['table'])
    return [sum(r[5] for r in o['table'] if r[0].count('1') >= k)/total for k in range(2, 6)]


# ---------------------------------------------------------------- all pairs alike (start values and fallback)
def bisect(fn, target, lo, hi):
    for _ in range(200):
        mid = (lo+hi)/2
        if hi-lo <= 1e-12+1e-10*abs(mid): return mid
        if fn(mid) < target: lo = mid
        else: hi = mid
    raise ArithmeticError('bisection did not converge')


def activity(mean, n):
    """The q for which a pair seen at least once is active in `mean` of n windows on average."""
    if not 1 <= mean <= n: raise ValueError('inconsistent counts')
    if mean == 1: return 0.
    if mean == n: return 1.
    return bisect(lambda q: 1. if q == 0 else float(n) if q == 1 else n*q/-math.expm1(n*math.log1p(-q)), mean, 0., 1.)


def profile(q):
    """rho_2..rho_5 if every pair is active with the same probability q."""
    if q == 0: return [0.]*4
    if q == 1: return [1.]*4
    den = -math.expm1(5*math.log1p(-q))
    return [sum(math.comb(5, j)*q**j*(1-q)**(5-j) for j in range(k, 6))/den for k in range(2, 6)]


def _windows(o):
    """Active windows, summed over the seen pairs."""
    return sum(r[0].count('1')*r[1] for r in o['table'])


def _event_rate(o):
    """Rate of a Poisson that is at least 1 and has the seen mean events per active window."""
    mean = o['M_obs']/_windows(o)
    if mean <= 1: return 0.
    return bisect(lambda r: 1. if r == 0 else r/-math.expm1(-r), mean, 0., mean)


def simple_b(o):
    """Arm B with all pairs alike: the seen activity, scaled up by the chance that an active
    window keeps at least one event."""
    if o['M_obs'] < _windows(o): raise ValueError('inconsistent counts')
    theta = activity(_windows(o)/o['D_obs'], 5)
    r, p = _event_rate(o), o['parameter']
    q = theta/(p if r == 0 else -math.expm1(-r)/-math.expm1(-r/p))
    return profile(q) if q <= 1 else [1.]*4


# ---------------------------------------------------------------- the Beta model
def _moment(a, b, r, s):
    """E[q^r (1-q)^s] for q ~ Beta(a, b), as a product of positive factors."""
    v = 1.
    for t in range(r): v *= (a+t)
    for t in range(s): v *= (b+t)
    for t in range(r+s): v /= (a+b+t)
    return v


def cell_probs(a, b, d, n):
    """P(J = j), j = 0..n, when a pair is seen active in a window with probability q*d.
    Written without subtractions, so it is exact also at the edges."""
    out = []
    for j in range(n+1):
        acc = 0.
        for s in range(n-j+1): acc += COMB[n-j][s]*(1-d)**(n-j-s)*_moment(a, b, n-s, s)
        out.append(COMB[n][j]*d**j*acc)
    return out


def beta_profile(a, b):
    """rho_2..rho_5 of the fitted model."""
    pk = cell_probs(a, b, 1., 5)
    den = math.fsum(pk[1:])
    if den <= 0: return [0.]*4
    out = [math.fsum(pk[k:])/den for k in range(2, 6)]
    if any(not -1e-9 <= x <= 1+1e-9 for x in out): raise ArithmeticError('profile outside [0, 1]')
    return [min(1., max(0., x)) for x in out]


def _ab(z):
    mu, size = 1/(1+math.exp(-z[0])), math.exp(z[1])
    return mu*size, (1-mu)*size


def _best(objective, starts, bounds):
    """The best of several L-BFGS-B runs (Byrd et al. 1995); None if none converged."""
    best = None
    for z0 in starts:
        try:
            r = minimize(objective, z0, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 500})
        except (ValueError, ArithmeticError):
            continue
        if np.isfinite(r.fun) and r.fun < INVALID/2 and r.success and (best is None or r.fun < best.fun): best = r
    return best


def _log_likelihood(counts, pr, n):
    """Log-likelihood of the pairs with j = 1..n active windows; None outside the model."""
    seen = math.fsum(pr[1:])
    if seen <= 0: return None
    s = 0.
    for j in range(1, n+1):
        if counts[j]:
            if pr[j] <= 0: return None
            s += counts[j]*(math.log(pr[j])-math.log(seen))
    return s


def mle_counts(counts, n):
    """Fit to counts[j] = (weighted) number of pairs with j of n visible windows active."""
    D = sum(counts[1:])
    q = activity(sum(j*counts[j] for j in range(1, n+1))/D, n)

    def objective(z):
        a, b = _ab(z)
        if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)): return INVALID
        s = _log_likelihood(counts, cell_probs(a, b, 1., n), n)
        return INVALID if s is None else -s
    q0 = min(max(q, 1e-6), 1-1e-6)
    best = _best(objective, [[math.log(q0/(1-q0)), math.log(size)] for size in (.2, 2., 20., 200., 2000., 20000.)], [MU_BOUND, SIZE_BOUND])
    return profile(q) if best is None else beta_profile(*_ab(best.x))


def mle_b(o):
    """Arm B: activity and events per active window are fitted together, given the keep probability p."""
    counts = [0]*6
    for row in o['table']: counts[row[0].count('1')] += row[1]
    D, S, M, p = o['D_obs'], _windows(o), o['M_obs'], o['parameter']
    rate = _event_rate(o)/p

    def objective(z):
        a, b = _ab(z); lam = math.exp(z[2])
        if not (a > 0 and b > 0 and math.isfinite(a) and math.isfinite(b)): return INVALID
        d = -math.expm1(-p*lam)/-math.expm1(-lam) if lam > 0 else p      # an active window keeps an event
        if not 0 < d <= 1: return INVALID
        s = _log_likelihood(counts, cell_probs(a, b, d, 5), 5)
        if s is None: return INVALID
        mu = p*lam                                                       # kept events per seen window
        return -s-((M-S)*math.log(mu)-S*mu-S*(math.log(-math.expm1(-mu))-math.log(mu)))
    q = min(max(activity(S/D, 5), 1e-6), 1-1e-6)
    best = _best(objective, [[math.log(q/(1-q)), math.log(size), math.log(min(max(rate, 1e-6), 1e3))] for size in (2., 20., 200.)],
                 [MU_BOUND, SIZE_BOUND, RATE_BOUND])
    return simple_b(o) if best is None else beta_profile(*_ab(best.x[:2]))


def mle(o):
    """The statistical model for any arm."""
    if o['arm'] == 'B': return mle_b(o)
    n = len(o['table'][0][0].replace('?', ''))                           # visible windows
    counts = [0.]*(n+1)
    for row in o['table']: counts[row[0].count('1')] += row[5] if o['arm'] == 'S' else row[1]
    return mle_counts(counts, n)
