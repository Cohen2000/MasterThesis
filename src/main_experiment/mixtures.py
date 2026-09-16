"""Two Beta-mixture development candidates for the suffix and event-sampling arms.

Model family after Dorazio & Royle (2003), Mixture Models for Estimating the Size
of a Closed Population When Capture Rates Vary among Individuals,
https://doi.org/10.1111/1541-0420.00042. That paper motivates a Beta mixture over
individual detection rates; the coupling to our samplers and to the persistence
profile below is our own derivation, and it carries no guarantee about MAE.

Shared latent structure
-----------------------
Each dyad e carries q_e ~ Beta(a,b). Given q_e the five archive windows are
conditionally independent and exchangeable with activity probability q_e, so the
true number of active windows is K_e | q_e ~ Bin(5, q_e). The estimand is the
full-archive profile conditioned on at least one truly active window,

    rho_k = Pr{K >= k} / Pr{K >= 1},   K ~ BetaBinomial(5, a, b),  k = 2..5,

which is what predict_profile returns. It is deliberately NOT the observed profile.

Cell probabilities
------------------
With a per-window observation probability q*d, the count of observed-active
windows out of n satisfies Pr{J=j} = C(n,j) E[(qd)^j (1-qd)^(n-j)]. Substituting
1 - qd = (1-q) + q(1-d) gives the all-positive form

    Pr{J=j} = C(n,j) d^j sum_{s=0}^{n-j} C(n-j,s) (1-d)^(n-j-s) E[q^(n-s)(1-q)^s],
    E[q^r (1-q)^s] = B(a+r,b+s)/B(a,b) = prod(a+t)_r prod(b+t)_s / prod(a+b+t)_(r+s),

in which no term is ever subtracted, so the result carries full precision for
every (a,b,d). The equivalent expansion in the plain moments E[q^r] is the same
identity but cancels catastrophically when qd concentrates near one. Checked
against exact rational arithmetic over a wide (a,b,d) grid in
tests/frozen_main/test_mixtures.py; worst relative error 8.3e-16. No quadrature
and no regularisation is used anywhere. predict_profile does clamp its four
outputs into [0,1]; the clamp is a guard against last-bit excursions only and
raises if it ever has to move a value by more than CLAMP_TOL.

Candidate 1 -- suffix (arm H), d = 1
------------------------------------
The suffix arm observes windows 3,4,5 in full: every event in an observed window
is retained, so there is no thinning and d = 1. A dyad appears in the table iff at
least one of the three observed windows is active, hence

    J | J>=1  ~  zero-truncated BetaBinomial(3, a, b).

Sufficiency: the pattern table already gives n_j, the number of dyads with exactly
j active observed windows. Under this model the event layer is independent of q
given activity, so the event counts carry no additional information about (a,b)
and are correctly unused. The table alone is sufficient for this likelihood.

Identifiability: conditioned on J>=1 the data are two free cell probabilities and
the model has two parameters, so the fit is exactly saturated -- zero degrees of
freedom and no possible goodness-of-fit test. Data that are under-dispersed
relative to a Binomial lie outside the Beta-Binomial family and drive the fit to
the homogeneous boundary. Extrapolating a shape estimated on three windows to a
five-window profile is an untestable extrapolation. Both are reported, not hidden.

Candidate 2 -- Bernoulli event sampling (arm B), d = d(lambda,p)
---------------------------------------------------------------
Given a truly active window the true event count is N ~ ZTP(lambda), and arm B
retains each event independently with the known probability p. For k >= 1,

    Pr{K=k} = sum_{n>=k} [lambda^n e^-lambda / (n! (1-e^-lambda))] C(n,k) p^k (1-p)^(n-k)
            = (p lambda)^k e^(-p lambda) / (k! (1 - e^-lambda)).

The closed form does NOT extend to k = 0: summing the same series from n = 1
omits the n = 0 term of the untruncated Poisson, so

    Pr{K=0} = (e^(lambda(1-p)) - 1) / (e^lambda - 1),

which is the k = 0 expression above minus 1/(e^lambda - 1). Only k >= 1 enters
anywhere below, and d(lambda,p) = 1 - Pr{K=0} reduces to the same expression
either way, so the distinction is a statement about the formula rather than
about any computed quantity. It is pinned by a test.

With that,

  (i)  Pr{window observed active | truly active} = (1-e^(-p lambda))/(1-e^-lambda)
       = d(lambda,p), exactly the retention factor of the task statement;
  (ii) Pr{K=k | truly active, K>=1} = ZTP(p lambda) exactly.

From (ii) the retained events of an observed-active window are ZTP(p lambda),
whose sufficient statistic is the sample mean. Therefore M_obs / S is sufficient
for lambda given S, and the event *sums* in the table suffice: per-dyad event
counts are never needed. From (i), Pr{window observed active | q} = q d, so
J | q ~ Bin(5, q d) and the sample is conditioned on J >= 1.

The working log-likelihood, up to a constant free of the parameters, is

  l(a,b,lambda) = sum_j n_j log[C(5,j) E((qd)^j (1-qd)^(5-j))]
                  - D_obs log[1 - E((1-qd)^5)]
                  + M log(p lambda) - S p lambda - S log(1 - e^(-p lambda)).

The dropped constant is -sum log(k_i!), which depends only on the unobserved
per-dyad counts and not on the parameters, so it does not move the argmax.
The two parts are coupled through lambda because d depends on lambda. This is
precisely why lambda may not be estimated from the event part alone and then
substituted: fit_events optimises (a,b,lambda) jointly and reports the
event-only lambda alongside, so the difference is visible rather than assumed.

Homogeneous limit
-----------------
kappa = a+b -> infinity at fixed mu = a/(a+b) collapses the mixture onto a point
mass, recovering the existing homogeneous correctors. The limit is approached at
order 1/kappa, so the upper bound kappa = 1e6 is an approximation to that
boundary and not the boundary itself: at mu = 0.3, d = 0.7 the cell probabilities
still differ from the exact Binomial by about 6e-7. A fit that lands there is
reported as boundary_homogeneous, meaning "numerically indistinguishable from the
homogeneous model at this sample size", not "exactly homogeneous".
"""
import math
from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import minimize

# Fixed before any performance check; see docs/MAIN_EXPERIMENT_IMPLEMENTATION.md.
LOGIT_BOUND=12.0                    # mu in [6.1e-6, 1-6.1e-6]
LOG_KAPPA_BOUNDS=(math.log(1e-3),math.log(1e6))   # 1e6 is the homogeneous boundary
LOG_LAMBDA_BOUNDS=(math.log(1e-6),math.log(1e3))
BOUNDARY_TOL=1e-6                   # distance in transformed units that counts as "at the bound"
AGREEMENT_TOL=1e-4                  # nll spread across starts that still counts as agreement
FLAT_DECADE_TOL=0.5                 # profile nll rise per decade of kappa below which
                                    # the concentration is called weakly identified
START_KAPPAS=(2.,20.,200.)
PENALTY=1e18                        # objective value returned on the invalid region
CLAMP_TOL=1e-9                      # a larger clamp is an error, not a repair
# Pre-registered before the main evaluation: an observation whose fit did not
# converge, or whose starts disagree, does not use the mixture prediction at all;
# it falls back to the existing homogeneous corrector and is counted. Boundary and
# weak-identifiability cases keep their fit, because there the fitted value is the
# answer the model actually implies. The condition is evaluated by is_unreliable()
# on the independent flags, never on the summary label.
UNRELIABLE_STATUSES=('not_converged','starts_disagree')
COMB=[[math.comb(n,k) for k in range(n+1)] for n in range(6)]


def joint_moment(a,b,r,s):
    """E[q^r (1-q)^s] = B(a+r,b+s)/B(a,b), as a product of positive factors."""
    v=1.
    for t in range(r): v*=(a+t)
    for t in range(s): v*=(b+t)
    for t in range(r+s): v/=(a+b+t)
    return v


def cell_probs(a,b,d,n):
    """Pr{J=j}, j=0..n, for J|q ~ Bin(n, q d) with q ~ Beta(a,b). Exact and positive.

    Writing 1 - qd = (1-q) + q(1-d) turns the binomial expansion into

        E[(qd)^j (1-qd)^(n-j)]
            = d^j sum_{s=0}^{n-j} C(n-j,s) (1-d)^(n-j-s) E[q^(n-s) (1-q)^s],

    in which every factor is non-negative, so no cancellation can occur for any
    (a,b,d). The naive alternating form in the moments m_r is mathematically the
    same identity but loses all significant digits when qd concentrates near one:
    at a=1e5, b=1e-2, d=1 it returns 6.7e-16 for Pr{J=0} where the true value is
    2.5e-26. Checked against exact rational arithmetic in
    tests/frozen_main/test_mixtures.py.
    """
    out=[]
    for j in range(n+1):
        acc=0.
        for s in range(n-j+1):
            acc+=COMB[n-j][s]*(1-d)**(n-j-s)*joint_moment(a,b,n-s,s)
        out.append(COMB[n][j]*d**j*acc)
    return out


def predict_profile(a,b):
    """rho_2..rho_5 for K ~ BetaBinomial(5,a,b) conditioned on K>=1.

    The clamp guards against last-bit excursions outside [0,1] only. Anything
    larger would be a genuine numerical failure and raises instead of being
    silently repaired.
    """
    pk=cell_probs(a,b,1.,5)
    den=math.fsum(pk[1:])
    if den<=0: return [0.,0.,0.,0.]
    raw=[math.fsum(pk[k:])/den for k in range(2,6)]
    out=[]
    for x in raw:
        y=min(1.,max(0.,x))
        if abs(y-x)>CLAMP_TOL: raise ArithmeticError(f'profile outside [0,1] by {x-y:.3e}')
        out.append(y)
    return out


def _pack(mu,kappa): return [math.log(mu/(1-mu)),math.log(kappa)]
def _unpack(z):
    mu=1/(1+math.exp(-z[0])); kappa=math.exp(z[1]); return mu*kappa,(1-mu)*kappa


def window_counts(o):
    """n_j = number of observed dyads with exactly j active observed windows."""
    n=3 if o['arm']=='H' else 5
    c=[0]*(n+1)
    for pat,d,_ in o['table']: c[pat.count('1')]+=d
    return c,n


def _ztp_event_nll(M,S,mu):
    """-log L of S iid ZTP(mu) with total M, dropping the parameter-free factorials.

    Written so that the M=S, mu->0 corner stays finite instead of -inf + inf.
    """
    if S<=0: return 0.
    return -((M-S)*math.log(mu)-S*mu-S*(math.log(-math.expm1(-mu))-math.log(mu)))


@dataclass
class Fit:
    prediction: list
    status: str
    a: float=float('nan')
    b: float=float('nan')
    mu: float=float('nan')
    kappa: float=float('nan')
    lam: float=float('nan')
    lam_event_only: float=float('nan')
    nll: float=float('nan')
    nll_spread: float=float('nan')
    flat_per_decade: float=float('nan')
    flags: dict=field(default_factory=dict)
    n_starts: int=0
    n_accepted: int=0
    seconds: float=0.
    notes: list=field(default_factory=list)


def diagnose(z,bounds,spread,flat):
    """Independent diagnostic flags, plus a label derived from them.

    These conditions are not mutually exclusive and must not be reported as if
    they were: a fit can sit on a parameter bound *and* have disagreeing starts.
    An earlier version returned the first matching label, so a boundary hit
    silently hid a disagreement, and the fallback rule -- which keys on the
    disagreement -- never fired for those fits. The flags are therefore computed
    independently and the label is only a summary of them.
    """
    hom=abs(z[1]-bounds[1][1])<BOUNDARY_TOL
    at_any=any(abs(z[i]-bounds[i][0])<BOUNDARY_TOL or abs(z[i]-bounds[i][1])<BOUNDARY_TOL
               for i in range(len(z)))
    flags={'boundary_homogeneous':bool(hom),
           'boundary_other':bool(at_any and not hom),
           'starts_disagree':bool(spread>AGREEMENT_TOL),
           # flat is NaN when every profile re-optimisation failed; that is an
           # absent diagnosis, not evidence of good identifiability.
           'flatness_unavailable':bool(flat!=flat),
           'weakly_identified':bool(flat==flat and flat<FLAT_DECADE_TOL)}
    if flags['boundary_homogeneous']: label='boundary_homogeneous'
    elif flags['boundary_other']: label='boundary_other'
    elif flags['starts_disagree']: label='starts_disagree'
    elif flags['weakly_identified']: label='weakly_identified'
    elif flags['flatness_unavailable']: label='flatness_unavailable'
    else: label='converged'
    return label,flags


def is_unreliable(status,flags):
    """The pre-registered fallback condition, evaluated on the flags.

    Reading it off the summary label would reinstate exactly the masking bug:
    a disagreeing fit that also sits on a bound is labelled boundary_* but is
    still unreliable.
    """
    return status=='not_converged' or bool(flags.get('starts_disagree'))


def _solve(nll,starts,bounds):
    """Run every start and accept only results that are genuinely optimiser output.

    A finite objective is not sufficient evidence of a fit: the invalid region
    returns PENALTY, which is finite, so a run that never left it would otherwise
    be accepted and would win whenever every start failed. Acceptance therefore
    requires that the optimiser reports success and that the objective is not in
    the penalty region. Rejections are returned rather than dropped.
    """
    best=None; values=[]; rejected=[]
    for z0 in starts:
        try:
            r=minimize(nll,z0,method='L-BFGS-B',bounds=bounds,
                       options={'ftol':1e-12,'gtol':1e-10,'maxiter':500})
        except (FloatingPointError,ValueError,ArithmeticError) as e:
            rejected.append(f'exception:{type(e).__name__}'); continue
        if not np.isfinite(r.fun): rejected.append('nonfinite_objective'); continue
        if r.fun>=PENALTY/2: rejected.append('penalty_region'); continue
        if not r.success:
            rejected.append(f'status{int(r.status)}:{str(r.message)[:40]}'); continue
        values.append(float(r.fun))
        if best is None or r.fun<best.fun: best=r
    return best,values,rejected


def _widen(bounds,decades):
    """Widen every box constraint by the given number of decades in log space."""
    out=[]
    for i,(lo,hi) in enumerate(bounds):
        d=decades*math.log(10)
        out.append((lo-d,hi+d) if i else (lo-decades,hi+decades))
    return out


def bound_sensitivity(o,mu0,lam0=None,decades=2.0):
    """How much does the answer depend on the artificial parameter box?

    Refits with every bound widened by `decades` and reports the movement of the
    objective and of the predicted persistence. A fit that is pinned by a bound
    shows up here as a large profile shift, which is a problem to report rather
    than a result to keep.
    """
    base=fit_events(o,mu0,lam0) if lam0 is not None else fit_suffix(o,mu0)
    wide=(fit_events(o,mu0,lam0,decades=decades) if lam0 is not None
          else fit_suffix(o,mu0,decades=decades))
    if base.status=='empty_sample' or wide.status=='empty_sample':
        return {'status':'empty_sample'}
    shift=max(abs(x-y) for x,y in zip(base.prediction,wide.prediction))
    return {'status':base.status,'status_widened':wide.status,
            'nll':base.nll,'nll_widened':wide.nll,
            'nll_improvement':float(base.nll-wide.nll),
            'max_profile_shift':float(shift),
            'kappa':base.kappa,'kappa_widened':wide.kappa}


def fit_suffix(o,homogeneous_mu,decades=0.):
    """Candidate 1: zero-truncated Beta-Binomial on the three observed windows."""
    import time; t0=time.perf_counter()
    counts,n=window_counts(o)
    D=sum(counts[1:])
    if D<=0: return Fit([0.,0.,0.,0.],'empty_sample')
    def nll(z):
        a,b=_unpack(z)
        if not (a>0 and b>0 and math.isfinite(a) and math.isfinite(b)): return PENALTY
        p=cell_probs(a,b,1.,n)
        tr=math.fsum(p[1:])
        if tr<=0: return PENALTY
        s=0.
        for j in range(1,n+1):
            if counts[j]:
                if p[j]<=0: return PENALTY
                s+=counts[j]*(math.log(p[j])-math.log(tr))
        return -s
    bounds=_widen([(-LOGIT_BOUND,LOGIT_BOUND),LOG_KAPPA_BOUNDS],decades) if decades else [(-LOGIT_BOUND,LOGIT_BOUND),LOG_KAPPA_BOUNDS]
    mu0=min(max(homogeneous_mu,1e-6),1-1e-6)
    starts=[_pack(mu0,k) for k in START_KAPPAS]
    best,values,rejected=_solve(nll,starts,bounds)
    if best is None:
        return Fit([0.,0.,0.,0.],'not_converged',n_starts=len(starts),
                   n_accepted=0,notes=rejected,seconds=time.perf_counter()-t0)
    a,b=_unpack(best.x)
    flat=_flatness(nll,best,bounds,1)
    spread=max(values)-min(values)
    st,flags=diagnose(best.x,bounds,spread,flat)
    return Fit(predict_profile(a,b),st,flags=flags,a=a,b=b,mu=a/(a+b),kappa=a+b,nll=float(best.fun),
               nll_spread=spread,flat_per_decade=flat,n_starts=len(starts),
               n_accepted=len(values),notes=rejected,seconds=time.perf_counter()-t0)


def fit_events(o,homogeneous_mu,homogeneous_lambda,decades=0.):
    """Candidate 2: Beta-mixed activity with the ZTP event layer and known retention p."""
    import time; t0=time.perf_counter()
    counts,n=window_counts(o)
    D=sum(counts[1:]); S=sum(j*counts[j] for j in range(1,n+1)); M=o['M_obs']; p=o['parameter']
    if D<=0: return Fit([0.,0.,0.,0.],'empty_sample')
    def nll(z):
        a,b=_unpack(z[:2]); lam=math.exp(z[2])
        if not (a>0 and b>0 and math.isfinite(a) and math.isfinite(b)): return PENALTY
        mu_ev=p*lam
        d=-math.expm1(-mu_ev)/-math.expm1(-lam) if lam>0 else p
        if not (0<d<=1): return PENALTY
        pr=cell_probs(a,b,d,n)
        tr=math.fsum(pr[1:])
        if tr<=0: return PENALTY
        s=0.
        for j in range(1,n+1):
            if counts[j]:
                if pr[j]<=0: return PENALTY
                s+=counts[j]*(math.log(pr[j])-math.log(tr))
        return -s+_ztp_event_nll(M,S,mu_ev)
    bounds=[(-LOGIT_BOUND,LOGIT_BOUND),LOG_KAPPA_BOUNDS,LOG_LAMBDA_BOUNDS]
    if decades: bounds=_widen(bounds,decades)
    mu0=min(max(homogeneous_mu,1e-6),1-1e-6)
    lam0=min(max(homogeneous_lambda,1e-6),1e3)
    starts=[_pack(mu0,k)+[math.log(lam0)] for k in START_KAPPAS]
    best,values,rejected=_solve(nll,starts,bounds)
    if best is None:
        return Fit([0.,0.,0.,0.],'not_converged',n_starts=len(starts),
                   n_accepted=0,notes=rejected,seconds=time.perf_counter()-t0)
    a,b=_unpack(best.x[:2]); lam=math.exp(best.x[2])
    flat=_flatness(nll,best,bounds,1)
    spread=max(values)-min(values)
    st,flags=diagnose(best.x,bounds,spread,flat)
    return Fit(predict_profile(a,b),st,flags=flags,a=a,b=b,mu=a/(a+b),kappa=a+b,lam=lam,
               lam_event_only=homogeneous_lambda,nll=float(best.fun),nll_spread=spread,
               flat_per_decade=flat,n_starts=len(starts),n_accepted=len(values),
               notes=rejected,seconds=time.perf_counter()-t0)


def _flatness(nll,best,bounds,index):
    """Profile rise of the nll one decade away in the given coordinate.

    A success flag says nothing about identifiability. This re-optimises every
    other coordinate with the chosen one displaced by a decade and reports the
    smaller of the two rises; a small value means the likelihood is flat and the
    concentration is only weakly identified.
    """
    rises=[]
    for step in (-math.log(10),math.log(10)):
        z=list(best.x); z[index]=min(max(z[index]+step,bounds[index][0]),bounds[index][1])
        if abs(z[index]-best.x[index])<1e-9: continue
        free=[i for i in range(len(z)) if i!=index]
        def sub(v):
            w=list(z)
            for i,k in enumerate(free): w[k]=v[i]
            return nll(w)
        try:
            r=minimize(sub,[best.x[i] for i in free],method='L-BFGS-B',
                       bounds=[bounds[i] for i in free],options={'ftol':1e-12,'maxiter':300})
        except (FloatingPointError,ValueError,ArithmeticError):
            continue
        # A failed or penalty-region profile optimisation says nothing about
        # identifiability; treating it as a rise would report a flat likelihood
        # as well determined, or a failed one as flat.
        if not r.success or not np.isfinite(r.fun) or r.fun>=PENALTY/2:
            continue
        rises.append(float(r.fun)-float(best.fun))
    return min(rises) if rises else float('nan')
