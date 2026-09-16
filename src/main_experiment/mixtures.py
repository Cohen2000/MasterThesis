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
and no regularisation is used anywhere.

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
retains each event independently with the known probability p. For k >= 0,

    Pr{K=k} = sum_{n>=k} [lambda^n e^-lambda / (n! (1-e^-lambda))] C(n,k) p^k (1-p)^(n-k)
            = (p lambda)^k e^(-p lambda) / (k! (1 - e^-lambda)),

so that

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
mass, recovering the existing homogeneous correctors. The upper bound on log kappa
is therefore a genuine model boundary and is reported as such.
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
    """rho_2..rho_5 for K ~ BetaBinomial(5,a,b) conditioned on K>=1."""
    pk=cell_probs(a,b,1.,5)
    den=math.fsum(pk[1:])
    if den<=0: return [0.,0.,0.,0.]
    return [min(1.,max(0.,math.fsum(pk[k:])/den)) for k in range(2,6)]


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
    n_starts: int=0
    seconds: float=0.
    notes: list=field(default_factory=list)


def _classify(z,bounds,spread,flat):
    at=[abs(z[i]-bounds[i][0])<BOUNDARY_TOL or abs(z[i]-bounds[i][1])<BOUNDARY_TOL
        for i in range(len(z))]
    if abs(z[1]-bounds[1][1])<BOUNDARY_TOL: return 'boundary_homogeneous'
    if any(at): return 'boundary_other'
    if spread>AGREEMENT_TOL: return 'starts_disagree'
    if flat<FLAT_DECADE_TOL: return 'weakly_identified'
    return 'converged'


def _solve(nll,starts,bounds):
    best=None; values=[]
    for z0 in starts:
        try: r=minimize(nll,z0,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-12,'gtol':1e-10,'maxiter':500})
        except (FloatingPointError,ValueError): continue
        if not np.isfinite(r.fun): continue
        values.append(float(r.fun))
        if best is None or r.fun<best.fun: best=r
    return best,values


def fit_suffix(o,homogeneous_mu):
    """Candidate 1: zero-truncated Beta-Binomial on the three observed windows."""
    import time; t0=time.perf_counter()
    counts,n=window_counts(o)
    D=sum(counts[1:])
    if D<=0: return Fit([0.,0.,0.,0.],'empty_sample')
    def nll(z):
        a,b=_unpack(z)
        if not (a>0 and b>0 and math.isfinite(a) and math.isfinite(b)): return 1e18
        p=cell_probs(a,b,1.,n)
        tr=math.fsum(p[1:])
        if tr<=0: return 1e18
        s=0.
        for j in range(1,n+1):
            if counts[j]:
                if p[j]<=0: return 1e18
                s+=counts[j]*(math.log(p[j])-math.log(tr))
        return -s
    bounds=[(-LOGIT_BOUND,LOGIT_BOUND),LOG_KAPPA_BOUNDS]
    mu0=min(max(homogeneous_mu,1e-6),1-1e-6)
    starts=[_pack(mu0,k) for k in START_KAPPAS]
    best,values=_solve(nll,starts,bounds)
    if best is None: return Fit([0.,0.,0.,0.],'not_converged',n_starts=len(starts),
                                seconds=time.perf_counter()-t0)
    a,b=_unpack(best.x)
    flat=_flatness(nll,best,bounds,1)
    spread=max(values)-min(values)
    st=_classify(best.x,bounds,spread,flat)
    return Fit(predict_profile(a,b),st,a=a,b=b,mu=a/(a+b),kappa=a+b,nll=float(best.fun),
               nll_spread=spread,flat_per_decade=flat,n_starts=len(starts),
               seconds=time.perf_counter()-t0)


def fit_events(o,homogeneous_mu,homogeneous_lambda):
    """Candidate 2: Beta-mixed activity with the ZTP event layer and known retention p."""
    import time; t0=time.perf_counter()
    counts,n=window_counts(o)
    D=sum(counts[1:]); S=sum(j*counts[j] for j in range(1,n+1)); M=o['M_obs']; p=o['parameter']
    if D<=0: return Fit([0.,0.,0.,0.],'empty_sample')
    def nll(z):
        a,b=_unpack(z[:2]); lam=math.exp(z[2])
        if not (a>0 and b>0 and math.isfinite(a) and math.isfinite(b)): return 1e18
        mu_ev=p*lam
        d=-math.expm1(-mu_ev)/-math.expm1(-lam) if lam>0 else p
        if not (0<d<=1): return 1e18
        pr=cell_probs(a,b,d,n)
        tr=math.fsum(pr[1:])
        if tr<=0: return 1e18
        s=0.
        for j in range(1,n+1):
            if counts[j]:
                if pr[j]<=0: return 1e18
                s+=counts[j]*(math.log(pr[j])-math.log(tr))
        return -s+_ztp_event_nll(M,S,mu_ev)
    bounds=[(-LOGIT_BOUND,LOGIT_BOUND),LOG_KAPPA_BOUNDS,LOG_LAMBDA_BOUNDS]
    mu0=min(max(homogeneous_mu,1e-6),1-1e-6)
    lam0=min(max(homogeneous_lambda,1e-6),1e3)
    starts=[_pack(mu0,k)+[math.log(lam0)] for k in START_KAPPAS]
    best,values=_solve(nll,starts,bounds)
    if best is None: return Fit([0.,0.,0.,0.],'not_converged',n_starts=len(starts),
                                seconds=time.perf_counter()-t0)
    a,b=_unpack(best.x[:2]); lam=math.exp(best.x[2])
    flat=_flatness(nll,best,bounds,1)
    spread=max(values)-min(values)
    st=_classify(best.x,bounds,spread,flat)
    return Fit(predict_profile(a,b),st,a=a,b=b,mu=a/(a+b),kappa=a+b,lam=lam,
               lam_event_only=homogeneous_lambda,nll=float(best.fun),nll_spread=spread,
               flat_per_decade=flat,n_starts=len(starts),seconds=time.perf_counter()-t0)


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
            if np.isfinite(r.fun): rises.append(float(r.fun)-float(best.fun))
        except (FloatingPointError,ValueError): pass
    return min(rises) if rises else float('nan')
