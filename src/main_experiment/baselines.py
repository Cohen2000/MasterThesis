import math
import numpy as np
from .common import LEGACY_H
from .observation import validate

def bisect(fn,target,lo,hi):
    for _ in range(200):
        mid=(lo+hi)/2
        if hi-lo<=1e-12+1e-10*abs(mid): return mid
        if fn(mid)<target: lo=mid
        else: hi=mid
    raise ArithmeticError('bisection did not converge')

def zbin_mean(q,n):
    if q==0: return 1.
    if q==1: return float(n)
    return n*q/-math.expm1(n*math.log1p(-q))

def activity(mean,n):
    if not 1<=mean<=n: raise ValueError('inconsistent S/D')
    if mean==1: return 0.
    if mean==n: return 1.
    return bisect(lambda q:zbin_mean(q,n),mean,0.,1.)

def profile(q):
    if q==0: return [0.]*4
    if q==1: return [1.]*4
    den=-math.expm1(5*math.log1p(-q))
    return [sum(math.comb(5,j)*q**j*(1-q)**(5-j) for j in range(k,6))/den for k in range(2,6)]

def plugin(o):
    if o['D_obs']==0: raise ValueError('empty sample requires fold median')
    return [sum(r[1] for r in o['table'] if r[0].count('1')>=k)/o['D_obs'] for k in range(2,6)]


def h_bounds(o):
    """Per-dyad bounds on K for the recent-cap arm, aggregated over the sample.

    A listed dyad with fewer than five retrieved events has its complete history,
    so K = J, its number of observed active windows. A dyad with exactly five
    retrieved events has a complete history from its earliest observed window b
    onwards -- every later event is more recent than the retrieved one in b, so it
    was retrieved too -- but windows 1..b-1 are unknown. Hence J <= K <= J+b-1.
    L_k and U_k are the shares of sampled dyads whose lower and upper bound reach
    k. They bound the profile of the *sampled dyads*, not the full archive.
    """
    validate(o)
    if o['arm']!='H': raise ValueError('bounds exist only for the recent-cap arm')
    D=o['D_obs']
    if D==0: raise ValueError('empty sample requires fold median')
    lo=[0]*4; hi=[0]*4
    for pat,d,e,c in o['table']:
        J=pat.count('1'); b=pat.index('1')+1
        for i,k in enumerate(range(2,6)):
            if J>=k: lo[i]+=d; hi[i]+=d
            elif J+b-1>=k: hi[i]+=c
    return [x/D for x in lo],[x/D for x in hi]


def h_midpoint(o):
    """Fixed bound-midpoint reference (L_k+U_k)/2; no tuning, no clipping."""
    lo,hi=h_bounds(o)
    return [(a+b)/2 for a,b in zip(lo,hi)]


def corrector(o):
    validate(o)
    D=o['D_obs']; M=o['M_obs']; S=sum(r[0].count('1')*r[1] for r in o['table'])
    if D==0: raise ValueError('empty sample requires fold median')
    if o['arm']=='R': return plugin(o)
    if o['arm']=='S':
        A=o['Walk_A']; den=sum(A)
        return [sum(A[k-1:])/den for k in range(2,6)]
    if o['arm']=='H': return h_midpoint(o)
    # Development variant only: the homogeneous three-to-five-window extrapolation.
    if o['arm']==LEGACY_H: return profile(activity(S/D,3))
    theta=activity(S/D,5); mean=M/S; p=o['parameter']
    if mean<1: raise ValueError('inconsistent M/S')
    r=0. if mean==1 else bisect(lambda r:1. if r==0 else r/-math.expm1(-r),mean,0.,mean)
    d=p if r==0 else -math.expm1(-r)/-math.expm1(-r/p)
    q=theta/d
    return profile(q) if q<=1 else [1.]*4

def all_baselines(o,median,extra):
    validate(o)
    if not o['D_obs']:
        return {m:{'prediction':list(median),'valid':False,'replacement':'empty_training_median'}
                for m in ['plugin','corrector','median','extratrees']}
    result={m:{'prediction':list(p),'valid':True,'replacement':None} for m,p in
            [('plugin',plugin(o)),('median',median),('extratrees',extra)]}
    try: result['corrector']={'prediction':corrector(o),'valid':True,'replacement':None}
    except (ArithmeticError,FloatingPointError,OverflowError):
        result['corrector']={'prediction':plugin(o),'valid':False,'replacement':'numerical_plugin'}
    return result
