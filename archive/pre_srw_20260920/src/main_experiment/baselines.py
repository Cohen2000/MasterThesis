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


def h_extrapolator(o):
    """Homogeneous ZT-Binomial working model, not an unbiased estimator.

    J | J>0 ~ Binomial(n_visible,q) truncated at zero. Its sufficient statistic
    is mean(J). The MLE solves n*q/(1-(1-q)**n)=mean(J), including limiting
    q=0/1. Full-archive rho conditions on K>0 for K~Binomial(5,q).
    """
    validate(o)
    if o['arm']!='H' or not o['D_obs']: raise ValueError('nonempty H input required')
    n=sum(o['Temporal_access'])
    mean=sum(p.count('1')*d for p,d,e in o['table'])/o['D_obs']
    q=activity(mean,n)
    return {'prediction':profile(q),'q':q,'visible_windows':n,
            'status':'boundary_zero' if q==0 else 'boundary_one' if q==1 else 'ok',
            'model':'homogeneous_zero_truncated_binomial','working_model':True}


def corrector(o):
    validate(o)
    D=o['D_obs']; M=o['M_obs']; S=sum(r[0].count('1')*r[1] for r in o['table'])
    if D==0: raise ValueError('empty sample requires fold median')
    if o['arm']=='R': return plugin(o)
    if o['arm']=='S':
        A=o['Walk_A']; den=sum(A)
        return [sum(A[k-1:])/den for k in range(2,6)]
    if o['arm']=='H': return h_extrapolator(o)['prediction']
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
