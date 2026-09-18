"""Strict final-answer parsing and hierarchical error aggregation; no transport."""
import json
import math
import re
import numpy as np
from .common import SAMPLER_DRAWS, LLM_REPEATS

KEYS=tuple(f'rho_{k}' for k in range(2,6))
FENCE=re.compile(r'\A```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*\Z',re.S)


def strip_fence(raw):
    """Remove one surrounding markdown code fence, if the whole answer is one.

    Decided before the main run, after a smoke test showed the model wrapping its
    JSON in ```json ... ``` in about half of the answers despite the instruction
    not to. A fence is a transport wrapper, not content: everything inside is
    still validated exactly as strictly as before -- the key set, finiteness, the
    [0,1] range, monotonicity and the rejection of duplicate keys all still apply,
    and nothing but a single enclosing fence is ever removed. Counting a fenced
    but otherwise perfect answer as invalid would measure markdown compliance
    rather than estimation, so the evaluation reports it as `valid_after_fence`
    and the share is stated alongside the results.
    """
    m=FENCE.match(raw.strip())
    return (m.group(1),True) if m else (raw,False)


def parse_final(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError('duplicate_key')
            d[k]=v
        return d
    def constant(_): raise ValueError('non_json_number')
    text,fenced=strip_fence(raw)
    try:
        obj=json.loads(text.strip(),object_pairs_hook=pairs,parse_constant=constant)
        if type(obj) is not dict or set(obj)!=set(KEYS): return None,'keys_or_object'
        values=[obj[k] for k in KEYS]
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in values): return None,'nonfinite_or_type'
        if any(not 0<=x<=1 for x in values): return None,'range'
        if any(a<b for a,b in zip(values,values[1:])): return None,'monotonicity'
        return values,('valid_after_fence' if fenced else 'valid')
    except (ValueError,TypeError,AttributeError,OverflowError) as e:
        return None,'invalid_json:'+str(e)


EVALUATION_VERSION='validity-conditional-mae-v1-20260918'


def resolve(o,median=None,record=None):
    """Failures have no estimate. median is accepted only for caller compatibility."""
    base={'prediction':None,'valid':None,'replacement':None,'started':False,'terminal':False}
    if o['D_obs']==0:
        return {**base,'status':'empty','valid':False,'terminal':True}
    if record is None or not record.get('started',False):
        return {**base,'status':'not_started'}
    if not record.get('terminal',False):
        return {**base,'status':'in_progress','started':True}
    values,reason=parse_final(record.get('final_text',''))
    if record.get('refusal'): values=None; reason='refusal'
    if record.get('technical_error'): values=None; reason='technical_error'
    return {**base,'prediction':values,'status':'terminal','terminal':True,
            'valid':values is not None,'validation_reason':reason,'started':True,
            'limit_hit':bool(record.get('limit_hit',False)),
            'technical_error':bool(record.get('technical_error',False))}


def errors(prediction,truth):
    if prediction is None: return {'AE2':None,'ProfileAE':None,'signed_rho2':None}
    diff=np.asarray(prediction)-np.asarray(truth)
    return {'AE2':float(abs(diff[0])),'ProfileAE':float(np.mean(abs(diff))),'signed_rho2':float(diff[0])}


def cell_variance(a):
    """Direct variance of the mean; no estimated additive components."""
    a=np.asarray(a,float)
    if a.ndim!=2 or not a.size or not np.isfinite(a).all():
        raise ValueError('finite draw x repeat array required')
    s,r=a.shape
    total=float(np.var(a.mean(1),ddof=1)/s) if s>1 else (
          float(np.var(a[0],ddof=1)/r) if r>1 else 0.)
    return {'total':total,'deterministic_draw':s==1}


def paired_summary(cells,expected=None):
    """Equal-source summary of complete cells, clustered on sampler draws."""
    for source,values in cells.items():
        a=np.asarray(values,float)
        if (a.ndim!=2 or a.shape[0]!=(expected or {}).get(source,SAMPLER_DRAWS)
                or a.shape[1]<1 or not np.isfinite(a).all()):
            raise ValueError('incomplete or invalid cell; no complete main result')
    return conditional_summary(cells,expected,complete=True)


def conditional_summary(cells,expected=None,complete=False):
    """Source-equal mean of valid-answer means; NaN denotes an unavailable answer.

    MCSE uses a cluster ratio linearization. This is conditional accuracy, never
    an imputed full-panel loss. No source can silently disappear from the mean.
    `complete=True` is for finite deterministic references / validity indicators.
    """
    if not cells: raise ValueError('no sources')
    sources={}
    for source,values in cells.items():
        a=np.asarray(values,float)
        draws=(expected or {}).get(source,SAMPLER_DRAWS)
        if a.ndim!=2 or a.shape[0]!=draws or a.shape[1]<1 or np.isinf(a).any():
            raise ValueError('invalid cell shape or infinite value')
        mask=np.isfinite(a); count=mask.sum(axis=1); n=int(count.sum())
        mu=float(np.nansum(a)/n) if n else None
        variance=None
        if n:
            if draws>1 and (complete or np.count_nonzero(count)>=2):
                residual=np.nansum(a,axis=1)-mu*count
                variance=float(draws/(draws-1)*np.sum(residual**2)/n**2)
            elif draws==1 and n>1:
                variance=float(np.var(a[mask],ddof=1)/n)
            elif draws==1 and complete: variance=0.
        sources[source]={'mean':mu,'mcse':math.sqrt(variance) if variance is not None else None,
                         'draws':draws,'valid_answers':n,'planned_answers':int(a.size),
                         'draws_with_valid_answers':int(np.count_nonzero(count)),
                         'deterministic_draw':draws==1}
    means=[r['mean'] for r in sources.values()]
    ses=[r['mcse'] for r in sources.values()]
    all_sources=all(m is not None for m in means)
    return {'mean':float(np.mean(means)) if all_sources else None,
            'mcse':math.sqrt(sum(x*x for x in ses))/len(ses) if all(x is not None for x in ses) else None,
            'between_source_sd':float(np.std(means,ddof=1)) if all_sources and len(means)>1 else None,
            'sources':sources,'sources_with_valid_answers':sum(m is not None for m in means),
            'conditional_on':'valid answers; fixed sources, training, models and calibrated L'}
