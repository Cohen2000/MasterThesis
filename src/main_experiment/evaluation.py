"""Strict final-answer parsing and hierarchical error aggregation; no transport."""
import json
import math
import re
import numpy as np
from .baselines import plugin
from .common import SAMPLES_PER_ARM, LLM_REPEATS

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


def resolve(o,median,record=None):
    if o['D_obs']==0:
        return {'prediction':list(median),'status':'empty','valid':False,'replacement':'training_median','started':False}
    if record is None or not record.get('started',False):
        return {'prediction':None,'status':'not_started','valid':None,'replacement':None,'started':False}
    if not record.get('terminal',False):
        return {'prediction':None,'status':'in_progress','valid':None,'replacement':None,'started':True}
    values,reason=parse_final(record.get('final_text',''))
    if record.get('refusal'): values=None; reason='refusal'
    return {'prediction':values if values is not None else plugin(o),'status':'terminal',
            'valid':values is not None,'validation_reason':reason,
            'replacement':None if values is not None else 'plugin','started':True,
            'limit_hit':bool(record.get('limit_hit',False)),
            'technical_error':bool(record.get('technical_error',False))}


def errors(prediction,truth):
    if prediction is None: return {'AE2':None,'ProfileAE':None,'signed_rho2':None}
    diff=np.asarray(prediction)-np.asarray(truth)
    return {'AE2':float(abs(diff[0])),'ProfileAE':float(np.mean(abs(diff))),'signed_rho2':float(diff[0])}


def paired_summary(cells):
    """Cells: source -> sample x repeat error differences. Complete cells only.

    Independent replication is the sampler draw, so the variance is taken over
    the per-draw means of the LLM repeats. Repeated answers and dyads are never
    counted as extra independent units. Sources are equally weighted; the caller
    keeps the real and synthetic strata apart.
    """
    means=[]; variance=[]; sources={}
    for source,values in cells.items():
        a=np.asarray(values,float)
        if a.shape!=(SAMPLES_PER_ARM,LLM_REPEATS) or not np.isfinite(a).all():
            raise ValueError('incomplete or invalid cell; no complete main result')
        independent=a.mean(1)
        v=float(np.var(independent,ddof=1)/len(independent))
        means.append(float(a.mean())); variance.append(v)
        sources[source]={'mean':means[-1],'mcse':math.sqrt(v)}
    if not means: raise ValueError('no sources')
    return {'mean':float(np.mean(means)),'mcse':math.sqrt(sum(variance))/len(means),
            'sources':sources,'conditional_on':'fixed sources, training, models and calibrated L'}
