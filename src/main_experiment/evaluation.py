"""Strict final-answer parsing and hierarchical error aggregation; no transport."""
import json
import math
import re
import numpy as np
from .baselines import plugin
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


def cell_variance(a):
    """Monte-Carlo variance of one source's mean, and its two components.

    a has shape (sampler draws, repeats). With at least two draws the variance of
    the mean is estimated from the draw means (it contains both components).
    With a single, deterministic draw the sampler contributes nothing and the
    variance is that of the repeat mean. The components are reported separately:
    model = within-draw repeat variance / (draws * repeats);
    sampler = between-draw variance net of the model part, floored at zero.
    """
    a=np.asarray(a,float)
    s,r=a.shape
    within=float(np.mean(np.var(a,axis=1,ddof=1))) if r>1 else 0.
    model=within/(s*r)
    if s>1:
        total=float(np.var(a.mean(1),ddof=1)/s)
        sampler=max(total-model,0.)
    else:
        total=model; sampler=0.
    return {'total':total,'model':model,'sampler':sampler,'deterministic_draw':s==1}


def paired_summary(cells,expected=None):
    """Cells: source -> draw x repeat error values. Complete cells only.

    expected maps source -> number of distinct sampler draws (SAMPLER_DRAWS, or 1
    for a deterministic draw); without it every cell must have SAMPLER_DRAWS rows.
    Repeated answers and dyads are never counted as extra independent units.
    Sources are equally weighted; the caller keeps the real and synthetic strata apart.
    """
    means=[]; total=[]; model=[]; sampler=[]; sources={}
    for source,values in cells.items():
        a=np.asarray(values,float)
        draws=(expected or {}).get(source,SAMPLER_DRAWS)
        if a.ndim!=2 or a.shape[0]!=draws or a.shape[1]<1 or not np.isfinite(a).all():
            raise ValueError('incomplete or invalid cell; no complete main result')
        v=cell_variance(a)
        means.append(float(a.mean())); total.append(v['total']); model.append(v['model']); sampler.append(v['sampler'])
        sources[source]={'mean':means[-1],'mcse':math.sqrt(v['total']),
                         'mcse_model_repeats':math.sqrt(v['model']),
                         'mcse_sampler':math.sqrt(v['sampler']),
                         'draws':draws,'deterministic_draw':v['deterministic_draw']}
    if not means: raise ValueError('no sources')
    n=len(means)
    return {'mean':float(np.mean(means)),'mcse':math.sqrt(sum(total))/n,
            'mcse_model_repeats':math.sqrt(sum(model))/n,
            'mcse_sampler':math.sqrt(sum(sampler))/n,
            'between_source_se':float(np.std(means,ddof=1)/math.sqrt(n)) if n>1 else None,
            'sources':sources,'conditional_on':'fixed sources, training, models and calibrated L'}
