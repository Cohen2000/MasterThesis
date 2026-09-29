"""When an answer counts: the strict answer format check for every language model.

An answer is valid only if its final text is one JSON object with exactly the keys
rho_2..rho_5, finite numbers in [0, 1], never increasing, optionally wrapped in one
code block. Invalid answers are never repaired. valid_profile() applies the same
conditions to ExtraTrees output by limiting its values.
"""
import json
import math
import re

KEYS=tuple(f'rho_{k}' for k in range(2,6))
FENCE=re.compile(r'\A```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*\Z',re.S)


# Allow exactly one ``` code fence around the whole answer (models often add one).
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


# The validity rule shared by all language models: returns (values, reason).
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


def valid_profile(p):
    """ExtraTrees output as a valid profile: each value limited to [0, 1], then each rho_k
    capped at rho_(k-1), so the profile never increases. A valid profile is unchanged."""
    out=[]
    for x in p:
        x=min(1.,max(0.,float(x)))
        out.append(min(x,out[-1]) if out else x)
    return out
