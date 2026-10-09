"""When an answer counts. An answer of a language model is valid only if its final text is exactly
one JSON object with the four values rho_2..rho_5, each between 0 and 1 and never increasing. One
surrounding code block (```json ... ```) is allowed. Invalid answers are never repaired."""
import json
import math
import re

KEYS = tuple(f'rho_{k}' for k in range(2, 6))
FENCE = re.compile(r'\A```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*\Z', re.S)


def parse_final(raw):
    """(values, reason): the four values of a valid answer, or None and why it is invalid."""
    def pairs(items):
        d = {}
        for k, v in items:
            if k in d: raise ValueError('duplicate_key')
            d[k] = v
        return d

    def constant(_): raise ValueError('non_json_number')
    fenced = FENCE.match(raw.strip())
    try:
        obj = json.loads((fenced.group(1) if fenced else raw).strip(), object_pairs_hook=pairs, parse_constant=constant)
        if type(obj) is not dict or set(obj) != set(KEYS): return None, 'keys_or_object'
        values = [obj[k] for k in KEYS]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in values): return None, 'nonfinite_or_type'
        if any(not 0 <= x <= 1 for x in values): return None, 'range'
        if any(a < b for a, b in zip(values, values[1:])): return None, 'monotonicity'
        return values, ('valid_after_fence' if fenced else 'valid')
    except (ValueError, TypeError, AttributeError, OverflowError) as e:
        return None, 'invalid_json:'+str(e)


def valid_profile(p):
    """ExtraTrees output as a valid profile: each value limited to [0, 1] and to at most the value before it."""
    out = []
    for x in p:
        x = min(1., max(0., float(x)))
        out.append(min(x, out[-1]) if out else x)
    return out
