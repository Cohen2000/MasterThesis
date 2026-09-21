"""Observations: the observed-only summary of one sampler draw, its text block,
its prompt messages, and the 134 features of the learned reference.

An observation lists, for every observed window pattern (1 = at least one
observed event in the window, 0 = none, ? = window not retrievable), the number
of observed dyads and events. Dyads without an observed event are absent.
"""
import math
import numpy as np
from .common import ARMS, ROOT, H_FRACTION, H_SENSITIVITY

PARAMETER_NAME = {'R': 'n_panel', 'S': 'L', 'H': 'n_panel_history', 'B': 'p'}
INTEGER_PARAMETER_ARMS = ('R', 'S', 'H')
PROMPTS = ROOT/'config/main_experiment'
RULE_FILES = {'R': 'rule_R.txt', 'S': 'rule_S.txt', 'H': 'rule_H_time_v3.txt', 'B': 'rule_B.txt'}
AUXILIARY_FILES = {'S': 'aux_S.txt'}
HEADER = 'pattern,dyads,events'
ALL_PATTERNS = [f'{p:05b}' for p in range(1, 32)]

FEATURE_VERSION = 'features-v6-panel888-20260921'
BASE_FEATURE_NAMES = (['N_obs', 'D_obs', 'M_obs']+[f'window_{i}' for i in range(1, 6)]+
                      [f'access_{i}' for i in range(1, 6)]+
                      [f'{p}_{x}' for p in ALL_PATTERNS for x in ('dyads', 'events')]+
                      [f'A_{i}' for i in range(1, 6)]+[f'arm_{a}' for a in ARMS]+
                      [PARAMETER_NAME[a] for a in ARMS]+['history_fraction'])
DERIVED_FEATURE_NAMES = ([f'share_{p}_dyads' for p in ALL_PATTERNS]+[f'share_window_{i}' for i in range(1, 6)]+
                         ['events_per_dyad']+[f'plugin_rho_{k}' for k in range(2, 6)]+
                         [f'corrector_rho_{k}' for k in range(2, 6)])
FEATURE_NAMES = BASE_FEATURE_NAMES+DERIVED_FEATURE_NAMES
assert len(FEATURE_NAMES) == 134 and len(set(FEATURE_NAMES)) == 134


def access_for(arm, h=H_FRACTION):
    """Retrievable windows. For H, the last round(5h) windows (h in .40/.60/.80
    makes the time cutoff coincide with a window boundary)."""
    if arm == 'H':
        if h not in H_SENSITIVITY: raise ValueError('unsupported history fraction')
        n = round(5*h)
        return [0]*(5-n)+[1]*n
    return [1]*5


def patterns_for(arm, h=H_FRACTION):
    n = sum(access_for(arm, h))
    return ['?'*(5-n)+format(p, f'0{n}b') for p in range(1, 2**n)]


def make(g, arm, budget, counts, traversals):
    """Observed-only summary of a draw; `counts` are observed events per dyad and window."""
    per_dyad = counts.sum(1)
    occupied = per_dyad > 0
    pattern = (counts > 0)@np.array([16, 8, 4, 2, 1])
    dyads = np.bincount(pattern[occupied], minlength=32)
    events = np.zeros(32, dtype=np.int64)
    np.add.at(events, pattern[occupied], per_dyad[occupied])
    h = budget.get('history_fraction', H_FRACTION)
    access = access_for(arm, h)
    if any(counts[:, j].any() for j, a in enumerate(access) if not a): raise ValueError('inaccessible events')
    table = []
    for p in patterns_for(arm, h):
        index = int(p.replace('?', '0'), 2)
        table.append((p, int(dyads[index]), int(events[index])))
    # Walk_A_j: traversals of observed dyads with K_e = j (S only, raw counts summing to L).
    walk_a = np.bincount(g.K, weights=traversals, minlength=6)[1:6].tolist() if arm == 'S' else None
    obs = {'arm': arm, 'N_obs': int(len(np.unique(g.ends[occupied]))), 'D_obs': int(occupied.sum()),
           'M_obs': int(counts.sum()), 'Temporal_access': access,
           'Events_per_window': [int(x) if a else None for x, a in zip(counts.sum(0), access)],
           'Walk_A': walk_a, 'parameter': budget[PARAMETER_NAME[arm]], 'table': table}
    if arm == 'H': obs['history_fraction'] = h
    validate(obs)
    return obs


def _count(x):
    return type(x) is int and x >= 0


def validate(o):
    """Internal consistency of an observation (raises ValueError)."""
    arm = o['arm']
    if arm not in ARMS: raise ValueError('arm')
    if arm == 'H' and 'history_fraction' not in o: raise ValueError('missing history fraction')
    h = o.get('history_fraction', H_FRACTION)
    for k in ('N_obs', 'D_obs', 'M_obs'):
        if not _count(o[k]): raise ValueError(k)
    if [r[0] for r in o['table']] != patterns_for(arm, h): raise ValueError('table patterns/order')
    if any(len(r) != 3 for r in o['table']): raise ValueError('table width')
    D = M = 0
    for pattern, d, e in o['table']:
        if not _count(d) or not _count(e) or (d == 0) != (e == 0) or e < d*pattern.count('1'):
            raise ValueError('inconsistent counts')
        D += d; M += e
    if (D, M) != (o['D_obs'], o['M_obs']): raise ValueError('table totals')
    access = access_for(arm, h)
    if o['Temporal_access'] != access or len(o['Events_per_window']) != 5: raise ValueError('access')
    for j, (a, e) in enumerate(zip(access, o['Events_per_window'])):
        if not a:
            if e is not None: raise ValueError('missing vs zero')
            continue
        if not _count(e): raise ValueError('window counts')
        active = sum(r[1] for r in o['table'] if r[0][j] == '1')
        if e < active or (active == 0) != (e == 0): raise ValueError('window/table mismatch')
    if sum(e or 0 for e in o['Events_per_window']) != M: raise ValueError('window sum')
    N = o['N_obs']
    if D == 0:
        if N != 0 or M != 0: raise ValueError('empty counts')
    elif N < 2 or N > 2*D or D > N*(N-1)//2: raise ValueError('endpoint count')
    parameter = o['parameter']
    if arm in INTEGER_PARAMETER_ARMS:
        if not _count(parameter): raise ValueError('integer parameter')
        if arm in ('R', 'H') and N > parameter: raise ValueError('panel smaller than observed nodes')
    elif not isinstance(parameter, (int, float)) or not 0 < parameter <= 1: raise ValueError('p')
    A = o['Walk_A']
    if arm == 'S':
        if A is None or len(A) != 5 or any(not math.isfinite(x) or x < 0 for x in A): raise ValueError('A')
        if bool(D) != bool(sum(A)): raise ValueError('walk mass')
        if any(x != int(x) for x in A) or sum(A) != parameter: raise ValueError('SRW traversal counts must sum to L')
    elif A is not None: raise ValueError('inapplicable A')


def _format(x):
    if x is None: return 'NA'
    return format(x, '.17g') if type(x) is float else str(x)


def serialize(o):
    """Text block shown to the model."""
    validate(o)
    lines = ['W=5', 'Temporal_access='+','.join(map(str, o['Temporal_access']))]
    lines += [f'{k}={o[k]}' for k in ('N_obs', 'D_obs', 'M_obs')]
    lines += ['Events_per_window='+','.join(map(_format, o['Events_per_window'])),
              PARAMETER_NAME[o['arm']]+'='+_format(o['parameter'])]
    if o['arm'] == 'S': lines.append('Walk_A='+','.join(map(_format, o['Walk_A'])))
    if o['arm'] == 'H': lines.append('History_fraction='+_format(o['history_fraction']))
    lines += [HEADER]+[','.join(map(str, row)) for row in o['table']]
    return '\n'.join(lines)


def parse(text):
    """Inverse of serialize; validates the result."""
    lines = text.splitlines()
    if not lines or lines[0] != 'W=5' or HEADER not in lines: raise ValueError('input block')
    header = lines.index(HEADER)
    pairs = [line.split('=', 1) for line in lines[1:header]]
    fields = dict(pairs)
    if len(fields) != len(pairs): raise ValueError('duplicate field')
    names = set(fields) & set(PARAMETER_NAME.values())
    if len(names) != 1: raise ValueError('parameter count')
    name = names.pop()
    arm = next(a for a, p in PARAMETER_NAME.items() if p == name)
    expected = {'Temporal_access', 'N_obs', 'D_obs', 'M_obs', 'Events_per_window', name}
    if arm == 'S': expected.add('Walk_A')
    if arm == 'H': expected.add('History_fraction')
    if set(fields) != expected: raise ValueError('unexpected input fields')
    rows = []
    for line in lines[header+1:]:
        cells = line.split(',')
        rows.append((cells[0], *map(int, cells[1:])))
    o = {'arm': arm,
         'parameter': int(fields[name]) if arm in INTEGER_PARAMETER_ARMS else float(fields[name]),
         **{k: int(fields[k]) for k in ('N_obs', 'D_obs', 'M_obs')},
         'Temporal_access': list(map(int, fields['Temporal_access'].split(','))),
         'Events_per_window': [None if x == 'NA' else int(x) for x in fields['Events_per_window'].split(',')],
         'Walk_A': list(map(float, fields['Walk_A'].split(','))) if arm == 'S' else None,
         'table': rows}
    if arm == 'H': o['history_fraction'] = float(fields['History_fraction'])
    validate(o)
    return o


def features(o):
    """134 features: raw observed counts and design parameters, then scale-free
    shares and the plug-in / arm-specific corrector profiles."""
    from .baselines import plugin, corrector
    validate(o)
    table = {r[0].replace('?', '0'): r[1:] for r in o['table']}

    def cell(p): return table.get(p, (0, 0))
    f = [o[k] for k in ('N_obs', 'D_obs', 'M_obs')]
    f += [x or 0 for x in o['Events_per_window']]+o['Temporal_access']
    f += [x for p in ALL_PATTERNS for x in cell(p)]
    f += o['Walk_A'] or [0]*5
    f += [int(o['arm'] == a) for a in ARMS]
    f += [o['parameter'] if a == o['arm'] else 0 for a in ARMS]
    f += [o.get('history_fraction', 1.)]
    if len(f) != len(BASE_FEATURE_NAMES): raise AssertionError('base feature count')
    D = o['D_obs']; M = o['M_obs']
    f += [cell(p)[0]/D if D else 0. for p in ALL_PATTERNS]
    f += [(x or 0)/M if M else 0. for x in o['Events_per_window']]
    f += [M/D if D else 0.]
    if D:
        plug = plugin(o)
        try: corrected = corrector(o)
        except (ArithmeticError, FloatingPointError, OverflowError, ValueError): corrected = plug
    else:
        plug = corrected = [0.]*4
    f += list(plug)+list(corrected)
    if len(f) != len(FEATURE_NAMES): raise AssertionError('feature count')
    return np.array(f, float)


def messages(block):
    """System and user message: common task text, the arm's sampling rule,
    auxiliary statistics (S only) and the observation block."""
    o = parse(block)
    system = (PROMPTS/'system.txt').read_text().rstrip('\n')
    parts = [(PROMPTS/'user_prefix.txt').read_text().rstrip('\n'),
             'Sampling rule: '+(PROMPTS/RULE_FILES[o['arm']]).read_text().rstrip('\n')]
    if o['arm'] in AUXILIARY_FILES:
        parts.append('Auxiliary statistics: '+(PROMPTS/AUXILIARY_FILES[o['arm']]).read_text().rstrip('\n'))
    parts.append(block)
    parts.append('Return the four full-archive estimates in the specified JSON format.')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': '\n'.join(parts)}]
