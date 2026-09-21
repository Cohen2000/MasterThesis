"""Observations: the observed-only summary of one sampler draw, its text block,
its prompt messages, and the features of the learned reference.

Every arm yields the same kind of table: distinct observed dyads grouped by
their window pattern (1 = at least one observed event in the window, 0 = none,
? = window not retrievable) with dyad and event counts, plus the arm's design
parameter and, for H, the history fraction. Dyads without an observed event are
absent. S2 additionally reports, per pattern row, the walk's traversals of those
dyads and the inverse-degree-product weight of these traversals; S1 shows the
same walk without any walker information.
"""
import numpy as np
from .common import ARMS, ROOT, H_FRACTION, H_SENSITIVITY

PARAMETER_NAME = {'R': 'n_panel', 'S1': 'L', 'S2': 'L', 'H': 'n_panel_history', 'B': 'p'}
PARAMETERS = ('n_panel', 'L', 'n_panel_history', 'p')          # one feature slot each
INTEGER_PARAMETER_ARMS = ('R', 'S1', 'S2', 'H')
PROMPTS = ROOT/'config/main_experiment'
RULE_FILES = {'R': 'rule_R.txt', 'S1': 'rule_S1.txt', 'S2': 'rule_S2.txt', 'H': 'rule_H.txt', 'B': 'rule_B.txt'}
HEADER = 'pattern,dyads,events'
S2_HEADER = HEADER+',traversals,inverse_degree_weight'
ALL_PATTERNS = [f'{p:05b}' for p in range(1, 32)]

FEATURE_VERSION = 'features-v8-panel888-20260921'
BASE_FEATURE_NAMES = (['N_obs', 'D_obs', 'M_obs']+[f'window_{i}' for i in range(1, 6)]+
                      [f'access_{i}' for i in range(1, 6)]+
                      [f'{p}_{x}' for p in ALL_PATTERNS for x in ('dyads', 'events')]+
                      [f'arm_{a}' for a in ARMS]+list(PARAMETERS)+['history_fraction'])
DERIVED_FEATURE_NAMES = ([f'share_{p}_dyads' for p in ALL_PATTERNS]+[f'share_window_{i}' for i in range(1, 6)]+
                         ['events_per_dyad']+[f'plugin_rho_{k}' for k in range(2, 6)]+
                         [f'corrector_rho_{k}' for k in range(2, 6)]+
                         [f'share_{p}_traversals' for p in ALL_PATTERNS]+
                         [f'share_{p}_inverse_degree_weight' for p in ALL_PATTERNS])
FEATURE_NAMES = BASE_FEATURE_NAMES+DERIVED_FEATURE_NAMES
assert len(FEATURE_NAMES) == 192 and len(set(FEATURE_NAMES)) == 192


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


def rounded(x):
    """Floats are shown and stored with 12 significant digits."""
    return float(f'{x:.12g}')


def make(g, arm, budget, counts, traversals=None):
    """Observed-only summary of a draw; `counts` are observed events per dyad and window.

    `traversals` (per-dyad walk traversal counts) is used for S2 only.
    """
    per_dyad = counts.sum(1)
    occupied = per_dyad > 0
    pattern = (counts > 0)@np.array([16, 8, 4, 2, 1])
    h = budget.get('history_fraction', H_FRACTION)
    access = access_for(arm, h)
    if any(counts[:, j].any() for j, a in enumerate(access) if not a): raise ValueError('inaccessible events')
    if arm == 'S2':
        degree = np.bincount(g.ends.ravel(), minlength=g.N)
        weight = traversals/(degree[g.ends[:, 0]]*degree[g.ends[:, 1]])
    table = []
    for p in patterns_for(arm, h):
        rows = occupied & (pattern == int(p.replace('?', '0'), 2))
        row = (p, int(rows.sum()), int(per_dyad[rows].sum()))
        if arm == 'S2': row += (int(traversals[rows].sum()), rounded(weight[rows].sum()))
        table.append(row)
    obs = {'arm': arm, 'N_obs': int(len(np.unique(g.ends[occupied]))), 'D_obs': int(occupied.sum()),
           'M_obs': int(counts.sum()), 'Temporal_access': access,
           'Events_per_window': [int(x) if a else None for x, a in zip(counts.sum(0), access)],
           'parameter': budget[PARAMETER_NAME[arm]], 'table': table}
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
    if any(len(r) != (5 if arm == 'S2' else 3) for r in o['table']): raise ValueError('table width')
    D = M = 0
    for pattern, d, e, *walker in o['table']:
        if not _count(d) or not _count(e) or (d == 0) != (e == 0) or e < d*pattern.count('1'):
            raise ValueError('inconsistent counts')
        if walker:                       # S2: every observed dyad was traversed at least once
            t, w = walker
            if not _count(t) or t < d or (d == 0) != (t == 0): raise ValueError('traversal counts')
            if not isinstance(w, float) or not 0 <= w <= t or (t == 0) != (w == 0): raise ValueError('weights')
        D += d; M += e
    if (D, M) != (o['D_obs'], o['M_obs']): raise ValueError('table totals')
    if arm == 'S2' and sum(r[3] for r in o['table']) != o['parameter']: raise ValueError('traversals must sum to L')
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


def _format(x):
    if x is None: return 'NA'
    return format(x, '.12g') if type(x) is float else str(x)


def serialize(o):
    """Text block shown to the model."""
    validate(o)
    lines = ['W=5', 'Temporal_access='+','.join(map(str, o['Temporal_access']))]
    lines += [f'{k}={o[k]}' for k in ('N_obs', 'D_obs', 'M_obs')]
    lines += ['Events_per_window='+','.join(map(_format, o['Events_per_window'])),
              PARAMETER_NAME[o['arm']]+'='+_format(o['parameter'])]
    if o['arm'] == 'H': lines.append('History_fraction='+_format(o['history_fraction']))
    lines.append(S2_HEADER if o['arm'] == 'S2' else HEADER)
    lines += [','.join(map(_format, row)) for row in o['table']]
    return '\n'.join(lines)


def parse(text):
    """Inverse of serialize; validates the result."""
    lines = text.splitlines()
    header_line = S2_HEADER if S2_HEADER in lines else HEADER
    if not lines or lines[0] != 'W=5' or header_line not in lines: raise ValueError('input block')
    header = lines.index(header_line)
    pairs = [line.split('=', 1) for line in lines[1:header]]
    fields = dict(pairs)
    if len(fields) != len(pairs): raise ValueError('duplicate field')
    names = set(fields) & set(PARAMETERS)
    if len(names) != 1: raise ValueError('parameter count')
    name = names.pop()
    arm = 'S2' if header_line == S2_HEADER else next(a for a, p in PARAMETER_NAME.items() if p == name)
    if PARAMETER_NAME[arm] != name: raise ValueError('parameter does not match the table')
    expected = {'Temporal_access', 'N_obs', 'D_obs', 'M_obs', 'Events_per_window', name}
    if arm == 'H': expected.add('History_fraction')
    if set(fields) != expected: raise ValueError('unexpected input fields')
    rows = []
    for line in lines[header+1:]:
        cells = line.split(',')
        row = (cells[0], *map(int, cells[1:4]))
        if arm == 'S2': row += (float(cells[4]),)
        rows.append(row)
    o = {'arm': arm,
         'parameter': int(fields[name]) if arm in INTEGER_PARAMETER_ARMS else float(fields[name]),
         **{k: int(fields[k]) for k in ('N_obs', 'D_obs', 'M_obs')},
         'Temporal_access': list(map(int, fields['Temporal_access'].split(','))),
         'Events_per_window': [None if x == 'NA' else int(x) for x in fields['Events_per_window'].split(',')],
         'table': rows}
    if arm == 'H': o['history_fraction'] = float(fields['History_fraction'])
    validate(o)
    return o


def features(o):
    """192 features, all computed from the observation block: raw observed counts
    and design parameters, scale-free shares, the plug-in and arm-corrector profiles,
    and (S2 only, zero otherwise) per-pattern traversal and weight shares."""
    from .baselines import plugin, corrector
    validate(o)
    table = {r[0].replace('?', '0'): r[1:] for r in o['table']}
    walker = o['arm'] == 'S2' and o['D_obs'] > 0

    def cell(p): return table.get(p, (0, 0, 0, 0.))
    f = [o[k] for k in ('N_obs', 'D_obs', 'M_obs')]
    f += [x or 0 for x in o['Events_per_window']]+o['Temporal_access']
    f += [x for p in ALL_PATTERNS for x in cell(p)[:2]]
    f += [int(o['arm'] == a) for a in ARMS]
    f += [o['parameter'] if PARAMETER_NAME[o['arm']] == name else 0 for name in PARAMETERS]
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
    if walker:
        traversals = sum(r[3] for r in o['table']); weight = sum(r[4] for r in o['table'])
        f += [cell(p)[2]/traversals for p in ALL_PATTERNS]+[cell(p)[3]/weight for p in ALL_PATTERNS]
    else:
        f += [0.]*(2*len(ALL_PATTERNS))
    if len(f) != len(FEATURE_NAMES): raise AssertionError('feature count')
    return np.array(f, float)


def messages(block):
    """System and user message: common task text, the arm's sampling rule and the observation block."""
    o = parse(block)
    system = (PROMPTS/'system.txt').read_text().rstrip('\n')
    parts = [(PROMPTS/'user_prefix.txt').read_text().rstrip('\n'),
             'Sampling rule: '+(PROMPTS/RULE_FILES[o['arm']]).read_text().rstrip('\n')]
    parts.append(block)
    parts.append('Return the four full-archive estimates in the specified JSON format.')
    return [{'role': 'system', 'content': system}, {'role': 'user', 'content': '\n'.join(parts)}]
