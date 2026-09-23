"""Serialized observation blocks, prompts and released-evidence ET features.

Rows group observed dyads by window pattern. S adds crawl traversals and
inverse-event sums; S_obs contains the inverse-event sums without the crawl log.
"""
import numpy as np
from .common import ARMS, ROOT, H_FRACTION, H_SENSITIVITY

PARAMETER_NAME = {'R': 'n_panel', 'S': 'L', 'S_obs': 'L', 'H': 'n_panel_history', 'B': 'p'}
PARAMETERS = ('n_panel', 'L', 'n_panel_history', 'p')          # one feature slot each
INTEGER_PARAMETER_ARMS = ('R', 'S', 'S_obs', 'H')
PROMPTS = ROOT/'config/main_experiment'
RULE_FILES = {'R': 'rule_R.txt', 'S': 'rule_S.txt', 'S_obs': 'rule_S_obs.txt', 'H': 'rule_H.txt', 'B': 'rule_B.txt'}
HEADER = 'pattern,dyads,events'
S_OBS_HEADER = HEADER+',inv_events'
S_HEADER = S_OBS_HEADER+',traversals,traversals_per_event'
ALL_PATTERNS = [f'{p:05b}' for p in range(1, 32)]

FEATURE_VERSION = 'features-v10-access-contract-20260923'
# Only released evidence enters ET; sampler calibration fields are intentionally absent.
FEATURE_NAMES = ([f'arm_{a}' for a in ARMS] +
                 [f'share_{p}_dyads' for p in ALL_PATTERNS] +
                 [f'share_{p}_events' for p in ALL_PATTERNS] +
                 [f'share_window_{i}' for i in range(1, 6)] +
                 ['events_per_dyad'] + [f'anchor_rho_{k}' for k in range(2, 6)] +
                 ['log1p_N_obs', 'log1p_D_obs', 'log1p_M_obs', 'p'] +
                 [f'share_{p}_{name}' for name in ('inv_events', 'traversals', 'traversals_per_event')
                  for p in ALL_PATTERNS])
assert len(FEATURE_NAMES) == 174 and len(set(FEATURE_NAMES)) == 174
# Compatibility names for lightweight downstream imports; these are not used as
# model features and intentionally contain no hidden calibration fields.
BASE_FEATURE_NAMES = FEATURE_NAMES
DERIVED_FEATURE_NAMES = []


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

    `traversals` (per-dyad walk traversal counts) is released only for S.
    """
    per_dyad = counts.sum(1)
    occupied = per_dyad > 0
    pattern = (counts > 0)@np.array([16, 8, 4, 2, 1])
    h = budget.get('history_fraction', H_FRACTION)
    access = access_for(arm, h)
    if any(counts[:, j].any() for j, a in enumerate(access) if not a): raise ValueError('inaccessible events')
    if arm in ('S', 'S_obs'):
        inv_events = 1 / g.m
        if traversals is None: raise ValueError('walk traversal counts missing')
    table = []
    for p in patterns_for(arm, h):
        rows = occupied & (pattern == int(p.replace('?', '0'), 2))
        row = (p, int(rows.sum()), int(per_dyad[rows].sum()))
        if arm in ('S', 'S_obs'): row += (rounded(inv_events[rows].sum()),)
        if arm == 'S': row += (int(traversals[rows].sum()),
                              rounded((traversals[rows] * inv_events[rows]).sum()))
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
    width = 6 if arm == 'S' else 4 if arm == 'S_obs' else 3
    if any(len(r) != width for r in o['table']): raise ValueError('table width')
    D = M = 0
    for pattern, d, e, *walker in o['table']:
        if not _count(d) or not _count(e) or (d == 0) != (e == 0) or e < d*pattern.count('1'):
            raise ValueError('inconsistent counts')
        if arm in ('S', 'S_obs'):
            inv = walker[0]
            if not isinstance(inv, float) or not np.isfinite(inv): raise ValueError('inv_events type')
            if d == 0 and inv != 0 or d > 0 and not (d*d/e - 1e-9 <= inv <= d + 1e-9):
                raise ValueError('inv_events bounds')
        if arm == 'S':
            t, w = walker[1:]
            if not _count(t) or t < d or (d == 0) != (t == 0): raise ValueError('traversal counts')
            if not isinstance(w, float) or not np.isfinite(w) or (d == 0 and w != 0) or (d > 0 and not (inv-1e-9 <= w <= t+1e-9)):
                raise ValueError('traversals_per_event bounds')
        D += d; M += e
    if (D, M) != (o['D_obs'], o['M_obs']): raise ValueError('table totals')
    if arm == 'S' and o['parameter'] is not None and sum(r[4] for r in o['table']) != o['parameter']:
        raise ValueError('traversals must sum to L')
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
        if parameter is not None:
            if not _count(parameter): raise ValueError('integer parameter')
            if arm in ('R', 'H') and N > parameter: raise ValueError('panel smaller than observed nodes')
    elif not isinstance(parameter, (int, float)) or not 0 < parameter <= 1: raise ValueError('p')


def _format(x):
    if x is None: return 'NA'
    return format(x, '.12g') if type(x) is float else str(x)


def serialize(o):
    """Text block shown to the model."""
    validate(o)
    lines = ['W=5', 'Arm='+o['arm'], 'Temporal_access='+','.join(map(str, o['Temporal_access']))]
    lines += [f'{k}={o[k]}' for k in ('N_obs', 'D_obs', 'M_obs')]
    lines += ['Events_per_window='+','.join(map(_format, o['Events_per_window']))]
    if o['arm'] == 'B': lines.append('p='+_format(o['parameter']))
    lines.append(S_HEADER if o['arm'] == 'S' else S_OBS_HEADER if o['arm'] == 'S_obs' else HEADER)
    lines += [','.join(map(_format, row)) for row in o['table']]
    return '\n'.join(lines)


def parse(text):
    """Inverse of serialize; validates the result."""
    lines = text.splitlines()
    header_line = next((h for h in (S_HEADER, S_OBS_HEADER, HEADER) if h in lines), None)
    if not lines or lines[0] != 'W=5' or header_line not in lines: raise ValueError('input block')
    header = lines.index(header_line)
    pairs = [line.split('=', 1) for line in lines[1:header]]
    fields = dict(pairs)
    if len(fields) != len(pairs): raise ValueError('duplicate field')
    arm = fields.get('Arm')
    if arm not in ARMS or header_line != (S_HEADER if arm == 'S' else S_OBS_HEADER if arm == 'S_obs' else HEADER):
        raise ValueError('arm/header mismatch')
    expected = {'Arm', 'Temporal_access', 'N_obs', 'D_obs', 'M_obs', 'Events_per_window'}
    if arm == 'B': expected.add('p')
    if set(fields) != expected: raise ValueError('unexpected input fields')
    rows = []
    for line in lines[header+1:]:
        cells = line.split(',')
        if len(cells) != (6 if arm == 'S' else 4 if arm == 'S_obs' else 3): raise ValueError('row width')
        row = (cells[0], int(cells[1]), int(cells[2]))
        if arm in ('S', 'S_obs'): row += (float(cells[3]),)
        if arm == 'S': row += (int(cells[4]), float(cells[5]))
        rows.append(row)
    o = {'arm': arm,
         'parameter': float(fields['p']) if arm == 'B' else None,
         **{k: int(fields[k]) for k in ('N_obs', 'D_obs', 'M_obs')},
         'Temporal_access': list(map(int, fields['Temporal_access'].split(','))),
         'Events_per_window': [None if x == 'NA' else int(x) for x in fields['Events_per_window'].split(',')],
         'table': rows}
    if arm == 'H': o['history_fraction'] = sum(o['Temporal_access'])/5   # released via Temporal_access; .6 in main
    if arm == 'S': o['parameter'] = sum(row[4] for row in rows)
    validate(o)
    return o


def features(o):
    """Released-evidence features only; hidden calibration fields never enter ET."""
    from .baselines import anchor_profile
    validate(o)
    table = {r[0].replace('?', '0'): r[1:] for r in o['table']}
    def cell(p): return table.get(p, (0, 0, 0, 0.))
    f = [int(o['arm'] == a) for a in ARMS]
    D = o['D_obs']; M = o['M_obs']
    f += [cell(p)[0]/D if D else 0. for p in ALL_PATTERNS]
    f += [cell(p)[1]/M if M else 0. for p in ALL_PATTERNS]
    f += [(x or 0)/M if M else 0. for x in o['Events_per_window']]
    f += [M/D if D else 0.]
    f += list(anchor_profile(o) if D else [0.]*4)
    f += [np.log1p(x) for x in (o['N_obs'], D, M)]
    f += [o['parameter'] if o['arm'] == 'B' else 0.]
    for index in (3, 4, 5):
        total = sum(r[index] for r in o['table']) if o['arm'] in ('S', 'S_obs') and (index == 3 or o['arm'] == 'S') else 0.
        f += [(cell(p)[index-1]/total if total and len(cell(p)) >= index else 0.) for p in ALL_PATTERNS]
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
