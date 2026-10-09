"""The sample text: what every method gets to see of a sample, the prompt and the ExtraTrees features.

A sample is written as a small table: one row per pattern of active windows (for example 01101)
with the number of pairs and events that show it, plus what the sampling arm states about itself.
Language models read this text; every other method reads the same text back with parse().
"""
import numpy as np
from .common import H_WINDOWS, ROOT

PROMPTS = ROOT/'config/prompts'
HEAD = 'pattern,dyads,events'
WALK_HEAD = HEAD+',inv_events,traversals,traversals_per_event'      # arm S adds the walk's log
PATTERNS = [f'{p:05b}' for p in range(1, 32)]


def _number(x):
    return 'NA' if x is None else format(x, '.12g') if type(x) is float else str(x)


def text(g, arm, size, counts, visits=None):
    """The sample text of one draw. `counts` are the seen events per pair and window."""
    events = counts.sum(1)
    seen = events > 0
    pattern = (counts > 0)@np.array([16, 8, 4, 2, 1])
    hidden = 5-H_WINDOWS if arm == 'H' else 0                        # windows the arm cannot see
    access = [0]*hidden+[1]*(5-hidden)
    rows = []
    for p in range(1, 2**(5-hidden)):
        of = seen & (pattern == p)
        row = ['?'*hidden+format(p, f'0{5-hidden}b'), int(of.sum()), int(events[of].sum())]
        if arm == 'S':      # 1/events of each pair, the walk's visits, and visits/events
            share = 1/g.m[of]
            row += [float(f'{share.sum():.12g}'), int(visits[of].sum()), float(f'{(visits[of]*share).sum():.12g}')]
        rows.append(row)
    lines = ['W=5', f'Arm={arm}', 'Temporal_access='+','.join(map(str, access)),
             f'N_obs={len(np.unique(g.ends[seen]))}', f'D_obs={int(seen.sum())}', f'M_obs={int(counts.sum())}',
             'Events_per_window='+','.join(_number(int(x) if a else None) for x, a in zip(counts.sum(0), access))]
    if arm == 'B': lines.append('p='+_number(size['B']))
    if arm in 'RH': lines.append(f'n_panel={size[arm]}')
    lines.append(WALK_HEAD if arm == 'S' else HEAD)
    return '\n'.join(lines+[','.join(map(_number, row)) for row in rows])


def parse(block):
    """The sample text as numbers. `parameter` is p for arm B and the number of steps for arm S."""
    lines = block.splitlines()
    head = lines.index(WALK_HEAD if WALK_HEAD in lines else HEAD)
    field = dict(line.split('=', 1) for line in lines[1:head])
    arm = field['Arm']
    table = []
    for line in lines[head+1:]:
        c = line.split(',')
        table.append((c[0], int(c[1]), int(c[2]), *((float(c[3]), int(c[4]), float(c[5])) if arm == 'S' else ())))
    o = {'arm': arm, **{k: int(field[k]) for k in ('N_obs', 'D_obs', 'M_obs')},
         'Temporal_access': list(map(int, field['Temporal_access'].split(','))),
         'Events_per_window': [None if x == 'NA' else int(x) for x in field['Events_per_window'].split(',')],
         'parameter': float(field['p']) if arm == 'B' else sum(r[4] for r in table) if arm == 'S' else None, 'table': table}
    if 'n_panel' in field: o['n_panel'] = int(field['n_panel'])
    return o


def prompt(block):
    """The two messages every language model receives: the task, the arm's sampling rule and the sample."""
    read = lambda name: (PROMPTS/name).read_text().rstrip('\n')
    user = [read('task.txt'), 'Sampling rule: '+read(block.splitlines()[1][4:]+'.txt'), block,
            'Return the four full-archive estimates in the specified JSON format.']
    return [{'role': 'system', 'content': read('system.txt')}, {'role': 'user', 'content': '\n'.join(user)}]


def features(o, start):
    """The numbers ExtraTrees sees: shares of pairs and events per pattern, shares of events per
    window, the start estimate, sizes, what the arm states, and for arm S the shares of the walk's
    log per pattern. Nothing that the sample text does not contain."""
    table = {r[0].replace('?', '0'): r[1:] for r in o['table']}
    cell = lambda p, i: table[p][i] if p in table and len(table[p]) > i else 0.
    D, M, walk = o['D_obs'], o['M_obs'], o['arm'] == 'S'
    f = [int(o['arm'] == a) for a in ('R', 'S', 'S_obs', 'H', 'B')]       # S_obs: an arm that was dropped; its
    f += [cell(p, 0)/D if D else 0. for p in PATTERNS]                    # empty column keeps the layout
    f += [cell(p, 1)/M if M else 0. for p in PATTERNS]
    f += [(x or 0)/M if M else 0. for x in o['Events_per_window']]
    f += [M/D if D else 0.]
    f += list(start)
    f += [np.log1p(x) for x in (o['N_obs'], D, M)]
    f += [o['parameter'] if o['arm'] == 'B' else 0.]
    for i in (2, 3, 4):
        total = sum(r[i+1] for r in o['table']) if walk else 0.
        f += [cell(p, i)/total if total else 0. for p in PATTERNS]
    if 'n_panel' in o: f.append(np.log1p(o['n_panel']))
    return np.array(f, float)
