"""Synthetic networks with known persistence: two generators, the 8 test networks, the 400 training networks.

DAR switches every pair of a fixed backbone on or off per window; with probability alpha a pair
copies its previous state (memory). Activity-driven (AD): in each round an active node contacts one
other node, at random or, with memory, preferably a node it already knows.
"""
import numpy as np
import pandas as pd
from .common import rng
from .data import graph

DAR = dict(N=500, E=5000, chi=.2, nu=1.)                            # settings of the test networks
AD = dict(N=500, tail=1.8, eps=.01, eta=1., rounds=1000, c=1.)       # tail = exponent of the activity distribution - 1
TEST = ('dar_a0_r1', 'dar_a08_r1', 'dar_a0_r2', 'dar_a08_r2', 'ad_memoryless_r1', 'ad_memory_r1', 'ad_memoryless_r2', 'ad_memory_r2')
POOL = 'pool-v3-panel888-20260921'      # fixed label of the training networks (part of their seeds)


# DAR(1) link activity: Williams, Mazzarisi, Lillo & Latora (2022). The events per active window, 1 + Poisson(nu), are our own.
def dar_latents(r, N, E, chi, nu):
    """Backbone and all random numbers of one run, shared by every alpha."""
    candidates = np.column_stack(np.triu_indices(N, 1))
    edges = candidates[np.sort(r.choice(len(candidates), E, replace=False))]
    initial = r.random(E) < chi
    copy = r.random((4, E)); refresh = r.random((4, E)) < chi
    counts = 1+r.poisson(nu, (5, E))
    offsets = np.r_[0, np.cumsum(counts.ravel())]
    pos = r.random(int(offsets[-1]))
    return dict(edges=edges, initial=initial, copy=copy, refresh=refresh, counts=counts, offsets=offsets, pos=pos, E=E)


def dar_rows(L, alpha):
    """Copy the previous state with probability alpha, otherwise draw anew; then place the events."""
    E = L['E']; states = np.empty((5, E), bool); states[0] = L['initial']
    for j in range(1, 5): states[j] = np.where(L['copy'][j-1] < alpha, states[j-1], L['refresh'][j-1])
    rows = []
    for j, e in zip(*np.where(states)):
        ix = j*E+e; a, b = L['offsets'][ix:ix+2]
        for t in (j+L['pos'][a:b])/5: rows.append((int(L['edges'][e, 0]), int(L['edges'][e, 1]), float(t)))
    return rows, states


# Activity-driven networks: Perra et al. (2012); memory rule P(new contact) = c / (n + c): Karsai, Perra & Vespignani (2014).
def ad_latents(r, N, tail, eps, eta, rounds):
    activities = eta*(eps**(-tail)+r.random(N)*(1-eps**(-tail)))**(-1/tail)
    return dict(activities=activities, activation=r.random((rounds, N)), decision=r.random((rounds, N)),
                partner=r.random((rounds, N)), N=N, rounds=rounds)


def ad_rows(L, mode, c):
    """Rounds in which every active node makes one contact; a pair counts once per round."""
    N = L['N']; rounds = L['rounds']
    memory = [set() for _ in range(N)]; rows = []; mutual = 0
    for t in range(rounds):
        contacts = set(); initiations = 0
        for i in np.flatnonzero(L['activation'][t] < L['activities']):
            if mode == 'memoryless':
                k = int(L['partner'][t, i]*(N-1)); j = k if k < i else k+1      # a random other node
            else:
                old = memory[i]
                new = len(old) == 0 or (len(old) < N-1 and L['decision'][t, i] < c/(len(old)+c))
                choices = [j for j in range(N) if j != i and j not in old] if new else sorted(old)
                j = choices[int(L['partner'][t, i]*len(choices))]
            contacts.add((min(int(i), j), max(int(i), j))); initiations += 1
        mutual += initiations-len(contacts)
        for i, j in sorted(contacts):
            rows.append((i, j, (t+.5)/rounds)); memory[i].add(j); memory[j].add(i)
    return rows, mutual


def _graph(key, rows):
    return graph(key, pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))


def test_networks():
    """The 8 synthetic test networks: two runs per generator, each without and with memory on the
    same random numbers (common random numbers, Glasserman & Yao 1992)."""
    for run in (1, 2):
        L = dar_latents(rng('graph', f'dar_pair_r{run}'), **DAR)
        for mode, alpha in (('a0', 0.), ('a08', .8)): yield _graph(f'dar_{mode}_r{run}', dar_rows(L, alpha)[0])
    for run in (1, 2):
        L = ad_latents(rng('graph', f'ad_pair_r{run}'), **{k: AD[k] for k in ('N', 'tail', 'eps', 'eta', 'rounds')})
        for mode in ('memoryless', 'memory'): yield _graph(f'ad_{mode}_r{run}', ad_rows(L, mode, AD['c'])[0])


def training_specs():
    """Settings of the 400 synthetic training networks of ExtraTrees: 200 DAR and 200 AD networks,
    spread evenly over size, activity and memory. The ranges were fixed before any result."""
    r = rng('pool_definition', POOL+':train')
    sizes = [200, 300, 500, 800, 1200]
    cut = lambda lo, hi: np.linspace(lo, hi, 6)                    # five equal bins of a range
    chi, alpha, eta = cut(.04, .45), cut(0., .9), cut(.2, 1.)
    dar_sizes = sizes*40; r.shuffle(dar_sizes)
    ad_grid = [(n, k) for n in range(5) for k in range(5)]*8; r.shuffle(ad_grid)
    specs = []
    for i in range(5):                                             # DAR: 8 networks per (chi, alpha) bin
        for j in range(5):
            for N in dar_sizes[8*(5*i+j):8*(5*i+j)+8]:
                degree = float(r.uniform(4., 24.))
                specs.append(dict(family='dar', N=N, E=min(int(round(degree*N/2)), N*(N-1)//2),
                                  chi=float(r.uniform(chi[i], chi[i+1])), alpha=float(r.uniform(alpha[j], alpha[j+1])),
                                  nu=float(r.uniform(.25, 3.))))
    for m, mode in enumerate(('memoryless', 'memory')):            # AD: 20 networks per (mode, eta) bin
        for j in range(5):
            for n, k in ad_grid[20*(5*m+j):20*(5*m+j)+20]:
                spec = dict(family='ad', N=sizes[n], rounds=[500, 750, 1000, 1500, 2000][k], mode=mode,
                            tail=float(r.uniform(.8, 2.4)), eps=float(r.uniform(.005, .05)), eta=float(r.uniform(eta[j], eta[j+1])))
                if mode == 'memory': spec['c'] = float(r.uniform(.5, 4.))
                specs.append(spec)
    for number, spec in enumerate(specs):
        spec['key'] = f"{POOL}_{spec['family']}_train_{number % 200 + 1:04d}"
    return specs


def training_network(spec):
    """One synthetic training network, generated from its own seed."""
    r, p = rng('pool', spec['key']), spec
    if p['family'] == 'dar':
        return _graph(p['key'], dar_rows(dar_latents(r, p['N'], p['E'], p['chi'], p['nu']), p['alpha'])[0])
    L = ad_latents(r, p['N'], p['tail'], p['eps'], p['eta'], p['rounds'])
    return _graph(p['key'], ad_rows(L, p['mode'], p.get('c', AD['c']))[0])
