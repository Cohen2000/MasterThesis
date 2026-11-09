"""The four samplers: how a sample of a network is taken, and how large it is.

  R  node sampling: random users; every pair of two sampled users is seen with its full history.
  S  random walk: every pair the walk passes is seen with its full history.
  H  time truncation: random users, but only events in the last three of the five windows are seen.
  B  event sampling: every event is kept with probability p, independently.

The size of each sampler (users, steps or p) is set per network so that it sees, on average,
10% of the active (pair, window) cells. Node, time and event sampling: Rocha, Masuda & Holme (2017).
A time-shuffled copy uses the random numbers of its original network, so the two differ only in their timing.
"""
import numpy as np
from .common import BUDGET, H_WINDOWS, LABEL, W, family, rng, seed
from .data import window_counts
from .walk import walk_length


def panel_size(cells, target, N):
    """Number of nodes n whose pairs hold `target` of the cells on average (the closest n; the smaller on a tie)."""
    n = np.arange(N+1, dtype=np.int64)
    expected = cells*(n.astype(float)*(n-1))/(N*(N-1))
    return int(np.argmin(abs(expected-target)))


def late_counts(g):
    """Events per pair and window in the part of the time span that arm H sees."""
    keep = g.t >= g.horizon[0]+(1-H_WINDOWS/W)*(g.horizon[1]-g.horizon[0])
    return window_counts(g.pair[keep], g.w[keep], g.D)


def keep_probability(g, target):
    """p for which the expected number of cells that keep at least one event is `target`.
    Found by halving; kept to 12 digits so that it is the same on every machine."""
    n = g.counts[g.counts > 0].astype(float)
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo+hi)/2
        if (float(np.sum(-np.expm1(n*np.log1p(-mid)))) if mid < 1 else float(len(n))) < target: lo = mid
        else: hi = mid
    return float(f'{hi:.12g}')


def sizes(g, walk):
    """The size of every arm for one network: nodes (R, H), steps (S) and keep probability (B)."""
    target = BUDGET*g.cells
    late = int((late_counts(g) > 0).sum())
    return {'R': panel_size(g.cells, target, g.N), 'S': walk_length(g, walk, target),
            'H': panel_size(late, target, g.N) if late else g.N, 'B': keep_probability(g, target)}


def draws(g, arm, size, n):
    """How many samples to draw: an H sample of all nodes is always the same, so it is drawn once."""
    return 1 if arm == 'H' and size['H'] == g.N else n


def draw(g, arm, index, domain, size, walk=None):
    """One sample: the seen events per pair and window, and for S the walk's visits per pair."""
    stream = (domain, family(g.key), LABEL[arm], index)
    if arm == 'S':
        visits = walk.run([seed(*stream)], int(size['S']), True)[2][0]
        return g.counts*(visits > 0)[:, None], visits
    if arm == 'B':
        keep = rng(*stream).random(g.M) < size['B']
        return window_counts(g.pair[keep], g.w[keep], g.D), None
    nodes = np.zeros(g.N, dtype=bool)
    nodes[rng(*stream).permutation(g.N)[:size[arm]]] = True
    seen = nodes[g.ends[:, 0]] & nodes[g.ends[:, 1]]
    return (g.counts if arm == 'R' else late_counts(g))*seen[:, None], None
