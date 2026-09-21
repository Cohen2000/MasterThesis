"""Small canonical graphs used by several test modules."""
import itertools
import numpy as np
import pandas as pd
from main_experiment.data import canonical


def graph(rows, key='fixture', proximity=False):
    return canonical(key, pd.DataFrame(rows, columns=['u', 'v', 't']), proximity, horizon=(0, 1))[0]


def tiny():
    """Three dyads, six events, all five windows used."""
    return graph([('a', 'b', 0.), ('b', 'a', .2), ('a', 'b', .4), ('a', 'c', .6), ('a', 'c', .8), ('b', 'c', 1.)])


def complete6():
    """Complete graph on six nodes; every dyad has events in all windows (H tests)."""
    return graph([(str(a), str(b), t) for a, b in itertools.combinations(range(6), 2)
                  for t in [0., .199, .2, .399, .4, .47, .6, .8, 1.]], 'fixture_h')


def ring(n=40, per=6, seed=0):
    """A sparse random-time ring, large enough for integer panel rounding."""
    r = np.random.default_rng(seed)
    rows = [(f'n{i}', f'n{(i+1+i % 3) % n}', float(r.uniform(0, 1))) for i in range(n) for _ in range(per)]
    return graph(rows)
