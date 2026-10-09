"""Networks: raw event files, the clean graph with five time windows, and time-shuffled twins.

A network is a list of events "node u with node v at time t". Pairs are undirected, events of a
node with itself are dropped, and the time span is cut into five equal windows. K is the number
of windows in which a pair is active; rho_k is the share of pairs with K >= k (k = 2..5).
"""
from dataclasses import dataclass, replace
import gzip
import io
import zipfile
import numpy as np
import pandas as pd
import yaml
from .common import RAW, ROOT, TWIN, W, rng, sha

NETWORKS = yaml.safe_load((ROOT/'config/networks.yaml').read_text())


@dataclass
class Graph:
    key: str
    u: np.ndarray        # first node of every event (u < v)
    v: np.ndarray
    t: np.ndarray        # time of every event
    w: np.ndarray        # window 0..4 of every event
    pair: np.ndarray     # pair index of every event
    ends: np.ndarray     # (D, 2) the two nodes of every pair
    counts: np.ndarray   # (D, 5) events per pair and window
    horizon: tuple       # (first time, last time)

    @property
    def N(self): return int(self.ends.max())+1             # nodes
    @property
    def D(self): return len(self.ends)                     # pairs
    @property
    def M(self): return len(self.t)                        # events
    @property
    def m(self): return self.counts.sum(axis=1)            # events per pair
    @property
    def K(self): return (self.counts > 0).sum(axis=1)      # active windows per pair
    @property
    def cells(self): return int((self.counts > 0).sum())   # active (pair, window) cells
    @property
    def truth(self): return [float(np.mean(self.K >= k)) for k in range(2, 6)]


def window_of(t, horizon):
    """Window of each time stamp; a time stamp exactly on a cut belongs to the later window."""
    cuts = np.array([horizon[0]+(horizon[1]-horizon[0])*j/W for j in range(1, W)])
    return np.searchsorted(cuts, t, side='right').astype(np.int64)


def window_counts(pair, w, D):
    return np.bincount(pair*W+w, minlength=D*W).reshape(-1, W).astype(np.int64)


def graph(key, events, proximity=False, horizon=None):
    """The clean graph of an event table with columns u, v, t."""
    x = events[['u', 'v', 't']].copy()
    x.u = x.u.astype(str); x.v = x.v.astype(str)
    x = x[x.u != x.v].copy()
    lo = np.minimum(x.u.to_numpy(), x.v.to_numpy()); hi = np.maximum(x.u.to_numpy(), x.v.to_numpy())
    x['u'] = lo; x['v'] = hi
    if proximity: x = x.drop_duplicates(['u', 'v', 't'])     # a sensor reports the same contact several times
    x = x.sort_values(['u', 'v', 't'], kind='stable').reset_index(drop=True)
    bounds = tuple(map(float, horizon or (x.t.min(), x.t.max())))
    labels = np.unique(np.r_[x.u.to_numpy(), x.v.to_numpy()])
    u = np.searchsorted(labels, x.u).astype(np.int64)
    v = np.searchsorted(labels, x.v).astype(np.int64)
    ends, pair = np.unique(np.column_stack([u, v]), axis=0, return_inverse=True)
    pair = pair.astype(np.int64)
    t = x.t.to_numpy(float)
    w = window_of(t, bounds)
    return Graph(key, u, v, t, w, pair, ends, window_counts(pair, w, len(ends)), bounds)


def read_events(spec, raw=RAW):
    """The rows of a raw file as a table u, v, t. Rows that cannot be read are skipped."""
    path = raw/spec['file']
    if sha(path) != spec['sha256']: raise ValueError(f'{path} is not the file of the study')
    if 'zip' in spec:
        with zipfile.ZipFile(path) as z: lines = io.TextIOWrapper(z.open(spec['zip']), encoding='utf-8').readlines()
    else:
        with (gzip.open if path.suffix == '.gz' else open)(path, 'rt') as f: lines = f.readlines()
    cu, cv, ct = spec['columns']
    rows = []
    for line in lines[spec.get('skip', 0):]:
        text = line.strip()
        if not text or text[0] in '#%': continue
        cells = text.split(spec.get('sep'))
        try:
            u, v, t = cells[cu].strip(), cells[cv].strip(), float(cells[ct])
        except (ValueError, IndexError):
            continue
        if not u or not v or not np.isfinite(t): continue
        if spec.get('no_device') and '-' in u+v: continue
        rows.append(('u:'+u, 'i:'+v, t) if spec.get('two_kinds') else (u, v, t))
    return pd.DataFrame(rows, columns=['u', 'v', 't'])


def network(key, raw=RAW):
    """One of the 19 real networks, built from its raw file."""
    spec = NETWORKS[key]
    return graph(key, read_events(spec, raw), spec.get('proximity', False))


def twin(g):
    """The time-shuffled twin: the same pairs with the same number of events, but the time stamps
    are shuffled over all events (shuffle P[w,t] of Gauvin et al. 2022)."""
    t = rng('pwt_productive', g.key).permutation(g.t)
    w = window_of(t, g.horizon)
    return replace(g, key=g.key+TWIN, t=t, w=w, counts=window_counts(g.pair, w, g.D))
