"""The random walk of arm S: it starts at a random node and repeatedly follows a random event of
the node it is at, so a pair is chosen in proportion to its number of events. The steps run in walk.cpp.

Random walk on a weighted graph: Masuda, Porter & Lambiotte (2017).
"""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import LABEL, seed, sha


class Walk:
    def __init__(self, g):
        source = Path(__file__).with_name('walk.cpp')
        library = Path(tempfile.gettempdir())/f'walk_{sha(source)[:16]}.so'
        if not library.exists():        # compiled once
            tmp = library.with_suffix(f'.{id(self)}.so')
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(tmp)], check=True)
            tmp.replace(library)
        self.kernel = ctypes.CDLL(str(library)).walks
        self.kernel.argtypes = [ctypes.c_int64, ctypes.c_int64]+[ctypes.c_void_p]*8+[ctypes.c_int64, ctypes.c_int64]+[ctypes.c_void_p]*4
        self.kernel.restype = None
        self.g = g
        # Neighbours of every node, with the running sum of the events of its pairs.
        u, v = g.ends.T
        nodes = np.r_[u, v]
        order = np.argsort(nodes, kind='stable')
        self.neighbors = np.ascontiguousarray(np.r_[v, u][order], dtype=np.int64)
        self.edges = np.ascontiguousarray(np.tile(np.arange(g.D), 2)[order], dtype=np.int64)
        degree = np.bincount(nodes, minlength=g.N).astype(np.int64)
        self.ptr = np.r_[0, np.cumsum(degree)].astype(np.int64)
        total = np.cumsum(np.asarray(g.m, dtype=np.int64)[self.edges])
        self.cumulative = np.ascontiguousarray(total-np.r_[0, total][np.repeat(self.ptr[:-1], degree)], dtype=np.int64)
        # A first visit of a pair adds its K active windows to what the walk has seen; a walk
        # cannot see more than the cells of the connected part it starts in.
        self.cells = np.ascontiguousarray(g.K, dtype=np.int64)
        adjacency = coo_matrix((np.ones(len(nodes)), (nodes, np.r_[v, u])), shape=(g.N, g.N)).tocsr()
        _, part = connected_components(adjacency, directed=False)
        self.reachable = np.ascontiguousarray(np.bincount(part[u], weights=self.cells).astype(np.int64)[part])

    def run(self, seeds, L, visits=False):
        """Walks of L steps. Returns the cells seen after each step (summed over the walks), the
        cells seen per walk and, on request, how often each walk passed each pair."""
        seeds = np.asarray(seeds, dtype=np.uint64)
        seen = np.zeros(L+1, dtype=np.int64)
        per_walk = np.zeros(len(seeds), dtype=np.int64)
        counts = np.zeros((len(seeds), self.g.D), dtype=np.int64) if visits else None
        done = np.zeros(len(seeds), dtype=np.int64)
        arrays = [self.ptr, self.neighbors, self.edges, self.cumulative, self.cells, self.reachable, seeds]
        self.kernel(self.g.N, self.g.D, *[a.ctypes.data for a in arrays], None, len(seeds), L, seen.ctypes.data,
                    per_walk.ctypes.data, counts.ctypes.data if visits else None, done.ctypes.data)
        return np.cumsum(seen), per_walk, counts


def walk_length(g, walk, target):
    """The number of steps L after which 256 test walks have seen `target` cells on average.
    The search doubles L until the target is reached (at most min(100 pairs, 1,000,000) steps)
    and then takes the L that comes closest."""
    cap = min(100*g.D, 1_000_000)
    seeds = [seed('walk_calibration_cells', g.key, LABEL['S'], i) for i in range(1, 257)]
    goal, L = 256*target, 1
    while True:
        seen = walk.run(seeds, L)[0]
        if seen[-1] >= goal or L == cap: break
        L = min(2*L, cap)
    if seen[-1] < goal: return cap
    lo, hi = 0, L
    while hi-lo > 1:
        mid = (lo+hi)//2
        if seen[mid] >= goal: hi = mid
        else: lo = mid
    return min((lo, hi), key=lambda l: (abs(int(seen[l])-goal), l))
