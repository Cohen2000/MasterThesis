"""Weighted-walk kernel and v9 budget calibration, frozen for the v10 gate audit."""
import ctypes
from pathlib import Path
import subprocess
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import seed, sha

AUDIT_ARM_ID = 'S-interaction-p888-access-v10-20260923'
CALIBRATION_WALKS = 256
VALIDATION_WALKS = 1024
EXTENDED_VALIDATION_WALKS = 4096
MAX_RELATIVE_MCSE = .01

class Walk:
    """Integer-weighted walk on the dyad support, implemented in walk_kernel.cpp.

    A first traversal of dyad e adds K_e to the walk's discovered volume, the
    quantity calibrated against T. Production audit weights equal event counts.
    """

    def __init__(self, g, build_dir, transition_weights=None):
        build = Path(build_dir); build.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).with_name('walk_v10_kernel.cpp')
        library = build/f'walk_{sha(source)[:16]}.so'
        if not library.exists():
            tmp = library.with_suffix('.tmp.so')
            subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(source), '-o', str(tmp)], check=True)
            tmp.replace(library)
        self.kernel = ctypes.CDLL(str(library)).walks
        self.kernel.argtypes = ([ctypes.c_int64, ctypes.c_int64]+[ctypes.c_void_p]*7+
                                [ctypes.c_int64, ctypes.c_int64]+[ctypes.c_void_p]*4)
        self.kernel.restype = None
        self.g = g
        # Adjacency in CSR form: for node x, neighbours[ptr[x]:ptr[x+1]] via dyads edges[...].
        u, v = g.ends.T
        nodes = np.r_[u, v]
        order = np.argsort(nodes, kind='stable')
        self.neighbors = np.ascontiguousarray(np.r_[v, u][order], dtype=np.int64)
        self.edges = np.ascontiguousarray(np.tile(np.arange(g.D), 2)[order], dtype=np.int64)
        self.degree = np.bincount(nodes, minlength=g.N).astype(np.int64)
        self.ptr = np.r_[0, np.cumsum(self.degree)].astype(np.int64)
        # Integer edge weights are passed explicitly; production uses event counts.
        edge_weight = np.asarray(g.m if transition_weights is None else transition_weights, dtype=np.int64)
        if edge_weight.shape != (g.D,) or np.any(edge_weight <= 0):
            raise ValueError('transition weights must be positive integers per dyad')
        self.transition_weights = edge_weight
        neighbor_degree = edge_weight[self.edges]
        start = np.repeat(self.ptr[:-1], self.degree)
        total = np.cumsum(neighbor_degree)
        self.cumulative = np.ascontiguousarray(total-np.r_[0, total][start], dtype=np.int64)
        adjacency = coo_matrix((np.ones(len(nodes)), (nodes, np.r_[v, u])), shape=(g.N, g.N)).tocsr()
        self.n_components, self.components = connected_components(adjacency, directed=False)
        self.weight = np.ascontiguousarray(g.K, dtype=np.int64)
        totals = np.bincount(self.components[u], weights=self.weight).astype(np.int64)
        # Largest volume a walk from each start node can discover (its component's cells).
        self.component_volume = np.ascontiguousarray(totals[self.components])

    def run(self, seeds, L, traversals=False):
        """Returns (cumulative discovered volume summed over paths by step,
        final volume per path, traversal counts per path and dyad or None,
        executed transitions per path)."""
        if not isinstance(L, int) or L < 0: raise ValueError('invalid L')
        seeds = np.asarray(seeds, dtype=np.uint64)
        delta = np.zeros(L+1, dtype=np.int64)
        volumes = np.zeros(len(seeds), dtype=np.int64)
        counts = np.zeros((len(seeds), self.g.D), dtype=np.int64) if traversals else None
        executed = np.zeros(len(seeds), dtype=np.int64)
        arrays = [self.ptr, self.neighbors, self.edges, self.cumulative, self.weight, self.component_volume, seeds]
        self.kernel(self.g.N, self.g.D, *[a.ctypes.data for a in arrays], len(seeds), L,
                    delta.ctypes.data, volumes.ctypes.data,
                    counts.ctypes.data if counts is not None else None, executed.ctypes.data)
        return np.cumsum(delta), volumes, counts, executed



def walk_length(g, walk, T):
    """Smallest-error L such that 256 common-seed walks discover 256*T cells in total.

    The summed discovery curve is monotone in L, so the search doubles L until the
    target (or the cap C = min(100 D, 10^6)) is reached and then bisects.
    """
    C = min(100*g.D, 1_000_000)
    seeds = [seed('walk_calibration_cells', g.key, AUDIT_ARM_ID, i) for i in range(1, CALIBRATION_WALKS+1)]
    target = CALIBRATION_WALKS*T
    bound = 1
    while True:
        total = walk.run(seeds, bound)[0]
        if total[-1] >= target or bound == C: break
        bound = min(2*bound, C)
    reached = bool(total[-1] >= target)
    if not reached:
        L = C
    else:
        lo, hi = 0, bound
        while hi-lo > 1:
            mid = (lo+hi)//2
            if total[mid] >= target: hi = mid
            else: lo = mid
        L = min((lo, hi), key=lambda l: (abs(int(total[l])-target), l))
    return {'L': L, 'C': C, 'calibration_paths': CALIBRATION_WALKS,
            'calibration_mean': float(total[L]/CALIBRATION_WALKS), 'search_limit_reached_without_budget': not reached}


def validate_walk_length(g, walk, L, T):
    """Mean discovered cells of independent validation walks at the fixed L (no adaptation)."""
    def volumes(first, last):
        seeds = [seed('walk_validation_cells', g.key, AUDIT_ARM_ID, i) for i in range(first, last+1)]
        return walk.run(seeds, L)[1].tolist()
    values = volumes(1, VALIDATION_WALKS)
    mcse = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    if mcse/T > MAX_RELATIVE_MCSE:
        values += volumes(VALIDATION_WALKS+1, EXTENDED_VALIDATION_WALKS)
        mcse = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    return {'validation_n': len(values), 'validation_mean': float(np.mean(values)), 'validation_mcse': mcse,
            'validation_volumes': values}
