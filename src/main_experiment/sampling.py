"""The four observation mechanisms and their budget calibration.

Every arm is matched on the expected number of observed active dyad-windows,
T = 0.10 * sum_e K_e, computed per graph (surrogates separately from parents):

  R  uniform node panel of size n; every dyad with both endpoints in the panel is
     observed with its complete history. n minimises |pi(n) sum_e K_e - T| with
     pi(n) = n(n-1)/(N(N-1)).
  S  one degree-biased random walk: uniform start vertex, next vertex v with
     probability d_v / sum_{x in N(u)} d_x (d = distinct neighbours in the
     full-archive support), exactly L transitions, no burn-in or restart; every
     traversed dyad is observed once with its complete history. L is calibrated
     on 256 walks and validated on 1024 (4096 if the relative MCSE exceeds 1%).
  H  uniform node panel, but only events with t >= t_start + (1-h)(t_end-t_start)
     are retrievable (h = 0.60). n minimises |pi(n) sum_e J_e - T|, where J_e is
     the number of windows with a retrievable event.
  B  every event record is kept independently with probability p, solved so
     that sum over active cells of 1-(1-p)^(events in cell) equals T.

Common random numbers: the sampler stream is keyed by the parent source, so a
surrogate uses its parent's node permutation (R, H), walk stream (S) and
per-record uniforms (B). Only the full-archive quantities above are used to
set n, L and p; no realised sample is ever used.
"""
import ctypes
from decimal import Decimal
from pathlib import Path
import subprocess
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import (ARM_ID, BUDGET_TOLERANCE, COVERAGE_FRACTION, DESIGN_VERSION, H_FRACTION,
                     parent_source, rng, sampler_id, seed, sha)
from .data import window_counts

CALIBRATION_WALKS = 256
VALIDATION_WALKS = 1024
EXTENDED_VALIDATION_WALKS = 4096
MAX_RELATIVE_MCSE = 0.01


class Walk:
    """Degree-biased random walk on the dyad support, implemented in walk_kernel.cpp.

    A first traversal of dyad e adds K_e to the walk's discovered volume, the
    quantity calibrated against T. Event multiplicities never affect transitions.
    """

    def __init__(self, g, build_dir):
        build = Path(build_dir); build.mkdir(parents=True, exist_ok=True)
        source = Path(__file__).with_name('walk_kernel.cpp')
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
        # Cumulative neighbour degrees within each vertex's adjacency range.
        neighbour_degree = self.degree[self.neighbors]
        start = np.repeat(self.ptr[:-1], self.degree)
        total = np.cumsum(neighbour_degree)
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


# ---------------------------------------------------------------- analytic arms R, H, B
def panel_size(total, T, N):
    """Integer panel size n in 0..N whose expected observed volume pi(n)*total is closest to T.

    Ties choose the smaller n.
    """
    n = np.arange(N+1, dtype=np.int64)
    expected = total*(n.astype(float)*(n-1))/(N*(N-1))
    best = int(np.argmin(abs(expected-T)))
    return best, float(expected[best])


def history_start(g, h=H_FRACTION):
    """Earliest retrievable time of arm H: t_start + (1-h)(t_end-t_start)."""
    if not 0 < h <= 1: raise ValueError('history fraction must be in (0,1]')
    start_fraction = float(Decimal(1)-Decimal(str(h)))
    return g.horizon[0]+start_fraction*(g.horizon[1]-g.horizon[0])


def history_counts(g, h=H_FRACTION):
    """Events per dyad and window that are retrievable by arm H."""
    keep = g.t >= history_start(g, h)
    return window_counts(g.pair[keep], g.w[keep], g.D)


def h_parameters(g, T, h=H_FRACTION):
    retrievable = history_counts(g, h)
    J_total = int((retrievable > 0).sum())
    n, expected = panel_size(J_total, T, g.N)
    if not J_total: n = g.N          # nothing retrievable: the maximal panel still observes nothing
    pi = n*(n-1)/(g.N*(g.N-1))
    relative_error = float((expected-T)/T)
    return {'history_fraction': h, 'history_start': history_start(g, h), 'query_time': float(g.horizon[1]),
            'J_total': J_total, 'n_panel_history': n, 'h_node_share': n/g.N, 'h_panel_dyad_inclusion': pi,
            'h_expected_cells': expected, 'h_expected_events': float(pi*retrievable.sum()),
            'h_relative_budget_error': relative_error, 'h_target_unreachable': bool(J_total < T),
            'h_saturated': bool(n == g.N), 'h_within_tolerance': bool(abs(relative_error) <= BUDGET_TOLERANCE)}


def bernoulli_p(g, T):
    """Retention probability p with expected observed active cells equal to T.

    A cell with n events is observed with probability 1-(1-p)^n; the expectation
    increases in p, so bisection (80 halvings) finds p to machine precision. p is
    then kept to 12 significant digits, so that the Bernoulli draws do not depend
    on last-bit floating-point differences between machines.
    """
    n = g.counts[g.counts > 0].astype(float)

    def expected_cells(p):
        return float(np.sum(-np.expm1(n*np.log1p(-p)))) if p < 1 else float(len(n))
    lo, hi = 0., 1.
    for _ in range(80):
        mid = (lo+hi)/2
        if expected_cells(mid) < T: lo = mid
        else: hi = mid
    p = float(f'{hi:.12g}')
    return p, expected_cells(p)


def analytic_parameters(g, fraction=COVERAGE_FRACTION):
    """Parameters of R, H and B, each matched to T = fraction * sum_e K_e (main study: 0.10)."""
    if g.N < 2: raise ValueError('undefined budget')
    cells = g.cells
    T = fraction*cells
    n, expected_r = panel_size(cells, T, g.N)
    pi = n*(n-1)/(g.N*(g.N-1))
    p, expected_b = bernoulli_p(g, T)
    return {'design_version': DESIGN_VERSION, 'matched_quantity': 'expected_observed_active_dyad_windows',
            'coverage_fraction': fraction, 'active_dyad_windows': cells, 'T': T,
            'budget_tolerance': BUDGET_TOLERANCE,
            'n_panel': n, 'node_expected_cells': expected_r, 'node_relative_budget_error': float((expected_r-T)/T),
            'node_expected_events': pi*g.M, 'node_dyad_share': pi,
            **h_parameters(g, T),
            'p': p, 'bernoulli_expected_cells': expected_b, 'bernoulli_relative_budget_error': float((expected_b-T)/T),
            'bernoulli_expected_events': p*g.M, 'bernoulli_dyad_share': float(np.mean(-np.expm1(g.m*np.log1p(-p))))}


# ---------------------------------------------------------------- arm S calibration
def walk_length(g, walk, T):
    """Smallest-error L such that 256 common-seed walks discover 256*T cells in total.

    The summed discovery curve is monotone in L, so the search doubles L until the
    target (or the cap C = min(100 D, 10^6)) is reached and then bisects.
    """
    C = min(100*g.D, 1_000_000)
    seeds = [seed('walk_calibration_cells', g.key, ARM_ID['S'], i) for i in range(1, CALIBRATION_WALKS+1)]
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
        seeds = [seed('walk_validation_cells', g.key, ARM_ID['S'], i) for i in range(first, last+1)]
        return walk.run(seeds, L)[1].tolist()
    values = volumes(1, VALIDATION_WALKS)
    mcse = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    if mcse/T > MAX_RELATIVE_MCSE:
        values += volumes(VALIDATION_WALKS+1, EXTENDED_VALIDATION_WALKS)
        mcse = float(np.std(values, ddof=1)/np.sqrt(len(values)))
    return {'validation_n': len(values), 'validation_mean': float(np.mean(values)), 'validation_mcse': mcse,
            'validation_volumes': values}


def calibrate(g, build_dir, fraction=COVERAGE_FRACTION):
    """Complete budget of one graph: analytic arms plus the calibrated walk length.

    Returns (budget, walk engine). budget['budget_matched_by_arm'] records whether
    each arm's expectation lies within 5% of T; unmatched arms are reported, never
    dropped or re-tuned.
    """
    budget = analytic_parameters(g, fraction)
    T = budget['T']
    walk = Walk(g, build_dir)
    calibration = walk_length(g, walk, T)
    validation = validate_walk_length(g, walk, calibration['L'], T)
    ceiling = float(np.mean(walk.component_volume))    # expected reachable cells from a uniform start
    walk_reasons = []
    if calibration['search_limit_reached_without_budget']: walk_reasons.append('calibration_cap')
    if abs(validation['validation_mean']-T)/T > BUDGET_TOLERANCE: walk_reasons.append('validation_mean_outside_5_percent')
    if validation['validation_mcse']/T > MAX_RELATIVE_MCSE: walk_reasons.append('validation_mcse_above_1_percent')
    if ceiling < T: walk_reasons.append('component_structural_ceiling_below_target')
    by_arm = {'R': abs(budget['node_relative_budget_error']) <= BUDGET_TOLERANCE,
              'S': not walk_reasons,
              'H': budget['h_within_tolerance'],
              'B': abs(budget['bernoulli_relative_budget_error']) <= BUDGET_TOLERANCE}
    reasons = [f'S:{x}' for x in walk_reasons]
    if not by_arm['R']: reasons.append('R:node_panel_outside_5_percent')
    if not by_arm['H']: reasons.append('H:expected_volume_outside_5_percent')
    if not by_arm['B']: reasons.append('B:expected_volume_outside_5_percent')
    budget.update(calibration)
    budget.update({k: v for k, v in validation.items() if k != 'validation_volumes'})
    budget.update({'walk_type': 'degree_biased_random_walk', 'walk_components': walk.n_components,
                   'walk_expected_component_ceiling': ceiling, 'walk_structural_target_unreachable': ceiling < T,
                   'validation_relative_error': (validation['validation_mean']-T)/T,
                   'walk_budget_matched': not walk_reasons, 'walk_unmatched_reasons': walk_reasons,
                   'budget_matched_by_arm': by_arm, 'budget_matched': all(by_arm.values()),
                   'unmatched_reasons': reasons})
    return budget, walk, validation['validation_volumes']


# ---------------------------------------------------------------- drawing observations
def node_panel_mask(g, r, size):
    """Dyads whose two endpoints are among the first `size` nodes of a uniform permutation.

    Prefixes of one permutation are nested uniform panels, which gives R/H common
    random numbers across graphs with identical node sets and across h values.
    """
    panel = np.zeros(g.N, dtype=bool)
    panel[r.permutation(g.N)[:size]] = True
    return panel[g.ends[:, 0]] & panel[g.ends[:, 1]]


def history_panel_mask(g, index, domain, budget):
    if budget['h_saturated'] and index != 1:
        raise ValueError('a saturated H panel has only one draw')
    r = rng(domain, parent_source(g.key), draw_sampler_id('H', budget), index)
    return node_panel_mask(g, r, budget['n_panel_history'])


def draw_sampler_id(arm, budget):
    # Budgets built by h_parameters alone (H diagnostics) belong to the main 0.10 study.
    return sampler_id(arm, budget.get('coverage_fraction', COVERAGE_FRACTION))


def draw(g, arm, index, domain, budget, walk=None):
    """Observed events per dyad and window for one sampler draw.

    Returns (counts, traversals); traversals are the walk's per-dyad traversal
    counts for arm S and None otherwise.
    """
    if index < 1: raise ValueError('sample indices start at 1')
    stream = (domain, parent_source(g.key), draw_sampler_id(arm, budget), index)
    if arm == 'R':
        return g.counts*node_panel_mask(g, rng(*stream), budget['n_panel'])[:, None], None
    if arm == 'S':
        traversals = walk.run([seed(*stream)], int(budget['L']), True)[2][0]
        return g.counts*(traversals > 0)[:, None], traversals
    if arm == 'H':
        panel = history_panel_mask(g, index, domain, budget)
        return history_counts(g, budget['history_fraction'])*panel[:, None], None
    if arm == 'B':
        keep = rng(*stream).random(g.M) < budget['p']
        return window_counts(g.pair[keep], g.w[keep], g.D), None
    raise ValueError('unknown arm')
