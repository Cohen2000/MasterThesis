"""The five v10 observation arms and their budget calibration.

Every arm is matched on the expected number of observed active dyad-windows,
T = 0.10 * sum_e K_e, computed per graph (surrogates separately from parents):

  R  uniform node panel of size n; every dyad with both endpoints in the panel is
     observed with its complete history. n minimises |pi(n) sum_e K_e - T| with
     pi(n) = n(n-1)/(N(N-1)).
  S/S_obs  one interaction-following walk: a uniform start vertex, then an
     incident event record chosen uniformly at each step. Each dyad has
     transition weight m_e; exactly L steps, with no burn-in or restart.
     Every traversed dyad is retrieved with its complete history. The arms
     share draws; S releases the crawler log and S_obs withholds it.
     L is calibrated on 256 walks and validated on 1024 (4096 if needed).
  H  uniform node panel, but only events with t >= t_start + (1-h)(t_end-t_start)
     are retrievable (h = 0.60). n minimises |pi(n) sum_e J_e - T|, where J_e is
     the number of windows with a retrievable event.
  B  every event record is kept independently with probability p, solved so
     that sum over active cells of 1-(1-p)^(events in cell) equals T.

Common random numbers: the sampler stream is keyed by the parent source, so a
surrogate uses its parent's node permutation (R, H), walk stream (S/S_obs) and
per-record uniforms (B). Only the full-archive quantities above are used to
set n, L and p; no realised sample is ever used.
"""
from decimal import Decimal
import numpy as np
from .common import (BUDGET_TOLERANCE, COVERAGE_FRACTION, DESIGN_VERSION, H_FRACTION,
                     parent_source, rng, sampler_id, seed)
from .data import window_counts

from .walk_v10_audit import (Walk, walk_length, validate_walk_length,
                             MAX_RELATIVE_MCSE)


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
              'S': not walk_reasons, 'S_obs': not walk_reasons,
              'H': budget['h_within_tolerance'],
              'B': abs(budget['bernoulli_relative_budget_error']) <= BUDGET_TOLERANCE}
    reasons = [f'S:{x}' for x in walk_reasons]
    if not by_arm['R']: reasons.append('R:node_panel_outside_5_percent')
    if not by_arm['H']: reasons.append('H:expected_volume_outside_5_percent')
    if not by_arm['B']: reasons.append('B:expected_volume_outside_5_percent')
    budget.update(calibration)
    budget.update({k: v for k, v in validation.items() if k != 'validation_volumes'})
    budget.update({'walk_type': 'interaction_following_random_walk', 'walk_components': walk.n_components,
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
    counts for S/S_obs and None otherwise.
    """
    if index < 1: raise ValueError('sample indices start at 1')
    stream_arm = 'S' if arm == 'S_obs' else arm
    stream = (domain, parent_source(g.key), draw_sampler_id(stream_arm, budget), index)
    if arm == 'R':
        return g.counts*node_panel_mask(g, rng(*stream), budget['n_panel'])[:, None], None
    if arm in ('S', 'S_obs'):
        traversals = walk.run([seed(*stream)], int(budget['L']), True)[2][0]
        return g.counts*(traversals > 0)[:, None], traversals
    if arm == 'H':
        panel = history_panel_mask(g, index, domain, budget)
        return history_counts(g, budget['history_fraction'])*panel[:, None], None
    if arm == 'B':
        keep = rng(*stream).random(g.M) < budget['p']
        return window_counts(g.pair[keep], g.w[keep], g.D), None
    raise ValueError('unknown arm')
