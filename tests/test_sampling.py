"""Budgets and draws of the four arms, the walk kernel and common random numbers."""
import itertools
import math
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from helpers import complete6, graph, ring, tiny
from main_experiment.common import (ARMS, COVERAGE_FRACTION, H_SENSITIVITY, MAIN_KEYS, TRAIN, draws_for, planned_sizes,
                                    seed)
from main_experiment.sampling import (Walk, analytic_parameters, calibrate, draw, h_parameters, history_counts,
                                      history_panel_mask, history_start)
from main_experiment.surrogates import shuffle


def reference_walk(walk, state, L):
    """Pure-Python degree-biased SplitMix64 walk, the specification of walk_kernel.cpp."""
    mask = (1 << 64)-1

    def next_int():
        nonlocal state
        state = (state+0x9e3779b97f4a7c15) & mask; z = state
        z = ((z ^ (z >> 30))*0xbf58476d1ce4e5b9) & mask
        z = ((z ^ (z >> 27))*0x94d049bb133111eb) & mask
        return z ^ (z >> 31)

    def bounded(n):
        threshold = ((1 << 64)-n) % n
        while True:
            x = next_int()
            if x >= threshold: return x % n
    node = bounded(walk.g.N); traversals = np.zeros(walk.g.D, dtype=np.int64); volume = [0]
    for _ in range(L):
        a, b = walk.ptr[node:node+2]
        neighbour_degrees = [int(walk.degree[walk.neighbors[j]]) for j in range(a, b)]
        x = bounded(sum(neighbour_degrees))
        j = a
        while x >= neighbour_degrees[j-a]:              # first neighbour whose cumulative degree exceeds x
            x -= neighbour_degrees[j-a]; j += 1
        traversals[walk.edges[j]] += 1; node = walk.neighbors[j]
        volume.append(int(walk.g.K[traversals > 0].sum()))
    return traversals, volume




class AnalyticArmTests(unittest.TestCase):
    def test_r_and_b_expectations_are_exact(self):
        g = tiny(); b = analytic_parameters(g); n = b['n_panel']
        cells = []
        for panel in itertools.combinations(range(g.N), n):             # every panel of size n
            chosen = np.isin(g.ends[:, 0], panel) & np.isin(g.ends[:, 1], panel)
            cells.append(int((g.counts[chosen] > 0).sum()))
        self.assertAlmostEqual(np.mean(cells), b['node_expected_cells'])
        expected = 0.
        for bits in itertools.product([0, 1], repeat=g.M):              # every Bernoulli outcome
            kept = np.array(bits, bool)
            c = np.bincount(g.pair[kept]*5+g.w[kept], minlength=g.D*5)
            expected += int((c > 0).sum())*b['p']**kept.sum()*(1-b['p'])**(g.M-kept.sum())
        self.assertAlmostEqual(expected, b['T'])
        self.assertAlmostEqual(b['T'], COVERAGE_FRACTION*g.cells)

    def test_retention_probability_is_machine_independent(self):
        from main_experiment.sampling import bernoulli_p
        g = ring(n=80, per=6, seed=2)
        p, expected = bernoulli_p(g, .1*g.cells)
        self.assertEqual(p, float(f'{p:.12g}'))                      # 12 significant digits
        self.assertAlmostEqual(expected, .1*g.cells, delta=1e-6*g.cells)

    def test_realised_cells_track_the_expectation(self):
        g = ring(n=80, per=6, seed=2); b = analytic_parameters(g)
        for arm, key in (('R', 'node_expected_cells'), ('H', 'h_expected_cells'), ('B', 'bernoulli_expected_cells')):
            cells = [int((draw(g, arm, i, 'test', b)[0] > 0).sum()) for i in range(1, 401)]
            self.assertLess(abs(np.mean(cells)-b[key]), .1*b['T'], arm)

    def test_bernoulli_keeps_each_event_with_probability_p(self):
        g = tiny(); b = analytic_parameters(g)
        volumes = [draw(g, 'B', i, 'test', b)[0].sum() for i in range(1, 4001)]
        self.assertLess(abs(np.mean(volumes)-b['p']*g.M), 6*math.sqrt(g.M*b['p']*(1-b['p'])/4000))


class HistoryArmTests(unittest.TestCase):
    def test_cutoff_is_elapsed_time(self):
        g = complete6()
        for h in (*H_SENSITIVITY, .53):
            start = history_start(g, h); direct = np.zeros_like(g.counts)
            for e, w, t in zip(g.pair, g.w, g.t):
                if t >= start: direct[e, w] += 1
            np.testing.assert_array_equal(history_counts(g, h), direct)
        self.assertEqual(history_start(g, .6), .4)

    def test_affine_time_change_preserves_access(self):
        g = complete6()
        import pandas as pd
        from main_experiment.data import canonical
        shifted = canonical('shifted', pd.DataFrame({'u': g.u, 'v': g.v, 't': 100+20*g.t}), horizon=(100, 120))[0]
        for h in H_SENSITIVITY: np.testing.assert_array_equal(history_counts(g, h), history_counts(shifted, h))

    def test_integer_panel_calibration_and_exact_expectation(self):
        g = complete6(); T = .1*g.cells
        for h in H_SENSITIVITY:
            b = h_parameters(g, T, h); c = history_counts(g, h); n = b['n_panel_history']
            expected = [m*(m-1)/(g.N*(g.N-1))*(c > 0).sum() for m in range(g.N+1)]
            self.assertEqual(n, min(range(g.N+1), key=lambda m: abs(expected[m]-T)))
            volumes = [int((c[np.isin(g.ends[:, 0], p) & np.isin(g.ends[:, 1], p)] > 0).sum())
                       for p in itertools.combinations(range(g.N), n)]
            self.assertAlmostEqual(np.mean(volumes), b['h_expected_cells'])

    def test_panels_are_nested_across_h_and_saturation_is_drawn_once(self):
        g = complete6()
        small, large = h_parameters(g, .1*g.cells, .8), h_parameters(g, .1*g.cells, .4)
        for i in range(1, 20):
            self.assertFalse((history_panel_mask(g, i, 'test', small) & ~history_panel_mask(g, i, 'test', large)).any())
        saturated = h_parameters(g, 10*g.cells, .6)
        self.assertTrue(saturated['h_saturated'] and saturated['h_target_unreachable'])
        self.assertEqual(draws_for('H', saturated), 1)
        with self.assertRaises(ValueError): draw(g, 'H', 2, 'test', saturated)




class BudgetSensitivityTests(unittest.TestCase):

    def test_target_scales_with_the_fraction_and_streams_differ(self):
        g = ring(n=80, per=6, seed=2)
        main, high = analytic_parameters(g), analytic_parameters(g, .5)
        self.assertAlmostEqual(high['T'], .5*g.cells)
        self.assertAlmostEqual(high['bernoulli_expected_cells'], high['T'], places=6)
        same_p = high | {'p': main['p']}
        self.assertFalse(np.array_equal(draw(g, 'B', 1, 'sample', main)[0], draw(g, 'B', 1, 'sample', same_p)[0]))




if __name__ == '__main__':
    unittest.main()
