"""Budgets and draws of the four arms, the walk kernel and common random numbers."""
import itertools
import math
import tempfile
import unittest
from dataclasses import replace
import numpy as np
from helpers import complete6, graph, ring, tiny
from main_experiment.common import (COVERAGE_FRACTION, H_SENSITIVITY, MAIN_KEYS, TRAIN, draws_for, planned_sizes,
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


class DesignSizeTests(unittest.TestCase):
    def test_sizes_follow_from_the_budgets(self):
        free = {k: {'h_saturated': False} for k in set(TRAIN) | set(MAIN_KEYS)}
        s = planned_sizes(free)
        self.assertEqual((s['main_observations'], s['training_observations']), (288, 320))
        self.assertEqual((s['planned_calls'], s['qwen_calls']), (3456, 1728))
        free['sp_highschool2013'] = {'h_saturated': True}       # deterministic H: one draw only
        s = planned_sizes(free)
        self.assertEqual((s['main_observations'], s['training_observations']), (286, 316))
        self.assertEqual(draws_for('R', free['sp_highschool2013']), 3)


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


class WalkTests(unittest.TestCase):
    def walk_graph(self):
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 4)]
        return graph([(str(a), str(b), t) for i, (a, b) in enumerate(pairs) for t in np.linspace(0, 1, i+2)])

    def test_kernel_matches_the_reference_walk(self):
        g = tiny()
        with tempfile.TemporaryDirectory() as d:
            walk = Walk(g, d)
            for state in (0, 1, 42, 2**63-1):
                traversals, volume = reference_walk(walk, state, 100)
                total, _, counts, executed = walk.run([state], 100, True)
                np.testing.assert_array_equal(total, volume)
                np.testing.assert_array_equal(counts[0], traversals)
                self.assertEqual(executed[0], 100)
                total, _, _, executed = walk.run([state], 100)   # summary mode may stop at saturation
                np.testing.assert_array_equal(total, volume)
                self.assertLess(executed[0], 100)

    def test_event_multiplicities_do_not_change_paths(self):
        g = self.walk_graph(); counts = g.counts.copy(); counts[0] *= 1000
        with tempfile.TemporaryDirectory() as d:
            a = Walk(g, d).run(list(range(100)), 31, True)[2]
            b = Walk(replace(g, counts=counts), d).run(list(range(100)), 31, True)[2]
        np.testing.assert_array_equal(a, b)
        self.assertTrue((a.sum(1) == 31).all())

    def test_uniform_start_and_degree_biased_first_step(self):
        g = self.walk_graph(); degree = np.bincount(g.ends.ravel(), minlength=g.N)
        neighbour_sum = np.bincount(g.ends.ravel(), weights=degree[g.ends[:, ::-1].ravel()], minlength=g.N)
        # P(first dyad = {a, b}) = P(start a) d_b / S_a + P(start b) d_a / S_b
        expected = np.array([(degree[b]/neighbour_sum[a]+degree[a]/neighbour_sum[b])/g.N for a, b in g.ends])
        with tempfile.TemporaryDirectory() as d:
            first = Walk(g, d).run([seed('walk_test', sample_index=i) for i in range(30000)], 1, True)[2].mean(0)
        np.testing.assert_array_less(np.abs(first-expected), 6*np.sqrt(expected*(1-expected)/30000))

    def test_stationary_traversal_is_proportional_to_the_degree_product(self):
        g = self.walk_graph(); degree = np.bincount(g.ends.ravel(), minlength=g.N)
        directed = [(a, b) for a, b in g.ends]+[(b, a) for a, b in g.ends]
        P = np.zeros((len(directed), len(directed)))
        for i, (_, b) in enumerate(directed):
            onward = [j for j, (c, _) in enumerate(directed) if c == b]
            total = sum(degree[directed[j][1]] for j in onward)
            for j in onward: P[i, j] = degree[directed[j][1]]/total
        stationary = np.array([degree[a]*degree[b] for a, b in directed], float)
        stationary /= stationary.sum()
        np.testing.assert_allclose(stationary@P, stationary)

    def test_design_reference_recovers_the_uniform_profile_on_a_long_walk(self):
        from main_experiment.baselines import design_reference
        g = ring(n=40, per=6, seed=3)
        with tempfile.TemporaryDirectory() as d:
            walk = Walk(g, d)
            self.assertEqual(walk.n_components, 1)
            traversals = walk.run([seed('walk_test', sample_index=1)], 400000, True)[2][0]
        np.testing.assert_allclose(design_reference(g, traversals), g.truth, atol=.01)
        plugin_on_traversals = [float((traversals*(g.K >= k)).sum()/traversals.sum()) for k in range(2, 6)]
        product = np.bincount(g.ends.ravel(), minlength=g.N)
        product = product[g.ends[:, 0]]*product[g.ends[:, 1]]
        selection = [float((product*(g.K >= k)).sum()/product.sum()) for k in range(2, 6)]
        np.testing.assert_allclose(plugin_on_traversals, selection, atol=.01)   # unweighted traversals select by d_u d_v

    def test_walk_stays_in_its_component(self):
        g = graph([('a', 'b', 0.), ('a', 'b', 1.), ('c', 'd', 0.), ('d', 'e', .5), ('e', 'c', 1.)])
        with tempfile.TemporaryDirectory() as d:
            walk = Walk(g, d); _, _, counts, executed = walk.run(list(range(100)), 100, True)
        self.assertEqual(walk.n_components, 2)
        self.assertAlmostEqual(np.mean(walk.component_volume), .4*2+.6*3)
        self.assertTrue((executed == 100).all())
        for row in counts: self.assertEqual(len({walk.components[g.ends[i, 0]] for i in np.flatnonzero(row)}), 1)

    def test_unreachable_target_is_reported_not_forced(self):
        g = graph([(f'n{2*i}', f'n{2*i+1}', t) for i in range(12) for t in (.5, .7, .9)])
        with tempfile.TemporaryDirectory() as d:
            budget, _, volumes = calibrate(g, d)
        self.assertEqual(budget['L'], budget['C'])
        self.assertFalse(budget['budget_matched_by_arm']['S'])
        self.assertIn('S:calibration_cap', budget['unmatched_reasons'])
        self.assertEqual(np.mean(volumes), 3.)              # one dyad with K_e = 3 per component


class BudgetSensitivityTests(unittest.TestCase):
    def test_main_budget_keeps_its_identity_and_other_budgets_are_versioned(self):
        from main_experiment.common import BUDGET_GRID, observation_id, sampler_id
        self.assertEqual(sampler_id('R'), 'R-p888-20260921')
        self.assertEqual(observation_id('g', 'S', 2, .10), 'g__S-dbrw-p888-20260921__s2')
        ids = {sampler_id(arm, b) for arm in 'RSHB' for b in BUDGET_GRID}
        self.assertEqual(len(ids), 4*len(BUDGET_GRID))
        self.assertEqual(sampler_id('B', .025), 'B-p888-20260921-b025')

    def test_target_scales_with_the_fraction_and_streams_differ(self):
        g = ring(n=80, per=6, seed=2)
        main, high = analytic_parameters(g), analytic_parameters(g, .5)
        self.assertAlmostEqual(high['T'], .5*g.cells)
        self.assertAlmostEqual(high['bernoulli_expected_cells'], high['T'], places=6)
        same_p = high | {'p': main['p']}
        self.assertFalse(np.array_equal(draw(g, 'B', 1, 'sample', main)[0], draw(g, 'B', 1, 'sample', same_p)[0]))


class CommonRandomNumberTests(unittest.TestCase):
    def test_surrogate_draws_share_the_parent_streams(self):
        g = graph([(str(a), str((a+1) % 30), t) for a in range(30) for t in (0., .2, .4, .6, .8, 1.)],
                  'sp_hospital', proximity=True)
        s = shuffle(g); b = analytic_parameters(g) | {'L': 101}
        np.testing.assert_array_equal(history_panel_mask(g, 1, 'sample', b), history_panel_mask(s, 1, 'sample', b))
        with tempfile.TemporaryDirectory() as d:
            np.testing.assert_array_equal(draw(g, 'S', 1, 'sample', b, Walk(g, d))[1],
                                          draw(s, 'S', 1, 'sample', b, Walk(s, d))[1])
        # Same p and uniforms: the same event records are kept, their times differ.
        np.testing.assert_array_equal(draw(g, 'B', 1, 'sample', b)[0].sum(1), draw(s, 'B', 1, 'sample', b)[0].sum(1))
        np.testing.assert_array_equal(draw(g, 'R', 1, 'sample', b)[0].sum(1) > 0, draw(s, 'R', 1, 'sample', b)[0].sum(1) > 0)


if __name__ == '__main__':
    unittest.main()
