"""Plug-in and arm-specific corrector references, and the oracle decomposition."""
import math
import tempfile
import unittest
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import binom
from helpers import complete6, graph, tiny
from main_experiment.baselines import activity, all_references, corrector, h_extrapolator, profile
from main_experiment.history_diagnostics import decompose
from main_experiment.observation import make
from main_experiment.sampling import Walk, analytic_parameters, draw, h_parameters


class CorrectorTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual((profile(0), profile(1)), ([0.]*4, [1.]*4))
        for n in (3, 5): self.assertEqual((activity(1, n), activity(n, n)), (0, 1))
        g = graph([('a', 'b', t) for t in (0, .2, .4, .6, 1.)])
        self.assertEqual(corrector(make(g, 'B', {'p': .1}, g.counts)), [1.]*4)   # q = 1
        c = np.zeros_like(g.counts); c[0, 0] = 1
        self.assertEqual(corrector(make(g, 'B', {'p': .2}, c)), [0.]*4)          # theta = 0

    def test_s1_baseline_is_the_plugin_and_the_design_reference_is_the_inverse_traversal_weight_estimate(self):
        # S2 (the walker-annotated view of the same walk) is not an active arm
        # of the v9 design; design_reference itself remains the S1 oracle
        # diagnostic and is checked directly against the traversal counts.
        from main_experiment.baselines import design_reference, plugin
        g = tiny(); b = analytic_parameters(g) | {'L': 51}
        with tempfile.TemporaryDirectory() as d:
            counts, traversals = draw(g, 'S1', 1, 'test', b, Walk(g, d))
        s1 = make(g, 'S1', b, counts)
        self.assertEqual(corrector(s1), plugin(s1))                # S1 shows no walker information
        degree = np.bincount(g.ends.ravel(), minlength=g.N)
        w = traversals/(degree[g.ends[:, 0]]*degree[g.ends[:, 1]])
        np.testing.assert_allclose(design_reference(g, traversals), [w[g.K >= k].sum()/w.sum() for k in range(2, 6)])
        repeated = design_reference(g, 2*traversals)                  # repeated traversals keep their weight
        np.testing.assert_allclose(repeated, design_reference(g, traversals))

    def test_b_hurdle_matches_direct_likelihood_optimisation(self):
        for D, S, M, p in [(20, 37, 80, .4), (20, 90, 95, .2), (50, 75, 250, .8), (30, 50, 52, .7)]:
            per_dyad = [1]*D
            for j in range(S-D): per_dyad[j % D] += 1
            counts = np.zeros((D, 5), dtype=np.int64)
            for i, k in enumerate(per_dyad): counts[i, :k] = 1
            counts[0, 0] += M-S
            patterns = (counts > 0)@np.array([16, 8, 4, 2, 1])
            table = [(f'{q:05b}', int((patterns == q).sum()), int(counts[patterns == q].sum())) for q in range(1, 32)]
            o = {'arm': 'B', 'N_obs': D+1, 'D_obs': D, 'M_obs': M, 'Temporal_access': [1]*5,
                 'Events_per_window': counts.sum(0).tolist(), 'parameter': p, 'table': table}

            def nll(z):
                q, r = z; d = -np.expm1(-r)/-np.expm1(-r/p); theta = q*d
                if not 0 < theta < 1: return 1e100
                return -(S*np.log(theta)+(5*D-S)*np.log1p(-theta)-D*np.log(-np.expm1(5*np.log1p(-theta)))
                         + M*np.log(r)-S*np.log(np.expm1(r)))
            fits = [minimize(nll, start, bounds=[(1e-8, 1.), (1e-8, 20)], method='Nelder-Mead',
                             options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 5000})
                    for start in [(.2, .5), (.8, 2.), (.99, .1)]]
            best = min(fits, key=lambda f: f.fun)
            np.testing.assert_allclose(corrector(o), profile(best.x[0]), atol=2e-6)

    def test_h_extrapolator_is_the_zero_truncated_binomial_mle(self):
        g = complete6()
        for h in (.4, .6, .8):
            n = round(5*h); rng = np.random.default_rng(92)
            counts = np.zeros_like(g.counts); counts[:, 5-n:] = rng.random((g.D, n)) < .55
            fit = h_extrapolator(make(g, 'H', h_parameters(g, g.cells, h), counts))
            j = (counts > 0).sum(1); j = j[j > 0]
            best = minimize_scalar(lambda q: -np.sum(binom.logpmf(j, n, q)-math.log1p(-(1-q)**n)),
                                   bounds=(1e-8, 1-1e-8), method='bounded')
            self.assertAlmostEqual(fit['q'], best.x, places=5)
            np.testing.assert_allclose(fit['prediction'], binom.sf(np.arange(1, 5), 5, fit['q'])/(1-(1-fit['q'])**5))

    def test_h_extrapolator_boundaries_and_empty(self):
        g = complete6(); b = h_parameters(g, g.cells, .6)
        for windows, expected in (([4], [0.]*4), ([2, 3, 4], [1.]*4)):
            c = np.zeros_like(g.counts); c[:, windows] = 1
            self.assertEqual(h_extrapolator(make(g, 'H', b, c))['prediction'], expected)
        with self.assertRaises(ValueError): h_extrapolator(make(g, 'H', b, np.zeros_like(g.counts)))

    def test_empty_observation_uses_the_fold_median_everywhere(self):
        g = tiny()
        empty = make(g, 'B', {'p': .5}, np.zeros_like(g.counts))
        references = all_references(empty, {}, [.4]*4)
        self.assertTrue(all(r['prediction'] == [.4]*4 and r['status'] == 'empty_sample' for r in references.values()))


class DecompositionTests(unittest.TestCase):
    def test_dyad_disappearance_is_history_not_node_selection(self):
        full = np.array([[1, 1, 0, 0, 0], [1, 0, 1, 1, 1], [0, 0, 0, 0, 1]])
        censored = full.copy(); censored[:, :2] = 0
        d = decompose(full, censored, np.ones(3, dtype=bool))
        self.assertEqual((d['node_selection'], d['lost_panel_dyads']), ([0.]*4, 1))
        self.assertNotEqual(d['dyad_disappearance'], [0.]*4)
        np.testing.assert_allclose(np.add(d['dyad_disappearance'], d['within_dyad_history']), d['net_history'])
        self.assertFalse(decompose(full, np.zeros_like(full), np.ones(3, dtype=bool))['defined'])


if __name__ == '__main__':
    unittest.main()
