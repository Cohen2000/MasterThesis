"""Shared zero-truncated Beta-Binomial MLE: input contract, correctness, fallback."""
import copy
import math
import tempfile
import unittest
from unittest import mock
import numpy as np
from helpers import complete6, graph, tiny
from main_experiment import baselines, shared_mle
from main_experiment.mixtures import cell_probs
from main_experiment.observation import make
from main_experiment.sampling import Walk, analytic_parameters, draw, h_parameters


def rsh_observation(arm='R', seed_offset=0, n=30):
    """A moderately sized R/S1/H-style observation with a non-degenerate table."""
    rng = np.random.default_rng(7+seed_offset)
    g = graph([(f'a{i}', f'b{i}', float(t)) for i in range(n) for t in np.linspace(0, .99, 5)])
    for j in range(5):
        rng.random(g.D)  # keep the stream identical regardless of branch below
    b = analytic_parameters(g)
    if arm == 'H':
        counts, _ = draw(g, 'H', 1, 'test', b)
        return make(g, 'H', b, counts)
    if arm == 'S1':
        b = b | {'L': 500}
        with tempfile.TemporaryDirectory() as d:
            drawn, _ = draw(g, 'S1', 1, 'test', b, Walk(g, d))
        return make(g, 'S1', b, drawn)
    counts, _ = draw(g, 'R', 1, 'test', b)
    return make(g, 'R', b, counts)


class InputContractTests(unittest.TestCase):
    def test_r_ignores_n_panel(self):
        o = rsh_observation('R')
        base = shared_mle.fit(o).rho
        mutated = copy.deepcopy(o); mutated['parameter'] = 999999
        self.assertEqual(shared_mle.fit(mutated).rho, base)

    def test_h_ignores_n_panel_history_and_derives_m_from_temporal_access(self):
        o = rsh_observation('H')
        _, m = shared_mle.visible_counts(o)
        self.assertEqual(m, sum(o['Temporal_access']))
        base = shared_mle.fit(o).rho
        mutated = copy.deepcopy(o); mutated['parameter'] = 1
        self.assertEqual(shared_mle.fit(mutated).rho, base)

    def test_s1_uses_only_the_table_no_walk_or_degree_fields(self):
        o = rsh_observation('S1')
        self.assertNotIn('design_reference', o)
        # fit_rsh only ever reads o['table']; confirm by stripping every other
        # key it could not legitimately use and refitting from a bare table.
        bare = {'arm': 'S1', 'D_obs': o['D_obs'], 'table': o['table']}
        self.assertEqual(shared_mle.fit(bare).rho, shared_mle.fit(o).rho)

    def test_b_never_touches_full_event_counts_or_hidden_coverage(self):
        g = tiny()
        b = {'p': .5}
        counts = (g.counts > 0).astype(np.int64)
        o = make(g, 'B', b, counts)
        self.assertNotIn('target_coverage', o)
        self.assertNotIn('T', o)
        result = shared_mle.fit(o)
        self.assertTrue(math.isfinite(result.objective) or result.fallback_used)

    def test_fit_signature_never_receives_ground_truth(self):
        import inspect
        self.assertEqual(list(inspect.signature(shared_mle.fit).parameters), ['o'])


class ValidityTests(unittest.TestCase):
    def test_predictions_are_finite_in_unit_interval_and_monotone(self):
        for arm in ('R', 'S1', 'H'):
            r = shared_mle.fit(rsh_observation(arm)).rho
            self.assertTrue(all(math.isfinite(x) and 0 <= x <= 1 for x in r))
            self.assertTrue(all(a >= b-1e-12 for a, b in zip(r, r[1:])))

    def test_deterministic_repeated_fit(self):
        o = rsh_observation('R')
        self.assertEqual(shared_mle.fit(o).rho, shared_mle.fit(o).rho)


class LikelihoodTests(unittest.TestCase):
    def test_h_with_full_access_matches_the_r_style_fit(self):
        """H's fit reads only o['table']; an H observation whose Temporal_access
        happens to cover all 5 windows (m=5) is fit exactly like an R table with
        the same counts -- the extrapolation step becomes a no-op."""
        r_obs = rsh_observation('R')
        h_obs = {'arm': 'H', 'D_obs': r_obs['D_obs'], 'table': r_obs['table'],
                 'Temporal_access': [1, 1, 1, 1, 1]}
        rr = shared_mle.fit(r_obs); rh = shared_mle.fit(h_obs)
        self.assertEqual(rr.rho, rh.rho)
        self.assertEqual(rr.alpha, rh.alpha)
        self.assertEqual(rr.beta, rh.beta)

    def test_synthetic_known_parameter_recovery(self):
        a_true, b_true = 2.5, 4.
        rng = np.random.default_rng(11)
        q = rng.beta(a_true, b_true, size=4000)
        k = rng.binomial(5, q)
        counts = [0]*6
        for x in k: counts[x] += 1
        j_counts = [0]+counts[1:]
        a, b, status, flags, obj = shared_mle.fit_zt_bb(j_counts, 5)
        self.assertFalse(shared_mle.is_unreliable(status, flags))
        from main_experiment.mixtures import predict_profile
        fitted = predict_profile(a, b)
        truth_pk = cell_probs(a_true, b_true, 1., 5)
        truth_den = sum(truth_pk[1:])
        truth = [sum(truth_pk[i] for i in range(kk, 6))/truth_den for kk in range(2, 6)]
        for f, t in zip(fitted, truth):
            self.assertLess(abs(f-t), 0.03)

    def test_b_delegates_to_mixtures_fit_events(self):
        g = tiny(); p = .6
        counts = (g.counts > 0).astype(np.int64)
        o = make(g, 'B', {'p': p}, counts)
        result = shared_mle.fit(o)
        from main_experiment import mixtures
        mu0, lam0 = baselines.mixture_start(o)
        direct = mixtures.fit_events(o, mu0, lam0)
        if not mixtures.is_unreliable(direct.status, direct.flags):
            self.assertEqual(result.rho, direct.prediction)
            self.assertAlmostEqual(result.lam, direct.lam)


class FallbackTests(unittest.TestCase):
    def test_fallback_used_and_reported_on_optimizer_failure(self):
        o = rsh_observation('R')
        with mock.patch.object(shared_mle, '_solve', return_value=(None, [], ['forced'])):
            result = shared_mle.fit(o)
        self.assertTrue(result.fallback_used)
        self.assertTrue(math.isnan(result.alpha) and math.isnan(result.beta))
        counts, m = shared_mle.visible_counts(o)
        mean = sum(j*counts[j] for j in range(1, m+1))/sum(counts[1:])
        q = baselines.activity(mean, m)
        self.assertEqual(result.rho, baselines.profile(q))

    def test_b_fallback_uses_homogeneous_corrector(self):
        g = tiny()
        counts = (g.counts > 0).astype(np.int64)
        o = make(g, 'B', {'p': .5}, counts)
        with mock.patch('main_experiment.mixtures.fit_events') as fit_events:
            from main_experiment.mixtures import Fit
            fit_events.return_value = Fit([float('nan')]*4, 'not_converged', n_starts=3, n_accepted=0)
            result = shared_mle.fit(o)
        self.assertTrue(result.fallback_used)
        self.assertEqual(result.rho, baselines.corrector(o))

    def test_empty_sample_raises_like_plugin(self):
        g = tiny()
        empty = make(g, 'B', {'p': .5}, np.zeros_like(g.counts))
        with self.assertRaises(ValueError):
            shared_mle.fit(empty)


if __name__ == '__main__':
    unittest.main()
