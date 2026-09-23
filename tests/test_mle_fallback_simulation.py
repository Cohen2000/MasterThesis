"""Seeded checks of the shared MLE fallback on its stated working models."""
import unittest
import numpy as np

from main_experiment.data import Graph
from main_experiment.mixtures import predict_profile
from main_experiment.observation import make
from main_experiment.shared_mle import fit, fit_profile_from_counts


class MLEFallbackSimulation(unittest.TestCase):
    def test_h_three_visible_two_hidden_windows(self):
        rng = np.random.default_rng(34109)
        alpha, beta, n_visible, n_dyads = .8, 2.8, 3, 1200
        truth = predict_profile(alpha, beta)
        new, old, old_fallbacks = [], [], 0
        for _ in range(36):
            q = rng.beta(alpha, beta, n_dyads)
            observed = rng.binomial(n_visible, q)
            observed = observed[observed > 0]  # H's zero-truncated observed block
            counts = [0] * (n_visible + 1)
            for j in observed:
                counts[j] += 1
            result = fit_profile_from_counts(counts, n_visible)
            new.append(result.rho[0])
            old.append(result.old_rule_rho[0])
            old_fallbacks += result.old_rule_fallback_used
        self.assertGreater(old_fallbacks, 0)
        self.assertLess(np.mean((np.asarray(new) - truth[0]) ** 2),
                        np.mean((np.asarray(old) - truth[0]) ** 2))

    def test_b_beta_ztp_thinning_simulation(self):
        rng = np.random.default_rng(8192)
        alpha, beta, lam, p, n_dyads = .7, 2.3, 1.5, .4, 900
        truth = predict_profile(alpha, beta)
        ends = np.column_stack((np.arange(n_dyads), np.arange(1, n_dyads + 1))).astype(np.int64)
        pair = np.arange(n_dyads)
        event_matrix = np.zeros((n_dyads, 5), dtype=np.int64)
        for draw in range(8):
            q = rng.beta(alpha, beta, n_dyads)
            active = rng.binomial(5, q)
            counts = np.zeros_like(event_matrix)
            for i, k in enumerate(active):
                for w in range(k):
                    n = 0
                    while n == 0:
                        n = rng.poisson(lam)
                    counts[i, w] = rng.binomial(n, p)
                rng.shuffle(counts[i])
            graph = Graph('seeded-B', ends[:, 0], ends[:, 1], counts.sum(1).astype(float),
                          counts.sum(1), pair, ends, counts, (0., 4.))
            result = fit(make(graph, 'B', {'p': p}, counts))
            self.assertFalse(result.fallback_used)
            self.assertTrue(np.isfinite(result.rho).all())
            self.assertEqual(result.old_rule_rho, result.rho)
            event_matrix += counts
        self.assertGreater(int(event_matrix.sum()), n_dyads)
        self.assertEqual(len(truth), 4)


if __name__ == '__main__':
    unittest.main()
