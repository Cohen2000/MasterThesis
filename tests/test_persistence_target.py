import unittest

import numpy as np

from census import profile, spans, window_index


class PersistenceTarget(unittest.TestCase):
    def test_dyads_have_equal_weight_regardless_of_event_counts(self):
        pairs = np.array([0, 0, 0, 1, 1, 2, 2, 3, 4])
        windows = np.array([0, 1, 4, 0, 0, 3, 4, 2, 4])
        result = profile(spans(pairs, windows), 5)
        self.assertEqual(result, {1: 1.0, 2: 0.4, 3: 0.2, 4: 0.0, 5: 0.0})
        extra_pairs = np.concatenate([pairs, np.zeros(100, dtype=int)])
        extra_windows = np.concatenate([windows, np.zeros(100, dtype=int)])
        self.assertEqual(profile(spans(extra_pairs, extra_windows), 5), result)

    def test_five_window_boundaries(self):
        normalized = np.array([0, 0.2, 0.4, 0.6, 0.8, np.nextafter(1.0, 0.0)])
        self.assertEqual(window_index(normalized, 0, 0.2, 5).tolist(),
                         [0, 1, 2, 3, 4, 4])

    def test_preserved_endpoint_and_epsilon_compatibility(self):
        # Existing behavior is recorded explicitly, not advertised as strict
        # half-open normalization. See the Phase-1 limitation in reproducibility.
        self.assertEqual(window_index(np.array([1.0]), 0, 0.2, 5).tolist(), [4])
        self.assertEqual(window_index(np.array([0.2 - 1e-12]), 0, 0.2, 5).tolist(), [1])

    def test_derived_occupancy_identity(self):
        counts = np.array([1, 2, 3, 4, 5])
        result = profile(counts, 5)
        self.assertAlmostEqual(np.mean(counts / 5),
                               (1 + sum(result[k] for k in range(2, 6))) / 5)
