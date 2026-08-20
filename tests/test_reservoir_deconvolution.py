import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reservoir_deconvolution import (  # noqa: E402
    parse_window_count_histogram,
    temporal_mask_rate_factorized_eb,
    temporal_mask_rate_mixture_eb,
)


class Parsing(unittest.TestCase):
    def test_exact_vectors(self):
        x, f = parse_window_count_histogram(
            json.dumps({"1,0,2,0,0": 3, "0,1,0,0,0": 5}))
        self.assertEqual(x.shape, (2, 5))
        self.assertEqual(int(f.sum()), 8)


class MixtureEB(unittest.TestCase):
    def test_single_window_patterns_give_low_persistence(self):
        raw = json.dumps({"1,0,0,0,0": 200, "2,0,0,0,0": 30})
        estimate, diagnostics = temporal_mask_rate_mixture_eb(
            raw, sampling_fraction=0.05, grid_size=24)
        self.assertLess(estimate["rho_k2"], 1e-8)
        self.assertTrue(diagnostics["iterations"] > 0)

    def test_fully_observed_two_window_pattern_is_persistent(self):
        raw = json.dumps({"3,2,0,0,0": 300})
        estimate, _ = temporal_mask_rate_mixture_eb(
            raw, sampling_fraction=1.0, grid_size=18)
        self.assertGreater(estimate["rho_k2"], 0.999)
        self.assertLess(estimate["rho_k3"], 1e-8)

    def test_profile_is_monotone_and_bounded(self):
        raw = json.dumps({
            "1,0,0,0,0": 50, "0,1,0,1,0": 20,
            "1,1,1,0,0": 10, "0,0,1,1,1": 5,
        })
        estimate, _ = temporal_mask_rate_mixture_eb(
            raw, sampling_fraction=0.1, grid_size=20)
        profile = np.array([estimate[f"rho_k{k}"] for k in range(2, 6)])
        self.assertTrue(np.all((0 <= profile) & (profile <= 1)))
        self.assertTrue(np.all(profile[:-1] >= profile[1:]))

    def test_factorized_fit_converges_and_is_monotone(self):
        raw = json.dumps({
            "1,0,0,0,0": 80, "0,1,0,1,0": 20,
            "1,1,1,0,0": 10,
        })
        estimate, diagnostics = temporal_mask_rate_factorized_eb(
            raw, sampling_fraction=0.1, grid_size=16)
        profile = np.array([estimate[f"rho_k{k}"] for k in range(2, 6)])
        self.assertTrue(diagnostics["converged"])
        self.assertTrue(np.all(profile[:-1] >= profile[1:]))


if __name__ == "__main__":
    unittest.main()
