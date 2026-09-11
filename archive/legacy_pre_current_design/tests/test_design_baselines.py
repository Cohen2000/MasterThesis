import json
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from design_baselines import (  # noqa: E402
    discovery_diagnostics,
    model_assisted_hh,
    parse_nmask_hist,
    project_profile_decreasing,
)
from mask_estimator import mask_mle  # noqa: E402


class HistogramParsing(unittest.TestCase):
    def test_hex_masks_and_counts(self):
        n, masks, counts = parse_nmask_hist(
            json.dumps({"2,0c": 3, "1,01": 7}))
        np.testing.assert_array_equal(n, [1, 2])
        np.testing.assert_array_equal(masks, [1, 12])
        np.testing.assert_array_equal(counts, [7, 3])


class WeightedMaskMLE(unittest.TestCase):
    def test_weights_equal_expansion(self):
        n = np.array([2, 4])
        masks = np.array([0b00011, 0b01111])
        weights = np.array([3, 2])
        a, _ = mask_mle(n, masks, weights=weights)
        b, _ = mask_mle(np.repeat(n, weights), np.repeat(masks, weights))
        for key in a:
            self.assertAlmostEqual(a[key], b[key], places=12)


class HansenHurwitz(unittest.TestCase):
    def test_fully_resolved_profile_is_traversal_weighted(self):
        n = np.array([100, 100])
        masks = np.array([0b00001, 0b11111])
        counts = np.array([3, 1])
        out = model_assisted_hh(n, masks, counts)
        self.assertAlmostEqual(out["rho_k2"], 0.25, places=3)
        self.assertAlmostEqual(out["rho_k5"], 0.25, places=3)


class ShapeProjection(unittest.TestCase):
    def test_valid_profile_is_unchanged(self):
        x = np.array([0.8, 0.4, 0.2, 0.0])
        np.testing.assert_allclose(project_profile_decreasing(x), x)

    def test_violation_is_pooled(self):
        got = project_profile_decreasing([0.2, 0.6, 0.1, 0.0])
        np.testing.assert_allclose(got, [0.4, 0.4, 0.1, 0.0])

    def test_bounds_are_enforced(self):
        got = project_profile_decreasing([1.2, 0.8, -0.2, 0.1])
        self.assertTrue(np.all((0 <= got) & (got <= 1)))
        self.assertTrue(np.all(got[:-1] >= got[1:]))


class DiscoveryDiagnostics(unittest.TestCase):
    def test_outputs_are_diagnostics_not_rho(self):
        got = discovery_diagnostics(np.array([1, 2]), np.array([4, 2]))
        self.assertGreater(got["chao1_dyads"], got["observed_dyads"])
        self.assertNotIn("rho", got)


if __name__ == "__main__":
    unittest.main()
