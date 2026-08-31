import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_stability_probe import mean_pairwise_gap  # noqa: E402


class PairwiseGap(unittest.TestCase):
    def test_mean_absolute_gap_uses_all_pairs(self):
        # Pairwise gaps are 0.1, 0.2, and 0.1.
        self.assertAlmostEqual(mean_pairwise_gap([[0.1, 0.2, 0.3]]),
                               0.4 / 3)

    def test_groups_are_pooled_with_equal_pair_counts(self):
        self.assertAlmostEqual(
            mean_pairwise_gap([[0.0, 0.0, 0.0], [0.0, 0.5, 1.0]]),
            1.0 / 3)


if __name__ == "__main__":
    unittest.main()
