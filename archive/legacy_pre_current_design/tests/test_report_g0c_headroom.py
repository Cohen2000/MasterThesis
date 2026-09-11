import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_g0c_headroom import (  # noqa: E402
    candidate_bias_summary,
    correction_class,
)


class CorrectionClass(unittest.TestCase):
    def test_threshold_has_neutral_band(self):
        self.assertEqual(correction_class(-0.30), "upward")
        self.assertEqual(correction_class(-0.01), "none")
        self.assertEqual(correction_class(0.14), "downward")


class CandidateBiasSummary(unittest.TestCase):
    def test_empty_cases_are_explicit_and_excluded_from_bias(self):
        frame = pd.DataFrame({
            "sample_seed": [0, 0, 1, 1],
            "group_id": ["a", "b", "a", "b"],
            "rho_W5_k2": [0.2, 0.4, 0.2, 0.4],
            "est__plugin_rho_k2": [0.4, 0.5, None, 0.7],
        })
        got = candidate_bias_summary(frame).set_index("sample_seed")
        self.assertEqual(got.loc[0, "valid_cases"], 2)
        self.assertEqual(got.loc[1, "empty_cases"], 1)
        self.assertAlmostEqual(got.loc[0, "group_macro_rho2_bias"], 0.15)
        self.assertAlmostEqual(got.loc[1, "group_macro_rho2_bias"], 0.30)


if __name__ == "__main__":
    unittest.main()
