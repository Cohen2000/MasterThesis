import unittest

import numpy as np
import pandas as pd

from report_g0b_headroom import (
    NODE_PANEL,
    PPS_DYAD,
    candidate_bias_rows,
    choose_candidate,
    mismatch_pairing,
    portable_token_count,
    vectorized_true_window_counts,
)
from report_g0_headroom import true_window_counts
from walks import build_index


class G0bHeadroomTests(unittest.TestCase):
    def test_opposite_sign_candidate_has_priority_over_zero_bias_candidate(self):
        frame = pd.DataFrame({
            "strategy": [NODE_PANEL] * 4 + [PPS_DYAD] * 4,
            "sample_seed": [0, 0, 1, 1] * 2,
            "group_id": ["a", "b"] * 4,
            "est__plugin_rho_k2": [.49, .51, .48, .52, .7, .7, .7, .7],
            "rho_W5_k2": [.5] * 8,
        })
        biases = candidate_bias_rows(frame)
        self.assertEqual(choose_candidate(biases), PPS_DYAD)

    def test_pairing_uses_median_bidirectional_bias_shift(self):
        rows = []
        for walk, shifts in [("time_agnostic_t", (.1, .2)),
                             ("time_respecting", (.3, .4)),
                             ("recent_history_k20", (.05, .05))]:
            rows.extend([
                {"sample_arm": walk, "assumed_arm": PPS_DYAD,
                 "profile_mae_penalty": 0, "abs_rho2_bias_shift": shifts[0]},
                {"sample_arm": PPS_DYAD, "assumed_arm": walk,
                 "profile_mae_penalty": 0, "abs_rho2_bias_shift": shifts[1]},
            ])
        summary, pair = mismatch_pairing(pd.DataFrame(rows))
        self.assertEqual(pair, ("time_respecting", PPS_DYAD))
        score = summary.loc[summary.walk_arm == "time_respecting",
                            "median_abs_rho2_bias_shift"].iloc[0]
        self.assertAlmostEqual(score, .35)

    def test_portable_token_count_is_deterministic_and_content_sensitive(self):
        short = portable_token_count('{"1,01":2}')
        long = portable_token_count('{"1,01":2,"2,03":4}')
        self.assertGreater(short, 0)
        self.assertGreater(long, short)

    def test_vectorized_true_k_matches_existing_index_implementation(self):
        events = pd.DataFrame({
            "u": [0, 0, 0, 1], "v": [1, 1, 2, 2],
            "t": [0.0, .41, .99, .2],
        })
        expected = true_window_counts(build_index(events, T=1.0, W=5), W=5)
        self.assertEqual(vectorized_true_window_counts(events), expected)


if __name__ == "__main__":
    unittest.main()
