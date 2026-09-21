from fractions import Fraction
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from dataset_census import (count_feasibility, distinct_timestamp_counts,
                            occupancy_counts, parse_audited,
                            prepare_complete, relative_cutoff,
                            summarize_events, survival, window_sensitivity)


class CensusEvidence(unittest.TestCase):
    def test_direction_collapse_event_bounds_and_five_window_distribution(self):
        raw = pd.DataFrame([("a", "b", 0), ("b", "a", 20),
                            ("a", "b", 80), ("c", "d", 10),
                            ("d", "c", 11), ("e", "f", 100)],
                           columns=["u", "v", "t"])
        row = summarize_events(raw)
        self.assertEqual((row["n_nodes"], row["n_edges_full"], row["n_events"]), (6, 3, 6))
        self.assertEqual([row[f"rho_{k}"] for k in range(2, 6)], [1/3, 1/3, 0, 0])
        self.assertEqual([row[f"q{k}"] for k in range(1, 6)], [2/3, 0, 1/3, 0, 0])
        self.assertAlmostEqual(sum(row[f"q{k}"] for k in range(1, 6)), 1)
        self.assertEqual([row[f"p_m_ge_{k}"] for k in range(2, 6)], [2/3, 1/3, 0, 0])

    def test_maximum_timestamp_and_survival_identity_for_all_windows(self):
        pair = np.array([0, 0, 0, 1, 1, 2])
        times = np.array([0, 0.2, 1.0, 0.5, 0.8, 1.0])
        for W in range(2, 21):
            with self.subTest(W=W):
                K = occupancy_counts(pair, times, W)
                self.assertEqual(K[0], len({0, int(0.2*W), W-1}))
                self.assertEqual(K[-1], 1)
                rho = survival(K, W)
                self.assertAlmostEqual(float(np.mean(K / W)), (1 + sum(rho[2:])) / W)

    def test_relative_threshold_rounding_and_long_table_completeness(self):
        for W in range(2, 21):
            for tau in ("0.2", "0.4", "0.6", "0.8", "1.0"):
                exact = Fraction(tau) * W
                self.assertEqual(relative_cutoff(tau, W), -(-exact.numerator // exact.denominator))
        rows = pd.DataFrame(window_sensitivity("toy", np.array([0, 0, 1]),
                                               np.array([0, 1, 0.5]), np.array([2, 1])))
        self.assertEqual(len(rows), sum(W - 1 + 2 + 5 for W in range(2, 21)))
        self.assertEqual(set(rows.W), set(range(2, 21)))
        self.assertEqual(set(rows[rows.statistic == "rho"].query("W == 20").k), set(range(2, 21)))
        at_five = rows[(rows.W == 5) & (rows.statistic == "relative_survival")]
        self.assertEqual(at_five.value.tolist(), [1, 0.5, 0, 0, 0])

    def test_same_timestamp_duplicates_are_retained_but_counted(self):
        raw = pd.DataFrame([("a", "b", 0), ("b", "a", 0), ("a", "b", 0),
                            ("a", "b", 10), ("c", "d", 5), ("c", "d", 5)],
                           columns=["u", "v", "t"])
        row = summarize_events(raw)
        self.assertEqual((row["n_events"], row["duplicate_dyad_timestamp_events"]), (6, 3))
        self.assertEqual(row["p_m_ge_2"], 1.0)
        _, pairs, times, counts, _ = prepare_complete(raw)
        distinct = distinct_timestamp_counts(pairs, times)
        self.assertEqual(sorted(distinct.tolist()), [1, 2])
        bound = count_feasibility("toy", counts, row["rho_2"], distinct)
        self.assertEqual((bound["empirical_rho_2"], bound["p_m_ge_2"]), (0.5, 1.0))
        self.assertEqual(bound["p_distinct_timestamps_ge_2"], 0.5)
        self.assertLessEqual(bound["empirical_rho_2"], bound["p_distinct_timestamps_ge_2"])

    def test_count_feasibility_and_margins(self):
        row = count_feasibility("toy", np.array([1, 1, 2, 3, 5]))
        self.assertEqual(row["p_m_ge_2"], 0.6)
        self.assertTrue(row["target_0_15_count_feasible"])
        self.assertTrue(row["target_0_55_count_feasible"])
        self.assertAlmostEqual(row["target_0_55_margin"], 0.05)
        impossible = count_feasibility("toy", np.ones(5, dtype=int))
        self.assertFalse(impossible["target_0_15_count_feasible"])

    def test_parser_quality_bipartite_and_header_verification(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "toy.csv"
            path.write_text("user_id,item_id,timestamp\n#comment\n0,0,0\n"
                            "0,0,10\n0,1,nan\n0,1,inf\n0,1,no\n,1,3\nshort\n")
            fmt = {"columns": {"u": 0, "v": 1, "t": 2}, "delimiter": ",",
                   "skiprows": 1, "bipartite": True}
            audit = {"expected_fields": 3, "expected_header": "user_id,item_id,timestamp"}
            raw, quality = parse_audited(path, fmt, audit)
            self.assertEqual(quality["invalid_rows_removed"], 5)
            self.assertEqual(quality["data_rows"], 7)
            self.assertEqual((raw.iloc[0].u, raw.iloc[0].v), ("u:0", "i:0"))
            self.assertEqual(summarize_events(raw)["n_nodes"], 2)
            with self.assertRaises(ValueError):
                parse_audited(path, fmt, dict(audit, expected_header="wrong"))

    def test_self_events_do_not_extend_the_retained_horizon(self):
        raw = pd.DataFrame([("a", "a", -10), ("a", "b", 0), ("b", "a", 10)],
                           columns=["u", "v", "t"])
        row = summarize_events(raw)
        self.assertEqual(row["self_events_removed"], 1)
        self.assertEqual(row["observation_span_source_units"], 10)

