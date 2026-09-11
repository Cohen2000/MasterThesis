import json
import sys
from pathlib import Path
import unittest

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from benchmark_features import build_case_features
from generator import _pick_contiguous_windows, make_dcsbm_graph
from run_benchmark_walks import _strategy_spec
from benchmark_generators import (dar_event_stream, edge_rewire_surrogate,
                                  lifetime_resample, timestamp_shuffle,
                                  within_window_timestamp_shuffle)
from mask_estimator import mask_mle
from walks import build_index, run_walk


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.events = pd.DataFrame({
            "u": [0, 0, 0, 1, 1, 2, 2, 0],
            "v": [1, 1, 2, 2, 2, 3, 3, 3],
            "t": [0.02, 0.22, 0.35, 0.42, 0.62, 0.72, 0.88, 0.95],
        })

    def test_timestamp_shuffle_exact_invariants(self):
        s = timestamp_shuffle(self.events, seed=7)
        self.assertTrue(np.array_equal(np.sort(s.t), np.sort(self.events.t)))
        a = self.events.groupby(["u", "v"]).size().sort_index()
        b = s.groupby(["u", "v"]).size().sort_index()
        pd.testing.assert_series_equal(a, b)

    def test_dar_event_layer(self):
        g = nx.cycle_graph(30)
        e = dar_event_stream(g, alpha=0.8, chi=0.4, seed=4, W=5)
        self.assertGreater(len(e), 1)
        self.assertGreaterEqual(e.t.min(), 0.0)
        self.assertLessEqual(e.t.max(), 1.0)
        self.assertTrue({"u", "v", "t"}.issubset(e.columns))

    def test_recent_history_is_reverse_time_between_restarts(self):
        idx = build_index(self.events)
        log = run_walk(idx, "recent_history", max_budget=80, seed=3, history_k=3)
        previous = None
        for row in log.itertuples():
            if row.kind == 0:
                previous = None
            else:
                if previous is not None:
                    self.assertLess(row.t, previous)
                previous = row.t

    def test_mask_mle_full_observation(self):
        n = np.array([80, 80, 80, 80])
        m = np.array([0b00001, 0b00011, 0b00111, 0b11111])
        out, _ = mask_mle(n, m, W=5)
        self.assertAlmostEqual(out["rho_k2"], 0.75, places=2)
        self.assertAlmostEqual(out["rho_k3"], 0.50, places=2)

    def test_features_are_observable_only(self):
        idx = build_index(self.events)
        log = run_walk(idx, "time_agnostic_t", max_budget=80, seed=9)
        f = build_case_features(log, budget=80)
        legal = ("occ__", "pat__", "crawl__")
        model_features = {k for k in f if k.startswith(legal)}
        self.assertTrue(model_features)
        forbidden = {"coverage", "total_edges", "n_nodes_true", "alpha", "chi"}
        self.assertFalse(model_features & forbidden)
        self.assertIsInstance(json.loads(f["input__nw_exact_json"]), dict)

    def test_v2_shuffle_invariants(self):
        within = within_window_timestamp_shuffle(self.events, seed=9, W=5)
        self.assertTrue(np.array_equal(np.sort(self.events.t), np.sort(within.t)))
        def ew(d):
            x = d.copy(); x["w"] = np.minimum((x.t * 5).astype(int), 4)
            return x.groupby(["u", "v", "w"]).size().sort_index()
        self.assertTrue(ew(self.events).equals(ew(within)))

        life = lifetime_resample(self.events, seed=11)
        a = self.events.groupby(["u", "v"])["t"].agg(["min", "max"]).sort_index()
        b = life.groupby(["u", "v"])["t"].agg(["min", "max"]).sort_index()
        self.assertTrue(np.allclose(a, b))

        rewired = edge_rewire_surrogate(self.events, seed=13)
        self.assertEqual(len(rewired), len(self.events))
        self.assertTrue(np.array_equal(np.sort(rewired.t), np.sort(self.events.t)))

    def test_contiguous_window_picker(self):
        rng = np.random.default_rng(3)
        chosen = _pick_contiguous_windows(rng, np.array([5, 4, 8, 7, 2]), 3, 10)
        self.assertEqual(chosen, list(range(chosen[0], chosen[0] + 3)))

    def test_dar_v2_alpha_modes(self):
        g = make_dcsbm_graph(80, seed=5, average_degree=10)
        a = dar_event_stream(g, alpha=0.5, chi=0.1, seed=4,
                             alpha_concentration=6.0)
        b = dar_event_stream(g, alpha=0.5, chi=0.1, seed=4,
                             alpha_within=0.8, alpha_between=0.2)
        self.assertGreater(len(a), 1)
        self.assertGreater(len(b), 1)

    def test_strategy_aliases_and_recent_json(self):
        self.assertEqual(_strategy_spec("recent_history_k5", 20)["history_k"], 5)
        self.assertEqual(_strategy_spec("time_respecting_multistart3", 20)["starts"], 3)
        idx = build_index(self.events)
        log = run_walk(idx, "time_agnostic_t", max_budget=40, seed=2)
        f = build_case_features(log, budget=40, idx=idx, recent_limit=10)
        rows = json.loads(f["input__recent_events_json"])
        self.assertIsInstance(rows, list)
        self.assertLessEqual(len(rows), 10)
        self.assertIn("diag__edgebank_frequency_auc", f)


if __name__ == "__main__":
    unittest.main()
