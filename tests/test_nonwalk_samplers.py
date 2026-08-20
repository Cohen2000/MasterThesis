import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nonwalk_samplers import (  # noqa: E402
    ego_recent_k_snowball,
    node_panel_full_history,
    node_panel_size,
    temporal_nonstationarity_diagnostics,
    time_prefix_events,
    time_random_window_events,
    uniform_event_reservoir,
)
from run_nonwalk_screen import _strategy_seed_key  # noqa: E402


def toy_events():
    return pd.DataFrame({
        "u": [0, 0, 0, 1, 1, 2, 2, 3, 0, 1, 2, 3],
        "v": [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 0],
        "t": np.linspace(0.02, 0.98, 12),
    })


class EventReservoir(unittest.TestCase):
    def test_reproducible_uniform_prefix(self):
        events = toy_events()
        a = uniform_event_reservoir(events, 8, seed=4).log
        b = uniform_event_reservoir(events, 8, seed=4).log
        small = uniform_event_reservoir(events, 3, seed=4).log
        self.assertTrue(a.equals(b))
        self.assertEqual(set(small.event_id), set(a.head(3).event_id))
        self.assertEqual(len(set(a.event_id)), len(a))


class TimeWindows(unittest.TestCase):
    def test_prefix_is_chronological(self):
        log = time_prefix_events(toy_events().sample(frac=1, random_state=2), 5).log
        self.assertTrue(np.all(np.diff(log.t) >= 0))
        self.assertEqual(len(log), 5)

    def test_random_event_windows_are_contiguous_and_nested(self):
        events = toy_events()
        small = time_random_window_events(events, 3, seed=9).log
        large = time_random_window_events(events, 8, seed=9).log
        self.assertLessEqual(small.event_id.max() - small.event_id.min(), 2)
        self.assertTrue(set(small.event_id).issubset(set(large.event_id)))
        self.assertTrue(np.all(np.diff(large.t) >= 0))


class NodePanel(unittest.TestCase):
    def test_expected_budget_calibration(self):
        self.assertEqual(node_panel_size(10, 1000, 200), 5)

    def test_histories_are_induced_and_complete(self):
        events = toy_events()
        result = node_panel_full_history(events, 5, seed=3)
        nodes = set(result.log.u).union(result.log.v)
        expected = events[events.u.isin(nodes) & events.v.isin(nodes)]
        self.assertEqual(len(result.log), len(expected))
        self.assertEqual(set(result.log.event_id), set(expected.index))


class EgoRecent(unittest.TestCase):
    def test_k_sweep_shares_rng_key(self):
        self.assertEqual(_strategy_seed_key("ego_recent_k1"),
                         _strategy_seed_key("ego_recent_k20"))

    def test_depth_changes_history_not_query_prefix(self):
        events = toy_events()
        shallow = ego_recent_k_snowball(events, 12, seed=8, k=1)
        deep = ego_recent_k_snowball(events, 12, seed=8, k=5)
        q1 = shallow.diagnostics["query_node_order"]
        q5 = deep.diagnostics["query_node_order"]
        self.assertEqual(q1[:min(len(q1), len(q5))], q5[:min(len(q1), len(q5))])
        self.assertTrue(deep.log.event_id.is_unique)
        self.assertTrue(shallow.log.event_id.is_unique)

    def test_partial_response_is_flagged(self):
        result = ego_recent_k_snowball(toy_events(), 2, seed=2, k=None)
        self.assertEqual(len(result.log), 2)
        self.assertEqual(result.diagnostics["partial_query_count"], 1)
        self.assertTrue(result.log.partial_response.all())


class Diagnostics(unittest.TestCase):
    def test_stationarity_outputs_are_truth_only(self):
        got = temporal_nonstationarity_diagnostics(toy_events(), bins=4)
        self.assertTrue(got)
        self.assertTrue(all(key.startswith("diag__") for key in got))

    def test_fixed_horizon_keeps_empty_early_period(self):
        events = toy_events()
        events["t"] = 0.5 + events["t"] / 2
        got = temporal_nonstationarity_diagnostics(events, bins=4)
        self.assertEqual(got["diag__nodes_seen_first_half_share"], 0.0)
        self.assertTrue(np.isnan(got["diag__late_early_event_rate_ratio"]))


if __name__ == "__main__":
    unittest.main()
