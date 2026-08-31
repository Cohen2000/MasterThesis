import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nonwalk_samplers import (  # noqa: E402
    activity_proportional_dyad_full_history,
    ego_recent_k_snowball,
    event_sample_then_full_history,
    neighbourhood_crawl,
    node_panel_full_history,
    node_panel_size,
    temporal_nonstationarity_diagnostics,
    time_prefix_events,
    time_random_window_events,
    uniform_event_reservoir,
)
from run_nonwalk_screen import (  # noqa: E402
    _parse_strategy,
    _strategy_seed_key,
)


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

    def test_histories_are_incident_complete_and_whole_node_budgeted(self):
        events = toy_events()
        result = node_panel_full_history(events, 5, seed=3)
        nodes = set(result.diagnostics["panel_node_order"])
        expected = events[events.u.isin(nodes) | events.v.isin(nodes)]
        self.assertEqual(len(result.log), len(expected))
        self.assertEqual(set(result.log.event_id), set(expected.index))
        self.assertLessEqual(len(result.log), 5)
        self.assertEqual(result.diagnostics["partial_response_count"], 0)
        self.assertEqual(result.diagnostics["budget_slack"], 5 - len(result.log))


class ActivityProportionalDyad(unittest.TestCase):
    def test_selected_dyad_histories_are_complete_and_within_budget(self):
        events = toy_events()
        result = activity_proportional_dyad_full_history(events, 7, seed=4)
        self.assertLessEqual(len(result.log), 7)
        self.assertEqual(result.diagnostics["partial_response_count"], 0)
        for edge, observed in result.log.groupby(["u", "v"]):
            full = events[(events.u == edge[0]) & (events.v == edge[1])]
            self.assertEqual(len(observed), len(full))
        again = activity_proportional_dyad_full_history(events, 7, seed=4)
        self.assertTrue(result.log.equals(again.log))

    def test_oversize_dyad_does_not_force_an_empty_sample(self):
        events = pd.DataFrame({
            "u": [0] * 9 + [2, 4],
            "v": [1] * 9 + [3, 5],
            "t": np.linspace(0.0, 1.0, 11),
        })
        result = activity_proportional_dyad_full_history(events, 2, seed=1)
        self.assertEqual(len(result.log), 2)
        self.assertGreaterEqual(
            result.diagnostics["skipped_oversize_dyad_count"], 1)


class EventSampleThenFullHistory(unittest.TestCase):
    def test_uniform_event_discovery_returns_only_complete_histories(self):
        events = toy_events()
        result = event_sample_then_full_history(events, 7, seed=4)
        self.assertLessEqual(len(result.log), 7)
        self.assertEqual(result.diagnostics["partial_response_count"], 0)
        for edge, observed in result.log.groupby(["u", "v"]):
            full = events[(events.u == edge[0]) & (events.v == edge[1])]
            self.assertEqual(len(observed), len(full))
        self.assertEqual(
            result.diagnostics["phase1_unique_event_count"]
            + result.diagnostics["phase2_additional_unique_event_count"],
            len(result.log))
        again = event_sample_then_full_history(events, 7, seed=4)
        self.assertTrue(result.log.equals(again.log))

    def test_whole_lookup_stop_can_leave_empty_sample(self):
        events = pd.DataFrame({
            "u": [0] * 9 + [2],
            "v": [1] * 9 + [3],
            "t": np.linspace(0.0, 1.0, 10),
        })
        empty = None
        for seed in range(100):
            got = event_sample_then_full_history(events, 2, seed=seed)
            if got.log.empty:
                empty = got
                break
        self.assertIsNotNone(empty)
        self.assertTrue(
            empty.diagnostics["empty_due_to_first_nonfitting_lookup"])
        self.assertEqual(empty.diagnostics["phase1_unique_event_count"], 0)

    def test_strategy_name_parses(self):
        self.assertEqual(_parse_strategy("event_sample_then_full_history"),
                         ("event_sample_then_full_history", None))


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


def star_events():
    """A hub with eight leaves: the newest-event chain is narrow, the
    neighbourhood is wide, so the two frontier rules must diverge."""
    leaves = list(range(1, 9))
    return pd.DataFrame({
        "u": [0] * 8 + leaves,
        "v": leaves + [(i % 8) + 1 for i in leaves],
        "t": np.linspace(0.01, 0.99, 16),
    })


class NeighbourhoodCrawl(unittest.TestCase):
    def test_reproducible_and_within_budget(self):
        events = star_events()
        a = neighbourhood_crawl(events, 6, seed=5, k=4)
        b = neighbourhood_crawl(events, 6, seed=5, k=4)
        self.assertTrue(a.log.equals(b.log))
        self.assertEqual(len(a.log), 6)
        self.assertTrue(a.log.event_id.is_unique)

    def test_bfs_frontier_is_wider_than_the_ego_chain(self):
        events = star_events()
        bfs = neighbourhood_crawl(events, 16, seed=1, k=8, expansion="bfs")
        ego = ego_recent_k_snowball(events, 16, seed=1, k=8)
        self.assertNotEqual(bfs.diagnostics["query_node_order"],
                            ego.diagnostics["query_node_order"])
        self.assertLess(bfs.diagnostics["restart_count"],
                        ego.diagnostics["restart_count"])

    def test_full_burn_probability_reduces_to_bfs(self):
        events = star_events()
        bfs = neighbourhood_crawl(events, 12, seed=7, k=4, expansion="bfs")
        burned = neighbourhood_crawl(events, 12, seed=7, k=4,
                                     expansion="forest_fire", burn_prob=1.0)
        self.assertEqual(bfs.diagnostics["query_node_order"],
                         burned.diagnostics["query_node_order"])

    def test_burning_drops_part_of_the_discovered_frontier(self):
        got = neighbourhood_crawl(star_events(), 16, seed=2, k=8,
                                  expansion="forest_fire", burn_prob=0.35)
        self.assertLess(got.diagnostics["queued_node_count"],
                        got.diagnostics["discovered_node_count"])

    def test_rejects_an_unknown_expansion(self):
        with self.assertRaises(ValueError):
            neighbourhood_crawl(star_events(), 4, seed=0, k=2,
                                expansion="teleport")

    def test_strategy_names_parse_and_keep_their_own_seed_key(self):
        self.assertEqual(_parse_strategy("bfs_crawl_k20"), ("bfs_crawl", 20))
        self.assertEqual(_parse_strategy("forest_fire_kall"),
                         ("forest_fire", None))
        self.assertEqual(_strategy_seed_key("bfs_crawl_k5"),
                         _strategy_seed_key("bfs_crawl_k20"))
        self.assertNotEqual(_strategy_seed_key("bfs_crawl_k5"),
                            _strategy_seed_key("ego_recent_k5"))


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
