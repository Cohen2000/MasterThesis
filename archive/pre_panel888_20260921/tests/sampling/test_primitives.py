import unittest

import numpy as np
import pandas as pd

from nonwalk_samplers import node_panel_full_history, uniform_event_reservoir
from walks import build_index, run_walk


def toy_events():
    return pd.DataFrame({
        "u": [0, 0, 0, 1, 1, 2, 2, 3, 0, 1, 2, 3],
        "v": [1, 1, 2, 2, 3, 3, 4, 4, 4, 4, 4, 0],
        "t": np.linspace(0.02, 0.98, 12),
    })


class EventReservoir(unittest.TestCase):
    # Test body preserved from the historical nonwalk sampler suite.
    def test_reproducible_uniform_prefix(self):
        events = toy_events()
        a = uniform_event_reservoir(events, 8, seed=4).log
        b = uniform_event_reservoir(events, 8, seed=4).log
        small = uniform_event_reservoir(events, 3, seed=4).log
        self.assertTrue(a.equals(b))
        self.assertEqual(set(small.event_id), set(a.head(3).event_id))
        self.assertEqual(len(set(a.event_id)), len(a))

    def test_full_budget_retains_all_events(self):
        events = toy_events()
        sample = uniform_event_reservoir(events, len(events), seed=4)
        self.assertEqual(set(sample.log.event_id), set(events.index))


class PreservedPanelPrimitive(unittest.TestCase):
    # This checks complete histories under the EXISTING whole-node budget stop;
    # it makes no fixed-size uniform-panel inclusion claim.
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

    def test_seed_is_deterministic(self):
        events = toy_events()
        a = node_panel_full_history(events, 9, seed=3)
        b = node_panel_full_history(events, 9, seed=3)
        pd.testing.assert_frame_equal(a.log, b.log)


class SimpleWalkPrimitive(unittest.TestCase):
    def test_index_and_transitions_ignore_event_multiplicity(self):
        events = toy_events()
        more_events = pd.concat([events, events.iloc[:2]] * 2, ignore_index=True)
        a = run_walk(build_index(events), "time_agnostic", 60, seed=3)
        b = run_walk(build_index(more_events), "time_agnostic", 60, seed=3)
        pd.testing.assert_frame_equal(a, b)
        edges = {tuple(sorted((row.u, row.v))) for row in events.itertuples()}
        self.assertTrue(all((row.u, row.v) in edges
                            for row in a[a.kind == 1].itertuples()))
        self.assertTrue(a.t.isna().all())
        # Full-history retrieval must still be attached in future implementation.
