import unittest

import networkx as nx
import numpy as np
import pandas as pd

from benchmark_generators import (activity_memory_event_stream, dar_event_stream,
                                  family_from_events)
from generator import make_dcsbm_graph, make_instance


class ControlledTiming(unittest.TestCase):
    def test_empirical_backbone_counts_nodes_and_timestamps_are_preserved(self):
        # Small in-memory fixture; no empirical dataset or panel is loaded.
        original = pd.DataFrame([(i, i + 1, t) for i in range(20)
                                 for t in np.linspace(0.0, 0.999, 10)],
                                columns=["u", "v", "t"])
        family = family_from_events(original, "fixture")
        variants = [make_instance(family, rho, seed=12, span_layout="contiguous")
                    for rho in (0.15, 0.55)]
        expected = original.groupby(["u", "v"]).size()
        for variant in variants:
            pd.testing.assert_series_equal(variant.events.groupby(["u", "v"]).size(),
                                           expected)
            self.assertEqual(set(variant.events.u) | set(variant.events.v),
                             set(original.u) | set(original.v))
            np.testing.assert_array_equal(np.sort(variant.events.t), np.sort(original.t))
        repeated = make_instance(family, 0.15, seed=12, span_layout="contiguous")
        pd.testing.assert_frame_equal(variants[0].events, repeated.events)
        # Feasibility of the requested rho is a later scientific check, not a
        # reason to force achieved values to match a nominal target in this test.


class MechanisticGenerators(unittest.TestCase):
    def test_dcsbm_preserves_requested_nodes_and_seed(self):
        a = make_dcsbm_graph(40, seed=4)
        b = make_dcsbm_graph(40, seed=4)
        self.assertEqual(set(a), set(range(40)))
        self.assertEqual(set(a.edges), set(b.edges))
        self.assertEqual(len(list(nx.selfloop_edges(a))), 0)

    def test_dar_event_layer(self):
        # Invariant retained from the historical test_benchmark.py.
        graph = nx.cycle_graph(30)
        events = dar_event_stream(graph, alpha=0.8, chi=0.4, seed=4, W=5)
        self.assertGreater(len(events), 1)
        self.assertGreaterEqual(events.t.min(), 0.0)
        self.assertLessEqual(events.t.max(), 1.0)
        self.assertTrue({"u", "v", "t"}.issubset(events.columns))
        pd.testing.assert_frame_equal(events, dar_event_stream(
            graph, alpha=0.8, chi=0.4, seed=4, W=5))

    def test_dar_full_memory_keeps_active_edges_in_all_windows(self):
        events = dar_event_stream(nx.cycle_graph(40), alpha=1, chi=0.5,
                                  seed=4, W=5, event_rate=0)
        self.assertTrue(events.groupby(["u", "v"]).size().eq(5).all())
        self.assertTrue((events.u < events.v).all())

    def test_activity_driven_modes_are_seeded_and_undirected(self):
        for memory in (None, 1.0):
            with self.subTest(memory=memory):
                options = dict(n=30, seed=8, slots_per_window=10,
                               mean_activity=0.15, memory_beta=memory)
                events = activity_memory_event_stream(**options)
                pd.testing.assert_frame_equal(events, activity_memory_event_stream(**options))
                self.assertTrue((events.u < events.v).all())
                self.assertTrue(((events.t >= 0) & (events.t < 1)).all())
