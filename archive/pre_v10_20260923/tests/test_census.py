import gzip
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from census import normalize, parse_events
from dataset_census import census_datasets, summarize_events


class EmpiricalLoading(unittest.TestCase):
    def test_gzip_comments_and_bipartite_namespaces(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "events.csv.gz"
            with gzip.open(path, "wt") as stream:
                stream.write("u,v,t\n# comment\n1,1,0\n1,1,2\n2,1,4\n")
            raw = parse_events(path, {"delimiter": ",", "skiprows": 1,
                                     "bipartite": True,
                                     "columns": {"u": 0, "v": 1, "t": 2}})
        self.assertEqual(len(raw), 3)
        self.assertEqual(raw.iloc[0].u, "u:1")
        self.assertEqual(raw.iloc[0].v, "i:1")
        self.assertEqual(normalize(raw).pair.nunique(), 2)

    def test_undirected_dyads_self_loops_and_duplicate_events(self):
        raw = pd.DataFrame([("a", "b", 0), ("b", "a", 1),
                            ("a", "b", 1), ("c", "c", 2)],
                           columns=["u", "v", "t"])
        pairs = normalize(raw)
        self.assertEqual(len(pairs), 3)
        self.assertEqual(pairs.pair.nunique(), 1)

    def test_counts_and_physical_window_length(self):
        raw = pd.DataFrame([("a", "b", 100), ("b", "a", 125),
                            ("a", "b", 185), ("b", "c", 110),
                            ("b", "c", 112), ("c", "d", 165),
                            ("c", "d", 200), ("a", "d", 150),
                            ("b", "d", 195), ("isolated", "isolated", 170)],
                           columns=["u", "v", "t"])
        row = summarize_events(raw)
        self.assertEqual((row["n_nodes"], row["n_edges_full"], row["n_events"]),
                         (4, 5, 9))
        self.assertAlmostEqual(row["events_per_edge"], 1.8)
        self.assertEqual([row[f"p_m_ge_{k}"] for k in range(2, 6)],
                         [0.6, 0.2, 0.0, 0.0])
        self.assertEqual([row[f"rho_{k}"] for k in range(2, 6)],
                         [0.4, 0.2, 0.0, 0.0])
        self.assertEqual(row["observation_span_source_units"], 100)
        self.assertEqual(row["window_duration_source_units"], 20)
        self.assertAlmostEqual(row["mean_occupancy_derived"], 0.32)

    def test_absent_registry_entries_remain_visible_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            registry = root / "datasets.yaml"
            registry.write_text("datasets:\n  absent:\n    label: Absent\n"
                                "    file: missing.csv\n", encoding="utf-8")
            result = census_datasets(registry, root)
            self.assertEqual(result.dataset.tolist(), ["absent"])
            self.assertEqual(result.status.tolist(), ["absent"])
            self.assertFalse((root / "missing.csv").exists())

    def test_unknown_dataset_key_is_rejected(self):
        # Only a temporary miniature registry is read by the test suite.
        with tempfile.TemporaryDirectory() as directory:
            registry = Path(directory) / "datasets.yaml"
            registry.write_text("datasets: {}\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                census_datasets(registry, Path(directory), ["unknown"])
