import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nonwalk_prompt_contract import (  # noqa: E402
    MECHANISMS,
    assert_prompt_metadata_parity,
    metadata_only_columns,
    prompt_parity_columns,
    render_nonwalk_prompt,
)


class PromptContract(unittest.TestCase):
    def row(self):
        return pd.Series({
            "strategy": "uniform_event_reservoir", "target_budget": 800,
            "budget": 800, "n_nodes_true": 100,
            "input__nw_exact_json": '{"1,1":2}',
            "input__nmask_exact_json": '{"1,01":2}',
            "input__window_counts_exact_json": '{"1,0,0,0,0":2}',
            "input__recent_events_json": '[[0,1,0.1]]',
            "observed_walk_nodes": 2, "observed_walk_edges": 1,
            "observed_timed_edges": 1,
            "crawl__restart_fraction": 0.0,
            "crawl__edge_revisit_rate": 0.0,
            "crawl__discovery_010": 1.0, "crawl__discovery_050": 1.0,
            "crawl__discovery_100": 1.0, "crawl__node_hits_q25": 1.0,
            "crawl__node_hits_q50": 1.0, "crawl__node_hits_q75": 1.0,
            "crawl__node_hits_q90": 1.0, "crawl__edge_hits_q25": 1.0,
            "crawl__edge_hits_q50": 1.0, "crawl__edge_hits_q75": 1.0,
            "crawl__edge_hits_q90": 1.0, "crawl__observed_degree_mean": 1.0,
            "crawl__observed_degree_max": 1.0, "crawl__dt_q25": float("nan"),
            "crawl__dt_q50": float("nan"), "crawl__dt_q75": float("nan"),
            "crawl__dt_q90": float("nan"), "crawl__observed_time_span": 0.0,
            "crawl__first_node_collision_frac": 1.0,
            "pat__adjacent_observed_C": float("nan"),
            "pat__noncontiguous_edge_share": 0.0,
            "pat__mean_mask_width": 0.0,
            "pat__lifetime_mean": float("nan"),
            **{f"pat__lifetime_q{q}": float("nan") for q in (25, 50, 75, 90)},
            "pat__iet_mean": float("nan"),
            **{f"pat__iet_q{q}": float("nan") for q in (25, 50, 75, 90)},
            **{f"pat__first_w{w}": float(w == 0) for w in range(5)},
            **{f"pat__last_w{w}": float(w == 0) for w in range(5)},
            **{f"pat__event_share_w{w}": float(w == 0) for w in range(5)},
        })

    def test_no_sample_contains_identical_metadata_but_no_histogram(self):
        row = self.row()
        full = render_nonwalk_prompt(row, True)
        control = render_nonwalk_prompt(row, False)
        assert_prompt_metadata_parity(full, control, row)
        self.assertIn(row.input__nmask_exact_json, full)
        self.assertIn(row.input__window_counts_exact_json, full)
        self.assertNotIn(row.input__nmask_exact_json, control)
        self.assertNotIn(row.input__window_counts_exact_json, control)

    def test_feature_contract_excludes_truth_and_oracles(self):
        frame = pd.DataFrame(columns=[
            "target_budget", "budget", "n_nodes_true", "rho_W5_k2",
            "oracle__x", "diag__x", "occ__a", "pat__mask_01",
            "pat__iet_mean", "crawl__discovery_010",
            "crawl__discovery_025",
        ])
        frame["wcnt__w0_mean"] = []
        cols = prompt_parity_columns(frame, "window_counts_crawl_temporal")
        self.assertIn("pat__iet_mean", cols)
        self.assertIn("crawl__discovery_010", cols)
        self.assertIn("wcnt__w0_mean", cols)
        self.assertNotIn("crawl__discovery_025", cols)
        self.assertNotIn("rho_W5_k2", cols)
        self.assertEqual(metadata_only_columns(frame),
                         ["target_budget", "budget", "n_nodes_true"])

    def test_every_nonwalk_strategy_has_a_renderable_mechanism(self):
        expected = {
            "uniform_event_reservoir", "time_prefix_events",
            "time_random_window_events", "node_panel_full_history",
            "ego_recent_k1", "ego_recent_k5", "ego_recent_k20",
            "ego_recent_kall",
        }
        self.assertEqual(set(MECHANISMS), expected)
        for strategy in sorted(expected):
            row = self.row().copy()
            row["strategy"] = strategy
            prompt = render_nonwalk_prompt(row, True)
            self.assertIn("SAMPLING MECHANISM", prompt)
            self.assertIn(strategy, prompt)


if __name__ == "__main__":
    unittest.main()
