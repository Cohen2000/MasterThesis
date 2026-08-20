"""Input-factor design of the V2.1 prompt builder.

The historical input axis was a cumulative ladder, so no single add-on block
was ever isolated. `render_input` now composes a base representation with an
independent factor set. These tests pin the two properties that matter: the
frozen ladder names must keep rendering exactly what they rendered before, and
the one-factor-at-a-time cells must differ from the reference cell by exactly
one block.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_llm_prompts_v2 import (  # noqa: E402
    INPUT_FACTORS, INPUT_LADDER, INPUT_OFAT, INPUT_CRAWL, INPUT_RECENT,
    INPUT_TEMPORAL, canonical_input_kind, crawl_block, parse_input_kind,
    render_input, temporal_block,
)


def synthetic_row():
    """A row carrying every column the input blocks read, with dummy values."""
    row = {
        "budget": 800,
        "input__nw_exact_json": '{"1,1":10,"2,2":3}',
        "input__nmask_exact_json": '{"1,1":10,"2,3":3}',
        "input__recent_events_json": "[[0,1,0.11],[1,2,0.42]]",
        "observed_walk_nodes": 120,
        "observed_walk_edges": 300,
        "observed_timed_edges": 280,
    }
    for name in ("crawl__restart_fraction", "crawl__edge_revisit_rate",
                 "crawl__discovery_010", "crawl__discovery_050",
                 "crawl__discovery_100", "crawl__observed_degree_mean",
                 "crawl__observed_degree_max", "crawl__observed_time_span",
                 "crawl__first_node_collision_frac",
                 "pat__adjacent_observed_C", "pat__noncontiguous_edge_share",
                 "pat__mean_mask_width", "pat__lifetime_mean",
                 "pat__iet_mean"):
        row[name] = 0.25
    for q in (25, 50, 75, 90):
        for stem in ("crawl__node_hits_q", "crawl__edge_hits_q", "crawl__dt_q",
                     "pat__lifetime_q", "pat__iet_q"):
            row[f"{stem}{q}"] = 0.1 * q
    for w in range(5):
        for stem in ("pat__first_w", "pat__last_w", "pat__event_share_w"):
            row[f"{stem}{w}"] = 0.2
    return row


class InputKindNaming(unittest.TestCase):
    def test_legacy_ladder_names_resolve_to_cumulative_factor_sets(self):
        self.assertEqual(parse_input_kind("nw"), ("nw", ()))
        self.assertEqual(parse_input_kind("mask"), ("mask", ()))
        self.assertEqual(parse_input_kind("mask_crawl_full"),
                         ("mask", ("crawl",)))
        self.assertEqual(parse_input_kind("mask_crawl_temporal"),
                         ("mask", ("crawl", "temporal")))
        self.assertEqual(parse_input_kind("mask_crawl_temporal_recent"),
                         ("mask", ("crawl", "temporal", "recent")))

    def test_ofat_names_isolate_single_factors(self):
        self.assertEqual(parse_input_kind("mask_crawl"), ("mask", ("crawl",)))
        self.assertEqual(parse_input_kind("mask_temporal"),
                         ("mask", ("temporal",)))
        self.assertEqual(parse_input_kind("mask_recent"),
                         ("mask", ("recent",)))
        self.assertEqual(parse_input_kind("mask_all"),
                         ("mask", INPUT_FACTORS))

    def test_factor_order_in_the_name_does_not_matter(self):
        self.assertEqual(parse_input_kind("mask_recent_crawl"),
                         parse_input_kind("mask_crawl_recent"))

    def test_canonical_name_round_trips(self):
        for kind in INPUT_OFAT:
            base, factors = parse_input_kind(kind)
            self.assertEqual(canonical_input_kind(base, factors), kind)

    def test_unknown_names_are_rejected(self):
        for bad in ("", "mask_", "masks", "mask_bogus", "mask_crawl_crawl",
                    "edges_crawl"):
            with self.assertRaises(ValueError, msg=bad):
                parse_input_kind(bad)


class InputRendering(unittest.TestCase):
    def setUp(self):
        self.row = synthetic_row()

    def render(self, kind):
        return render_input(self.row, kind)

    def test_renamed_cells_render_the_historical_text(self):
        # These two OFAT cells are the frozen suite's cells under a new name,
        # so their existing answers stay usable.
        self.assertEqual(self.render("mask_crawl"),
                         self.render("mask_crawl_full"))
        self.assertEqual(self.render("mask_all"),
                         self.render("mask_crawl_temporal_recent"))

    def test_rendering_depends_only_on_the_factor_set(self):
        self.assertEqual(self.render("mask_recent_crawl"),
                         self.render("mask_crawl_recent"))

    def test_each_ofat_cell_adds_exactly_its_own_block(self):
        reference = self.render("mask")
        headers = {"crawl": INPUT_CRAWL, "temporal": INPUT_TEMPORAL,
                   "recent": INPUT_RECENT}
        for factor, header in headers.items():
            text = self.render(f"mask_{factor}")
            self.assertNotIn(header, reference)
            self.assertIn(header, text)
            for other, other_header in headers.items():
                if other != factor:
                    self.assertNotIn(other_header, text)
            self.assertTrue(text.startswith(reference))

    def test_full_cell_contains_every_block(self):
        text = self.render("mask_all")
        for header in (INPUT_CRAWL, INPUT_TEMPORAL, INPUT_RECENT):
            self.assertIn(header, text)

    def test_nw_base_is_a_reduction_not_an_addition(self):
        self.assertNotIn(self.row["input__nmask_exact_json"],
                         self.render("nw"))
        self.assertIn(self.row["input__nw_exact_json"], self.render("nw"))

    def test_blocks_are_strict_json(self):
        import json
        json.loads(crawl_block(self.row))
        json.loads(temporal_block(self.row))

    def test_every_declared_kind_renders(self):
        for kind in set(INPUT_LADDER) | set(INPUT_OFAT):
            self.assertTrue(self.render(kind))


if __name__ == "__main__":
    unittest.main()
