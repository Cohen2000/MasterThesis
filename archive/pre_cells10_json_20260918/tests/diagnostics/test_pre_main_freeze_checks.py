import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from census import window_index
from walks import build_index, run_walk

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location(
    "pre_main_freeze_checks", ROOT / "scripts/pre_main_freeze_checks.py")
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)


def toy_backbone():
    """Complete horizon [0,100], so W=5 windows are 20 source units wide.

    Dyads in census order: ab m=3 K5=2 (windows 0,4), bc m=1 K5=1, cd m=2 K5=1
    (95 and the closed endpoint 100 share window 4), ad m=1 K5=1, ac m=4 K5=4.
    """
    raw = pd.DataFrame([("a", "b", 0), ("b", "a", 10), ("a", "b", 90), ("b", "c", 50),
                        ("c", "d", 95), ("d", "c", 100), ("a", "d", 20), ("a", "c", 5),
                        ("a", "c", 30), ("c", "a", 70), ("a", "c", 99)],
                       columns=["u", "v", "t"])
    return checks.prepare_backbone("toy", raw)


def observe(bb, mask):
    windows = checks.complete_windows(bb)
    _, rho_full = checks.complete_targets(bb)
    return checks.observation_summary(bb, windows, rho_full, mask)


def labelled(bb, values):
    names = [f"{bb.node_labels[u]}{bb.node_labels[v]}" for u, v in zip(bb.dyad_u, bb.dyad_v)]
    return dict(zip(names, np.asarray(values).tolist()))


class PersistenceProfile(unittest.TestCase):
    def test_hand_checkable_occupancy_and_profiles(self):
        bb = toy_backbone()
        self.assertEqual(labelled(bb, bb.m), {"ab": 3, "bc": 1, "cd": 2, "ad": 1, "ac": 4})
        K_full, rho_full = checks.complete_targets(bb)
        self.assertEqual(labelled(bb, K_full[5]), {"ab": 2, "bc": 1, "cd": 1, "ad": 1, "ac": 4})
        self.assertEqual(rho_full[5].tolist(), [0.4, 0.2, 0.2, 0.0])
        self.assertEqual(rho_full[4].tolist(), [0.4, 0.2, 0.2])
        # W=10: ab occupies windows 0, 1 and 9.
        self.assertEqual(rho_full[10].tolist(), [0.4, 0.4, 0.2, 0, 0, 0, 0, 0, 0])
        windows = checks.complete_windows(bb)
        for W in checks.W_GRID:
            np.testing.assert_array_equal(
                checks.occupancy(bb.pair, windows[W], bb.n_dyads, W), K_full[W])
        K = np.array([0, 3, 1, 0, 5])
        np.testing.assert_allclose(checks.plugin_profile(K, 5), [2 / 3, 2 / 3, 1 / 3, 1 / 3])
        np.testing.assert_allclose(checks.roster_profile(K, 5), [0.4, 0.4, 0.2, 0.2])
        self.assertTrue(np.isnan(checks.plugin_profile(np.zeros(3, dtype=int), 5)).all())

    def test_normalized_error_formula(self):
        self.assertAlmostEqual(checks.normalized_profile_error(
            [0.25, 0.25, 0.25], [0.5, 0.25, 0.0], 4), 0.125)
        estimate, truth = [0.4, 0.3, 0.1, 0.0], [0.5, 0.2, 0.2, 0.1]
        self.assertAlmostEqual(checks.normalized_profile_error(estimate, truth, 5), 0.08)
        self.assertAlmostEqual(checks.profile_mae_w5(estimate, truth), 0.1)
        with self.assertRaises(ValueError):
            checks.normalized_profile_error([0.1, 0.1], truth, 5)


class NodeSample(unittest.TestCase):
    def test_fixed_size_uniform_sample(self):
        N, n, draws = 10, 4, 2000
        nodes, pair = np.zeros(N), 0
        for seed in range(draws):
            sample = checks.uniform_node_order(N, seed)[:n]
            self.assertEqual(len(np.unique(sample)), n)
            nodes[sample] += 1
            pair += {2, 7} <= set(sample.tolist())
        np.testing.assert_allclose(nodes / draws, n / N, atol=0.05)
        self.assertAlmostEqual(pair / draws, n * (n - 1) / (N * (N - 1)), delta=0.03)

    def test_induced_subgraph_with_full_histories(self):
        bb = toy_backbone()
        ids = {label: i for i, label in enumerate(bb.node_labels)}
        abc = checks.induced_dyads(bb, [ids["a"], ids["b"], ids["c"]])
        self.assertEqual(labelled(bb, abc), {"ab": True, "bc": True, "cd": False,
                                             "ad": False, "ac": True})
        row, _, _, _ = observe(bb, abc[bb.pair])
        self.assertEqual((row["n_observed_events"], row["n_observed_dyads"],
                          row["n_observed_nodes"]), (8, 3, 3))
        # Two non-adjacent nodes induce nothing, although each has incident dyads.
        self.assertFalse(checks.induced_dyads(bb, [ids["b"], ids["d"]]).any())
        order = np.array([ids["c"], ids["a"], ids["b"], ids["d"]])
        prefix = checks.node_prefix_events(bb, order)
        self.assertEqual(prefix.tolist(), [0, 0, 4, 8, 11])
        for size in range(5):
            self.assertEqual(prefix[size], bb.m[checks.induced_dyads(bb, order[:size])].sum())


class CoverageAndTimeAxis(unittest.TestCase):
    def test_event_dyad_and_node_coverage(self):
        bb = toy_backbone()
        ids = {label: i for i, label in enumerate(bb.node_labels)}
        row, _, _, _ = observe(bb, checks.induced_dyads(bb, [ids["a"], ids["b"], ids["c"]])[bb.pair])
        self.assertEqual((row["event_coverage"], row["dyad_coverage"], row["node_coverage"]),
                         (8 / 11, 3 / 5, 3 / 4))
        self.assertEqual([row[f"rho_hat_{k}"] for k in range(2, 6)], [2 / 3, 1 / 3, 1 / 3, 0])
        self.assertAlmostEqual(row["profile_mae_w5"], (4 / 15 + 2 / 15 + 2 / 15) / 4)
        suffix, _, _, _ = observe(bb, bb.t_norm >= 0.9)
        self.assertEqual((suffix["event_coverage"], suffix["dyad_coverage"],
                          suffix["node_coverage"]), (4 / 11, 3 / 5, 1.0))
        self.assertEqual((checks.event_sample_size(11, "0.25"),
                          checks.event_sample_size(11, "0.50")), (3, 6))
        events = pd.DataFrame({"u": bb.dyad_u[bb.pair], "v": bb.dyad_v[bb.pair], "t": bb.t_norm})
        mask = checks.sampled_event_mask(bb, events, 3, seed=5)
        sampled, _, _, _ = observe(bb, mask)
        self.assertEqual((sampled["n_observed_events"], sampled["event_coverage"]), (3, 3 / 11))
        self.assertEqual(sampled["n_observed_dyads"], len(np.unique(bb.pair[mask])))

    def test_observed_timestamps_stay_on_the_complete_axis(self):
        bb = toy_backbone()
        mask = bb.t_norm >= 0.9
        np.testing.assert_array_equal(np.sort(bb.t_norm[mask]), [0.9, 0.95, 0.99, 1.0])
        row, _, K, seen = observe(bb, mask)
        # All suffix events lie in the last complete-axis window.
        self.assertEqual([row[f"rho_hat_{k}"] for k in range(2, 6)], [0, 0, 0, 0])
        # Rescaling the suffix to its own range would spread the cd dyad over two windows.
        own = (bb.t_norm[mask] - bb.t_norm[mask].min()) / np.ptp(bb.t_norm[mask])
        rescaled = checks.occupancy(bb.pair[mask], window_index(own, 0, 0.2, 5), bb.n_dyads, 5)
        self.assertGreater(checks.plugin_profile(rescaled, 5)[0], 0)
        K_full, _ = checks.complete_targets(bb)
        full = checks.full_history_mask(bb, [0, 4])
        _, _, K_history, seen = observe(bb, full)
        np.testing.assert_array_equal(K_history[5][seen], K_full[5][seen])


class RecencyPurity(unittest.TestCase):
    def test_cutoff_choice_and_ties(self):
        bb = toy_backbone()
        self.assertEqual(checks.recency_cutoff(bb.t_norm, "0.25"), (0.95, 3))
        # 6 and 5 retained events are equally close to 5.5; the later cutoff wins.
        self.assertEqual(checks.recency_cutoff(bb.t_norm, "0.50"), (0.7, 5))

    def test_vanished_dyads_roster_and_survivors(self):
        bb = toy_backbone()
        K_full, rho_full = checks.complete_targets(bb)
        cutoff, summary, purity = checks.recency_purity(
            bb, checks.complete_windows(bb), rho_full, K_full, "0.25")
        self.assertEqual(cutoff, 0.95)
        self.assertEqual(purity["cutoff_source_units"], 95.0)
        # Suffix {95, 99, 100}: cd and ac survive; ab, bc and ad vanish.
        self.assertEqual((purity["vanished_dyad_fraction"], purity["event_coverage"]), (3 / 5, 3 / 11))
        self.assertEqual((purity["cutoff_w5_window"], purity["w5_windows_in_suffix"]), (4, 1))
        self.assertEqual([purity[f"plugin_rho_{k}"] for k in range(2, 6)], [0, 0, 0, 0])
        self.assertEqual([purity[f"roster_rho_{k}"] for k in range(2, 6)], [0, 0, 0, 0])
        self.assertEqual([purity[f"survivor_full_history_rho_{k}"] for k in range(2, 6)],
                         [0.5, 0.5, 0.5, 0.0])
        self.assertAlmostEqual(purity["plugin_profile_mae_w5"], 0.2)
        self.assertAlmostEqual(purity["survivor_full_history_profile_mae_w5"], (0.1 + 0.3 + 0.3) / 4)
        self.assertEqual(summary[0]["n_observed_events"], 3)


class RandomWalk(unittest.TestCase):
    def test_repeated_traversals_never_duplicate_events(self):
        raw = pd.DataFrame([("x", "y", t) for t in range(5)], columns=["u", "v", "t"])
        bb = checks.prepare_backbone("pair", raw)
        nodes, dyads = checks.simple_random_walk(bb, checks.walk_index(bb), 40, seed=3)
        self.assertEqual((len(nodes), len(dyads)), (41, 40))
        mask = checks.full_history_mask(bb, dyads)
        row, _, _, _ = observe(bb, mask)
        self.assertEqual((row["n_observed_events"], row["event_coverage"]), (5, 1.0))
        structure = checks.graph_structure(bb)
        walk = checks.walk_row(bb, structure, False, nodes, dyads, 40)
        self.assertEqual((walk["unique_traversed_dyads"], walk["event_coverage"]), (1, 1.0))
        self.assertEqual(walk["repeated_dyad_step_fraction"], 39 / 40)
        self.assertEqual(walk["revisit_fraction"], 39 / 40)

    def test_walk_matches_repository_walk_on_all_events(self):
        bb = toy_backbone()
        nodes, dyads = checks.simple_random_walk(bb, checks.walk_index(bb), 200, seed=11)
        events = pd.DataFrame({"u": bb.dyad_u[bb.pair], "v": bb.dyad_v[bb.pair], "t": bb.t_norm})
        reference = run_walk(build_index(events), "time_agnostic", 201, seed=11)
        np.testing.assert_array_equal(nodes, reference["node"].to_numpy())
        at, cumulative = checks.discovery_curve(dyads, bb.m)
        self.assertEqual(cumulative[-1], bb.m[np.unique(dyads)].sum())
        self.assertLessEqual(cumulative[-1], bb.n_events)

    def test_discovery_curve_and_median_calibration(self):
        curve = checks.discovery_curve(np.array([2, 2, 0, 2, 1]), np.array([5, 1, 3]))
        self.assertEqual(checks.events_by_step(curve, np.arange(1, 6)).tolist(), [3, 3, 8, 8, 9])
        coverage = [[0.125, 0.125, 0.5], [0.0, 0.375, 0.75], [0.25, 0.25, 0.5], [0.0, 0.25, 1.0]]
        self.assertEqual(checks.closest_median_parameter([10, 20, 30], coverage, "0.25"), (20, 0.25))
        # Equal distance on both sides: the smaller budget is chosen.
        self.assertEqual(checks.closest_median_parameter([1, 2], [[0.125, 0.375]] * 4, "0.25")[0], 1)


class TimingVariants(unittest.TestCase):
    def setUp(self):
        rows = [(f"n{i}", f"n{(i + 1) % 12}", float((37 * (4 * i + j)) % 1000))
                for i in range(12) for j in range(1 + i % 4)]
        self.bb = checks.prepare_backbone("ring", pd.DataFrame(rows, columns=["u", "v", "t"]))

    def variants(self):
        out = {}
        for target, seed, instance, labels, error in checks.timing_variants(self.bb):
            self.assertEqual(error, "")
            events = instance.events
            out[target] = (labels[events.u.to_numpy()], labels[events.v.to_numpy()],
                           events.t.to_numpy(float))
        return out

    def test_generated_variants_satisfy_all_invariants(self):
        variants = self.variants()
        self.assertEqual(sorted(variants), ["0.15", "0.35"])
        for u, v, t in variants.values():
            result = checks.timing_invariants(self.bb, u, v, t)
            self.assertFalse(result["hard_failure"])
        again = self.variants()
        for target in variants:
            for a, b in zip(variants[target], again[target]):
                np.testing.assert_array_equal(a, b)

    def test_each_violation_is_a_hard_failure(self):
        u, v, t = self.variants()["0.35"]
        base = pd.Series(np.minimum(u, v) + "|" + np.maximum(u, v))
        counts = base.value_counts()
        three, two = counts[counts == 3].index[0], counts[counts == 2].index[0]
        moved_u, moved_v = u.copy(), v.copy()
        row = int(np.flatnonzero(base == three)[0])
        moved_u[row], moved_v[row] = two.split("|")
        shifted = t.copy()
        shifted[0] += 1e-6
        renamed_u = u.copy()
        renamed_u[0] = "new-node"
        cases = {
            "same_exact_per_dyad_counts": (moved_u, moved_v, t),
            "same_global_timestamp_multiset": (u, v, shifted),
            "same_node_set": (renamed_u, v, t),
            "same_total_events": (u[1:], v[1:], t[1:]),
        }
        for broken, arrays in cases.items():
            with self.subTest(broken=broken):
                result = checks.timing_invariants(self.bb, *arrays)
                self.assertFalse(result[broken])
                self.assertTrue(result["hard_failure"])
        # Swapping one event between a 3-event and a 2-event dyad keeps the multiset.
        moved = checks.timing_invariants(self.bb, moved_u, moved_v, t)
        self.assertTrue(moved["same_per_dyad_count_multiset"])
        self.assertTrue(moved["same_collapsed_dyad_set"])


class GroupedLeaveOneBackboneOut(unittest.TestCase):
    def profiles(self, a_natural=0.9):
        rows = []
        for dataset, base in (("A", a_natural), ("B", 0.5), ("C", 0.3)):
            for kind, shift in (("natural", 0.0), ("rho2_015", -0.1), ("rho2_035", -0.05)):
                value = base + shift if not (dataset == "A" and kind != "natural") else 0.7 + shift
                rows.append({"dataset": dataset, "graph_kind": kind, "valid": True,
                             "rho_2": value, "rho_3": value / 2, "rho_4": value / 4,
                             "rho_5": value / 8})
        return pd.DataFrame(rows)

    def test_held_out_backbone_and_its_variants_are_excluded(self):
        result = checks.grouped_loso_constant(self.profiles()).set_index(
            ["held_out_dataset", "graph_kind"])
        a = result.loc[("A", "natural")]
        self.assertEqual((a.reference_datasets, a.n_reference_datasets), ("B;C", 2))
        self.assertAlmostEqual(a.pred_rho_2, 0.4)
        self.assertAlmostEqual(result.loc[("A", "rho2_015")].pred_rho_2, 0.3)
        self.assertAlmostEqual(result.loc[("B", "rho2_035")].pred_rho_2, (0.65 + 0.25) / 2)
        self.assertFalse(result.held_out_in_reference.any())
        changed = checks.grouped_loso_constant(self.profiles(a_natural=0.1)).set_index(
            ["held_out_dataset", "graph_kind"])
        self.assertAlmostEqual(changed.loc[("A", "natural")].pred_rho_2, 0.4)
        self.assertNotAlmostEqual(changed.loc[("B", "natural")].pred_rho_2,
                                  result.loc[("B", "natural")].pred_rho_2)

    def test_invalid_variants_are_not_references(self):
        profiles = self.profiles()
        profiles.loc[(profiles.dataset == "C") & (profiles.graph_kind == "rho2_015"), "valid"] = False
        result = checks.grouped_loso_constant(profiles).set_index(["held_out_dataset", "graph_kind"])
        self.assertEqual(result.loc[("A", "rho2_015")].reference_datasets, "B")


if __name__ == "__main__":
    unittest.main()
