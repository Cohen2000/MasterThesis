import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prompt_contract_g1 as C  # noqa: E402
from report_freeze_g2 import (  # noqa: E402
    censoring_recovery,
    delta_table,
    schema_check,
    select_replication_graphs,
)


def toy_primary(n_graphs=12):
    rows = []
    rng = np.random.default_rng(0)
    for arm in C.ARMS:
        for g in range(n_graphs):
            rows.append({
                "case_id": f"g{g}|{arm}", "instance_id": f"g{g}",
                "group_id": f"grp{g % 4}", "strategy": arm,
                "coverage": float(rng.uniform(0.01, 0.4)),
                "n_nodes_true": 500, "n_events_true": 9000,
                "input__nmask_exact_json": '{"1,01":5,"3,07":2}',
                "rho_W5_k2": 0.4, "rho_W5_k3": 0.3,
                "rho_W5_k4": 0.2, "rho_W5_k5": 0.1,
                "est__plugin_rho_k2": 0.25, "est__plugin_rho_k3": 0.2,
                "est__plugin_rho_k4": 0.15, "est__plugin_rho_k5": 0.05,
            })
    return pd.DataFrame(rows)


class OutputSchemaTest(unittest.TestCase):
    """G2.5: an accidental format difference would become a condition effect."""

    def test_task_block_is_byte_identical_everywhere(self):
        result = schema_check(toy_primary(3))
        self.assertEqual(int(result.distinct_task_blocks.iloc[0]), 1)
        self.assertTrue(
            bool(result.byte_identical_across_all_arms_and_conditions.iloc[0]))

    def test_every_arm_and_condition_was_actually_checked(self):
        result = schema_check(toy_primary(3))
        # 5 arms x 6 non-mismatch conditions + 2 arms x mismatched, per graph.
        self.assertEqual(int(result.combinations_checked.iloc[0]), 3 * 32)

    def test_a_perturbed_task_block_is_caught(self):
        original = C.TASK
        try:
            C.TASK = original + "\nExtra line."
            self.assertGreater(
                int(schema_check(toy_primary(2)).distinct_task_blocks.iloc[0]),
                0)
        finally:
            C.TASK = original
        # After restoring, the check passes again -- the guard is not sticky.
        self.assertEqual(
            int(schema_check(toy_primary(2)).distinct_task_blocks.iloc[0]), 1)

    def test_the_requested_keys_appear_in_every_condition(self):
        row = toy_primary(1).iloc[0].to_dict()
        for condition in C.CONDITIONS:
            stated = (C.MISMATCH_PAIR[1] if condition == "mismatched" else None)
            arm_row = dict(row, strategy=C.MISMATCH_PAIR[0])
            text = C.build_prompt(arm_row, condition, stated)
            for key in ("rho_k2", "rho_k5", "C_one_step", "lo90", "hi90"):
                self.assertIn(key, text)


class SubsetSelectionTest(unittest.TestCase):
    """G2.4: fixed by a recorded rule, never by eye."""

    def test_selection_is_deterministic(self):
        primary = toy_primary()
        first = select_replication_graphs(primary, 4)
        second = select_replication_graphs(primary.sample(frac=1, random_state=7),
                                           4)
        self.assertEqual(list(first.instance_id), list(second.instance_id))

    def test_selection_spans_distinct_groups(self):
        chosen = select_replication_graphs(toy_primary(), 4)
        self.assertEqual(len(chosen), 4)
        self.assertEqual(chosen.group_id.nunique(), 4)

    def test_selection_does_not_cluster_at_one_end_of_coverage(self):
        primary = toy_primary(16)
        chosen = select_replication_graphs(primary, 4)
        span = chosen.mean_coverage.max() - chosen.mean_coverage.min()
        whole = (primary.groupby("instance_id").coverage.mean().max() -
                 primary.groupby("instance_id").coverage.mean().min())
        self.assertGreater(span, 0.3 * whole)


class DeltaTest(unittest.TestCase):
    def test_delta_is_truth_minus_naive(self):
        table, per_case = delta_table(toy_primary(2))
        self.assertAlmostEqual(float(per_case.delta_i.iloc[0]), 0.4 - 0.25)

    def test_pooled_row_is_present_and_counts_every_case(self):
        primary = toy_primary(3)
        table, _ = delta_table(primary)
        pooled = table[table.arm == "POOLED"].iloc[0]
        self.assertEqual(int(pooled.cases), len(primary))


class CensoringRecoveryTest(unittest.TestCase):
    def _ladder(self, naive, mask):
        return pd.DataFrame([
            {"arm": "a", "estimator": "naive read-off", "rho2_bias": naive},
            {"arm": "a", "estimator": "mask MLE (uniform; censoring-aware, "
                                     "mechanism-agnostic)", "rho2_bias": mask},
        ])

    def test_defined_when_the_anchor_moves_toward_zero(self):
        row = censoring_recovery(self._ladder(-0.30, -0.05)).iloc[0]
        self.assertEqual(row.normalization, "defined")
        self.assertTrue(bool(row.anchor_moves_toward_zero))

    def test_degenerate_when_the_anchor_moves_away_from_zero(self):
        # This is the arms A and B case: no censoring to recover, so a
        # censoring correction inflates an already-correct profile.
        row = censoring_recovery(self._ladder(-0.01, 0.29)).iloc[0]
        self.assertIn("DEGENERATE", row.normalization)
        self.assertFalse(bool(row.anchor_moves_toward_zero))

    def test_degenerate_when_the_denominator_vanishes(self):
        row = censoring_recovery(self._ladder(-0.10, -0.10)).iloc[0]
        self.assertIn("DEGENERATE", row.normalization)


if __name__ == "__main__":
    unittest.main()
