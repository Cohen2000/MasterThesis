"""Small invariants for the repeated OFAT assembly/evaluation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate_llm_ofat import record_score  # noqa: E402
from make_llm_ofat import (CELLS, generation_seed_for_rep,  # noqa: E402
                           include_in_plan)


class OfatSeeds(unittest.TestCase):
    def test_rep1_keeps_the_frozen_hf_seed(self):
        self.assertEqual(generation_seed_for_rep("prompt", 1), 0)

    def test_later_reps_are_deterministic_and_distinct(self):
        a = generation_seed_for_rep("prompt", 2)
        self.assertEqual(a, generation_seed_for_rep("prompt", 2))
        self.assertNotEqual(a, generation_seed_for_rep("prompt", 3))
        self.assertGreater(a, 0)


class OfatWorkInventory(unittest.TestCase):
    def test_only_deepseek_rebuilds_all_six_cells(self):
        rows = [{"input_kind": cell, "rep": rep}
                for rep in (1, 2, 3) for cell in CELLS]
        # Per one case; multiply by 36 for the real run.
        expected = {"codex": 2, "gemini": 6,
                    "deepseek": 6, "qwen": 2}
        for plan, count in expected.items():
            self.assertEqual(sum(include_in_plan(plan, row) for row in rows),
                             count, plan)

    def test_real_new_generation_total_is_648(self):
        per_case = {}
        rows = [{"input_kind": cell, "rep": rep}
                for rep in (1, 2, 3) for cell in CELLS]
        for plan in ("codex", "gemini", "deepseek", "qwen"):
            per_case[plan] = sum(include_in_plan(plan, row) for row in rows)
        total = 36 * (per_case["codex"] + per_case["gemini"]
                      + per_case["deepseek"] + 2 * per_case["qwen"])
        self.assertEqual(total, 648)


class OfatFrozenLoss(unittest.TestCase):
    truth = {"rho_W5_k2": "0.2", "rho_W5_k3": "0.1",
             "rho_W5_k4": "0.05", "rho_W5_k5": "0.0"}

    def test_invalid_component_gets_loss_one(self):
        record = {"answer": '{"rho_k2":0.2,"rho_k3":0.1,'
                            '"rho_k4":2,"rho_k5":0.0}'}
        scored = record_score(record, self.truth)
        self.assertFalse(scored["valid"])
        self.assertAlmostEqual(scored["penalized"], 0.25)
        self.assertIsNone(scored["complete"])

    def test_complete_profile_is_scored_without_clipping(self):
        record = {"answer": '{"rho_k2":0.3,"rho_k3":0.2,'
                            '"rho_k4":0.1,"rho_k5":0.0}'}
        scored = record_score(record, self.truth)
        self.assertTrue(scored["valid"])
        self.assertAlmostEqual(scored["complete"], 0.0625)


if __name__ == "__main__":
    unittest.main()
