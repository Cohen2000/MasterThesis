import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from make_codex_prompts_g3 import (  # noqa: E402
    CUT_LADDER,
    PROTECTED_ARMS,
    PROTECTED_CONDITIONS,
    apply_cuts,
)


def toy_records():
    arms = ["time_agnostic_t", "time_respecting", "recent_history_k20",
            "node_panel_full_history", "event_sample_then_full_history"]
    out = []
    for arm in arms:
        for condition in ("hidden", "direction_only", "mechanism",
                          "mechanism_direction"):
            out.append({"prompt_id": f"{arm}|{condition}", "strategy": arm,
                        "condition": condition, "subset": "factorial"})
        out.append({"prompt_id": f"{arm}|ic", "strategy": arm,
                    "condition": "irrelevant_context",
                    "subset": "irrelevant_context"})
        out.append({"prompt_id": f"{arm}|dh", "strategy": arm,
                    "condition": "disclosed_historical",
                    "subset": "disclosed_historical"})
        out.append({"prompt_id": f"{arm}|sr", "strategy": arm,
                    "condition": "hidden", "subset": "seed_replication"})
    return out


class CutTest(unittest.TestCase):
    def test_the_applied_cut_keeps_the_full_factorial(self):
        keep, _ = apply_cuts(
            toy_records(),
            {"irrelevant_context", "disclosed_historical", "seed_replication"},
            None)
        conditions = {r["condition"] for r in keep}
        self.assertEqual(conditions, {"hidden", "direction_only", "mechanism",
                                      "mechanism_direction"})
        self.assertEqual(len({r["strategy"] for r in keep}), 5)

    def test_cutting_a_protected_condition_is_refused(self):
        records = [r for r in toy_records() if r["condition"] == "hidden"]
        for r in records:
            r["subset"] = "factorial"
        with self.assertRaises(RuntimeError):
            apply_cuts(records, {"factorial"}, None)

    def test_cutting_a_protected_arm_is_refused(self):
        for arm in PROTECTED_ARMS:
            with self.assertRaises(RuntimeError):
                apply_cuts(toy_records(), set(), arm)

    def test_cutting_an_unprotected_walk_arm_is_allowed(self):
        keep, log = apply_cuts(toy_records(), set(), "recent_history_k20")
        self.assertNotIn("recent_history_k20", {r["strategy"] for r in keep})
        for arm in PROTECTED_ARMS:
            self.assertIn(arm, {r["strategy"] for r in keep})

    def test_the_log_accounts_for_every_removed_prompt(self):
        records = toy_records()
        cut = {"irrelevant_context", "disclosed_historical", "seed_replication"}
        keep, log = apply_cuts(records, cut, None)
        self.assertEqual(len(keep) + int(log.prompts.sum()), len(records))
        self.assertEqual(set(log.removed), cut)

    def test_the_log_says_what_still_carries_a_cut_subset(self):
        _, log = apply_cuts(toy_records(), {"seed_replication"}, None)
        carried = log.set_index("removed").carried_by["seed_replication"]
        self.assertIn("qwen", carried.lower())

    def test_the_ladder_order_is_the_brief_order(self):
        self.assertEqual(CUT_LADDER[:2],
                         ("irrelevant_context", "disclosed_historical"))
        self.assertEqual(CUT_LADDER[-1], "direction_only")
        for condition in PROTECTED_CONDITIONS:
            self.assertNotIn(condition, CUT_LADDER)


if __name__ == "__main__":
    unittest.main()
