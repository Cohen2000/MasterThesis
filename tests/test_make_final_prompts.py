import json
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import prompt_contract_g1 as C  # noqa: E402
from make_final_prompts import FACTORIAL, build  # noqa: E402


def toy_config():
    return {"final_run": {
        "arms": {a: {"family": "walk", "budget": 800} for a in C.ARMS},
        "subsets": {
            "seed_replication": {"conditions": ["hidden", "mechanism"]},
            "irrelevant_context": {"arms": list(C.MISMATCH_PAIR)},
        }}}


def toy_cases(n_graphs=3, slots=1):
    rows = []
    for arm in C.ARMS:
        for g in range(n_graphs):
            for slot in range(slots):
                rows.append({
                    "case_id": f"g{g}|{arm}|s{slot}", "instance_id": f"g{g}",
                    "group_id": f"grp{g % 2}", "strategy": arm,
                    "seed_slot": slot, "budget": 800,
                    "coverage": 0.05 + 0.01 * g,
                    "n_nodes_true": 500, "n_events_true": 9000,
                    "input__nmask_exact_json":
                        '{"%d,01":5,"3,07":2}' % (g + slot + 1),
                })
    return pd.DataFrame(rows)


class PromptSetTest(unittest.TestCase):
    def setUp(self):
        self.primary = toy_cases(3, 1)
        self.records = build(self.primary, self.primary, toy_config(), [],
                             run_historical=False)

    def test_one_record_per_prompt_not_per_call(self):
        # Generations are a runner-level repeat; putting them in prompt_id
        # would make the frozen prompt set depend on the model roster.
        ids = [r["prompt_id"] for r in self.records]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertFalse(any(i.endswith("|g0") for i in ids))

    def test_the_identical_sample_reaches_every_condition_of_a_case(self):
        """The single most load-bearing property of the design."""
        by_case = {}
        for record in self.records:
            body = record["prompt"]
            if "OBSERVED DATA" not in body:
                continue          # metadata_only shows no sample by design
            sample = body[body.index("OBSERVED DATA"):body.index("TASK\n")]
            by_case.setdefault(record["case_id"], set()).add(sample)
        self.assertTrue(by_case)
        for case_id, samples in by_case.items():
            self.assertEqual(len(samples), 1, f"{case_id} varies by condition")

    def test_every_arm_gets_the_whole_factorial(self):
        for arm in C.ARMS:
            got = {r["condition"] for r in self.records
                   if r["strategy"] == arm and r["subset"] == "factorial"}
            self.assertEqual(got, set(FACTORIAL))

    def test_mismatched_only_on_the_frozen_pair_and_bidirectional(self):
        rows = [r for r in self.records if r["condition"] == "mismatched"]
        self.assertEqual({r["strategy"] for r in rows}, set(C.MISMATCH_PAIR))
        for row in rows:
            self.assertNotEqual(row["stated_arm"], row["strategy"])
            self.assertIn(row["stated_arm"], C.MISMATCH_PAIR)

    def test_metadata_only_is_one_prompt_per_graph_not_per_arm(self):
        rows = [r for r in self.records if r["condition"] == "metadata_only"]
        self.assertEqual(len(rows), self.primary.instance_id.nunique())

    def test_historical_bridge_is_walks_only_and_opt_in(self):
        self.assertFalse([r for r in self.records
                          if r["condition"] == "disclosed_historical"])
        with_bridge = build(self.primary, self.primary, toy_config(), [],
                            run_historical=True)
        arms = {r["strategy"] for r in with_bridge
                if r["condition"] == "disclosed_historical"}
        self.assertEqual(arms, set(C.WALK_ARMS))

    def test_replication_uses_extra_slots_only(self):
        cases = toy_cases(3, 3)
        records = build(cases[cases.seed_slot == 0], cases, toy_config(),
                        ["g0"], run_historical=False)
        rep = [r for r in records if r["subset"] == "seed_replication"]
        self.assertTrue(rep)
        self.assertTrue(all(r["seed_slot"] > 0 for r in rep))
        self.assertTrue(all(r["instance_id"] == "g0" for r in rep))

    def test_every_prompt_records_a_hash_of_its_own_text(self):
        import hashlib
        for record in self.records:
            self.assertEqual(
                record["prompt_sha256"],
                hashlib.sha256(record["prompt"].encode()).hexdigest())


class FrozenPromptFileTest(unittest.TestCase):
    """Guards the generated file itself, when it is present."""

    PATH = Path("results/final_run_g2/prompts.jsonl")

    def setUp(self):
        if not self.PATH.exists():
            self.skipTest("frozen prompt file not generated in this checkout")
        self.records = [json.loads(l) for l in
                        self.PATH.read_text().splitlines() if l.strip()]

    def test_counts_match_the_g2_freeze(self):
        counts = {}
        for record in self.records:
            counts[record["subset"]] = counts.get(record["subset"], 0) + 1
        self.assertEqual(counts["factorial"], 640)
        self.assertEqual(counts["mismatched"], 64)
        self.assertEqual(counts["metadata_only"], 32)
        self.assertEqual(counts["seed_replication"], 240)
        self.assertEqual(len(self.records), 1136)

    def test_task_block_is_constant_across_the_whole_file(self):
        tails = {r["prompt"][r["prompt"].index("TASK\n"):]
                 for r in self.records}
        self.assertEqual(len(tails), 1)


if __name__ == "__main__":
    unittest.main()
