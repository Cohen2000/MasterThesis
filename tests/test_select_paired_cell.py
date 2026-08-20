import json
import os
import tempfile
import unittest

import select_paired_cell as selector


COMPLETE = {
    "rho_k2": 0.4, "rho_k3": 0.3, "rho_k4": 0.2, "rho_k5": 0.1,
    "mean_occupancy": 0.5, "C_one_step": 0.6,
    "lifetime_mean_over_T": 0.2, "lo90": 0.3, "hi90": 0.5,
}
STRATEGIES = ["time_agnostic_t", "time_respecting", "recent_history_k20"]
BANDS = ["low", "high"]


class TestSelectPairedCell(unittest.TestCase):
    def write(self, rows, suffix=".jsonl"):
        fd, path = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def write_cases(self, case_ids):
        fd, path = tempfile.mkstemp(suffix=".csv")
        with os.fdopen(fd, "w") as fh:
            fh.write("case_id,strategy,coverage_band\n")
            for i, case_id in enumerate(case_ids):
                fh.write(f"{case_id},{STRATEGIES[i % 3]},{BANDS[i % 2]}\n")
        self.addCleanup(os.unlink, path)
        return path

    def build(self, n=12, answered=None):
        """n cases, both cells present, `answered` complete in the baseline."""
        case_ids = [f"c{i:02d}" for i in range(n)]
        answered = case_ids if answered is None else answered
        prompts = self.write(
            [{"prompt_id": f"p_{c}", "case_id": c, "condition": "disclosed",
              "input_kind": "mask"} for c in case_ids]
            + [{"prompt_id": f"x_{c}", "case_id": c,
                "condition": "disclosed_examples", "input_kind": "mask"}
               for c in case_ids])
        answers = self.write(
            [{"prompt_id": f"p_{c}", "case_id": c, "condition": "disclosed",
              "input_kind": "mask", "answer": json.dumps(COMPLETE),
              "finish_reason": "stop"} for c in answered])
        return prompts, self.write_cases(case_ids), answers, case_ids

    def run_select(self, prompts, cases, answers, limit=0, prefer=None):
        ids, _, _ = selector.select(
            prompts, cases, [answers], "disclosed_examples", "mask",
            "disclosed", "mask", limit, prefer)
        return ids

    def test_pairs_to_completely_answered_cases_only(self):
        prompts, cases, answers, _ = self.build(n=6, answered=["c00", "c03"])
        self.assertEqual(sorted(self.run_select(prompts, cases, answers)),
                         ["x_c00", "x_c03"])

    def test_incomplete_baseline_record_does_not_enter_the_cell(self):
        prompts, cases, _, _ = self.build(n=4)
        answers = self.write([
            {"prompt_id": "p_c00", "case_id": "c00", "condition": "disclosed",
             "input_kind": "mask", "answer": json.dumps(COMPLETE),
             "finish_reason": "stop"},
            # length-truncated: retryable, so it is not an answered case
            {"prompt_id": "p_c01", "case_id": "c01", "condition": "disclosed",
             "input_kind": "mask", "answer": "", "finish_reason": "length"},
        ])
        self.assertEqual(self.run_select(prompts, cases, answers), ["x_c00"])

    def test_limit_is_deterministic_and_stratified(self):
        prompts, cases, answers, _ = self.build(n=12)
        first = self.run_select(prompts, cases, answers, limit=6)
        self.assertEqual(first, self.run_select(prompts, cases, answers, limit=6))
        self.assertEqual(len(first), 6)
        strategies = {STRATEGIES[int(i[3:]) % 3] for i in first}
        self.assertEqual(strategies, set(STRATEGIES))

    def test_smaller_limit_is_a_prefix_so_subsets_nest(self):
        prompts, cases, answers, _ = self.build(n=12)
        self.assertEqual(self.run_select(prompts, cases, answers, limit=4),
                         self.run_select(prompts, cases, answers, limit=8)[:4])

    def test_prefer_from_puts_the_other_runs_cases_first(self):
        prompts, cases, answers, _ = self.build(n=12)
        other = self.write([
            {"prompt_id": f"p_{c}", "case_id": c, "condition": "disclosed",
             "input_kind": "mask", "answer": json.dumps(COMPLETE),
             "finish_reason": "stop"} for c in ["c09", "c10", "c11"]])
        picked = self.run_select(prompts, cases, answers, limit=3,
                                 prefer=[other])
        self.assertEqual(sorted(picked), ["x_c09", "x_c10", "x_c11"])

    def test_unpairable_case_is_an_error_not_a_silent_gap(self):
        prompts = self.write([
            {"prompt_id": "p_c00", "case_id": "c00", "condition": "disclosed",
             "input_kind": "mask"},
        ])
        cases = self.write_cases(["c00"])
        answers = self.write([
            {"prompt_id": "p_c00", "case_id": "c00", "condition": "disclosed",
             "input_kind": "mask", "answer": json.dumps(COMPLETE),
             "finish_reason": "stop"}])
        with self.assertRaises(SystemExit):
            self.run_select(prompts, cases, answers)

    def test_empty_baseline_is_an_error(self):
        prompts, cases, _, _ = self.build(n=4)
        with self.assertRaises(SystemExit):
            self.run_select(prompts, cases, self.write([]))


if __name__ == "__main__":
    unittest.main()
