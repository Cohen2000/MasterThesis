#!/usr/bin/env python3
"""Invariants of the frozen scoring rule in llm_eval_frozen.

The properties pinned here are the ones a later refactor could quietly break:
invalid predictions cost full loss instead of being clipped, complete-case and
penalized metrics come apart exactly where a run failed to answer, a prompt
that was never attempted is not a failure, and raw predictions are never
repaired.
"""

import json
import tempfile
import unittest
from pathlib import Path

from llm_eval_frozen import (PROFILE_TRUTH, cell_index, load_answers,
                             readoff_rho2, score, spearman, valid_unit)


def answer(**profile):
    """A record whose final JSON carries the given prediction values."""
    payload = {"rho_k2": 0.5, "rho_k3": 0.4, "rho_k4": 0.3, "rho_k5": 0.2,
               "mean_occupancy": 0.4, "C_one_step": 0.5,
               "lifetime_mean_over_T": 0.3, "lo90": 0.4, "hi90": 0.6}
    payload.update(profile)
    return {"prompt_id": "p", "finish_reason": "stop",
            "answer": "reasoning text\n" + json.dumps(payload)}


def truth_case(case_id="c", **values):
    row = {"case_id": case_id, "rho_W5_k2": "0.5", "rho_W5_k3": "0.4",
           "rho_W5_k4": "0.3", "rho_W5_k5": "0.2", "C_one_step": "0.5",
           "strategy": "time_agnostic_t"}
    row.update({k: str(v) for k, v in values.items()})
    return row


class ValidityTest(unittest.TestCase):
    def test_rejects_values_outside_the_unit_interval(self):
        for bad in (-0.01, 1.01, float("nan"), float("inf"), None, "0.5",
                    True, False):
            self.assertFalse(valid_unit(bad), bad)

    def test_accepts_the_closed_unit_interval(self):
        for good in (0, 1, 0.0, 1.0, 0.5):
            self.assertTrue(valid_unit(good), good)


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.cases = {"c": truth_case()}
        self.mapping = {"c": "p"}

    def test_exact_prediction_scores_zero(self):
        m = score({"p": answer()}, self.mapping, ["c"], self.cases)
        self.assertAlmostEqual(m["profile_mae_penalized"], 0.0)
        self.assertAlmostEqual(m["profile_mae_complete"], 0.0)
        self.assertEqual(m["validity"], 1.0)

    def test_out_of_range_component_costs_full_loss_and_is_not_clipped(self):
        # 1.5 must not become 1.0; the component is invalid and costs 1.0.
        m = score({"p": answer(rho_k2=1.5)}, self.mapping, ["c"], self.cases)
        self.assertAlmostEqual(m["profile_mae_penalized"], 0.25)
        self.assertEqual(m["validity"], 0.0)
        self.assertNotEqual(m["profile_mae_complete"],
                            m["profile_mae_complete"])  # NaN: no valid case

    def test_missing_component_costs_full_loss(self):
        record = answer()
        payload = json.loads(record["answer"].split("\n", 1)[1])
        payload.pop("rho_k5")
        record["answer"] = json.dumps(payload)
        m = score({"p": record}, self.mapping, ["c"], self.cases)
        self.assertAlmostEqual(m["profile_mae_penalized"], 0.25)

    def test_unanswered_prompt_is_missing_rather_than_wrong(self):
        # No record at all for the only case: nothing to score.
        self.assertIsNone(score({}, self.mapping, ["c"], self.cases))

        # A prompt exists for `absent` but the run never attempted it. That is
        # reported separately and must not be penalized as a wrong answer.
        mapping = {**self.mapping, "absent": "p_absent"}
        cases = {**self.cases, "absent": truth_case("absent")}
        m = score({"p": answer()}, mapping, ["c", "absent"], cases)
        self.assertEqual(m["n"], 1)
        self.assertEqual(m["missing"], 1)
        self.assertAlmostEqual(m["profile_mae_penalized"], 0.0)

    def test_case_without_a_prompt_in_this_cell_is_not_counted(self):
        # Ablation cells cover fewer cases than `mask`; those absences are a
        # property of the design, not of the run.
        m = score({"p": answer()}, self.mapping, ["c", "not_in_cell"],
                  {**self.cases, "not_in_cell": truth_case("not_in_cell")})
        self.assertEqual(m["n"], 1)
        self.assertEqual(m["missing"], 0)

    def test_truncated_record_scores_as_failure_not_as_a_gap(self):
        truncated = {"prompt_id": "p", "finish_reason": "length",
                     "answer": '{"rho_k2": 0.5'}
        m = score({"p": truncated}, self.mapping, ["c"], self.cases)
        self.assertEqual(m["n"], 1)
        self.assertEqual(m["response_rate"], 0.0)
        self.assertAlmostEqual(m["profile_mae_penalized"], 1.0)

    def test_non_monotone_profile_is_recorded_not_repaired(self):
        # rho_k3 above rho_k2 violates the requested ordering. It stays as
        # submitted and is scored against the truth, only flagged.
        m = score({"p": answer(rho_k2=0.2, rho_k3=0.9)}, self.mapping, ["c"],
                  self.cases)
        self.assertEqual(m["violation_rate"], 1.0)
        self.assertEqual(m["validity"], 1.0)
        self.assertAlmostEqual(m["profile_mae_penalized"],
                               (abs(0.2 - 0.5) + abs(0.9 - 0.4)) / 4)

    def test_bias_keeps_its_sign(self):
        m = score({"p": answer(rho_k2=0.3)}, self.mapping, ["c"], self.cases)
        self.assertAlmostEqual(m["rho2_bias"], -0.2)


class LoadAnswersTest(unittest.TestCase):
    def test_prefers_a_complete_record_over_a_later_truncated_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            with open(path, "w") as fh:
                fh.write(json.dumps(answer()) + "\n")
                fh.write(json.dumps({"prompt_id": "p",
                                     "finish_reason": "length",
                                     "answer": "cut off"}) + "\n")
            merged = load_answers(["answers.jsonl"], root=tmp)
            self.assertEqual(merged["p"]["finish_reason"], "stop")

    def test_merges_shards_and_skips_smoke_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for shard, pid in enumerate(["a", "b"]):
                record = dict(answer(), prompt_id=pid)
                with open(Path(tmp) / f"answers.shard{shard}.jsonl", "w") as fh:
                    fh.write(json.dumps(record) + "\n")
            with open(Path(tmp) / "answers_smoke.jsonl", "w") as fh:
                fh.write(json.dumps(dict(answer(), prompt_id="s")) + "\n")
            merged = load_answers(["answers*.jsonl"], root=tmp)
            self.assertEqual(sorted(merged), ["a", "b"])


class HelpersTest(unittest.TestCase):
    def test_spearman_endpoints(self):
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [1, 2, 3, 4]), 1.0)
        self.assertAlmostEqual(spearman([1, 2, 3, 4], [4, 3, 2, 1]), -1.0)

    def test_readoff_counts_dyads_seen_in_at_least_two_windows(self):
        # hex masks: 01 = one window, 03 and 11 = two windows.
        row = {"input__nmask_exact_json":
               json.dumps({"1,01": 6, "1,03": 3, "2,11": 1})}
        self.assertAlmostEqual(readoff_rho2(row), 4 / 10)

    def test_cell_index_groups_cases_by_condition_and_input(self):
        prompts = {
            "p1": {"prompt_id": "p1", "case_id": "c", "condition": "disclosed",
                   "input_kind": "mask"},
            "p2": {"prompt_id": "p2", "case_id": "c", "condition": "disclosed",
                   "input_kind": "nw"},
        }
        cells = cell_index(prompts)
        self.assertEqual(cells[("disclosed", "mask")], {"c": "p1"})
        self.assertEqual(cells[("disclosed", "nw")], {"c": "p2"})


class TruthColumnTest(unittest.TestCase):
    def test_profile_truth_columns_match_the_frozen_target(self):
        self.assertEqual(PROFILE_TRUTH,
                         ["rho_W5_k2", "rho_W5_k3", "rho_W5_k4", "rho_W5_k5"])


if __name__ == "__main__":
    unittest.main()
