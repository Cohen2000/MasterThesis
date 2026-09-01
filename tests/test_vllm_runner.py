import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from run_llm_v2 import PRED_KEYS, is_complete_record  # noqa: E402
from run_llm_vllm_g3 import (  # noqa: E402
    build_record,
    completed_ids,
    load_prompts,
)

GOOD_JSON = json.dumps({k: 0.5 for k in PRED_KEYS})


def fake_output(text, finish_reason="stop", n_out=120, n_in=900):
    """Shape of a vllm RequestOutput, only the fields the runner reads."""
    return SimpleNamespace(
        prompt_token_ids=list(range(n_in)),
        outputs=[SimpleNamespace(text=text, finish_reason=finish_reason,
                                 token_ids=list(range(n_out)))])


def fake_args(max_new_tokens=65536):
    return SimpleNamespace(model="Qwen/Qwen3.6-27B", thinking="on",
                           temperature=1.0, top_p=0.95, top_k=20,
                           max_new_tokens=max_new_tokens)


def fake_row(prompt_id="p1"):
    return {"prompt_id": prompt_id, "case_id": "c1", "condition": "mechanism",
            "input_kind": "mask", "strategy": "time_agnostic_t",
            "subset": "factorial", "stated_arm": "time_agnostic_t",
            "seed_slot": 0, "gen_seed": 12345, "rep": 0,
            "prompt_sha256": "abc", "prompt": "text"}


class RecordSchemaTest(unittest.TestCase):
    """The evaluation reads these records through the frozen helpers, so the
    schema has to satisfy them rather than merely look similar."""

    def test_a_good_answer_counts_as_complete(self):
        rec = build_record(fake_row(), fake_output("reasoning\n" + GOOD_JSON),
                           fake_args(), "0.28.0", 3.2)
        self.assertTrue(is_complete_record(rec))

    def test_a_truncated_answer_is_incomplete_and_therefore_retryable(self):
        rec = build_record(fake_row(),
                           fake_output("reasoning without json",
                                       finish_reason="length"),
                           fake_args(), "0.28.0", 3.2)
        self.assertEqual(rec["finish_reason"], "length")
        self.assertFalse(is_complete_record(rec))

    def test_a_full_length_answer_that_still_parsed_is_not_resurrected(self):
        # finish_reason length is disqualifying on its own, by the frozen rule.
        rec = build_record(fake_row(), fake_output(GOOD_JSON, "length"),
                           fake_args(), "0.28.0", 3.2)
        self.assertFalse(is_complete_record(rec))

    def test_a_missing_key_is_incomplete(self):
        partial = json.dumps({k: 0.5 for k in PRED_KEYS[:-1]})
        rec = build_record(fake_row(), fake_output(partial), fake_args(),
                           "0.28.0", 3.2)
        self.assertFalse(is_complete_record(rec))

    def test_out_of_range_values_are_recorded_not_repaired(self):
        wild = json.dumps({k: (5.0 if k == "rho_k2" else 0.5)
                           for k in PRED_KEYS})
        rec = build_record(fake_row(), fake_output(wild), fake_args(),
                           "0.28.0", 3.2)
        # structurally complete; validity is an outcome the evaluation scores
        self.assertTrue(is_complete_record(rec))
        self.assertIn('"rho_k2":5.0', rec["answer"].replace(" ", ""))

    def test_the_record_carries_the_seed_and_the_stack(self):
        rec = build_record(fake_row(), fake_output(GOOD_JSON), fake_args(),
                           "0.28.0", 3.2)
        self.assertEqual(rec["seed"], 12345)
        self.assertEqual(rec["backend"], "vllm")
        self.assertIn("0.28.0", rec["engine"])
        self.assertEqual(rec["prompt_sha256"], "abc")

    def test_usage_is_reported_the_way_the_evaluation_reads_it(self):
        rec = build_record(fake_row(), fake_output(GOOD_JSON, n_out=77,
                                                   n_in=900),
                           fake_args(), "0.28.0", 3.2)
        self.assertEqual(rec["usage"]["completion_tokens"], 77)
        self.assertEqual(rec["total_tokens"], 977)


class ResumeTest(unittest.TestCase):
    def _file(self, records):
        tmp = Path(tempfile.mkdtemp()) / "a.jsonl"
        tmp.write_text("".join(json.dumps(r) + "\n" for r in records))
        return tmp

    def test_only_complete_records_are_skipped(self):
        good = build_record(fake_row("p1"), fake_output(GOOD_JSON),
                            fake_args(), "0.28.0", 1.0)
        bad = build_record(fake_row("p2"), fake_output("nope", "length"),
                           fake_args(), "0.28.0", 1.0)
        path = self._file([good, bad])
        self.assertEqual(completed_ids(path), {"p1"})

    def test_a_missing_file_is_not_an_error(self):
        self.assertEqual(completed_ids(Path("/nonexistent/a.jsonl")), set())

    def test_a_corrupt_line_does_not_abort_resume(self):
        path = self._file([build_record(fake_row("p1"),
                                        fake_output(GOOD_JSON), fake_args(),
                                        "0.28.0", 1.0)])
        path.write_text(path.read_text() + "{not json\n")
        self.assertEqual(completed_ids(path), {"p1"})


class ShardingTest(unittest.TestCase):
    def _prompts(self, n):
        tmp = Path(tempfile.mkdtemp()) / "p.jsonl"
        tmp.write_text("".join(
            json.dumps(fake_row(f"p{i:03d}")) + "\n" for i in range(n)))
        return str(tmp)

    def test_shards_partition_the_set_exactly_once(self):
        path = self._prompts(50)
        seen = []
        for i in range(4):
            seen += [r["prompt_id"] for r in load_prompts(path, i, 4, None)]
        self.assertEqual(len(seen), 50)
        self.assertEqual(len(set(seen)), 50)

    def test_sharding_is_stable_under_input_order(self):
        path = self._prompts(20)
        a = [r["prompt_id"] for r in load_prompts(path, 1, 4, None)]
        lines = Path(path).read_text().splitlines()
        Path(path).write_text("\n".join(reversed(lines)) + "\n")
        b = [r["prompt_id"] for r in load_prompts(path, 1, 4, None)]
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
