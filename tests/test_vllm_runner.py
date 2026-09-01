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
    templated_token_ids,
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


class FakeTokenizer:
    """Records how apply_chat_template was called, and refuses a second BOS."""

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True, **kwargs):
        self.calls.append({"msgs": msgs, "tokenize": tokenize,
                           "add_generation_prompt": add_generation_prompt,
                           "kwargs": kwargs})
        thinking = kwargs.get("enable_thinking")
        tag = "" if thinking is None else f"<think:{thinking}>"
        return f"<BOS><|user|>{msgs[0]['content']}{tag}<|assistant|>"

    def encode(self, text, add_special_tokens=True):
        if add_special_tokens:
            raise AssertionError(
                "the chat template already carries the special tokens; "
                "encoding with add_special_tokens=True prepends a second BOS")
        return [len(text)]


class ChatTemplateTest(unittest.TestCase):
    """The bug this exists to prevent: passing the raw prompt string to vLLM.

    The HF path wraps every prompt with apply_chat_template and the Qwen
    enable_thinking switch. A vLLM runner that skips it makes the model
    continue the text instead of answering, and silently turns --thinking off
    into no switch at all. Measured on a first attempt: 171 of 256 non-thinking
    generations ran into the token cap and completeness fell to 33% against a
    historical 53%.
    """

    def test_the_prompt_is_wrapped_as_a_user_message(self):
        tok = FakeTokenizer()
        templated_token_ids(tok, "PROMPT BODY", "off")
        self.assertEqual(len(tok.calls), 1)
        call = tok.calls[0]
        self.assertEqual(call["msgs"],
                         [{"role": "user", "content": "PROMPT BODY"}])
        self.assertTrue(call["add_generation_prompt"])
        self.assertFalse(call["tokenize"])

    def test_thinking_off_reaches_the_template(self):
        tok = FakeTokenizer()
        templated_token_ids(tok, "p", "off")
        self.assertIs(tok.calls[0]["kwargs"]["enable_thinking"], False)

    def test_thinking_on_reaches_the_template(self):
        tok = FakeTokenizer()
        templated_token_ids(tok, "p", "on")
        self.assertIs(tok.calls[0]["kwargs"]["enable_thinking"], True)

    def test_thinking_none_omits_the_switch(self):
        tok = FakeTokenizer()
        templated_token_ids(tok, "p", "none")
        self.assertNotIn("enable_thinking", tok.calls[0]["kwargs"])

    def test_special_tokens_are_not_added_twice(self):
        # FakeTokenizer.encode raises if add_special_tokens is left at True.
        tok = FakeTokenizer()
        self.assertEqual(templated_token_ids(tok, "p", "off"), [len(
            "<BOS><|user|>p<think:False><|assistant|>")])

    def test_the_two_thinking_modes_produce_different_encodings(self):
        tok = FakeTokenizer()
        on = templated_token_ids(tok, "same prompt", "on")
        off = templated_token_ids(tok, "same prompt", "off")
        self.assertNotEqual(
            on, off,
            "thinking on and off encode identically -- the switch is inert")


if __name__ == "__main__":
    unittest.main()
