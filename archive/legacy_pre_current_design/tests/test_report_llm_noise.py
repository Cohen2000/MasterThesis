"""Bucketing rules for noise-probe answers.

The probe asks every arm the identical prompt_ids, so any label collision is
total rather than occasional: one configuration silently replaces another
instead of merely mixing with it.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_llm_noise import load_by_model, model_key


def record(model, thinking, prompt_id, answer="{}"):
    return {"model": model, "thinking": thinking,
            "prompt_id": prompt_id, "answer": answer,
            "finish_reason": "stop"}


class ModelKeyTest(unittest.TestCase):
    def test_thinking_on_gets_its_own_bucket(self):
        off = model_key(record("Qwen/Qwen3.6-27B", "off", "p1"), "x.jsonl")
        on = model_key(record("Qwen/Qwen3.6-27B", "on", "p1"), "x.jsonl")
        self.assertNotEqual(off, on)

    def test_non_reasoning_labels_are_unchanged(self):
        for flag in ("off", "none", None, ""):
            self.assertEqual(
                model_key(record("gemini-3.1-flash-lite", flag, "p1"), "x"),
                "gemini-3.1-flash-lite")

    def test_missing_model_falls_back_to_filename(self):
        self.assertEqual(model_key({"thinking": "off"}, "answers_probe.jsonl"),
                         "answers_probe")


class LoadByModelTest(unittest.TestCase):
    def test_two_thinking_modes_do_not_overwrite_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for mode, text in (("off", '{"a": 1}'), ("on", '{"a": 2}')):
                path = root / f"answers_qwen_{mode}.jsonl"
                with open(path, "w") as fh:
                    for pid in ("p1", "p2"):
                        fh.write(json.dumps(
                            record("Qwen/Qwen3.6-27B", mode, pid, text)) + "\n")
            buckets = load_by_model(["answers_qwen_*.jsonl"], root=root)
            self.assertEqual(len(buckets), 2)
            for recs in buckets.values():
                self.assertEqual(sorted(recs), ["p1", "p2"])
            answers = {k: v["p1"]["answer"] for k, v in buckets.items()}
            self.assertEqual(sorted(answers.values()), ['{"a": 1}', '{"a": 2}'])


if __name__ == "__main__":
    unittest.main()
