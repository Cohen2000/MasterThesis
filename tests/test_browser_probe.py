import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from browser_probe import REQUIRED_KEYS, _api_answered, parse_pasted  # noqa: E402

IDS = {"0031d7cb776a", "0fe3d8d44b83"}


def write(text):
    tmp = TemporaryDirectory()
    path = Path(tmp.name) / "answers_pasted.txt"
    path.write_text(text)
    return tmp, path


class ParsePasted(unittest.TestCase):
    def test_a_markdown_heading_in_a_reply_does_not_start_a_section(self):
        tmp, path = write(
            "### 0031d7cb776a\n\n### Result\nprose\n{\"rho_k2\": 0.4}\n")
        with tmp:
            got = parse_pasted(path, IDS)
        self.assertEqual(list(got), ["0031d7cb776a"])
        self.assertIn("### Result", got["0031d7cb776a"])

    def test_empty_sections_are_dropped_not_scored(self):
        tmp, path = write("### 0031d7cb776a\n\n\n### 0fe3d8d44b83\n\nanswer\n")
        with tmp:
            got = parse_pasted(path, IDS)
        self.assertEqual(list(got), ["0fe3d8d44b83"])

    def test_text_before_the_first_id_is_ignored(self):
        tmp, path = write("# notes to self\n### 0fe3d8d44b83\n\nanswer\n")
        with tmp:
            got = parse_pasted(path, IDS)
        self.assertEqual(got, {"0fe3d8d44b83": "answer"})


class ApiAnchors(unittest.TestCase):
    def test_failed_attempts_are_not_counted_as_anchors(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            complete = {
                "prompt_id": "0031d7cb776a",
                "finish_reason": "stop",
                "required_keys": REQUIRED_KEYS,
                "answer": json.dumps({key: 0.0 for key in REQUIRED_KEYS}),
            }
            failed = {
                "prompt_id": "0fe3d8d44b83",
                "finish_reason": None,
                "required_keys": REQUIRED_KEYS,
                "answer": None,
            }
            path.write_text(json.dumps(complete) + "\n" +
                            json.dumps(failed) + "\n")
            self.assertEqual(_api_answered(str(path)), {"0031d7cb776a"})


if __name__ == "__main__":
    unittest.main()
