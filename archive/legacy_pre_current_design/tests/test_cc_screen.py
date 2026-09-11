"""Guards of the Claude Code screening runner.

These decide whether a run keeps spending money, so they are tested against
the exact shapes the CLI emitted during the run that exhausted the plan on
2026-07-28. The first version matched prose only and missed all of them.
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/cc_screen"))

from run_cc_screen import attempts_by_id, limit_info, wait_seconds  # noqa: E402

# verbatim shapes from logs/notools/1c6e0c752a89.jsonl, trimmed to the fields
# the detector looks at
RATE_LIMIT_EVENT = json.dumps({
    "type": "rate_limit_event",
    "rate_limit_info": {"status": "rejected", "resetsAt": 1785258600,
                        "rateLimitType": "five_hour",
                        "overageStatus": "rejected"},
})
REFUSAL_RESULT = json.dumps({
    "type": "result", "subtype": "success", "is_error": True,
    "api_error_status": 429, "terminal_reason": "api_error",
    "total_cost_usd": 0,
    "result": "You've hit your monthly spend limit · raise it at "
              "claude.ai/settings/usage?from=cc_cli_limit_message",
})
GOOD_RESULT = json.dumps({
    "type": "result", "subtype": "success", "is_error": False,
    "total_cost_usd": 0.146287, "num_turns": 1,
    "result": "rho_k2 is 0.42 based on 429 observed pairs",
})


class LimitDetection(unittest.TestCase):
    def test_rate_limit_event_gives_reason_and_reset(self):
        reason, resets = limit_info(RATE_LIMIT_EVENT)
        self.assertIsNotNone(reason)
        self.assertEqual(resets, 1785258600)

    def test_429_result_object_is_detected(self):
        reason, _ = limit_info(REFUSAL_RESULT)
        self.assertIsNotNone(reason)

    def test_meta_carries_the_status_when_the_stream_is_lost(self):
        reason, _ = limit_info("", {"api_error_status": 429})
        self.assertIsNotNone(reason)

    def test_full_stream_is_detected(self):
        stream = "\n".join([RATE_LIMIT_EVENT, REFUSAL_RESULT])
        reason, resets = limit_info(stream)
        self.assertIsNotNone(reason)
        self.assertEqual(resets, 1785258600)

    def test_prose_fallback_for_wording_the_json_misses(self):
        reason, _ = limit_info("", {}, "You've hit your monthly spend limit")
        self.assertIsNotNone(reason)

    def test_successful_run_is_not_flagged(self):
        self.assertEqual(limit_info(GOOD_RESULT), (None, None))

    def test_answer_text_about_429_pairs_is_not_flagged(self):
        # prompt data is full of counts; a bare "429" must never trip the guard
        reason, _ = limit_info("", {}, "429 observed pairs, rate 0.42 per step")
        self.assertIsNone(reason)


class WaitSeconds(unittest.TestCase):
    def test_future_reset_is_honoured(self):
        self.assertAlmostEqual(wait_seconds(time.time() + 600), 660, delta=5)

    def test_past_reset_does_not_spin(self):
        self.assertGreaterEqual(wait_seconds(time.time() - 10_000), 60)

    def test_missing_reset_falls_back(self):
        self.assertEqual(wait_seconds(None), 1800.0)

    def test_absurd_reset_is_capped(self):
        self.assertLessEqual(wait_seconds(time.time() + 10 ** 7), 6 * 3600)


class AttemptCounting(unittest.TestCase):
    def test_counts_every_record_not_just_failures(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as fh:
            for pid in ("aaa", "aaa", "bbb"):
                fh.write(json.dumps({"prompt_id": pid}) + "\n")
            fh.write("not json\n")  # a torn line must not abort the count
            path = fh.name
        self.assertEqual(attempts_by_id(path), {"aaa": 2, "bbb": 1})
        Path(path).unlink()

    def test_missing_file_is_empty(self):
        self.assertEqual(attempts_by_id("/nonexistent/answers.jsonl"), {})

    def test_plan_limit_refusal_does_not_burn_attempt(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as fh:
            fh.write(json.dumps({"prompt_id": "aaa", "limit_refusal": True,
                                 "finish_reason": "error: plan limit"}) + "\n")
            fh.write(json.dumps({"prompt_id": "aaa",
                                 "finish_reason": "error: timeout"}) + "\n")
            path = fh.name
        self.assertEqual(attempts_by_id(path), {"aaa": 1})
        Path(path).unlink()


class TimeoutPartial(unittest.TestCase):
    def test_timeout_stdout_is_bytes_even_in_text_mode(self):
        """Guards the assumption the partial-capture fix depends on.

        text=True does not decode TimeoutExpired.stdout, so an
        isinstance(x, str) filter throws the partial stream away.
        """
        script = ("import sys, time\n"
                  "print('PARTIAL'); sys.stdout.flush(); time.sleep(30)\n")
        with self.assertRaises(subprocess.TimeoutExpired) as cm:
            subprocess.run([sys.executable, "-c", script],
                           capture_output=True, text=True, timeout=2)
        out = cm.exception.stdout
        self.assertNotIsInstance(out, str)
        self.assertIn("PARTIAL", out.decode("utf-8", "replace"))


if __name__ == "__main__":
    unittest.main()
