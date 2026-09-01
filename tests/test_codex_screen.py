"""Guards of the Codex screening runner.

Provenance of the fixtures matters here, so it is stated per fixture:

VERIFIED    shapes copied from a real `codex exec --json` stream (codex-cli
            0.146.0, 2026-07-30, unauthenticated probe): thread.started,
            turn.started, error, item.completed with an error item, and
            turn.failed -- the whole sequence exited with returncode 0.
CONSTRUCTED shapes for a successful turn and for an exhausted rate-limit
            window could not be observed without credentials. The parser is
            therefore written to tolerate variants, and these tests pin that
            tolerance rather than one exact schema. Re-check them against the
            first real log in ~/Dokumente/codex_screen/logs/.
"""
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts/codex_screen"))

from run_codex_screen import (  # noqa: E402
    ALWAYS_OFF, NO_TOOL_FEATURES, attempts_by_id, build_cmd, done_ids,
    estimate_usd, iter_events, limit_info, parse_events, wait_seconds,
)

# VERIFIED -- the 401 probe, trimmed. Note the interleaved plain-text ERROR
# line: the CLI writes those alongside the JSONL and they must not break
# parsing.
FAILED_STREAM = "\n".join([
    '{"type":"thread.started","thread_id":"019fb105-c39e-7e63-a482-53829dd12f6d"}',
    '{"type":"turn.started"}',
    "2026-07-30T03:16:17.993300Z ERROR codex_api::endpoint::responses_websocket: "
    "failed to connect to websocket: HTTP error: 401 Unauthorized",
    '{"type":"error","message":"Reconnecting... 2/5 (unexpected status 401 '
    'Unauthorized: Missing bearer or basic authentication in header)"}',
    '{"type":"item.completed","item":{"id":"item_0","type":"error",'
    '"message":"Falling back from WebSockets to HTTPS transport."}}',
    '{"type":"turn.failed","error":{"message":"unexpected status 401 '
    'Unauthorized: Missing bearer or basic authentication in header"}}',
])

# CONSTRUCTED -- a bare successful turn.
OK_STREAM = "\n".join([
    '{"type":"thread.started","thread_id":"t-42"}',
    '{"type":"turn.started"}',
    '{"type":"item.completed","item":{"type":"reasoning","text":"thinking"}}',
    '{"type":"item.completed","item":{"type":"agent_message",'
    '"text":"Final answer: {\\"rho_k2\\": 0.42}"}}',
    '{"type":"turn.completed","usage":{"input_tokens":3000,'
    '"cached_input_tokens":1000,"output_tokens":8000}}',
])

# CONSTRUCTED -- the tools arm, with two kinds of execution event.
TOOL_STREAM = "\n".join([
    '{"type":"thread.started","thread_id":"t-7"}',
    '{"type":"item.completed","item":{"type":"command_execution",'
    '"command":"python3 estimate.py"}}',
    '{"type":"item.completed","item":{"type":"file_change","path":"x.py"}}',
    '{"type":"item.completed","item":{"type":"agent_message","text":"done"}}',
    '{"type":"turn.completed","usage":{"total_tokens":50000}}',
])


class EventParsing(unittest.TestCase):
    def test_plain_text_error_lines_do_not_break_parsing(self):
        kinds = [e.get("type") for e in iter_events(FAILED_STREAM)]
        self.assertEqual(kinds, ["thread.started", "turn.started", "error",
                                 "item.completed", "turn.failed"])

    def test_turn_failed_is_an_error_even_though_the_cli_exits_zero(self):
        """The regression that makes returncode useless for this CLI."""
        text, meta = parse_events(FAILED_STREAM)
        self.assertIsNone(text)
        self.assertIsNotNone(meta["failed"])
        self.assertIn("401", meta["failed"])

    def test_agent_message_is_the_answer(self):
        text, meta = parse_events(OK_STREAM)
        self.assertIn("rho_k2", text)
        self.assertIsNone(meta["failed"])
        self.assertEqual(meta["thread_id"], "t-42")

    def test_tokens_are_summed_without_double_counting_cached_input(self):
        _, meta = parse_events(OK_STREAM)
        self.assertEqual(meta["total_tokens"], 11000)

    def test_legacy_prompt_completion_token_names_are_supported(self):
        stream = json.dumps({"type": "turn.completed", "usage": {
            "prompt_tokens": 120, "completion_tokens": 30}})
        _, meta = parse_events(stream)
        self.assertEqual(meta["total_tokens"], 150)

    def test_reported_total_is_used_as_is(self):
        _, meta = parse_events(TOOL_STREAM)
        self.assertEqual(meta["total_tokens"], 50000)

    def test_reasoning_items_are_not_counted_as_tool_use(self):
        _, meta = parse_events(OK_STREAM)
        self.assertEqual(meta["n_tool_events"], 0)

    def test_tool_events_are_counted_and_named(self):
        _, meta = parse_events(TOOL_STREAM)
        self.assertEqual(meta["n_tool_events"], 2)
        self.assertEqual(meta["tool_item_types"],
                         ["command_execution", "file_change"])

    def test_unknown_item_type_counts_as_tool_use(self):
        """A new tool must show up in the audit, not slip through it."""
        stream = ('{"type":"item.completed","item":'
                  '{"type":"some_future_tool_v9"}}')
        _, meta = parse_events(stream)
        self.assertEqual(meta["n_tool_events"], 1)

    def test_output_last_message_file_wins_over_the_stream(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as fh:
            fh.write("authoritative final message")
            path = fh.name
        text, _ = parse_events(OK_STREAM, path)
        self.assertEqual(text, "authoritative final message")
        Path(path).unlink()

    def test_empty_output_file_falls_back_to_the_stream(self):
        with tempfile.NamedTemporaryFile("w", suffix=".txt",
                                         delete=False) as fh:
            fh.write("   \n")
            path = fh.name
        text, _ = parse_events(OK_STREAM, path)
        self.assertIn("rho_k2", text)
        Path(path).unlink()

    def test_missing_output_file_is_not_an_exception(self):
        text, _ = parse_events(OK_STREAM, "/nonexistent/last_message.txt")
        self.assertIn("rho_k2", text)


class LimitDetection(unittest.TestCase):
    def test_exhausted_window_with_relative_reset(self):
        ev = json.dumps({"type": "token_count", "rate_limits": {
            "primary": {"used_percent": 100.0, "window_minutes": 300,
                        "resets_in_seconds": 600}}})
        reason, resets = limit_info(ev)
        self.assertIsNotNone(reason)
        self.assertAlmostEqual(resets, time.time() + 600, delta=5)

    def test_exhausted_window_with_absolute_reset(self):
        ev = json.dumps({"type": "token_count", "rate_limits": {
            "secondary": {"used_percent": 100, "resets_at": 1785258600}}})
        reason, resets = limit_info(ev)
        self.assertIsNotNone(reason)
        self.assertEqual(resets, 1785258600)

    def test_window_below_full_is_not_a_limit(self):
        ev = json.dumps({"type": "token_count", "rate_limits": {
            "primary": {"used_percent": 99.4, "resets_in_seconds": 600}}})
        self.assertEqual(limit_info(ev), (None, None))

    def test_429_in_an_error_event_is_detected(self):
        ev = json.dumps({"type": "turn.failed", "error": {
            "message": "unexpected status 429 Too Many Requests"}})
        reason, _ = limit_info(ev)
        self.assertIsNotNone(reason)

    def test_the_401_probe_is_not_mistaken_for_a_rate_limit(self):
        self.assertEqual(limit_info(FAILED_STREAM), (None, None))

    def test_successful_stream_is_not_flagged(self):
        self.assertEqual(limit_info(OK_STREAM), (None, None))

    def test_prose_fallback_covers_wording_the_events_miss(self):
        reason, _ = limit_info("", {}, "You have hit your usage limit")
        self.assertIsNotNone(reason)

    def test_answer_text_is_never_scanned_for_prose(self):
        """Prompt data is full of counts and rates.

        limit_info only ever sees stderr and the finish_reason, never the
        model's answer -- this pins that a bare number in the answer cannot
        reach the prose fallback through the stream path.
        """
        stream = json.dumps({"type": "item.completed", "item": {
            "type": "agent_message",
            "text": "429 pairs observed, rate limit of the estimator is 0.4"}})
        self.assertEqual(limit_info(stream), (None, None))


class WaitSeconds(unittest.TestCase):
    def test_future_reset_is_honoured(self):
        self.assertAlmostEqual(wait_seconds(time.time() + 600), 660, delta=5)

    def test_past_reset_does_not_spin(self):
        self.assertGreaterEqual(wait_seconds(time.time() - 10_000), 60)

    def test_missing_reset_falls_back(self):
        self.assertEqual(wait_seconds(None), 1800.0)

    def test_absurd_reset_is_capped(self):
        self.assertLessEqual(wait_seconds(time.time() + 10 ** 7), 6 * 3600)


class CostEstimate(unittest.TestCase):
    def test_no_price_means_no_number(self):
        """A guessed price would be a fabricated column in the results."""
        usage = {"input_tokens": 1000, "output_tokens": 2000}
        self.assertIsNone(estimate_usd(usage, 0.0, 0.0))

    def test_cached_input_is_not_charged_twice(self):
        usage = {"input_tokens": 1_000_000, "cached_input_tokens": 1_000_000,
                 "output_tokens": 1_000_000}
        self.assertAlmostEqual(estimate_usd(usage, 1.0, 10.0), 11.0, places=6)

    def test_missing_usage_is_not_an_exception(self):
        self.assertIsNone(estimate_usd(None, 1.0, 10.0))


class CommandShape(unittest.TestCase):
    class Args:
        model = "gpt-5.6-sol"
        effort = "high"
        reasoning_summary = "detailed"
        sandbox = "workspace-write"

    def test_notools_disables_execution(self):
        cmd = build_cmd(self.Args(), "notools")
        for feat in NO_TOOL_FEATURES:
            self.assertIn(feat, cmd, f"{feat} not disabled in notools arm")
        self.assertIn("read-only", cmd)
        self.assertNotIn("workspace-write", cmd)

    def test_tools_arm_keeps_execution(self):
        cmd = build_cmd(self.Args(), "tools")
        self.assertNotIn("shell_tool", cmd)
        self.assertIn("workspace-write", cmd)

    def test_state_and_network_are_off_in_both_arms(self):
        for arm in ("notools", "tools"):
            cmd = build_cmd(self.Args(), arm)
            for feat in ALWAYS_OFF:
                self.assertIn(feat, cmd, f"{feat} enabled in {arm}")
            self.assertIn('web_search="disabled"', cmd)
            self.assertIn("tools.web_search=false", cmd)

    def test_user_config_and_history_cannot_leak_in(self):
        cmd = build_cmd(self.Args(), "tools")
        for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral"):
            self.assertIn(flag, cmd)

    def test_only_config_keys_the_cli_accepts_are_passed(self):
        """`tools.view_image` does not exist in 0.146.0.

        With --strict-config an unknown key aborts every call before the model
        is reached, which is how it was caught -- cheaply, but only because the
        flag is there.
        """
        cmd = build_cmd(self.Args(), "notools")
        self.assertIn("--strict-config", cmd)
        self.assertNotIn("tools.view_image=false", cmd)

    def test_danger_full_access_is_not_reachable(self):
        cmd = build_cmd(self.Args(), "tools")
        self.assertNotIn("danger-full-access", cmd)


class AttemptCounting(unittest.TestCase):
    def test_notools_resume_retries_a_structurally_complete_tool_leak(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                         delete=False) as fh:
            fh.write(json.dumps({
                "prompt_id": "leaked", "finish_reason": "stop",
                "answer": '{"rho_k2": 0.5}', "n_tool_events": 24,
                "required_keys": ["rho_k2"],
            }) + "\n")
            fh.write(json.dumps({
                "prompt_id": "clean", "finish_reason": "stop",
                "answer": '{"rho_k2": 0.5}', "n_tool_events": 0,
                "required_keys": ["rho_k2"],
            }) + "\n")
            path = fh.name
        self.assertEqual(done_ids(path, "notools"), {"clean"})
        self.assertEqual(done_ids(path, "tools"), {"leaked", "clean"})

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


class TimeoutPartial(unittest.TestCase):
    def test_timeout_stdout_is_bytes_even_in_text_mode(self):
        """Guards the assumption the partial-capture path depends on.

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


class WaitForResetBudgetTest(unittest.TestCase):
    """A long run must survive many plan windows, and only many *in a row*
    should stop it.

    The counter was previously cumulative with a default of 8, so a run long
    enough to sit out nine limits stopped even though every one of them had
    reset normally. The G3 Codex pass is ~29 h of wall clock against a 5 h
    window, so that ceiling would have been hit routinely.
    """

    def _source(self):
        path = (Path(__file__).resolve().parents[1] / "scripts" /
                "codex_screen" / "run_codex_screen.py")
        return path.read_text()

    def test_the_counter_resets_after_a_usable_answer(self):
        source = self._source()
        anchor = source.index("usable = usable_record(rec, args.arm)")
        window = source[anchor:anchor + 400]
        self.assertIn("waits = 0", window,
                      "the wait counter is not reset on success, so it is "
                      "still a lifetime budget")

    def test_the_default_allows_a_long_run(self):
        source = self._source()
        marker = '"--max-waits", type=int, default='
        value = int(source.split(marker)[1].split(",")[0].strip())
        self.assertGreaterEqual(value, 32)

    def test_a_five_hour_window_fits_under_the_sleep_cap(self):
        # wait_seconds clamps the sleep; a 5 h reset must not be truncated
        # into an early retry that would burn a wait for nothing.
        resets_at = time.time() + 5 * 3600
        delay = wait_seconds(resets_at)
        self.assertGreater(delay, 5 * 3600)
        self.assertLess(delay, 6 * 3600 + 1)

    def test_an_unparseable_reset_still_retries_rather_than_spinning(self):
        delay = wait_seconds(None)
        self.assertGreaterEqual(delay, 60)
        self.assertLessEqual(delay, 6 * 3600)

    def test_a_stale_reset_timestamp_does_not_busy_loop(self):
        delay = wait_seconds(time.time() - 10_000)
        self.assertGreaterEqual(delay, 60)


if __name__ == "__main__":
    unittest.main()
