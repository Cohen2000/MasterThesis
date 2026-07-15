#!/usr/bin/env python3
"""Tests for the resume validation in run_llm_v2.py.

Run: PYTHONPATH=src python -m unittest tests.test_run_llm_v2 -v
"""

import io
import json
import os
import tempfile
import unittest
import urllib.error
from email.message import Message
from unittest import mock

import run_llm_v2 as runner

COMPLETE = {"rho_k2": 0.4, "rho_k3": 0.3, "rho_k4": 0.2, "rho_k5": 0.1,
            "mean_occupancy": 0.5, "C_one_step": 0.6,
            "lifetime_mean_over_T": 0.2, "lo90": 0.3, "hi90": 0.5}


def record(prompt_id="p1", answer="", finish_reason="stop", **kw):
    rec = {"prompt_id": prompt_id, "answer": answer,
           "finish_reason": finish_reason, "reasoning": None}
    rec.update(kw)
    return rec


class TestExtractLastJson(unittest.TestCase):
    def test_plain_object(self):
        text = "Reasoning...\n" + json.dumps(COMPLETE)
        self.assertEqual(runner.extract_last_json(text), COMPLETE)

    def test_code_fenced_object(self):
        text = "```json\n" + json.dumps(COMPLETE) + "\n```"
        self.assertEqual(runner.extract_last_json(text), COMPLETE)

    def test_prefers_last_object_with_rho_k2(self):
        decoy = {"rho_k2": 0.9}
        text = (json.dumps(decoy) + "\nrevised estimate:\n"
                + json.dumps(COMPLETE) + "\nDone.")
        self.assertEqual(runner.extract_last_json(text), COMPLETE)

    def test_skips_trailing_object_without_rho_k2(self):
        text = json.dumps(COMPLETE) + "\n" + json.dumps({"note": "bye"})
        self.assertEqual(runner.extract_last_json(text), COMPLETE)

    def test_nested_braces(self):
        obj = dict(COMPLETE, meta={"inner": {"x": 1}})
        self.assertEqual(runner.extract_last_json("x " + json.dumps(obj)), obj)

    def test_unparseable_and_empty(self):
        self.assertIsNone(runner.extract_last_json("{rho_k2: 0.5"))
        self.assertIsNone(runner.extract_last_json(""))
        self.assertIsNone(runner.extract_last_json(None))
        self.assertIsNone(runner.extract_last_json("no json here"))

    def test_matches_evaluator_extraction(self):
        """Runner resume parsing must agree with eval_llm_v2 parsing."""
        try:
            import eval_llm_v2 as ev
        except ImportError:
            self.skipTest("eval_llm_v2 deps (numpy/pandas) unavailable")
        samples = [
            "Reasoning...\n" + json.dumps(COMPLETE),
            "```json\n" + json.dumps(COMPLETE) + "\n```",
            json.dumps({"rho_k2": 0.9}) + "\n" + json.dumps(COMPLETE),
            json.dumps(COMPLETE) + "\n" + json.dumps({"note": 1}),
            "{rho_k2: broken", "", "prose only",
            "<think>{\"rho_k2\": 0.1}</think>\n" + json.dumps(COMPLETE),
        ]
        for s in samples:
            self.assertEqual(runner.extract_last_json(s),
                             ev.extract_last_json(s), msg=repr(s[:60]))


class TestIsCompleteRecord(unittest.TestCase):
    def test_complete_record(self):
        r = record(answer="thoughts\n" + json.dumps(COMPLETE))
        self.assertTrue(runner.is_complete_record(r))

    def test_null_values_still_complete(self):
        ans = json.dumps(dict(COMPLETE, C_one_step=None))
        self.assertTrue(runner.is_complete_record(record(answer=ans)))

    def test_error_record_retryable(self):
        r = record(answer=json.dumps(COMPLETE),
                   finish_reason="error: HTTP 500")
        self.assertFalse(runner.is_complete_record(r))

    def test_length_truncated_retryable(self):
        r = record(answer=json.dumps(COMPLETE), finish_reason="length")
        self.assertFalse(runner.is_complete_record(r))

    def test_empty_answer_retryable(self):
        self.assertFalse(runner.is_complete_record(record(answer="")))

    def test_unparsable_answer_retryable(self):
        r = record(answer="I think the answer is roughly 0.4.")
        self.assertFalse(runner.is_complete_record(r))

    def test_missing_key_retryable(self):
        partial = {k: v for k, v in COMPLETE.items() if k != "hi90"}
        r = record(answer=json.dumps(partial))
        self.assertFalse(runner.is_complete_record(r))

    def test_missing_finish_reason_ok_if_answer_complete(self):
        r = record(answer=json.dumps(COMPLETE), finish_reason=None)
        self.assertTrue(runner.is_complete_record(r))

    def test_values_not_validated_or_repaired(self):
        """Presence check only: out-of-range / non-monotonic stays complete."""
        bad = dict(COMPLETE, rho_k2=0.1, rho_k3=0.9, hi90=7.0)
        self.assertTrue(runner.is_complete_record(record(answer=json.dumps(bad))))


class TestDoneIds(unittest.TestCase):
    def write(self, records):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            for r in records:
                fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")
        self.addCleanup(os.unlink, path)
        return path

    def test_missing_file(self):
        self.assertEqual(runner.done_ids("/nonexistent/answers.jsonl"), set())

    def test_only_complete_records_count(self):
        good = "ok\n" + json.dumps(COMPLETE)
        path = self.write([
            record("p_ok", answer=good),
            record("p_err", answer="", finish_reason="error: timeout"),
            record("p_len", answer="<think>...", finish_reason="length"),
            record("p_empty", answer=""),
            record("p_nojson", answer="about 0.4"),
            record("p_partial", answer=json.dumps({"rho_k2": 0.4})),
            "{broken json line",
        ])
        self.assertEqual(runner.done_ids(path), {"p_ok"})

    def test_appended_retry_marks_done(self):
        good = json.dumps(COMPLETE)
        path = self.write([
            record("p1", answer="", finish_reason="error: HTTP 500"),
            record("p1", answer=good),
        ])
        self.assertEqual(runner.done_ids(path), {"p1"})

    def test_legacy_truncated_records_are_retried(self):
        """The old bug: finish_reason='length' counted as done."""
        path = self.write([record("p1", answer="<think> unfinished",
                                  finish_reason="length")])
        self.assertEqual(runner.done_ids(path), set())


class TestSplitThink(unittest.TestCase):
    def test_think_block_split_losslessly(self):
        final = json.dumps(COMPLETE)
        reasoning, answer = runner.split_think(
            "<think>step 1... step 2...</think>\n\n" + final)
        self.assertEqual(reasoning, "step 1... step 2...")
        self.assertEqual(answer, final)

    def test_no_think_block(self):
        reasoning, answer = runner.split_think("plain answer")
        self.assertIsNone(reasoning)
        self.assertEqual(answer, "plain answer")

    def test_truncated_think_stays_in_answer(self):
        reasoning, answer = runner.split_think("<think>never closed ...")
        self.assertIsNone(reasoning)
        self.assertEqual(answer, "<think>never closed ...")
        self.assertFalse(runner.is_complete_record(record(answer=answer)))

    def test_missing_open_tag(self):
        # Qwen3-style templates may emit the opening <think> as part of the
        # prompt, so generations can start mid-think.
        final = json.dumps(COMPLETE)
        reasoning, answer = runner.split_think("thoughts</think>" + final)
        self.assertEqual(reasoning, "thoughts")
        self.assertEqual(answer, final)


class TestRetryWaitSeconds(unittest.TestCase):
    def hdrs(self, retry_after=None):
        h = Message()
        if retry_after is not None:
            h["Retry-After"] = retry_after
        return h

    def test_no_suggestion(self):
        self.assertEqual(runner.retry_wait_seconds(self.hdrs(), ""), 0.0)
        self.assertEqual(runner.retry_wait_seconds(None, None), 0.0)

    def test_retry_after_header(self):
        self.assertEqual(runner.retry_wait_seconds(self.hdrs("7"), ""), 7.0)

    def test_invalid_retry_after_ignored(self):
        h = self.hdrs("Wed, 21 Oct 2026 07:28:00 GMT")
        self.assertEqual(runner.retry_wait_seconds(h, ""), 0.0)

    def test_gemini_retry_delay_body(self):
        body = ('{"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", '
                '"details": [{"retryDelay": "39s"}]}}')
        self.assertEqual(runner.retry_wait_seconds(self.hdrs(), body), 39.0)

    def test_max_of_both(self):
        body = '{"retryDelay": "5s"}'
        self.assertEqual(runner.retry_wait_seconds(self.hdrs("12"), body), 12.0)


def http_429(body=b'{"error": {"status": "RESOURCE_EXHAUSTED"}}',
             retry_after=None):
    h = Message()
    if retry_after is not None:
        h["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x/chat/completions", 429,
                                  "Too Many Requests", h, io.BytesIO(body))


def ok_response(content="done"):
    payload = json.dumps({
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        "model": "m"}).encode()
    m = mock.MagicMock()
    m.__enter__.return_value.read.return_value = payload
    return m


class TestApiCallRateLimit(unittest.TestCase):
    """Patient 429 handling for free-tier runs (rate_limit_max_wait > 0)."""

    def call(self, side_effects, retries=2, rate_limit_max_wait=0.0,
             sleep=1.0):
        waits = []
        with mock.patch("urllib.request.urlopen", side_effect=side_effects), \
             mock.patch.object(runner.time, "sleep", waits.append):
            res = runner.api_call(
                "http://x", "key", "m", "p", 0.0, 64, "none",
                timeout=5, retries=retries, sleep=sleep,
                rate_limit_max_wait=rate_limit_max_wait)
        return res, waits

    def test_429_does_not_consume_attempts(self):
        # more consecutive 429s than retries must still succeed
        effects = [http_429(), http_429(), http_429(), ok_response()]
        res, waits = self.call(effects, retries=2, rate_limit_max_wait=60)
        self.assertEqual(res["answer"], "done")
        self.assertEqual(res["finish_reason"], "stop")
        self.assertEqual(len(waits), 3)

    def test_wait_honors_server_delay_and_cap(self):
        body = b'{"error": {"details": [{"retryDelay": "39s"}]}}'
        effects = [http_429(body), http_429(body), ok_response()]
        res, waits = self.call(effects, retries=2, rate_limit_max_wait=30)
        self.assertEqual(res["answer"], "done")
        # server suggests 39s but the cap wins
        self.assertEqual(waits, [30.0, 30.0])

    def test_backoff_doubles_without_server_hint(self):
        effects = [http_429(), http_429(), http_429(), ok_response()]
        res, waits = self.call(effects, rate_limit_max_wait=3600, sleep=7)
        self.assertEqual(waits, [7.0, 14.0, 28.0])

    def test_default_behavior_unchanged(self):
        # rate_limit_max_wait=0: 429 consumes attempts and eventually raises
        effects = [http_429(retry_after="2"), http_429(retry_after="2")]
        with self.assertRaises(RuntimeError) as ctx:
            self.call(effects, retries=2, rate_limit_max_wait=0.0)
        self.assertIn("HTTP 429", str(ctx.exception))

    def test_gemini_reasoning_field_fallback(self):
        payload = json.dumps({
            "choices": [{"message": {"content": "final",
                                     "reasoning": "thoughts"},
                         "finish_reason": "stop"}]}).encode()
        m = mock.MagicMock()
        m.__enter__.return_value.read.return_value = payload
        res, _ = self.call([m])
        self.assertEqual(res["answer"], "final")
        self.assertEqual(res["reasoning"], "thoughts")


def sse(*events):
    """Encode chunks as an SSE byte-line iterable (like an HTTP response)."""
    lines = []
    for ev in events:
        data = ev if isinstance(ev, str) else json.dumps(ev)
        lines.append(f"data: {data}\n".encode())
        lines.append(b"\n")
    return lines


def chunk(content=None, reasoning=None, finish=None, usage=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return {"model": "m", "choices": [{"delta": delta,
                                       "finish_reason": finish}],
            "usage": usage}


class TestGeminiThinkingBody(unittest.TestCase):
    """Gemini 400s when reasoning_effort and thinking_config are both sent."""

    def test_thoughts_fold_effort_into_config(self):
        extra, effort = runner.gemini_thinking_body(True, "high")
        self.assertIsNone(effort)
        self.assertEqual(extra, {"google": {"thinking_config": {
            "include_thoughts": True, "thinking_level": "high"}}})

    def test_thoughts_without_effort(self):
        extra, effort = runner.gemini_thinking_body(True, None)
        self.assertIsNone(effort)
        self.assertEqual(extra, {"google": {"thinking_config": {
            "include_thoughts": True}}})

    def test_no_thoughts_keeps_top_level_effort(self):
        # NIM Mistral path must stay unchanged
        self.assertEqual(runner.gemini_thinking_body(False, "none"),
                         (None, "none"))
        self.assertEqual(runner.gemini_thinking_body(False, None),
                         (None, None))


class TestReadStream(unittest.TestCase):
    def test_reassembles_content_reasoning_and_usage(self):
        u = {"prompt_tokens": 5, "completion_tokens": 9}
        out = runner.read_stream(sse(
            chunk(reasoning="think "), chunk(reasoning="hard"),
            chunk(content="fin"), chunk(content="al"),
            chunk(finish="stop", usage=u), "[DONE]"))
        msg = out["choices"][0]["message"]
        self.assertEqual(msg["content"], "final")
        self.assertEqual(msg["reasoning_content"], "think hard")
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")
        self.assertEqual(out["usage"], u)
        self.assertEqual(out["model"], "m")

    def test_no_reasoning_field_when_absent(self):
        out = runner.read_stream(sse(chunk(content="hi"),
                                     chunk(finish="stop"), "[DONE]"))
        self.assertNotIn("reasoning_content", out["choices"][0]["message"])

    def test_length_finish_reason_survives(self):
        out = runner.read_stream(sse(chunk(reasoning="x"),
                                     chunk(finish="length"), "[DONE]"))
        self.assertEqual(out["choices"][0]["finish_reason"], "length")
        self.assertEqual(out["choices"][0]["message"]["content"], "")

    def test_ignores_keepalive_noise(self):
        lines = [b": keep-alive\n", b"\n"] + sse(chunk(content="ok"),
                                                 chunk(finish="stop"),
                                                 "[DONE]")
        out = runner.read_stream(lines)
        self.assertEqual(out["choices"][0]["message"]["content"], "ok")


class TestApiCallStreaming(unittest.TestCase):
    def test_stream_request_and_response(self):
        lines = sse(chunk(reasoning="deep"), chunk(content="done"),
                    chunk(finish="stop",
                          usage={"prompt_tokens": 1, "completion_tokens": 2}),
                    "[DONE]")
        m = mock.MagicMock()
        m.__enter__.return_value = iter(lines)
        sent = {}

        def fake_urlopen(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            return m

        with mock.patch("urllib.request.urlopen", side_effect=fake_urlopen):
            res = runner.api_call("http://x", "key", "m", "p", 1.0, 64,
                                  "on", timeout=5, retries=1, sleep=0,
                                  stream=True)
        self.assertTrue(sent["stream"])
        self.assertEqual(sent["stream_options"], {"include_usage": True})
        self.assertEqual(sent["chat_template_kwargs"], {"thinking": True})
        self.assertEqual(res["answer"], "done")
        self.assertEqual(res["reasoning"], "deep")
        self.assertEqual(res["finish_reason"], "stop")
        self.assertEqual(res["usage"]["completion_tokens"], 2)

    def test_mid_stream_disconnect_is_retryable(self):
        import http.client as hc

        def broken(req, timeout=None):
            m = mock.MagicMock()
            def boom():
                raise hc.IncompleteRead(b"partial")
            m.__enter__.return_value = iter_raises(boom)
            return m

        def iter_raises(fn):
            class It:
                def __iter__(self):
                    return self
                def __next__(self):
                    fn()
            return It()

        with mock.patch("urllib.request.urlopen", side_effect=broken), \
             mock.patch.object(runner.time, "sleep", lambda s: None):
            with self.assertRaises(RuntimeError) as ctx:
                runner.api_call("http://x", "key", "m", "p", 1.0, 64,
                                "none", timeout=5, retries=2, sleep=0.1,
                                stream=True)
        self.assertIn("IncompleteRead", str(ctx.exception))


class TestLoadPromptsSharding(unittest.TestCase):
    def test_shards_partition_and_are_stable(self):
        rows = [{"prompt_id": f"id{i:03d}", "prompt": "x"} for i in range(11)]
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        with os.fdopen(fd, "w") as fh:
            for r in sorted(rows, key=lambda r: r["prompt_id"], reverse=True):
                fh.write(json.dumps(r) + "\n")
        self.addCleanup(os.unlink, path)
        shards = [runner.load_prompts(path, i, 4, []) for i in range(4)]
        ids = [r["prompt_id"] for s in shards for r in s]
        self.assertEqual(sorted(ids), sorted(r["prompt_id"] for r in rows))
        self.assertEqual(len(ids), len(set(ids)))
        only = runner.load_prompts(path, 0, 1, ["id003", "id007"])
        self.assertEqual([r["prompt_id"] for r in only], ["id003", "id007"])


if __name__ == "__main__":
    unittest.main()
