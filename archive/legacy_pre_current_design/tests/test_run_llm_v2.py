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

    def test_attempt_stats_counts_valid_lines(self):
        path = self.write([
            record("p1", answer="", finish_reason="length"),
            record("p1", answer="", finish_reason="error: timeout"),
            record("p2", answer="", finish_reason="length"),
            "{broken",
        ])
        attempts, lengths = runner.attempt_stats(path)
        self.assertEqual(attempts, {"p1": 2, "p2": 1})
        self.assertEqual(lengths, {"p1": 1, "p2": 1})

    def test_select_todo_prioritizes_unseen_and_caps_length(self):
        good = json.dumps(COMPLETE)
        path = self.write([
            record("done", answer=good),
            record("capped", answer="", finish_reason="length"),
            record("retry", answer="", finish_reason="error: timeout"),
        ])
        rows = [{"prompt_id": x} for x in
                ("retry", "unseen_b", "done", "capped", "unseen_a")]
        done, todo, lengths = runner.select_todo(
            rows, path, max_length_attempts=1)
        self.assertEqual(done, {"done"})
        self.assertEqual([r["prompt_id"] for r in todo],
                         ["unseen_a", "unseen_b", "retry"])
        self.assertEqual(lengths["capped"], 1)

    def test_zero_length_cap_preserves_retry(self):
        path = self.write([
            record("old", answer="", finish_reason="length"),
        ])
        rows = [{"prompt_id": "old"}, {"prompt_id": "new"}]
        _, todo, _ = runner.select_todo(rows, path, max_length_attempts=0)
        self.assertEqual([r["prompt_id"] for r in todo], ["new", "old"])

    def test_resume_done_ids_unions_globbed_answer_files(self):
        good = json.dumps(COMPLETE)
        first = self.write([record("p1", answer=good)])
        second = self.write([
            record("p2", answer=good),
            record("retry", answer="", finish_reason="error: HTTP 429"),
        ])
        pattern = os.path.join(os.path.dirname(first), "*.jsonl")
        # Other test temp files can share /tmp, so assert inclusion rather
        # than exact equality for the broad temporary-directory glob.
        ids = runner.resume_done_ids([first, second, pattern])
        self.assertIn("p1", ids)
        self.assertIn("p2", ids)
        self.assertNotIn("retry", ids)


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

    def test_gemini_thought_block(self):
        final = json.dumps(COMPLETE)
        reasoning, answer = runner.split_think(
            "<thought>**Analysis** of the sample</thought>\n" + final)
        self.assertEqual(reasoning, "**Analysis** of the sample")
        self.assertEqual(answer, final)

    def test_gemini_multiple_thought_blocks_split_at_last(self):
        final = json.dumps(COMPLETE)
        text = ("<thought>part one</thought><thought>part two</thought>\n"
                + final)
        reasoning, answer = runner.split_think(text)
        self.assertIn("part one", reasoning)
        self.assertIn("part two", reasoning)
        self.assertNotIn("<thought>", reasoning)
        self.assertEqual(answer, final)

    def test_truncated_thought_stays_in_answer(self):
        # length-truncated Gemini record: open <thought> never closed
        reasoning, answer = runner.split_think("<thought>cut off mid...")
        self.assertIsNone(reasoning)
        self.assertEqual(answer, "<thought>cut off mid...")
        self.assertFalse(runner.is_complete_record(record(answer=answer)))


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

    def test_gemini_inline_thought_split_into_reasoning(self):
        res, _ = self.call([ok_response("<thought>why</thought>\nfinal")])
        self.assertEqual(res["reasoning"], "why")
        self.assertEqual(res["answer"], "final")

    def test_no_split_when_reasoning_content_present(self):
        payload = json.dumps({
            "choices": [{"message": {"content": "a </thought> b",
                                     "reasoning_content": "rc"},
                         "finish_reason": "stop"}]}).encode()
        m = mock.MagicMock()
        m.__enter__.return_value.read.return_value = payload
        res, _ = self.call([m])
        self.assertEqual(res["reasoning"], "rc")
        self.assertEqual(res["answer"], "a </thought> b")

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

    def test_official_deepseek_thinking_toggle_format(self):
        sent = {}
        ok = {"choices": [{"message": {"content": "done"},
                           "finish_reason": "stop"}]}

        def fake(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            m = mock.MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(ok).encode()
            return m

        with mock.patch("urllib.request.urlopen", side_effect=fake):
            runner.api_call(
                "http://x", "key", "deepseek-v4-flash", "p", 1.0, 8192,
                "off", timeout=5, retries=1, sleep=0,
                api_thinking_format="deepseek")
        self.assertEqual(sent["thinking"], {"type": "disabled"})
        self.assertNotIn("chat_template_kwargs", sent)


class TestApiCostGuard(unittest.TestCase):
    def test_cost_uses_detailed_cache_usage(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 200,
                 "prompt_cache_hit_tokens": 80,
                 "prompt_cache_miss_tokens": 20}
        got = runner.estimate_api_cost_usd(usage, 0.0028, 0.14, 0.28)
        expected = (80 * 0.0028 + 20 * 0.14 + 200 * 0.28) / 1_000_000
        self.assertAlmostEqual(got, expected)

    def test_cost_falls_back_to_all_cache_miss(self):
        usage = {"prompt_tokens": 100, "completion_tokens": 200}
        got = runner.estimate_api_cost_usd(usage, 0.0028, 0.14, 0.28)
        expected = (100 * 0.14 + 200 * 0.28) / 1_000_000
        self.assertAlmostEqual(got, expected)

    def test_request_reservation_assumes_full_output_budget(self):
        got = runner.request_cost_upper_bound("abcd", 8192, 0.14, 0.28)
        expected = (4 * 0.14 + 8192 * 0.28) / 1_000_000
        self.assertAlmostEqual(got, expected)

    def test_shared_budget_reserves_settles_and_reuses_headroom(self):
        with tempfile.TemporaryDirectory() as tmp:
            budget = runner.SharedApiBudget(
                os.path.join(tmp, "budget.json"), 1.0)
            first = budget.reserve(0.6)
            self.assertIsNotNone(first)
            self.assertIsNone(budget.reserve(0.5))
            self.assertAlmostEqual(budget.settle(first, 0.2), 0.2)
            second = budget.reserve(0.5)
            self.assertIsNotNone(second)
            budget.release(second)
            state = budget.snapshot()
            self.assertAlmostEqual(state["spent_usd"], 0.2)
            self.assertEqual(state["reservations"], {})

    def test_shared_budget_rejects_a_different_limit_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "budget.json")
            runner.SharedApiBudget(path, 1.0)
            with self.assertRaises(ValueError):
                runner.SharedApiBudget(path, 2.0)


class TestHFRepetitionControls(unittest.TestCase):
    """Degeneration controls must reach generate(), and stay off by default.

    The Qwen3.6 non-thinking smoke on the non-walk screen filled its entire
    8192-token budget with `+1+1+1...`: trigram diversity collapsed from 0.44
    to 0.001. The model card's answer is presence_penalty, which transformers
    does not have; these are the substitutes.
    """

    def build(self, **kw):
        hf = runner.HFModel.__new__(runner.HFModel)
        hf.max_new_tokens = 128
        hf.temperature = 0.7
        hf.top_p = 0.8
        hf.top_k = 20
        hf.seed = 0
        hf.thinking = "off"
        hf.repetition_penalty = kw.get("repetition_penalty")
        hf.no_repeat_ngram_size = kw.get("no_repeat_ngram_size")
        return hf

    def gen_kwargs(self, hf):
        captured = {}

        class Tok:
            eos_token_id = 7

            def apply_chat_template(self, msgs, **kw):
                return "text"

            def __call__(self, text, **kw):
                class Inputs(dict):
                    def to(self, device):
                        return self
                return Inputs(input_ids=DummyTensor())

            def decode(self, ids, **kw):
                return "answer"

        class DummyTensor:
            shape = (1, 3)

            def __getitem__(self, item):
                return self

        class Model:
            device = "cpu"

            def generate(self, **kw):
                captured.update(kw)
                return [[0, 1, 2, 3]]

        class Torch:
            @staticmethod
            def manual_seed(seed):
                pass

            @staticmethod
            def no_grad():
                import contextlib
                return contextlib.nullcontext()

        hf.tok = Tok()
        hf.model = Model()
        hf.torch = Torch()
        hf("prompt")
        return captured

    def test_defaults_send_neither_control(self):
        kw = self.gen_kwargs(self.build())
        self.assertNotIn("repetition_penalty", kw)
        self.assertNotIn("no_repeat_ngram_size", kw)

    def test_controls_reach_generate(self):
        kw = self.gen_kwargs(self.build(repetition_penalty=1.1,
                                        no_repeat_ngram_size=40))
        self.assertEqual(kw["repetition_penalty"], 1.1)
        self.assertEqual(kw["no_repeat_ngram_size"], 40)

    def test_penalty_of_one_is_treated_as_off(self):
        # 1.0 is transformers' no-op; sending it is harmless but noise
        kw = self.gen_kwargs(self.build(repetition_penalty=0))
        self.assertNotIn("repetition_penalty", kw)


class TestOutputBudget(unittest.TestCase):
    """`--max-tokens 0` must omit the field, not send a zero budget."""

    def send(self, max_tokens):
        sent = {}
        ok = {"choices": [{"message": {"content": "{}"},
                           "finish_reason": "stop"}]}

        def fake(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            m = mock.MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(ok).encode()
            return m

        with mock.patch("urllib.request.urlopen", side_effect=fake):
            runner.api_call("http://x", "key", "m", "p", 1.0, max_tokens,
                            "none", timeout=5, retries=1, sleep=0)
        return sent

    def test_explicit_budget_is_sent(self):
        self.assertEqual(self.send(8192)["max_tokens"], 8192)

    def test_reasoning_models_get_max_completion_tokens(self):
        """OpenAI reasoning models 400 on `max_tokens` and demand the other
        name. Same number, different key -- the caller picks."""
        sent = {}
        ok = {"choices": [{"message": {"content": "{}"},
                           "finish_reason": "stop"}]}

        def fake(req, timeout=None):
            sent.update(json.loads(req.data.decode()))
            m = mock.MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(ok).encode()
            return m

        with mock.patch("urllib.request.urlopen", side_effect=fake):
            runner.api_call("http://x", "key", "m", "p", -1, 4096, "none",
                            timeout=5, retries=1, sleep=0,
                            max_tokens_param="max_completion_tokens")
        self.assertEqual(sent["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", sent)
        self.assertNotIn("temperature", sent)

    def test_zero_budget_omits_the_field(self):
        body = self.send(0)
        self.assertNotIn("max_tokens", body)
        # a literal 0 would ask the server for an empty completion
        self.assertEqual(body["messages"][0]["content"], "p")


class TestTransientStatusHandling(unittest.TestCase):
    """A busy endpoint must not become a permanent error record.

    NVIDIA NIM answers HTTP 529 when the provider is overloaded. It was absent
    from the retryable set, so the first 529 raised immediately and the prompt
    was written out as failed after zero retries.
    """

    def http_error(self, code):
        import urllib.error

        def raiser(req, timeout=None):
            raise urllib.error.HTTPError(
                "http://x", code, "boom", {},
                io.BytesIO(b'{"message":"overloaded"}'))
        return raiser

    def call(self, retries=3, **kw):
        return runner.api_call("http://x", "key", "m", "p", 1.0, 64, "none",
                               timeout=5, retries=retries, sleep=0, **kw)

    def test_529_is_retried_before_giving_up(self):
        calls = []

        def counting(req, timeout=None):
            calls.append(1)
            return self.http_error(529)(req, timeout)

        with mock.patch("urllib.request.urlopen", side_effect=counting), \
             mock.patch.object(runner.time, "sleep", lambda s: None):
            with self.assertRaises(RuntimeError) as ctx:
                self.call(retries=3)
        self.assertEqual(len(calls), 3)
        self.assertIn("529", str(ctx.exception))

    def test_529_recovers_without_a_failure_record(self):
        state = {"n": 0}
        ok = {"choices": [{"message": {"content": "{}"},
                           "finish_reason": "stop"}]}

        def flaky(req, timeout=None):
            state["n"] += 1
            if state["n"] == 1:
                return self.http_error(529)(req, timeout)
            m = mock.MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(ok).encode()
            return m

        with mock.patch("urllib.request.urlopen", side_effect=flaky), \
             mock.patch.object(runner.time, "sleep", lambda s: None):
            res = self.call(retries=3)
        self.assertEqual(res["finish_reason"], "stop")

    def test_persistent_backpressure_gives_up_once_the_budget_is_spent(self):
        """Patient waiting must terminate.

        The backpressure branch does not consume retry attempts, so without a
        total ceiling one permanently overloaded endpoint parks the run on its
        first prompt indefinitely -- writing nothing, reporting nothing.
        """
        calls = []
        slept = []

        def always_529(req, timeout=None):
            calls.append(1)
            return self.http_error(529)(req, timeout)

        with mock.patch("urllib.request.urlopen", side_effect=always_529), \
             mock.patch.object(runner.time, "sleep", slept.append):
            with self.assertRaises(RuntimeError):
                self.call(retries=3, rate_limit_max_wait=60,
                          rate_limit_total_wait=10)
        self.assertLessEqual(sum(slept), 10 + 1e-9,
                             "total wait exceeded the stated budget")
        self.assertGreater(len(calls), 3, "should wait before spending retries")
        self.assertLess(len(calls), 30, "must terminate, not loop forever")

    def test_unbounded_budget_keeps_the_daily_quota_behaviour(self):
        state = {"n": 0}
        ok = {"choices": [{"message": {"content": "{}"},
                           "finish_reason": "stop"}]}

        def flaky(req, timeout=None):
            state["n"] += 1
            if state["n"] <= 8:
                return self.http_error(429)(req, timeout)
            m = mock.MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(ok).encode()
            return m

        with mock.patch("urllib.request.urlopen", side_effect=flaky), \
             mock.patch.object(runner.time, "sleep", lambda s: None):
            res = self.call(retries=2, rate_limit_max_wait=60)
        self.assertEqual(res["finish_reason"], "stop")
        self.assertEqual(state["n"], 9)

    def test_529_waits_patiently_under_rate_limit_budget(self):
        # With a wait budget, backpressure must not consume retry attempts,
        # so more than `retries` requests can be made before success.
        state = {"n": 0}
        ok = {"choices": [{"message": {"content": "{}"},
                           "finish_reason": "stop"}]}

        def flaky(req, timeout=None):
            state["n"] += 1
            if state["n"] <= 4:
                return self.http_error(529)(req, timeout)
            m = mock.MagicMock()
            m.__enter__.return_value.read.return_value = json.dumps(ok).encode()
            return m

        with mock.patch("urllib.request.urlopen", side_effect=flaky), \
             mock.patch.object(runner.time, "sleep", lambda s: None):
            res = self.call(retries=2, rate_limit_max_wait=60)
        self.assertEqual(state["n"], 5)
        self.assertEqual(res["finish_reason"], "stop")

    def test_client_errors_still_fail_immediately(self):
        for code in (400, 401, 404, 410):
            calls = []

            def counting(req, timeout=None, code=code):
                calls.append(1)
                return self.http_error(code)(req, timeout)

            with mock.patch("urllib.request.urlopen", side_effect=counting), \
                 mock.patch.object(runner.time, "sleep", lambda s: None):
                with self.assertRaises(RuntimeError):
                    self.call(retries=3)
            self.assertEqual(len(calls), 1, f"HTTP {code} must not retry")


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
