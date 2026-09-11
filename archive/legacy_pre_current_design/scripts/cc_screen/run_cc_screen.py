#!/usr/bin/env python3
"""Screening run: Claude Code over the frozen v2.1 prompts, with and without tools.

This is a SCREEN, not a reported model comparison. It answers one question:
does a stronger model (optionally with code execution) do materially better on
this task than the five models already benchmarked? If yes, a proper API run
with pinned parameters is worth paying for; if no, the money is saved.
Nothing produced here belongs in the main leaderboard -- the harness is a
product (hidden defaults, no seed, version drift), so it is not reproducible.

Two arms, identical in everything except tool availability:

  notools  --tools ""              bare model, closest to a plain chat call
  tools    --tools "Bash,Read,..." same model, may write and execute code

Both arms get the SAME frozen prompt text and the SAME neutral system prompt,
so the arm difference isolates tool access. That contrast is the interesting
one: the v2.1 traces show the models recognise the sampling-bias problem but
fail to execute an estimator, so tools are exactly the variable expected to
move the result.

Isolation: every prompt runs in a fresh empty temporary directory that is
deleted afterwards. The agent gets no repository, no ground truth, no prompt
list and no previous answers -- with tools enabled it could otherwise read the
truth columns straight out of results/ and the screen would be worthless.
Raw CLI output is kept per prompt so tool use stays auditable.

Answers are written in the run_llm_v2.py record schema, append-only, and
resume by prompt_id via the shared is_complete_record(): truncated, empty,
unparsable and schema-incomplete records stay retryable. Values are never
repaired here.

Examples
  # 1. two-prompt smoke, no model call is made without --limit removed
  python scripts/cc_screen/run_cc_screen.py --arm notools --limit 2
  # 2. full arm (84 prompts, condition=disclosed / input=mask)
  python scripts/cc_screen/run_cc_screen.py --arm notools
  python scripts/cc_screen/run_cc_screen.py --arm tools
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from run_llm_v2 import is_complete_record  # noqa: E402

# Identical in both arms so that --tools is the only difference. The API
# models in the frozen suite were called with no system message at all, so
# this stays deliberately thin.
SYSTEM_PROMPT = (
    "You are a careful quantitative analyst. Answer the user's question "
    "directly. If tools are available to you, you may use them. Follow the "
    "output format the user specifies exactly."
)

DEFAULT_TOOLS = "Bash,Read,Write,Edit,Glob,Grep"

# The tools arm runs multi-turn with real computation; the bare arm answers in
# one turn. Observed: notools 187-233 s, tools 588 s and one run past 1200 s.
DEFAULT_TIMEOUT = {"notools": 1500, "tools": 3300}

# Fallback only. The message actually observed on exhaustion was "You've hit
# your monthly spend limit", which matches none of the phrases one would guess
# -- hence the structured check below is the primary signal and this list is
# just a safety net for wording the JSON does not cover.
LIMIT_PATTERNS = (
    "usage limit", "spend limit", "rate limit", "limit reached",
    "limit will reset", "insufficient credit", "out of credits",
    "credit balance", "upgrade your plan", "too many requests",
)


def limit_info(raw, meta=None, *texts):
    """(reason, resets_at_epoch) when a call was refused for plan/rate limits.

    The CLI says so structurally, and that is checked first: a
    `rate_limit_event` whose status is "rejected", or a result object carrying
    `api_error_status: 429`. Both were present verbatim in the run that
    exhausted the plan while every prose pattern missed, so string matching is
    only the fallback. `resetsAt` is a unix timestamp and lets the caller wait
    out the window instead of giving up.
    """
    reason = resets = None
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        if obj.get("type") == "rate_limit_event":
            info = obj.get("rate_limit_info") or {}
            if str(info.get("status")).lower() in ("rejected", "exhausted"):
                reason = reason or ("rate_limit_event rejected "
                                    f"({info.get('rateLimitType')})")
                resets = info.get("resetsAt") or resets
        if obj.get("api_error_status") == 429:
            reason = reason or "api_error_status=429"
    if isinstance(meta, dict) and meta.get("api_error_status") == 429:
        reason = reason or "api_error_status=429"
    if reason:
        return reason, resets
    for t in texts:
        low = (t or "").lower()
        for pat in LIMIT_PATTERNS:
            if pat in low:
                return f"matched {pat!r}", None
    return None, None


def wait_seconds(resets_at, default=1800.0, cap=6 * 3600.0):
    """How long to sleep before retrying a refused call.

    `resets_at` is the unix timestamp the CLI reports for the rolling window.
    Clamped both ways: a stale or bogus timestamp must not spin (min) and must
    not park the run for a day (cap).
    """
    try:
        delay = float(resets_at) - time.time() + 60.0
    except (TypeError, ValueError):
        delay = default
    return max(60.0, min(delay, cap))


def load_prompts(path, condition, input_kind, only_ids, limit):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: r["prompt_id"])
    if condition:
        rows = [r for r in rows if r["condition"] == condition]
    if input_kind:
        rows = [r for r in rows if r["input_kind"] == input_kind]
    if only_ids:
        keep = set(only_ids)
        rows = [r for r in rows if r["prompt_id"] in keep]
    if limit:
        rows = rows[:limit]
    return rows


def done_ids(out_path):
    """prompt_ids already answered completely (shared runner semantics)."""
    ids = set()
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if is_complete_record(rec):
                        ids.add(rec["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def attempts_by_id(out_path):
    """Real model attempts already written per prompt_id, complete or not.

    Resume alone retries a failing prompt forever. One prompt here ran the
    model to its 64 000-token output ceiling twice, ~900 s and >1.50 USD an
    attempt, and will never finish -- that has to cost a bounded amount.

    Calls structurally refused by the plan/rate limit do not count: no model
    attempt took place, and a stopped/restarted wait must not silently exhaust
    ``--max-attempts`` for that prompt.
    """
    counts = {}
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("limit_refusal"):
                    continue
                refused, _ = limit_info(
                    "", {}, str(rec.get("finish_reason") or ""))
                if refused:
                    continue
                pid = rec.get("prompt_id")
                if pid:
                    counts[pid] = counts.get(pid, 0) + 1
    return counts


def spent_usd(out_path):
    """Reported cost already recorded in an answer file."""
    total = 0.0
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    total += json.loads(line).get("cost_usd") or 0.0
                except json.JSONDecodeError:
                    continue
    return total


def build_cmd(args, arm):
    # stream-json in BOTH arms: --output-format json returns only the final
    # result object, which makes the tools arm unauditable -- exactly the arm
    # where "did it reach the ground truth?" has to be answerable. The stream
    # carries every tool call, and keeping it identical across arms means
    # --tools stays the only difference.
    cmd = ["claude", "--print",
           "--model", args.model,
           "--output-format", "stream-json", "--verbose",
           "--system-prompt", SYSTEM_PROMPT,
           "--no-session-persistence",
           "--safe-mode"]
    if args.effort:
        cmd += ["--effort", args.effort]
    if args.per_prompt_budget_usd:
        cmd += ["--max-budget-usd", str(args.per_prompt_budget_usd)]
    if arm == "notools":
        cmd += ["--tools", ""]
    else:
        cmd += ["--tools", args.tools,
                "--permission-mode", args.permission_mode]
    return cmd


def extract_answer(payload):
    """Pull the final reply text out of the CLI's --output-format json blob.

    The exact field name is not contractual, so several are tried and the raw
    payload is kept in the log either way. Returns (text, meta) with text None
    if nothing usable was found -- the caller then records an error so the
    prompt stays retryable.
    """
    meta = {}
    if isinstance(payload, dict):
        for k in ("total_cost_usd", "duration_ms", "num_turns", "session_id",
                  "usage", "model", "subtype", "is_error", "api_error_status",
                  "terminal_reason"):
            if k in payload:
                meta[k] = payload[k]
        for k in ("result", "response", "text", "content", "answer", "output"):
            v = payload.get(k)
            if isinstance(v, str) and v.strip():
                return v, meta
            if isinstance(v, list):  # content blocks
                parts = [b.get("text", "") for b in v
                         if isinstance(b, dict) and b.get("type") == "text"]
                if any(p.strip() for p in parts):
                    return "\n".join(parts), meta
    elif isinstance(payload, str) and payload.strip():
        return payload, meta
    return None, meta


def parse_stream(raw):
    """Last result object of a --output-format stream-json run."""
    payload = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("type") == "result":
            payload = obj
        elif payload is None and isinstance(obj, dict) and "result" in obj:
            payload = obj
    return payload


def call_claude(cmd, prompt, timeout):
    """One isolated invocation: fresh empty cwd, removed afterwards.

    TMPDIR points into that directory too, so scratch files an agent writes
    cannot be picked up by the next prompt. Hard-coded /tmp paths still
    escape it -- see the independence note in README.md.
    """
    workdir = tempfile.mkdtemp(prefix="task_")
    env = {**os.environ, "TMPDIR": workdir, "TMP": workdir, "TEMP": workdir}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, input=prompt, cwd=workdir, timeout=timeout,
                              capture_output=True, text=True, env=env)
        raw, err = proc.stdout, proc.stderr
        # parse before branching on the exit code: a failed call still reports
        # what it spent, and dropping that meta silently under-counted the cost
        # cap by the most expensive calls of the run
        payload = parse_stream(raw)
        if payload is None:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = raw
        text, meta = extract_answer(payload)
        lat = round(time.time() - t0, 2)
        reason = None
        if proc.returncode != 0:
            detail = err.strip() or (text or "")
            reason = f"error: exit {proc.returncode} {detail[:200]}"
        elif meta.get("is_error"):
            # a refusal arrives as a normal-looking reply; without this it
            # would be stored as the model's answer
            reason = f"error: cli is_error ({(text or '')[:160]})"
        elif text is None:
            reason = "error: no answer text in cli output"
        if reason:
            return ({"answer": "", "reasoning": None, "finish_reason": reason,
                     "usage": None, "model_echo": None,
                     "latency_s": lat}, raw, meta, err)
        return ({"answer": text, "reasoning": None,
                 # the CLI ran to completion; whether the reply is usable is
                 # decided by is_complete_record on the JSON itself
                 "finish_reason": "stop",
                 "usage": meta.get("usage"),
                 "model_echo": meta.get("model"),
                 "latency_s": round(time.time() - t0, 2)}, raw, meta, err)
    except subprocess.TimeoutExpired as e:
        # keep whatever the stream produced before the kill -- it shows how
        # far the agent got and whether the wall was compute or an API stall.
        # TimeoutExpired.stdout is bytes even under text=True, so an
        # isinstance(str) guard silently discards every partial.
        partial = e.stdout
        if isinstance(partial, (bytes, bytearray)):
            partial = partial.decode("utf-8", "replace")
        elif not isinstance(partial, str):
            partial = ""
        return ({"answer": "", "reasoning": None,
                 "finish_reason": f"error: timeout after {timeout}s",
                 "usage": None, "model_echo": None,
                 "latency_s": round(time.time() - t0, 2)}, partial, {}, "")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["notools", "tools"], required=True)
    ap.add_argument("--prompts", default=str(REPO / "results/llm_v2/prompts.jsonl"))
    ap.add_argument("--condition", default="disclosed",
                    help="frozen condition to screen (default: disclosed)")
    ap.add_argument("--input-kind", default="mask",
                    help="frozen input kind (default: mask -> the 84-case "
                         "headline cell of the leaderboard)")
    ap.add_argument("--ids", default=None, help="comma-separated prompt_ids")
    ap.add_argument("--limit", type=int, default=0, help="first N prompts (smoke)")
    ap.add_argument("--model", default="opus")
    ap.add_argument("--effort", default=None,
                    choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--tools", default=DEFAULT_TOOLS,
                    help="tools arm only (default: %(default)s)")
    ap.add_argument("--permission-mode", default="bypassPermissions",
                    help="tools arm only; non-interactive runs stall without a "
                         "non-asking mode")
    ap.add_argument("--out-dir", default=str(Path.home() / "Dokumente/cc_screen"),
                    help="OUTSIDE the repo; never visible to the agent")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=None,
                    help="seconds per prompt (default: %d notools / %d tools)"
                         % (DEFAULT_TIMEOUT["notools"], DEFAULT_TIMEOUT["tools"]))
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-cost-usd", type=float, default=0.0,
                    help="stop once the file's total reported cost exceeds "
                         "this (0 = no cap); counts earlier runs too")
    ap.add_argument("--max-consecutive-errors", type=int, default=3,
                    help="stop after this many failures in a row (0 = never)")
    ap.add_argument("--max-attempts", type=int, default=3,
                    help="skip a prompt once it has this many records and "
                         "still no complete answer (0 = retry forever); a "
                         "call refused by the plan limit counts too")
    ap.add_argument("--ignore-usage-limit", action="store_true",
                    help="keep going when the output looks like plan "
                         "exhaustion (default: stop immediately)")
    ap.add_argument("--wait-for-reset", action="store_true",
                    help="on plan exhaustion sleep until the reported reset "
                         "and retry the same prompt instead of stopping")
    ap.add_argument("--max-waits", type=int, default=8,
                    help="give up after this many reset waits (default: %(default)s)")
    ap.add_argument("--per-prompt-budget-usd", type=float, default=0.0,
                    help="pass --max-budget-usd to each call (0 = off); caps a "
                         "single runaway prompt, one of which burned 64k output "
                         "tokens and produced no answer")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command and the first prompt, call nothing")
    args = ap.parse_args()

    if shutil.which("claude") is None:
        sys.exit("claude not found on PATH")

    out_dir = Path(args.out_dir)
    if REPO in out_dir.resolve().parents or out_dir.resolve() == REPO:
        sys.exit(f"--out-dir must lie outside the repository ({REPO})")
    # effort goes into the label so runs at different effort land in different
    # files: mixing them inside one arm would silently mix two experiments
    label = f"claude-code-{args.model}_{args.arm}"
    if args.effort:
        label += f"_{args.effort}"
    out_path = Path(args.out) if args.out else out_dir / f"answers_{label}.jsonl"
    log_dir = out_dir / "logs" / args.arm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    only = args.ids.split(",") if args.ids else None
    rows = load_prompts(args.prompts, args.condition, args.input_kind,
                        only, args.limit)
    if not rows:
        sys.exit("no prompts selected")
    have = done_ids(out_path)
    todo = [r for r in rows if r["prompt_id"] not in have]
    tries = attempts_by_id(out_path)
    burned = []
    if args.max_attempts:
        burned = [r["prompt_id"] for r in todo
                  if tries.get(r["prompt_id"], 0) >= args.max_attempts]
        todo = [r for r in todo if r["prompt_id"] not in burned]
    # Finish never-attempted cases before retrying earlier failures. Stable
    # sorting preserves the frozen prompt order inside both groups.
    todo.sort(key=lambda r: tries.get(r["prompt_id"], 0) > 0)
    cmd = build_cmd(args, args.arm)
    timeout = args.timeout or DEFAULT_TIMEOUT[args.arm]
    spent = spent_usd(out_path)  # carry earlier runs into the cost cap

    print(f"arm={args.arm}  model={args.model}  label={label}")
    print(f"selected={len(rows)}  complete={len(have)}  todo={len(todo)}"
          + (f"  giving_up_on={len(burned)}" if burned else ""))
    if burned:
        # a prompt the model cannot answer is a result, not a gap to hide:
        # every attempt stays in the file and belongs in the write-up
        print(f"  no complete answer after {args.max_attempts} attempts, "
              f"skipped: {', '.join(burned)}")
    print(f"timeout={timeout}s  spent_so_far={spent:.2f}USD"
          + (f"  cap={args.max_cost_usd:.2f}USD" if args.max_cost_usd else ""))
    print(f"out={out_path}")
    print("cmd=" + " ".join(repr(c) if c == "" else c for c in cmd))
    if args.dry_run:
        print("\n--- first prompt ---\n" + rows[0]["prompt"][:2000])
        return
    if not todo:
        print("nothing to do")
        return
    if args.max_cost_usd and spent >= args.max_cost_usd:
        print(f"\nSTOPPED EARLY: cost cap already reached "
              f"({spent:.2f} >= {args.max_cost_usd:.2f} USD)", file=sys.stderr)
        sys.exit(2)

    n_err = 0
    streak = 0
    waits = 0
    stopped = None
    with open(out_path, "a") as fh:
        for i, r in enumerate(todo, 1):
            while True:  # loops only to serve a reset wait
                res, raw, meta, err = call_claude(cmd, r["prompt"], timeout)
                if str(res["finish_reason"]).startswith("error"):
                    n_err += 1
                    streak += 1
                else:
                    streak = 0
                if raw:
                    (log_dir / f"{r['prompt_id']}.jsonl").write_text(raw)
                lim, resets = limit_info(raw, meta, err, res["finish_reason"])
                rec = {"id": r["prompt_id"], "prompt_id": r["prompt_id"],
                       "case_id": r["case_id"], "condition": r["condition"],
                       "input_kind": r["input_kind"], "strategy": r["strategy"],
                       "model": label, "backend": "claude_code",
                       "arm": args.arm, "cli_model": args.model,
                       "effort": args.effort,
                       "tools": "" if args.arm == "notools" else args.tools,
                       "thinking": None, "reasoning_effort": None,
                       "temperature": None, "top_p": None, "top_k": None,
                       "seed": None, "max_tokens": None,
                       "num_turns": meta.get("num_turns"),
                       "cost_usd": meta.get("total_cost_usd"),
                       "limit_refusal": bool(lim),
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                spent += meta.get("total_cost_usd") or 0.0
                ok = "ok " if is_complete_record(rec) else "INCOMPLETE"
                print(f"[{i}/{len(todo)}] {r['prompt_id']} {ok} "
                      f"turns={meta.get('num_turns')} {res['latency_s']}s "
                      f"total={spent:.2f}USD", flush=True)

                if lim and args.wait_for_reset and waits < args.max_waits:
                    waits += 1
                    delay = wait_seconds(resets)
                    until = time.strftime("%H:%M", time.localtime(time.time() + delay))
                    print(f"    plan limit ({lim}) -- sleeping {delay / 60:.0f} min "
                          f"until ~{until}, then retrying this prompt "
                          f"[wait {waits}/{args.max_waits}]", flush=True)
                    streak = 0  # a refused call is not a model failure
                    time.sleep(delay)
                    continue
                break

            if lim and not args.ignore_usage_limit:
                stopped = (f"plan limit ({lim}). The record stays retryable; "
                           f"rerun the same command after the reset, or use "
                           f"--wait-for-reset to sit it out automatically.")
                break
            if args.max_cost_usd and spent >= args.max_cost_usd:
                stopped = (f"cost cap reached ({spent:.2f} >= "
                           f"{args.max_cost_usd:.2f} USD)")
                break
            if args.max_consecutive_errors and streak >= args.max_consecutive_errors:
                stopped = (f"{streak} failures in a row -- last: "
                           f"{res['finish_reason'][:120]}")
                break
            if args.sleep:
                time.sleep(args.sleep)

    print(f"done, {n_err} errors, {spent:.2f} USD total; answers in {out_path}")
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
