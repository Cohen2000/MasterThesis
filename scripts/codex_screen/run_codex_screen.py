#!/usr/bin/env python3
"""Screening run: Codex CLI over the frozen v2.1 prompts, with and without tools.

Same purpose and same non-status as scripts/cc_screen: this is a SCREEN, not a
reported model comparison. It answers whether a frontier agentic model does
materially better on this task than the five models already benchmarked, so
that the decision to pay for a proper API run is informed. Nothing produced
here belongs in the main leaderboard -- the harness is a product (hidden
defaults, injected instructions, no seed, version drift), so it is not
reproducible.

Two arms, identical except tool availability:

  notools  shell/exec features disabled     bare model, closest to a chat call
  tools    read-only or workspace sandbox   same model, may write and run code

Both arms get the SAME frozen prompt text and the SAME preamble, so the arm
difference isolates tool access.

Four things about the Codex CLI drive this file and differ from `claude`:

1. `codex exec` exits 0 even when the turn failed. The 401 probe on 2026-07-30
   ended in `{"type":"turn.failed",...}` with returncode 0, so the exit code
   is worthless as a success signal and every call is judged from its events.
2. On a ChatGPT plan the CLI reports TOKENS, not dollars. The budget cap here
   is therefore token-based; `--usd-per-mtok-*` only derives an estimate and
   is off by default rather than guessing a price.
3. There is no `--system-prompt`. Codex injects its own developer messages
   (skills block, agent-team block) that cannot be removed -- verified with
   `codex debug prompt-input`, 9,243 characters remain even with every tool feature
   disabled. The shared preamble is therefore prepended to the user prompt,
   and the residual asymmetry against the Claude arm is a stated limitation,
   not a hidden one. `--verify` prints exactly what the model will see.
4. There is no `--tools ""`. Tool access is switched off via feature flags,
   which is a claim rather than a guarantee -- so every record stores how many
   tool events the stream actually contained, and the notools arm warns
   loudly if that number is not zero.

Isolation: every prompt runs in a fresh empty temporary directory that is
deleted afterwards, with TMPDIR redirected into it. The agent gets no
repository, no ground truth, no prompt list and no previous answers -- with
tools enabled it could otherwise read the truth columns straight out of
results/ and the screen would be worthless. Cross-prompt memory (`goals`,
`memories`) and network reach (`browser_use`, `web_search`) are disabled in
BOTH arms: they would break the independence of the 84 cases, which matters
more than the small capability they add. Raw event streams are kept per prompt
so tool use stays auditable.

Answers are written in the run_llm_v2.py record schema, append-only, and
resume by prompt_id via the shared is_complete_record(): truncated, empty,
unparsable and schema-incomplete records stay retryable. Values are never
repaired here.

Examples
  # 0. what will the model actually see? free, local, no API call
  python scripts/codex_screen/run_codex_screen.py --arm notools --verify
  # 1. two-prompt smoke
  python scripts/codex_screen/run_codex_screen.py --arm notools --limit 2
  # 2. screening subset, budget-capped
  python scripts/codex_screen/run_codex_screen.py --arm notools --limit 30 \
      --max-total-tokens 3000000 --wait-for-reset
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

# Byte-identical to scripts/cc_screen/run_cc_screen.py so the two CLIs are
# asked the same thing. Codex has no --system-prompt, so this is prepended to
# the user message instead; see the module docstring.
SYSTEM_PROMPT = (
    "You are a careful quantitative analyst. Answer the user's question "
    "directly. If tools are available to you, you may use them. Follow the "
    "output format the user specifies exactly."
)

DEFAULT_MODEL = "gpt-5.6-sol"
# gpt-5.6-sol ships with default_reasoning_level "low", described in the model
# catalogue as "Fast responses with lighter reasoning". Running the screen
# there would handicap the capability under test and bias the outcome towards
# "do not fund the API run". The effort is therefore explicit, high, and part
# of the output filename so two levels can never land in one file.
DEFAULT_EFFORT = "high"

# Off in BOTH arms: cross-prompt state and network reach. Independence of the
# 84 cases is worth more than the capability these add, and a web-reachable
# agent could in principle find the benchmark itself.
ALWAYS_OFF = (
    "goals", "memories",                      # carry state between prompts
    "browser_use", "browser_use_external", "browser_use_full_cdp_access",
    "computer_use", "image_generation",
    "multi_agent", "multi_agent_v2",          # sub-agents = unbounded cost
    "apps", "plugins",
)
# Additionally off in the notools arm: everything that can execute code.
NO_TOOL_FEATURES = (
    "shell_tool", "unified_exec", "code_mode_host", "code_mode",
    "shell_snapshot", "skill_search", "tool_suggest",
)

# Same walls as the Claude screen, for comparability of the timeout column.
DEFAULT_TIMEOUT = {"notools": 900, "tools": 2700}

# Event item types that are NOT tool use. Anything else counts as a tool
# event: an unknown type must show up in the audit rather than slip past it.
BENIGN_ITEMS = frozenset({
    "agent_message", "assistant_message", "message", "user_message",
    "reasoning", "error", "todo_list", "plan_update",
})

# Fallback only; the structured checks below are the primary signal. The
# Claude screen learned this the hard way -- its first detector matched prose
# and missed a real exhaustion event because the CLI said it in JSON.
LIMIT_PATTERNS = (
    "usage limit", "spend limit", "rate limit", "rate_limit",
    "limit reached", "limit will reset", "quota", "insufficient credit",
    "out of credits", "credit balance", "upgrade your plan",
    "too many requests", "429",
)


def iter_events(raw):
    """Every JSON object on its own line of a `codex exec --json` stream."""
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue  # the CLI also writes plain ERROR lines
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def _rate_limit_from(obj):
    """(reason, resets_at_epoch) if this event reports an exhausted window.

    Codex reports usage as percentages with a relative reset, unlike the
    absolute `resetsAt` the Claude CLI emits, so both forms are accepted. The
    exact nesting was not observable without credentials -- hence the walk
    over any dict that carries a `used_percent`, rather than one hard-coded
    path that a schema change would silently break.
    """
    stack = [obj]
    while stack:
        cur = stack.pop()
        if isinstance(cur, list):
            stack.extend(cur)
            continue
        if not isinstance(cur, dict):
            continue
        pct = cur.get("used_percent", cur.get("usedPercent"))
        if isinstance(pct, (int, float)) and pct >= 100:
            secs = cur.get("resets_in_seconds", cur.get("resetsInSeconds"))
            at = cur.get("resets_at", cur.get("resetsAt"))
            if isinstance(secs, (int, float)):
                at = time.time() + secs
            return f"rate limit window at {pct:.0f}%", at
        stack.extend(cur.values())
    return None, None


def limit_info(raw, meta=None, *texts):
    """(reason, resets_at_epoch) when a call was refused for plan/rate limits.

    Structured first: a rate-limit window reported at >=100%, or an error /
    turn.failed message carrying a 429. Prose matching is only the safety net
    for wording the events do not cover.
    """
    reason = resets = None
    for obj in iter_events(raw):
        r, at = _rate_limit_from(obj)
        if r and reason is None:
            reason, resets = r, at
        # only error-bearing events, never the answer: an agent_message
        # saying "429 pairs observed" would otherwise stop the whole run
        blob = None
        if obj.get("type") in ("error", "turn.failed"):
            blob = json.dumps(obj).lower()
        elif obj.get("type") == "item.completed":
            item = obj.get("item") or {}
            if isinstance(item, dict) and item.get("type") == "error":
                blob = json.dumps(item).lower()
        if blob and ("429" in blob or "rate limit" in blob
                     or "usage limit" in blob or "quota" in blob):
            reason = reason or "error event reports a 429/rate limit"
    if isinstance(meta, dict):
        if meta.get("status_code") == 429:
            reason = reason or "status_code=429"
        r, at = _rate_limit_from(meta)
        if r and reason is None:
            reason, resets = r, at
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

    Clamped both ways: a stale or bogus timestamp must not spin (min) and must
    not park the run for a day (cap).
    """
    try:
        delay = float(resets_at) - time.time() + 60.0
    except (TypeError, ValueError):
        delay = default
    return max(60.0, min(delay, cap))


def parse_events(raw, answer_file=None):
    """(text, meta) from a `codex exec --json` run.

    `--output-last-message` is trusted first: it is the CLI's own idea of the
    final reply and needs no guessing about event names. The stream is the
    fallback and always supplies the metadata -- including the tool-event
    count that makes the notools arm checkable instead of merely claimed.
    """
    meta = {"n_tool_events": 0, "tool_item_types": [], "failed": None,
            "thread_id": None, "usage": None, "total_tokens": None,
            "status_code": None}
    text = None
    seen_types = []
    for obj in iter_events(raw):
        kind = obj.get("type")
        if kind == "thread.started":
            meta["thread_id"] = obj.get("thread_id")
        elif kind == "turn.failed":
            err = obj.get("error") or {}
            meta["failed"] = (err.get("message") if isinstance(err, dict)
                              else str(err)) or "turn.failed"
        elif kind == "error" and meta["failed"] is None:
            # keep only the first: the CLI logs one per retry attempt
            meta["failed"] = obj.get("message") or "error event"
        elif kind == "item.completed":
            item = obj.get("item") or {}
            itype = item.get("type")
            if itype in ("agent_message", "assistant_message", "message"):
                t = item.get("text") or item.get("content")
                if isinstance(t, str) and t.strip():
                    text = t  # last message wins
            elif itype == "error" and meta["failed"] is None:
                meta["failed"] = item.get("message") or "error item"
            elif itype and itype not in BENIGN_ITEMS:
                meta["n_tool_events"] += 1
                seen_types.append(itype)
        if isinstance(obj.get("usage"), dict):
            meta["usage"] = obj["usage"]
        for k in ("status_code", "http_status"):
            if isinstance(obj.get(k), int):
                meta["status_code"] = obj[k]
    meta["tool_item_types"] = sorted(set(seen_types))
    u = meta["usage"] or {}
    tot = u.get("total_tokens")
    if tot is None:
        # input_tokens includes cached input, and output_tokens includes
        # reasoning tokens. Detail fields must not be counted a second time.
        input_tok = u.get("input_tokens", u.get("prompt_tokens"))
        output_tok = u.get("output_tokens", u.get("completion_tokens"))
        parts = [input_tok, output_tok]
        nums = [p for p in parts if isinstance(p, (int, float))]
        tot = sum(nums) if nums else None
    meta["total_tokens"] = tot

    if answer_file:
        try:
            got = Path(answer_file).read_text()
        except OSError:
            got = ""
        if got.strip():
            text = got
    return text, meta


def estimate_usd(usage, in_rate, out_rate):
    """USD estimate from published per-Mtok prices, or None if none given.

    Deliberately not defaulted: inventing a price would put a fabricated
    number into the results file. See README.
    """
    if not usage or not (in_rate or out_rate):
        return None
    ins = usage.get("input_tokens") or 0
    out = usage.get("output_tokens") or 0
    # cached_input_tokens is a detail subset of input_tokens, not extra input.
    return round((ins * in_rate + out * out_rate) / 1e6, 6)


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


def usable_record(record, arm):
    """Whether a record is complete and valid for the requested arm."""
    if not is_complete_record(record):
        return False
    # A structurally valid answer obtained after a tool call is still not a
    # valid observation for the deliberately tool-free treatment.
    return not (arm == "notools" and record.get("n_tool_events"))


def done_ids(out_path, arm=None):
    """prompt_ids already answered completely and validly for ``arm``."""
    ids = set()
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if usable_record(rec, arm):
                        ids.add(rec["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def attempts_by_id(out_path):
    """Records already written per prompt_id, complete or not.

    Resume alone retries a failing prompt forever. In the Claude screen three
    prompts drove the model into its output ceiling at ~1.60 USD an attempt
    and never finished; that has to cost a bounded amount.
    """
    counts = {}
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    pid = json.loads(line).get("prompt_id")
                except json.JSONDecodeError:
                    continue
                if pid:
                    counts[pid] = counts.get(pid, 0) + 1
    return counts


def spent(out_path):
    """(tokens, usd_estimate) already recorded in an answer file."""
    tok = 0
    usd = 0.0
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tok += rec.get("total_tokens") or 0
                usd += rec.get("cost_usd") or 0.0
    return tok, usd


def build_cmd(args, arm):
    """`codex exec` invocation, minus the per-call paths.

    --json in both arms so tool use is auditable. `--cd`,
    `--output-last-message` and the trailing `-` are appended per call by
    call_codex, which owns the temporary directory.
    """
    # Keep the executable selectable: CLI feature schemas can change between
    # releases, and an ablation must use the same harness version in every
    # cell.  Older unit-test fixtures intentionally omit codex_bin.
    codex_bin = getattr(args, "codex_bin", None) or "codex"
    cmd = [codex_bin, "exec",
           "--model", args.model,
           "--json", "--color", "never",
           # the user's own config, project rules and session history must not
           # leak into a benchmark run
           "--ignore-user-config", "--ignore-rules", "--ephemeral",
           "--skip-git-repo-check",
           "--strict-config",
           "-c", f'model_reasoning_effort="{args.effort}"',
           "-c", f'model_reasoning_summary="{args.reasoning_summary}"',
           "-c", 'approval_policy="never"',
           # --strict-config rejects unknown keys, which is how this list was
           # validated locally: `tools.view_image` does not exist in 0.146.0
           # and made every call abort before the model was reached.
           # The top-level mode is the authoritative switch that removes the
           # hosted web-search tool. tools.web_search=false alone proved
           # insufficient in one real 0.146.0 run.
           "-c", 'web_search="disabled"',
           "-c", "tools.web_search=false"]
    for feat in ALWAYS_OFF:
        cmd += ["--disable", feat]
    if arm == "notools":
        cmd += ["--sandbox", "read-only"]
        for feat in NO_TOOL_FEATURES:
            cmd += ["--disable", feat]
    else:
        cmd += ["--sandbox", args.sandbox]
    return cmd


def call_codex(base_cmd, prompt, timeout):
    """One isolated invocation: fresh empty cwd, removed afterwards.

    TMPDIR points into that directory too, so scratch files an agent writes
    cannot be picked up by the next prompt. Hard-coded /tmp paths still
    escape it -- see the independence note in README.md. The final `-` makes
    the CLI read the prompt from stdin; passing it as an argv instead would
    append stdin as a separate <stdin> block.
    """
    workdir = tempfile.mkdtemp(prefix="task_")
    answer_file = Path(workdir) / "last_message.txt"
    cmd = [*base_cmd, "--cd", workdir,
           "--output-last-message", str(answer_file), "-"]
    env = {**os.environ, "TMPDIR": workdir, "TMP": workdir, "TEMP": workdir}
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, input=prompt, cwd=workdir, timeout=timeout,
                              capture_output=True, text=True, env=env)
        raw, err = proc.stdout, proc.stderr
        text, meta = parse_events(raw, answer_file)
        lat = round(time.time() - t0, 2)
        reason = None
        # the exit code is NOT a success signal here: the 401 probe on
        # 2026-07-30 ended in turn.failed and still returned 0
        if meta["failed"]:
            reason = f"error: {str(meta['failed'])[:200]}"
        elif proc.returncode != 0:
            detail = err.strip() or (text or "")
            reason = f"error: exit {proc.returncode} {detail[:200]}"
        elif text is None or not text.strip():
            reason = "error: no answer text in cli output"
        if reason:
            return ({"answer": "", "reasoning": None, "finish_reason": reason,
                     "usage": meta.get("usage"), "model_echo": None,
                     "latency_s": lat}, raw, meta, err)
        return ({"answer": text, "reasoning": None,
                 # the CLI ran to completion; whether the reply is usable is
                 # decided by is_complete_record on the JSON itself
                 "finish_reason": "stop",
                 "usage": meta.get("usage"), "model_echo": None,
                 "latency_s": lat}, raw, meta, err)
    except subprocess.TimeoutExpired as e:
        # keep whatever the stream produced before the kill -- it shows how far
        # the agent got. TimeoutExpired.stdout is bytes even under text=True,
        # so an isinstance(str) guard silently discards every partial.
        partial = e.stdout
        if isinstance(partial, (bytes, bytearray)):
            partial = partial.decode("utf-8", "replace")
        elif not isinstance(partial, str):
            partial = ""
        _, meta = parse_events(partial)
        return ({"answer": "", "reasoning": None,
                 "finish_reason": f"error: timeout after {timeout}s",
                 "usage": meta.get("usage"), "model_echo": None,
                 "latency_s": round(time.time() - t0, 2)}, partial, meta, "")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def verify(args, arm, prompt):
    """Print what the model will actually see. Local, free, no API call.

    `codex debug prompt-input` renders the full input list, so the isolation
    claim and the injected-instruction asymmetry are checkable rather than
    asserted. Also greps for repository leakage.
    """
    workdir = tempfile.mkdtemp(prefix="verify_")
    cmd = ["codex", "debug", "prompt-input",
           "-c", f'model="{args.model}"',
           "-c", f'model_reasoning_effort="{args.effort}"',
           "-c", f'model_reasoning_summary="{args.reasoning_summary}"',
           "-c", "tools.web_search=false"]
    for feat in ALWAYS_OFF:
        cmd += ["--disable", feat]
    if arm == "notools":
        for feat in NO_TOOL_FEATURES:
            cmd += ["--disable", feat]
    cmd.append(prompt)
    try:
        proc = subprocess.run(cmd, cwd=workdir, capture_output=True,
                              text=True, timeout=120)
        if proc.returncode != 0:
            print(proc.stderr[:2000] or proc.stdout[:2000])
            return 1
        try:
            items = json.loads(proc.stdout)
        except json.JSONDecodeError:
            print(proc.stdout[:2000] or proc.stderr[:2000])
            return 1
        print(f"model-visible input: {len(items)} messages")
        for it in items:
            body = it.get("content") or []
            txt = "".join(b.get("text", "") for b in body
                          if isinstance(b, dict))
            head = txt[:100].replace("\n", " / ")
            print(f"  {it.get('role','?'):10s} {len(txt):7d} chars  {head}")
        blob = json.dumps(items)
        leaks = [t for t in ("MasterArbeit", "benchmark_v2", "cases_shard",
                             "llm_cases", "rho_W5_k2", "mean_span_frac",
                             "predictions.csv")
                 if t in blob]
        print("\nleakage check:",
              "CLEAN" if not leaks else f"LEAK -> {leaks}")
        injected = sum(len("".join(b.get("text", "") for b in
                                   (it.get("content") or [])
                                   if isinstance(b, dict)))
                       for it in items if it.get("role") == "developer")
        print(f"codex-injected developer instructions: {injected} chars "
              f"(cannot be removed; the Claude arm has none of these)")
        return 0 if not leaks else 2
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
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help="default: %(default)s (see `codex debug models`)")
    ap.add_argument("--codex-bin", default=os.environ.get("CODEX_BIN", "codex"),
                    help="Codex CLI executable (or set CODEX_BIN). Pin this "
                         "when extending an existing experiment.")
    ap.add_argument("--effort", default=DEFAULT_EFFORT,
                    choices=["low", "medium", "high", "xhigh", "max", "ultra"],
                    help="default: %(default)s -- the model's own default is "
                         "'low', which would handicap the capability screened")
    ap.add_argument("--reasoning-summary", default="detailed",
                    choices=["none", "concise", "detailed", "auto"],
                    help="use 'none' if the CLI rejects the value")
    ap.add_argument("--sandbox", default="workspace-write",
                    choices=["read-only", "workspace-write"],
                    help="tools arm only (default: %(default)s); "
                         "danger-full-access is deliberately not offered")
    ap.add_argument("--no-preamble", action="store_true",
                    help="send the frozen prompt alone, without the shared "
                         "analyst preamble the Claude arm got as a system "
                         "prompt (changes the treatment -- use knowingly)")
    ap.add_argument("--out-dir", default=str(Path.home() / "Dokumente/codex_screen"),
                    help="OUTSIDE the repo; never visible to the agent")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=None,
                    help="seconds per prompt (default: %d notools / %d tools)"
                         % (DEFAULT_TIMEOUT["notools"], DEFAULT_TIMEOUT["tools"]))
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--max-total-tokens", type=int, default=0,
                    help="stop once the file's total reported tokens exceed "
                         "this (0 = no cap); counts earlier runs too. On a "
                         "ChatGPT plan the CLI reports tokens, not dollars")
    ap.add_argument("--max-cost-usd", type=float, default=0.0,
                    help="only meaningful together with --usd-per-mtok-*")
    ap.add_argument("--usd-per-mtok-in", type=float, default=0.0,
                    help="published input price per million tokens; 0 = do "
                         "not estimate cost (no price is guessed here)")
    ap.add_argument("--usd-per-mtok-out", type=float, default=0.0,
                    help="published output price per million tokens")
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
    ap.add_argument("--verify", action="store_true",
                    help="render the model-visible input for the first "
                         "selected prompt and exit; makes no API call")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command and the first prompt, call nothing")
    args = ap.parse_args()

    codex_bin = shutil.which(args.codex_bin)
    if codex_bin is None:
        sys.exit(f"codex executable not found: {args.codex_bin}")
    args.codex_bin = codex_bin
    try:
        cli_version = subprocess.run(
            [codex_bin, "--version"], text=True, capture_output=True,
            timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        sys.exit(f"could not identify Codex CLI {codex_bin}: {exc}")

    out_dir = Path(args.out_dir)
    if REPO in out_dir.resolve().parents or out_dir.resolve() == REPO:
        sys.exit(f"--out-dir must lie outside the repository ({REPO})")
    only = args.ids.split(",") if args.ids else None
    rows = load_prompts(args.prompts, args.condition, args.input_kind,
                        only, args.limit)
    if not rows:
        sys.exit("no prompts selected")

    def full(row):
        if args.no_preamble:
            return row["prompt"]
        return SYSTEM_PROMPT + "\n\n" + row["prompt"]

    # Verification is read-only and should not create answer/log directories.
    if args.verify:
        sys.exit(verify(args, args.arm, full(rows[0])))

    # effort is part of the label: mixing two efforts inside one arm would
    # silently mix two experiments
    label = f"codex-{args.model}_{args.arm}_{args.effort}"
    out_path = Path(args.out) if args.out else out_dir / f"answers_{label}.jsonl"
    log_dir = out_dir / "logs" / args.arm
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    selected_ids = {r["prompt_id"] for r in rows}
    have = done_ids(out_path, args.arm) & selected_ids
    todo = [r for r in rows if r["prompt_id"] not in have]
    burned = []
    tries = attempts_by_id(out_path)
    if args.max_attempts:
        burned = [r["prompt_id"] for r in todo
                  if tries.get(r["prompt_id"], 0) >= args.max_attempts]
        todo = [r for r in todo if r["prompt_id"] not in burned]
    cmd = build_cmd(args, args.arm)
    timeout = args.timeout or DEFAULT_TIMEOUT[args.arm]
    tok_spent, usd_spent = spent(out_path)

    agents_md = Path.home() / ".codex" / "AGENTS.md"
    if agents_md.exists():
        print(f"WARNING: {agents_md} exists and may be injected into every "
              f"prompt. Check with --verify before trusting the run.")

    print(f"arm={args.arm}  model={args.model}  effort={args.effort}  label={label}")
    print(f"cli={cli_version}  binary={codex_bin}")
    print(f"selected={len(rows)}  complete={len(have)}  todo={len(todo)}"
          + (f"  giving_up_on={len(burned)}" if burned else ""))
    if burned:
        # a prompt the model cannot answer is a result, not a gap to hide:
        # every attempt stays in the file and belongs in the write-up
        print(f"  no complete answer after {args.max_attempts} attempts, "
              f"skipped: {', '.join(burned)}")
    print(f"timeout={timeout}s  tokens_so_far={tok_spent}"
          + (f"  cap={args.max_total_tokens}" if args.max_total_tokens else ""))
    print(f"out={out_path}")
    print("cmd=" + " ".join(cmd))
    if args.dry_run:
        print("\n--- first prompt (as sent) ---\n" + full(rows[0])[:2000])
        return
    if not todo:
        print("nothing to do")
        return
    if args.max_total_tokens and tok_spent >= args.max_total_tokens:
        sys.exit(f"token cap already reached ({tok_spent} >= "
                 f"{args.max_total_tokens}); no call made")
    if args.max_cost_usd and usd_spent >= args.max_cost_usd:
        sys.exit(f"cost cap already reached ({usd_spent:.2f} >= "
                 f"{args.max_cost_usd:.2f} USD estimated); no call made")

    n_err = 0
    n_tool_leak = 0
    streak = 0
    waits = 0
    stopped = None
    with open(out_path, "a") as fh:
        for i, r in enumerate(todo, 1):
            exhausted_attempts = False
            while True:  # loops only to serve a reset wait
                attempt = tries.get(r["prompt_id"], 0) + 1
                if args.max_attempts and attempt > args.max_attempts:
                    exhausted_attempts = True
                    print(f"    giving up on {r['prompt_id']}: "
                          f"{tries[r['prompt_id']]} attempts recorded",
                          flush=True)
                    break
                res, raw, meta, err = call_codex(cmd, full(r), timeout)
                if str(res["finish_reason"]).startswith("error"):
                    n_err += 1
                    streak += 1
                else:
                    streak = 0
                if raw:
                    # Failed attempts are evidence; a later retry must not
                    # overwrite their raw streams.
                    name = f"{r['prompt_id']}.attempt-{attempt:03d}.jsonl"
                    (log_dir / name).write_text(raw)
                usd = estimate_usd(meta.get("usage"), args.usd_per_mtok_in,
                                   args.usd_per_mtok_out)
                rec = {"id": r["prompt_id"], "prompt_id": r["prompt_id"],
                       "case_id": r["case_id"], "condition": r["condition"],
                       "input_kind": r["input_kind"], "strategy": r["strategy"],
                       "model": label, "backend": "codex_cli",
                       "arm": args.arm, "cli_model": args.model,
                       "cli_version": cli_version,
                       "cli_binary": codex_bin,
                       "effort": args.effort,
                       "reasoning_effort": args.effort,
                       "tools": "" if args.arm == "notools" else args.sandbox,
                       "thinking": None,
                       "temperature": None, "top_p": None, "top_k": None,
                       "seed": None, "max_tokens": None,
                       "rep": r.get("rep"),
                       "base_prompt_id": r.get("base_prompt_id"),
                       "prompt_sha256": r.get("prompt_sha256"),
                       "num_turns": None,
                       "n_tool_events": meta.get("n_tool_events"),
                       "tool_item_types": meta.get("tool_item_types"),
                       "total_tokens": meta.get("total_tokens"),
                       "thread_id": meta.get("thread_id"),
                       "cost_usd": usd,
                       "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                tries[r["prompt_id"]] = attempt
                tok_spent += meta.get("total_tokens") or 0
                usd_spent += usd or 0.0
                usable = usable_record(rec, args.arm)
                ok = "ok " if usable else "INCOMPLETE"
                print(f"[{i}/{len(todo)}] {r['prompt_id']} {ok} "
                      f"tools={meta.get('n_tool_events')} "
                      f"tok={meta.get('total_tokens')} {res['latency_s']}s "
                      f"total_tok={tok_spent}"
                      + (f" ~{usd_spent:.2f}USD" if usd_spent else ""),
                      flush=True)
                if args.arm == "notools" and meta.get("n_tool_events"):
                    # the arm contrast is the whole experiment; a leak here
                    # invalidates it and must not be discovered later
                    n_tool_leak += 1
                    print(f"    WARNING: notools arm used tools "
                          f"({meta['tool_item_types']}) -- the feature flags "
                          f"did not disable execution", flush=True)
                    # Fail closed. Continuing could silently contaminate more
                    # benchmark cells; the written record remains auditable
                    # and retryable after the configuration is corrected.
                    sys.exit(2)

                lim, resets = limit_info(raw, meta, err, res["finish_reason"])
                # A complete answer must never be repeated merely because the
                # CLI also emitted plan-limit metadata after the turn. Three
                # such successful calls were previously duplicated.
                if (not usable and lim and args.wait_for_reset
                        and waits < args.max_waits):
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

            if exhausted_attempts:
                continue
            if not usable and lim and not args.ignore_usage_limit:
                stopped = (f"plan limit ({lim}). The record stays retryable; "
                           f"rerun the same command after the reset, or use "
                           f"--wait-for-reset to sit it out automatically.")
                break
            if args.max_total_tokens and tok_spent >= args.max_total_tokens:
                stopped = (f"token cap reached ({tok_spent} >= "
                           f"{args.max_total_tokens})")
                break
            if args.max_cost_usd and usd_spent >= args.max_cost_usd:
                stopped = (f"cost cap reached ({usd_spent:.2f} >= "
                           f"{args.max_cost_usd:.2f} USD, estimated)")
                break
            if args.max_consecutive_errors and streak >= args.max_consecutive_errors:
                stopped = (f"{streak} failures in a row -- last: "
                           f"{res['finish_reason'][:120]}")
                break
            if args.sleep:
                time.sleep(args.sleep)

    print(f"done, {n_err} errors, {tok_spent} tokens"
          + (f", ~{usd_spent:.2f} USD estimated" if usd_spent else "")
          + f"; answers in {out_path}")
    if n_tool_leak:
        print(f"\nWARNING: {n_tool_leak} notools calls used tools. The arm "
              f"contrast is not valid until that is fixed.", file=sys.stderr)
    if stopped:
        print(f"\nSTOPPED EARLY: {stopped}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        # A completed record is flushed after every prompt. If Ctrl-C lands
        # during a call, that prompt has no record and resume retries it;
        # earlier completed prompts are never repeated.
        print("\nInterrupted by user. Completed prompts are saved; rerun the "
              "same command to resume.", file=sys.stderr)
        sys.exit(130)
