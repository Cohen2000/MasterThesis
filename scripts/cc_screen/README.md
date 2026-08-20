# Claude Code screening run (with / without tools)

A **screen**, not a reported model comparison. It answers one question before
any money is spent: does a stronger model — optionally with code execution —
do materially better on this task than the five models already benchmarked?

- better → a proper API run with pinned parameters is worth paying for
- not better → the API budget is saved and the negative result stands

Nothing produced here goes into the main leaderboard. The harness is a product
with hidden defaults, no seed and version drift, so it is not reproducible.
That is acceptable for a decision aid and not acceptable for a reported number.

## The two arms

Identical in everything except tool availability — same frozen prompt text,
same neutral system prompt, same model, same cell:

| arm | flag | what it is |
|---|---|---|
| `notools` | `--tools ""` | bare model, closest to a plain chat call |
| `tools` | `--tools "Bash,Read,Write,…"` | same model, may write and run code |

The contrast is the scientifically interesting part. The v2.1 traces show the
models *recognise* the sampling-bias problem (74–85 % name it) and some even
name the right estimator family, but they do not execute a correction of the
right magnitude. Tool access is therefore exactly the variable expected to
move the result:

- **tools ≈ notools** → the summary is genuinely information-poor at this
  budget. That bounds *any* method and strengthens the baseline story.
- **tools ≫ notools** → the information is there and LLMs simply cannot do the
  arithmetic unaided. The finding becomes "this task needs tools, not more
  reasoning".

## Which prompts

The design target is `condition=disclosed`, `input_kind=mask` → **84
prompts**, the headline cell of the leaderboard. The locally preserved screen
was deliberately stopped at 30 selected cases per arm because of product-plan
limits: `results/cc_screen_snapshot/README.md` is the authoritative status
record (Tools 30/30 parseable, NoTools 28/30). Chosen so each completed screen
case is directly paired against the existing models on the same prompt, and
because `disclosed` is the realistic condition: the sampling mechanism is
described, no target information is given.

84 prompts × 2 arms = 168 invocations.

## What "better" means

Same cell, from `results/llm_v21/eval/llm_leaderboard.md` (MAE on `rho_k2`):

| | MAE |
|---|---:|
| `baseline_et_stacked` (trained) | 0.083 |
| `baseline_mask_mle` (analytic) | 0.146 |
| `baseline_floor` | 0.191 |
| **constant predictor (LOO median)** | **0.186** |
| gemini-3.1-flash-lite_minimal | 0.217 |
| deepseek-v4-pro_nothink | 0.256 |
| mistral-small-4_high | 0.307 |

Decision thresholds:

- **MAE < 0.186** → first model to beat the constant, i.e. first model to show
  *any* skill. Fund the API run.
- **MAE ≈ 0.08 – 0.15** → competitive with the baselines. Big result, fund it.
- **MAE > 0.21** → no better than what is already benchmarked. Save the money
  and report the negative result as it stands.

## Isolation

Every prompt runs in a **fresh empty temporary directory** that is deleted
afterwards. The agent sees no repository, no ground truth, no prompt list and
no previous answers.

This matters: with tools enabled, an agent that could reach `results/` would
read the truth columns straight out of the benchmark and the screen would be
worthless. The script refuses an `--out-dir` inside the repository, and both
arms run with `--output-format stream-json`, so the full message stream —
every tool call with its arguments — is kept under `<out-dir>/logs/<arm>/`.

Audit a finished arm:

```bash
grep -ho '"file_path":"[^"]*"\|"command":"[^"]*"' \
     ~/Dokumente/cc_screen/logs/tools/*.jsonl | sort -u | less
# and the hard check: nothing may point at the benchmark
grep -l 'MasterArbeit\|benchmark_v2\|cases_shard\|llm_cases' \
     ~/Dokumente/cc_screen/logs/tools/*.jsonl
```

The second command must print nothing. If it prints a file, that prompt's
record is contaminated and has to be dropped.

**Independence caveat.** `TMPDIR` is redirected into the per-prompt directory,
but an agent that hard-codes `/tmp/foo.py` still writes outside it, so scratch
files from prompt *n* can in principle be found by prompt *n+1*. Observed in
practice. This does not leak ground truth — each prompt has different data, and
a stale script would hurt rather than help — but strict per-prompt independence
is not guaranteed and should be stated if the arm is ever reported.

## Running it

Prerequisite: `claude` on `PATH`, logged in.

```bash
cd /home/albert/Dokumente/MasterArbeit
source .venv/bin/activate
```

**1. Look before you call anything.** `--dry-run` makes no model call:

```bash
python scripts/cc_screen/run_cc_screen.py --arm notools --dry-run
```

**2. Two-prompt smoke per arm.** Check that answers parse and see the real cost:

```bash
python scripts/cc_screen/run_cc_screen.py --arm notools --limit 2
python scripts/cc_screen/run_cc_screen.py --arm tools   --limit 2
```

Every line prints `ok` or `INCOMPLETE`. `INCOMPLETE` means the reply had no
parseable final JSON with all nine keys — that prompt stays retryable and is
picked up again on the next call. Then extrapolate the bill:

```bash
python - <<'PY'
import json, glob
for f in sorted(glob.glob("/home/albert/Dokumente/cc_screen/answers_*.jsonl")):
    rs = [json.loads(l) for l in open(f)]
    c = [r["cost_usd"] for r in rs if r.get("cost_usd")]
    t = [r["num_turns"] for r in rs if r.get("num_turns")]
    print(f, len(rs), "records",
          f"avg_cost={sum(c)/len(c):.4f}" if c else "cost=n/a",
          f"-> 84 prompts ≈ {84*sum(c)/len(c):.2f}" if c else "",
          f"avg_turns={sum(t)/len(t):.1f}" if t else "")
PY
```

**3. Full arms.** Resume is automatic, so an interrupted run is just restarted
with the same command:

```bash
python scripts/cc_screen/run_cc_screen.py --arm notools
python scripts/cc_screen/run_cc_screen.py --arm tools
```

Run `notools` first. It is the cheap, fast arm and on its own it decides the
API-funding question; `tools` costs several times as much and answers the
separate scientific question.

Measured over the first 10 `notools` prompts (Opus, 2026-07-28):

| arm | turns | s / prompt | USD / prompt | 84 prompts |
|---|---:|---:|---:|---:|
| `notools` | 1 | 79–806 | 0.15–1.62 | ~10 h, ~45 USD |
| `tools` | 21+ | 600–1200 | ~1.5–3 | ~14–28 h, ~125–250 USD |

The spread is the point, not the mean. Cost is set almost entirely by output
tokens, and those ranged from 5 223 to 64 000 on the *same* kind of prompt —
a factor of 12. The 64 000-token run hit the model's output ceiling and
returned no usable answer at all, so a single prompt can cost more than five
others together and still count as a failure. Budget from the tail, not the
average.

### Budget on a subscription

The Pro five-hour window ran out after roughly 5 USD of Opus usage, i.e. about
8–12 prompts. Two consequences:

- **Prefer fewer prompts over a cheaper configuration.** `--limit 30` keeps
  every case paired against the existing baselines and costs about a third.
  Lowering `--effort` would also cut the bill, but it handicaps exactly the
  capability under test and biases the screen toward the negative answer —
  the one that says "do not fund the API run". Cutting `n` costs statistical
  power instead, which is the honest trade: at `n = 30` the decision gaps here
  (0.05–0.13 MAE) are still comfortably resolvable.
- **Let it sit out the window.** `--wait-for-reset` sleeps until the reported
  reset and retries, so 30 prompts finish unattended across ~3 windows.

The `tools` arm may take many turns per prompt — one observed run fitted a
nine-parameter latent-variable model by Nelder-Mead MLE over an 80 000-particle
simulator. Run both in `tmux`/`screen`.

### Resume both paired examples arms (tools first)

The examples wrapper asks for up to 30 paired cases per arm. The current
notools baseline has only 28 complete cases, so that arm can reach at most 28
without first repairing its two baseline non-responses; tools can reach 30.
Fresh cases run before earlier failures. Up to four passes then retry remaining
failures at the end, with at most five real model attempts per prompt. The fifth
attempt keeps the single remaining notools case eligible on the next resume.

Run both arms sequentially with one command. It resumes `tools` first and then
continues with the remaining `notools` cases. The wrapper defaults to a
50 USD cumulative cap for notools and 70 USD for tools; each cap applies only
to its respective answer file:

```bash
bash scripts/cc_screen/run_examples_both.sh
```

The defaults permit up to 120 USD of reported aggregate usage across both
files. Already recorded usage counts toward them. Override them with
`CC_EXAMPLES_NOTOOLS_COST_CAP` and `CC_EXAMPLES_TOOLS_COST_CAP` if desired.
The per-prompt budget defaults to 4 USD because some of the remaining cases
repeatedly exhausted the former 2 USD allowance; it can be overridden with
`CC_EXAMPLES_PER_PROMPT_CAP`.
When the CLI reports a plan reset, the runner waits until that reset and
resumes the same prompt. It does not poll or repeatedly invoke the service
while a limit is active. A plan-limit refusal is recorded for audit but does
not consume a prompt's real-attempt allowance.

Relevant overrides are `CC_EXAMPLES_LIMIT`, `CC_EXAMPLES_MAX_ATTEMPTS`,
`CC_EXAMPLES_RETRY_PASSES`, `CC_EXAMPLES_MAX_WAITS`, and
`CC_EXAMPLES_PER_PROMPT_CAP`.

**4. Evaluate against the existing models**, into a separate directory so the
frozen evaluation stays untouched:

```bash
python src/eval_llm_v2.py \
    --answers "results/llm_v21/eval_input/answers_*.jsonl" \
              "$HOME/Dokumente/cc_screen/answers_claude-code-*.jsonl" \
    --out-dir results/llm_v21/eval_cc_screen
```

Read `results/llm_v21/eval_cc_screen/llm_leaderboard.md`, section
`condition=disclosed, input=mask`.

## Useful options

| option | default | note |
|---|---|---|
| `--model` | `opus` | alias or full model name |
| `--effort` | unset | `low … max`; leave unset unless comparing effort |
| `--tools` | `Bash,Read,Write,Edit,Glob,Grep` | tools arm only |
| `--permission-mode` | `bypassPermissions` | tools arm only — a non-interactive run stalls in any asking mode |
| `--timeout` | 1500 notools / 3300 tools | seconds per prompt |
| `--max-cost-usd` | `0` (off) | stops once the file's total reported cost passes it; counts earlier runs |
| `--max-consecutive-errors` | `3` | stops after N failures in a row |
| `--max-attempts` | `3` | stops retrying one prompt that never completes (0 = forever) |
| `--wait-for-reset` | off | on plan exhaustion sleep until the reported reset and retry, instead of stopping |
| `--max-waits` | `8` | give up after this many reset waits |
| `--per-prompt-budget-usd` | `0` (off) | passes `--max-budget-usd` to each call; bounds one runaway prompt |
| `--ignore-usage-limit` | off | keep going instead of stopping on plan exhaustion |

`--effort` becomes part of the answer-file name (`…_notools_medium.jsonl`).
Two effort levels are two experiments and must not land in one file.
| `--condition` / `--input-kind` | `disclosed` / `mask` | other frozen cells |
| `--ids` | – | comma-separated `prompt_id`s |

## Stopping safely

Three guards, all of which exit with status `2` and print `STOPPED EARLY: …`.
In every case the unfinished prompt stays retryable and the same command
resumes exactly where it left off — nothing is lost and nothing is paid twice.

1. **Plan / usage limit.** Exhaustion arrives looking like an ordinary reply,
   so without this the run would burn through every remaining prompt writing
   error records. Detected structurally, from the stream itself:

   - a `rate_limit_event` whose `status` is `rejected`, which also carries
     `resetsAt` — the unix timestamp `--wait-for-reset` sleeps until;
   - a result object with `api_error_status: 429`.

   Prose matching is only a fallback. It has to be: the message actually seen
   was *"You've hit your monthly spend limit"*, which matches none of the
   phrases one would think to grep for. The first version of this guard did
   exactly that and let the run continue into three dead calls.

   With `--wait-for-reset` the run sleeps to the reset and retries the same
   prompt; a refused call does not count toward the consecutive-error streak,
   since nothing about it is a model failure.
2. **Cost cap.** `--max-cost-usd 40` stops as soon as the file's accumulated
   `cost_usd` passes 40 — across restarts, not just this session.
3. **Consecutive failures.** Three errors in a row (dropped connection,
   repeated timeouts) end the run rather than grinding through the rest.

A failed call is still parsed for its reported cost before the exit code is
acted on. The run that exhausted the plan had spent 1.62 USD on a 64 000-token
generation by the time it was refused; discarding that meta would have hidden
the single most expensive call of the run from `--max-cost-usd`.

**Prompts that never finish.** Resume retries an incomplete prompt on every
restart, so a prompt the model cannot answer is an unbounded bill. One in this
cell drove the model to its 64 000-token output ceiling twice, ~900 s and over
1.50 USD per attempt. `--max-attempts` stops trying it and names it in the
header. Every attempt stays in the answer file: a prompt with no answer is a
result and belongs in the write-up as a non-response, not as a silent gap.

**Timeouts cost money that is not in the ledger.** A killed call never emits
its result object, so no `cost_usd` is recorded although the tokens were
billed. `--max-cost-usd` therefore under-counts exactly the slowest calls.
`--per-prompt-budget-usd` is the fix that keeps the books straight: the CLI
stops itself and still reports what it spent.

A per-prompt timeout is not a run-level failure: the record is written as
`error: timeout after Ns`, counts as incomplete, and is retried later. The
partial message stream is kept in the log so it is visible how far the agent
got. The `tools` arm genuinely needs its larger budget — one observed prompt
ran a 590 s optimisation as a single step inside a longer session.

Tighter sandbox for the tools arm, if you would rather not grant blanket
permission: add `--allowedTools "Bash(python3 *) Write Read"`. The empty
temporary cwd is the main control, but this narrows Bash as well.

## Limitations to state if this is ever mentioned in the thesis

1. **Not reproducible.** No seed, no pinned parameters, hidden system-level
   defaults, model version can change between runs. Screening only.
2. **The `tools` arm is not the same experiment** as the API models. It has
   capabilities they do not have. Never mix it into a model ranking; report it
   as its own arm with its own claim ("tool-augmented upper bound").
3. **Single sample per prompt.** No variance estimate. With `n = 84` paired
   cases a difference of roughly 0.03–0.05 MAE is detectable; smaller gaps are
   not decision-relevant here.
4. **Subscription terms.** Whether a Pro subscription covers batch benchmark
   inference is worth checking rather than assuming — an API run with an
   explicit key avoids the question entirely.

## Contract compliance

- answer files are **append-only**; retries append a new record
- resume uses the shared `run_llm_v2.is_complete_record`: error, truncated,
  empty, unparsable and schema-incomplete records stay retryable
- no value is repaired, sorted or clamped anywhere in this script
- no frozen artifact is read for anything but the prompt text, and none is
  written
