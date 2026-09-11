# Codex CLI screening run

Sibling of `scripts/cc_screen`, same purpose and the same non-status: this is a
**screen**, not a reported model comparison. It answers one question — does a
frontier agentic model, optionally with code execution, do materially better on
the v2.1 task than the five models already benchmarked? A yes justifies paying
for a proper API run with pinned parameters; a no saves the money. Nothing
produced here belongs in the main leaderboard: the CLI is a product with hidden
defaults, injected instructions, no seed and version drift, so it is not
reproducible.

Verified against **codex-cli 0.146.0** on 2026-07-30.

## Two arms

| arm | how | what it tests |
|---|---|---|
| `notools` | execution features disabled, `--sandbox read-only` | the bare model, closest to a plain chat call |
| `tools` | `--sandbox workspace-write` | the same model, may write and run code |

Both arms get the same frozen prompt text and the same preamble, so tool access
is the only intended difference. That contrast is the interesting one: the v2.1
traces show models recognising the sampling-bias problem but failing to execute
an estimator, so tools are exactly the variable expected to move the result.

## Four ways Codex differs from `claude`, and what each cost

**1. `codex exec` exits 0 even when the turn failed.** The unauthenticated
probe ended in `{"type":"turn.failed","error":{...401...}}` and still returned
`0`. The exit code is therefore worthless as a success signal, and every call
is judged from its event stream instead. `test_turn_failed_is_an_error_even_though_the_cli_exits_zero`
pins this.

**2. On a ChatGPT plan the CLI reports tokens, not dollars.** The budget cap is
token-based (`--max-total-tokens`). `cost_usd` stays `null` unless you pass
`--usd-per-mtok-in` / `--usd-per-mtok-out` with the currently published prices.
No price is hard-coded here on purpose: a guessed number in a results column is
worse than an empty one.

`input_tokens` already contains its cached subset and `output_tokens` already
contains reasoning tokens. The runner therefore sums only input plus output;
the detail fields remain in `usage` but are not charged twice.

**3. There is no `--system-prompt`.** Codex injects its own developer messages
that cannot be removed. Measured with `--verify`, with every tool feature
disabled, **~9.2 kB** of skills and agent-team instructions remain. The shared
analyst preamble is therefore prepended to the user prompt instead. This is a
real asymmetry against the Claude arm and belongs in the write-up as a stated
limitation. Do not compare the two CLIs as if they had identical system
context — compare each arm against its own counterpart.

**4. There is no `--tools ""`.** Tool access is switched off via feature flags,
which is a claim, not a guarantee. So every record stores `n_tool_events` and
`tool_item_types` from the actual stream, and the `notools` arm prints a loud
warning plus a non-zero-exit summary if that count is ever above zero. Verify
rather than assume.

## Before spending anything: `--verify`

```bash
python scripts/codex_screen/run_codex_screen.py --arm notools --verify
```

Renders the complete model-visible input with `codex debug prompt-input` —
local, free, no API call — and greps it for repository leakage (`MasterArbeit`,
`benchmark_v2`, `cases_shard`, `llm_cases`, `rho_W5_k2`, `mean_span_frac`).
Expected output ends in `leakage check: CLEAN`. Run it again for `--arm tools`.

`--strict-config` is on in every call, which is how the option list was
validated for free: `tools.view_image` does not exist in 0.146.0 and aborted
every call before the model was reached. If a future version renames a key, the
run fails immediately and loudly instead of silently ignoring the override.

## Isolation

Every prompt runs in a fresh empty temporary directory, deleted afterwards,
with `TMPDIR` redirected into it, `--cd` pointed at it and
`--ignore-user-config --ignore-rules --ephemeral` set. The agent gets no
repository, no ground truth, no prompt list and no previous answers — with
tools enabled it could otherwise read the truth columns straight out of
`results/` and the screen would be worthless.

`goals` and `memories` are disabled in **both** arms: they carry state between
prompts and would break the independence of the 84 cases. `browser_use`,
`computer_use`, `multi_agent` and `tools.web_search` are off in both arms too —
a web-reachable agent could in principle find the benchmark itself, and
sub-agents make cost unbounded. The capability lost is small next to the
inferences those would invalidate.

Residual leaks worth knowing about: an agent writing to a hard-coded `/tmp`
path escapes the per-prompt `TMPDIR`, and `~/.codex/AGENTS.md` would be
injected into every prompt (the runner warns if it exists; it did not on
2026-07-30).

## Reasoning effort

`gpt-5.6-sol` ships with `default_reasoning_level = "low"`, described in the
model catalogue as *"Fast responses with lighter reasoning"*. Running the screen
there would handicap the exact capability under test and bias the outcome
towards "do not fund the API run". The default here is therefore
`--effort high`, explicit, and part of the output filename so two effort levels
can never end up in one answer file.

If money is tight, cut `n` (`--limit 20`), not effort. Fewer cases cost
statistical power; lower effort costs the validity of the answer. With the
leaderboard gaps at 0.05–0.13 MAE, n = 20–30 still resolves the decision.

## Options

| flag | default | note |
|---|---|---|
| `--arm` | required | `notools` / `tools` |
| `--model` | `gpt-5.6-sol` | see `codex debug models` |
| `--effort` | `high` | goes into the filename |
| `--reasoning-summary` | `detailed` | use `none` if the CLI rejects it |
| `--sandbox` | `workspace-write` | tools arm only; `danger-full-access` is not offered |
| `--condition` / `--input-kind` | `disclosed` / `mask` | the 84-case headline cell |
| `--limit` | 0 (all) | first N prompts |
| `--timeout` | 900 / 2700 s | same walls as the Claude screen |
| `--max-total-tokens` | 0 (off) | counts earlier runs in the same file |
| `--usd-per-mtok-in/out` | 0 (off) | only then is `cost_usd` filled |
| `--max-attempts` | 3 | stop retrying a prompt that never finishes |
| `--max-consecutive-errors` | 3 | stop after N failures in a row |
| `--wait-for-reset` | off | sleep out a plan limit and retry |
| `--max-waits` | 8 | give up after N waits |
| `--verify` | — | render the model-visible input, no API call |
| `--dry-run` | — | print command and first prompt |
| `--out-dir` | `~/Dokumente/codex_screen` | must lie outside the repo |

## Stopping safely

Press `Ctrl-C` at any time and rerun the identical command later. Records are
flushed after every completed prompt and resume is by `prompt_id`. If the
interrupt lands during a model call, that one prompt remains unrecorded and is
retried on restart; its already-consumed tokens may be absent from the ledger.
For zero wasted work, interrupt just after a `[N/M] ...` status line.

1. **Plan limits are detected structurally.** A rate-limit window reported at
   `used_percent >= 100` (relative `resets_in_seconds` and absolute `resets_at`
   both accepted), or a `429` inside an `error` / `turn.failed` event. Prose
   matching is only a fallback. Only error-bearing events are scanned, never
   the model's answer — a reply containing "429 pairs observed" must not stop
   the run, and a test pins that.
2. **Prompts that never finish.** Resume retries an incomplete prompt on every
   restart, so a prompt the model cannot answer is an unbounded bill. In the
   Claude arm three of thirty prompts drove the model into its output ceiling
   at ~1.60 USD an attempt and never produced an answer. `--max-attempts`
   stops trying and names them in the header. Every attempt stays in the answer
   file: **a prompt with no answer is a result** and belongs in the write-up as
   a non-response, not as a silent gap.
3. **Timeouts bill tokens that never reach the ledger.** A killed call may not
   emit its usage event, so `--max-total-tokens` under-counts exactly the
   slowest calls. The partial stream is still saved (`TimeoutExpired.stdout` is
   `bytes` even under `text=True` — an `isinstance(str)` filter silently
   discards it, which is a bug that already happened once in the Claude
   harness and is now pinned by a test).

## Records

`run_llm_v2.py` schema, append-only, one JSON object per line, resume by
`prompt_id` through the shared `is_complete_record()` — truncated, empty,
unparsable and schema-incomplete records stay retryable. Values are never
repaired: no clamping, no sorting of the `rho` profile. Codex-specific extra
columns: `n_tool_events`, `tool_item_types`, `total_tokens`, `thread_id`.

Raw event streams land in
`<out-dir>/logs/<arm>/<prompt_id>.attempt-NNN.jsonl` so every retry and its
tool use stays auditable after the fact:

```bash
grep -l 'MasterArbeit\|benchmark_v2\|cases_shard\|llm_cases' \
    ~/Dokumente/codex_screen/logs/tools/*.jsonl
```

must print nothing.

## Paired input-information ablation

`run_input_ablation.sh` evaluates the frozen 36-case ablation subset (12
cases per walk strategy) with matched case IDs across input formats. Existing
`disclosed/mask` answers are reused; only missing prompt IDs are appended.

```bash
# Full no-tool information ladder
bash scripts/codex_screen/run_input_ablation.sh notools

# Tool comparison on mask, temporal aggregates and recent raw events
bash scripts/codex_screen/run_input_ablation.sh tools
```

The no-tool ladder is `mask`, `nw`, `mask_crawl_full`,
`mask_crawl_temporal`, `mask_crawl_temporal_recent`. The tool arm retains
`mask`, `mask_crawl_temporal`, and `mask_crawl_temporal_recent`. New cells
have separate answer files; rerunning the command resumes by `prompt_id`.

## Paired worked-examples cell

`run_examples_cell.sh` runs `disclosed_examples / mask` on cases that the
matching `disclosed / mask` baseline arm answered completely. The default is
30 cases, stratified over walk strategy and coverage band. The Codex subset is
also nested with the available Claude Code baseline cases, so both screens can
be compared on overlapping cases.

```bash
# Tool-enabled Codex run with worked examples (30 paired cases)
bash scripts/codex_screen/run_examples_cell.sh tools

# Free check of command, case selection and first rendered prompt
CODEX_EXAMPLES_DRY_RUN=1 \
  bash scripts/codex_screen/run_examples_cell.sh tools

# Extend the same resumable output file to all 56 paired cases
CODEX_EXAMPLES_LIMIT=56 \
  bash scripts/codex_screen/run_examples_cell.sh tools
```

The default output is
`~/Dokumente/codex_screen/answers_codex-gpt-5.6-sol_tools_high_disclosed_examples_mask_paired.jsonl`.
Every completed prompt is flushed immediately; rerunning the identical command
continues by `prompt_id`. Set `CODEX_SCREEN_OUT_DIR` to use another output
directory and `CODEX_EXAMPLES_TOKEN_CAP` to change the default 9M-token cap.
