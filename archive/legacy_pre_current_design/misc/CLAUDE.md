# Project

Master's thesis benchmark for estimating temporal-graph persistence and related
full-network properties from partial temporal random-walk summaries.

The current active track is the frozen V2 estimator benchmark plus the V2.1 LLM
prompt and evaluation suite.

# Current state and entry point

The main comparison is being assembled on the frozen 32-graph panel. Targets
and scoring are settled; what still has to be recorded are the open gates
listed at the end of `docs/TARGET_EVALUATION_FREEZE.md` — input contract,
output schema, model configurations, the 96 cases with their walk budget, and
the prompt hash.

Two documents carry the current picture: `docs/TARGET_EVALUATION_FREEZE.md`
for targets, scoring and gates, and `docs/LLM_EVIDENCE.md` for what
has been measured about inputs, contexts and models. Both keep the historical
84-case evidence suite separate from the final 96-case panel; that distinction
matters, because transferring a result across the two is a design judgement
rather than a measurement. `docs/LLM_V21_RUNS.md` carries the model
matrix, the cluster runs and the token-budget ladder.

`LLM_EVIDENCE.md` also lists what has *not* been tested. Several of its numbers
come from exploratory scripts that are not part of the tested code base; it says
so where that applies. Treat its readings as candidate interpretations open to
revision, not as settled results.

Most generated evidence under `results/` is gitignored, so `git status`,
`git ls-files`, and a default `rg --files` inventory say nothing about whether
those files exist. Read the explicit paths.

# Repository map

- `src/`: benchmark generation, walks, prompt creation, model runners, evaluation
- `tests/`: lightweight unit and invariant tests
- `config/`: benchmark and dataset configuration
- `scripts/`: local runners
- `slurm/`: cluster environment and job files
- `docs/`: benchmark design, validation, and run notes
- `data/`: raw and generated benchmark data
- `results/`: generated cases, prompts, answers, metrics, and reports

Work from the repository root unless a command explicitly says otherwise.

# Environment

```bash
source .venv/bin/activate
```

For tests that import modules directly from `src/`, use:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Lightweight inspection, compilation, and unit tests are allowed on a cluster
login node. Benchmark generation, random-walk runs, model inference, and other
heavy computation must use SLURM. Never run heavy jobs on a login node.

# Configuration caveat

Runners default to `config/benchmark.yaml`; `config/benchmark_v21.yaml` is
never selected automatically and differs by adding the lifetime target to
several evaluation blocks and dropping `extra_trees:patterns` from
`headline_only_pairs`. Filenames are not a reliable guide to which
configuration is in force — checking CLI defaults and passing `--config`
explicitly avoids a whole class of silent mistakes.

Which of the two is authoritative is an open question tied to the lifetime
add-on run, so merging them is a decision to make deliberately rather than in
passing.

Both files also accept new presets. Adding one in a scratch location is a cheap
way to run a variation without touching the committed configuration.

# Frozen benchmark artifacts

Treat these as immutable unless the user explicitly requests regeneration:

- V2 benchmark instances, cases, and random walks
- `results/benchmark_v2/results/cases_shard_*.csv.gz`
- `results/benchmark_v2/results/predictions.csv.gz`
- `results/llm_v2/llm_cases.csv`
- `results/llm_v2/llm_examples.csv`
- `results/llm_v2/prompts.jsonl`

The frozen V2.1 LLM suite contains:

- 84 selected cases
- 420 prompts
- 3 access strategies

Do not regenerate walks, cases, examples, or prompts. Do not use `--overwrite`,
`RESET=1`, or destructive cleanup commands on frozen paths without explicit
permission. Do not delete, truncate, rename, or modify generated answers or
results unless explicitly requested.

# Historical V2.1 LLM output contract

The frozen 420-prompt V2.1 suite uses the nine-key contract below. It predates
the target hierarchy in `docs/TARGET_EVALUATION_FREEZE.md`, where mean
occupancy became profile-derived and lifetime moved to robustness reporting.
It is therefore evidence about the *input representation*, and not a template
that carries over to the final experiment by default — the new prompt's
requested keys are one of the open gates.

Every historical V2.1 prediction object uses these nine keys:

- `rho_k2`
- `rho_k3`
- `rho_k4`
- `rho_k5`
- `mean_occupancy`
- `C_one_step`
- `lifetime_mean_over_T`
- `lo90`
- `hi90`

Internal truth-column mapping:

- `rho_k2` -> `rho_W5_k2`
- `rho_k3` -> `rho_W5_k3`
- `rho_k4` -> `rho_W5_k4`
- `rho_k5` -> `rho_W5_k5`
- `mean_occupancy` -> `mean_span_frac`
- `C_one_step` -> `C_one_step`
- `lifetime_mean_over_T` -> `lifetime_mean_over_T`
- `lo90` and `hi90` are the predicted 90% interval bounds for the headline
  `rho_k2` estimate; they are not separate truth columns

All written JSON and JSONL must be strict JSON: use `null`, never `NaN` or
`Infinity`.

Prompts request the monotonic relation
`rho_k2 >= rho_k3 >= rho_k4 >= rho_k5`. Never sort or otherwise repair the
profile in raw model outputs. Preserve and evaluate model inconsistency as an
outcome, and report monotonicity violations separately.

Raw answer JSONL records are append-only and must never be rewritten to clamp,
sort, repair, or replace model predictions. The historical
`src/eval_llm_v2.py` converts numeric values to floats and clips them to
`[0, 1]` in its derived evaluation table; it does not repair monotonicity and
records `profile_violation`. That is a legacy V2.1 reporting policy, not the
frozen final policy. `docs/TARGET_EVALUATION_FREEZE.md` instead treats
out-of-range values as invalid and requires validity plus failure-penalized
loss. Never present the clipped historical metrics as if they already
implemented the frozen final scoring rule.

# Resume and answer validation

LLM runs resume by `prompt_id`, and failed attempts must not count as complete.
Answer files remain append-only; retries append a new record.

Resolved issue: `src/run_llm_v2.py:is_complete_record` now counts a record as
complete only when:

- it is not an error or length-truncated response;
- its answer contains a parseable final JSON object;
- all nine required keys are present.

It is covered by focused tests. Do not regress this behavior: truncated,
empty, unparsable, and schema-incomplete records remain retryable, and invalid
values are never silently repaired as part of resume validation. A record with
`finish_reason="stop"` is therefore not necessarily structurally complete.

# V2.1 evidence model matrix

Five API configurations are complete at 420/420 prompts: DeepSeek V4 Pro
non-thinking, Gemini 3.1 Flash Lite minimal and high reasoning, and Mistral
Small 4 reasoning `none` and `high`.

A partial Open-Weights snapshot (Qwen3.6-27B thinking/non-thinking,
DeepSeek-R1-Distill-Qwen-32B) sits in `results/llm_v21/cluster_snapshot/` and
uses the same frozen prompts. Its open prompts are largely token-limit
truncations rather than substantive failures, so complete-case and
failure-penalized numbers diverge there and are best read together.

Claude Code Opus and Codex CLI runs carry a harness prompt that is not part of
the frozen prompt and cannot be version-pinned, which is why they are kept
apart from the API rows. That is a statement about what claim they support,
not a verdict on their results — they are the strongest results in the suite.
Details in `docs/LLM_EVIDENCE.md`.

Existing SLURM filenames may refer to older Qwen2.5 or pilot runs. Inspect them
before editing and do not infer current experiment status from filenames.

# Secrets and external services

- Read API keys only from environment variables.
- Never print, write, expose, or commit API keys or authorization headers.
- Never place secrets in prompts, logs, result files, shell history, or job files.
- Do not start API calls, paid model runs, full benchmark runs, or SLURM jobs
  without explicit permission.

# Working on the code

Before changing something, it helps to know whether the target path is
generated, frozen, ignored, or append-only — that determines whether an edit is
routine or destroys evidence. Small focused changes are easier to review than
bundled refactors.

Committing, pushing, submitting jobs, calling paid APIs, or launching full runs
are actions the user decides on, not side effects of a task.

After a change, the usual checks are compilation for Python, `bash -n` for
shell and SLURM files, strict parsing plus a schema and row-count sanity check
for JSON/JSONL, and the smallest test that would actually catch a regression.
Smoke runs belong in temporary output paths. `git diff --stat` and a short note
on what changed, what was run, and what is still open make the result
reviewable.

Analysis that only reads data is cheap and safe; the walk and estimator
pipeline on the 32-graph panel, for instance, runs in under a minute locally.
Gauging the actual cost before reaching for SLURM is worth the few seconds.

# Documentation discipline

Some root and run documentation predates the current merged layout. Verify paths,
environment names, configuration defaults, and commands against the actual tree
before following or updating documentation. When code and documentation disagree,
report the mismatch instead of silently choosing one.
