# Project

Master's thesis benchmark for estimating temporal-graph persistence and related
full-network properties from partial temporal random-walk summaries.

The current active track is the frozen V2 estimator benchmark plus the V2.1 LLM
prompt and evaluation suite.

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

# Current configuration caveat

- Existing benchmark runners default to `config/benchmark.yaml`.
- `config/benchmark_v21.yaml` is not selected automatically by the current
  runners. Compared with `benchmark.yaml`, it adds the lifetime target to
  several evaluation blocks and removes `extra_trees:patterns` from
  `headline_only_pairs`.
- The lifetime add-on currently requires an explicit
  `--config config/benchmark_v21.yaml` argument.
- Never assume a configuration from its filename. Inspect CLI defaults and pass
  `--config ...` explicitly when required.
- Do not merge the two configurations or change which one is authoritative
  without explicit approval. Revisit this only after the lifetime add-on run is
  complete.

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

# LLM output contract

Every final prediction object uses these nine keys:

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
sort, repair, or replace model predictions. The current evaluator converts
numeric values to floats and clips them to `[0, 1]` in its derived evaluation
table; it does not repair monotonicity and records `profile_violation`. Treat
this clipping as an explicit existing evaluation policy. Do not change it
silently. A raw-value scoring policy must be implemented as a named and tested
evaluation change or ablation.

# Resume and answer validation

LLM runs resume by `prompt_id`, and failed attempts must not count as complete.
Answer files remain append-only; retries append a new record.

Known issue: the current `src/run_llm_v2.py` resume logic treats every record
whose `finish_reason` does not start with `error` as complete. This can wrongly
skip truncated (`finish_reason="length"`), empty, unparsable, or
schema-incomplete responses.

Before any full LLM run, fix and test resume validation so that a record counts
as complete only when:

- it is not an error or length-truncated response;
- its answer contains a parseable final JSON object;
- all nine required keys are present.

Do not silently repair invalid values as part of resume validation. Keep
truncated, empty, unparsable, and schema-incomplete records retryable. Add a
focused unit test before changing the runner. Keep the evaluator's retry
selection behavior aligned with the runner.

# Planned model matrix

This section describes intended experiments, not implementation status:

- Qwen3.6-27B: thinking and non-thinking on the cluster
- DeepSeek-R1-Distill-Qwen-32B: continuity baseline on the cluster
- DeepSeek V4 Pro: thinking and non-thinking through NVIDIA NIM
- Mistral Small 4: reasoning `none` and `high` through NVIDIA NIM
- GPT through the AIST endpoint later

Existing SLURM filenames may refer to older Qwen2.5 or pilot runs. Inspect them
before editing and do not assume they already implement this matrix.

# Secrets and external services

- Read API keys only from environment variables.
- Never print, write, expose, or commit API keys or authorization headers.
- Never place secrets in prompts, logs, result files, shell history, or job files.
- Do not start API calls, paid model runs, full benchmark runs, or SLURM jobs
  without explicit permission.

# Before editing

1. Run `git status --short` and inspect the relevant files and nearby tests.
2. Check whether any target path is generated, frozen, ignored, or append-only.
3. Explain the intended changes and any effect on reproducibility or frozen data.
4. Prefer the smallest focused change; avoid unrelated refactors and formatting.
5. Do not commit, push, submit jobs, call APIs, or launch full runs without
   explicit permission.

# After editing

1. Run `python -m py_compile` on every changed Python file.
2. Run the smallest relevant unit or invariant test. Use temporary output paths
   for smoke tests so frozen results are not modified.
3. Run `bash -n` on changed shell or SLURM scripts.
4. Validate changed JSON/JSONL with strict parsing and check expected schemas,
   unique IDs, and row counts where relevant.
5. Run `git diff --check` and show `git diff --stat`.
6. List changed files, tests run, results, and unresolved issues.

# Documentation discipline

Some root and run documentation predates the current merged layout. Verify paths,
environment names, configuration defaults, and commands against the actual tree
before following or updating documentation. When code and documentation disagree,
report the mismatch instead of silently choosing one.
