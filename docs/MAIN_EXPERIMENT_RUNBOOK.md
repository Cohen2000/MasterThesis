# Main experiment offline runbook

Authority: supplied freeze, [full extraction](MAIN_FREEZE_SOURCE.txt) and
[implementation decisions](MAIN_EXPERIMENT_IMPLEMENTATION.md). The user explicitly
confirmed synchronous AD contacts with one undirected event per dyad/round.

## Local execution and resume

From the repository root, using its existing `.venv`:

```bash
.venv/bin/python scripts/run_main_offline.py --out results/main_experiment/frozen_20260916
.venv/bin/python scripts/verify_main_offline.py --run results/main_experiment/frozen_20260916
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

The compact, reviewable acceptance evidence is derived from a finished run. It only
reformats existing artifacts; it samples nothing, trains nothing and calls no model,
and it refuses to run against a directory that is not marked `offline_ready`:

```bash
.venv/bin/python scripts/build_main_acceptance_evidence.py \
    --run results/main_experiment/frozen_20260916 \
    --out results/main_experiment/evidence
```

The resulting [offline acceptance report](../results/main_experiment/ACCEPTANCE.md)
and the `evidence/` directory are the only parts of `results/main_experiment/` that
are tracked in git; bulk artifacts stay local.

## Qwen execution on bwUniCluster

Everything below runs on allocated compute nodes; the login node is used only for
data transfer and package installation.

Workspace layout the job scripts assume:

```
$WS/src/main_experiment/     the package, uploaded from this repository
$WS/mainexp/                 scripts, run/, answers/, logs/
$WS/mainexp/run/             requests.jsonl and observations/sample from the offline run
$WS/models/Qwen3.6-35B-A3B/  the pinned snapshot
$WS/venv_mainexp/            the pinned environment
```

`run_qwen_batch.py` finds the package relative to its own location, so the package
must sit at `$WS/src` and the script at `$WS/mainexp`.

```bash
# once: environment and pinned model snapshot
bash cluster/install_vllm.sh                 # venv on module Python 3.12, vLLM 0.29.0
bash cluster/fetch_model.sh                  # pinned revision, resumable, curl-based
python cluster/verify_model.py               # shard index vs. what is on disk

# configuration probe on development inputs only
sbatch cluster/qwen_probe.sbatch

# one generation pass per submission, four shards
# arguments, not --export: see the note below
sbatch --array=0-3 qwen_main.sbatch thinking    1 4 16 16
sbatch --array=0-3 qwen_main.sbatch nonthinking 1 4 16 16
bash submit_all.sh          # all six passes at once

# collect and evaluate
python scripts/collect_qwen_answers.py --run <run> --answers <answers> --out responses.jsonl
python scripts/evaluate_main_responses.py --run <run> --responses responses.jsonl --out <eval>
```

Three things about this cluster that cost time to find and are easy to hit again:

* The modulefiles are Lmod `.lua`. A non-interactive shell loads classic Tcl
  modules instead and fails with `Magic cookie '#%Module' missing`, silently
  leaving the system Python 3.9 in place, which then resolves cp39 wheels. Every
  script therefore starts with `#!/bin/bash -l`.
* The Qwen Triton kernels are JIT-compiled and need `nvcc`. The cluster modules
  stop at CUDA 12.8 while torch here is cu130; the matching 13.4 toolchain ships
  inside the venv as `nvidia-cuda-nvcc`, so the jobs set `CUDA_HOME` to it.
  Without that the engine dies at warmup.
* Long wall times sit behind shorter ones at equal priority and do not get
  backfilled. Jobs use eight hours and rely on resume, which costs one model
  reload per pass.
* Passing variables with `--export` makes Slurm build a fresh login environment
  for the job, which fails here with `user env retrieval failed requeued held`.
  The jobs take positional arguments and inherit the environment instead.

Resume is per request id, and each generation index writes into its own directory.
A repeat written into another repeat's directory would see every id as done and
silently collapse the repeat-to-repeat variation the study measures.

## Baseline revision

The learned baseline is fitted on a frozen synthetic pool in addition to the real
sources, and two Beta-mixture correctors are checked on separate development data.
This never writes to `results/main_experiment/frozen_20260916`, and it makes no LLM
call, no API call and no paid job.

```bash
.venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_20260916 --stage pool
.venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_20260916 --stage train
.venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_20260916 --stage dev
```

`--stage all` runs the three in order. Every stage resumes: the pool skips graphs
whose observation file already exists, training reuses hash-verified model
manifests, and the development stage reuses the per-observation record table.
The pool takes about 450 s and the pooled fit about 250 s on one local core, so no
scheduler is involved and nothing runs on a login node.

Results and their limits are in the
[development report](../results/baseline_revision_20260916/DEVELOPMENT_REPORT.md),
with `development_summary.csv` as the machine-readable table and
`pool_definition.json` as the frozen pool. `freeze_unchanged.json` records the
re-derivation showing that the frozen observation blocks, prompts and planned calls
are byte-identical under the revision code. Pool graphs, calibration checkpoints,
fitted models and the per-observation record table stay local.

Run the identical pipeline command after interruption. Atomic per-source graphs,
calibration-prefix files, validation batches and observation files are reused.
The directory is locked against simultaneous writers. Input/code/configuration
and environment changes reject resume and require a new output directory;
no already-produced results are silently appended or duplicated. A killed stage
can be recomputed, while completed stages stay available. Models are local trusted
pickle files with verified SHA-256, never load untrusted model files.

A clean environment can be created from the run's `environment.lock.txt`.
A C++17 compiler is required for the small weighted-walk kernel. BLAS and
ExtraTrees each use one CPU thread; no GPU or inference dependency is needed.
The main pipeline does not access the network. On first setup only, run
`.venv/bin/python scripts/fetch_main_tokenizers.py` to obtain the pinned tokenizer
assets; it downloads no model weights. The Jinja dependency is needed for Qwen's
actual chat template. The empirical registry is `config/datasets.yaml`; all 16
listed training sources must exist in `data/raw`. CNS additionally requires the
original `bt_symmetric.csv` from Figshare v1 file 14000795. Its original MD5 and
SHA-256, cleaning counts and equality with the legacy export are checked.

The DOCX can be re-extracted, preserving mathematical hats/subscripts, with:

```bash
python3 scripts/extract_main_freeze.py '/home/albert/Downloads/Hauptexperiment_Freeze_Input_Auswertung_Modelle (2).docx' --out /tmp/freeze_check.txt
```

## Artifacts

- `graphs/*/manifest.json`, `canonical.csv`, `graph.npz`: canonical source export,
  raw/canonical hashes, transformations, fixed horizons and ground truth.
- `calibration/*/`: 256-path calibration, timing pilot, independent 1024/4096
  validation paths, frozen L, budget deviations and matching flags.
- `observations/{sample,training}/`: serialized observation blocks and exact prompts.
  Internal evaluation metadata is separate and never rendered into prompts/features.
- `models/*/`: six LOSO models and one all-real training model for synthetic tests,
  independent source labels, medians, weights, 88-feature order and checksums.
- `data_summary.csv`, `budget_summary.csv`, `baseline_observations.csv`,
  `baseline_summary.csv`, `prompt_sizes.csv`, `observation_status.csv`: compact tables.
- `requests.jsonl`: planned future requests, all with dispatch disabled. IDs and
  seeds are metadata; three repeats have identical messages. Empty observations
  would be `skipped_empty`, not failed requests.
- `seed_manifest.json`, `environment.lock.txt`, `preparation_inputs.json`,
  `tokenizer_manifest.json`, `checksums.json`, `report.json`: reproducibility evidence.

Qwen counts use the actual pinned chat template in both modes and are cross-checked
against explicit rendering. DeepSeek counts message texts; Sol uses o200k as a
proxy. Their provider framing remains a later technical release condition, as in
the freeze. No prompt may be truncated to resolve a size violation.

## Later response evaluation, without calling a model

```bash
.venv/bin/python scripts/evaluate_main_responses.py --run results/main_experiment/frozen_20260916 --out results/main_experiment/no_llm_execution
# Later, only after separately authorized execution:
.venv/bin/python scripts/evaluate_main_responses.py --run results/main_experiment/frozen_20260916 --responses /path/to/real_responses.jsonl --out results/main_experiment/real_evaluation
```

Input: exactly one record per logical request ID, with `started`, `terminal`,
`final_text`, `reasoning` (if available), `finish_reason`, `limit_hit`,
`technical_error`, `refusal`, provider model/fingerprint/usage and attempt metadata.
The parser reads only `final_text`. Nonterminal/ambiguous requests remain pending;
not-started requests get no fallback and no error estimate. Raw bytes are retained.
Duplicate/unknown IDs fail. Simulated responses must carry `mock:true` and require
`--mock` plus a separate output directory containing `mock` in its name.
Errors are computed per answer, then repeat/seed/source averages and paired MCSE.
Real and four synthetic conditions remain separate. Error tables include validity,
replacement and empty fractions; a conditional valid-only comparison uses the
same cases on both sides and does not replace the main result.

## CPU cluster option

`ssh3` was absent as executable, alias and function in this session. The existing
`ssh uc3` connection worked. Read-only checks found SLURM, account `tu`, CPU
partition `cpu`, maximum 72-hour walltime and 192 CPUs/node. An initial `sinfo`
query was denied; `scontrol show partition` succeeded. No compute ran on login
nodes and no cluster job was submitted: the complete local calibration/data run
was only a few minutes, so transferring data and setting up a second environment
would not improve this run.

If later needed, copy only this implementation, configs, the 16 data files,
CNS original and tokenizer artifacts to a separate cluster directory using the
existing `uc3` host. Install the exact environment lock in that directory first.
Check current partitions/account again; then from that directory:

```bash
sbatch scripts/run_main_offline_cpu.sbatch
# Same command resumes the same output location after interruption.
# Optional single source, still within the one-CPU job:
sbatch scripts/run_main_offline_cpu.sbatch --graph snap_mathoverflow
```

The provided job requests one CPU, 16 GB and two hours, no GPU. Set
`MAIN_OFFLINE_PYTHON` and `MAIN_OFFLINE_OUT` if using a different environment/path.
Do not run several writers against one output directory. Source-level jobs may
use separate directories, but no uncontrolled nested parallelism is enabled.

## Scope of readiness

This prepares the full offline study. It does not implement a production inference
transport or claim that API JSON/reasoning separation, streaming recovery, provider
usage or Qwen GPU serving has been tested. Pure retry/watchdog/reservation policies
are tested with mocks. Future execution still needs an authorized dispatcher that
persists attempts/stream fragments, reconciles uncertain submissions, respects the
specified worker limits and Sol batches <=64, and applies the frozen policies.
No API credentials have been read or printed; no LLM inference or smoke test ran.
