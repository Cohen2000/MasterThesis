# How persistent is a network you only partly see?

Temporal networks (who interacts with whom, and when) are rarely observed completely. This project asks how well the **persistence** of a network can be recovered from a small sample. Persistence here means the share of interacting pairs that are active in at least *k* of five time windows (`rho_k`, with `rho_2` as the main target). Classical estimators, a supervised model (ExtraTrees) and language models (Qwen, DeepSeek, GPT) all get the same sampled data.

## Setup

- **32 graphs:** 12 real interaction datasets, one matched surrogate per real source (12) and 8 synthetic graphs.
- **4 sampling designs, each seeing about 10% of the activity:**
  - **R:** a random panel of nodes
  - **S:** an interaction-following random walk, with its crawl log
  - **H:** a node panel that only sees the last 60% of history
  - **B:** random thinning of individual events
- Each design is drawn 3 times per graph; estimators and models receive only the released design information (e.g. panel size, crawl log, thinning probability).
- **Four estimators per design:** plugin, training median, a shared maximum-likelihood model (MLE) and ExtraTrees, next to Qwen 3.6 and the paid APIs. The reference is plugin for R and the MLE otherwise.
- **Metric:** absolute error of `rho_2`, averaged per graph, then equally over graphs (MAE_2). The 12 real sources are the main analysis; surrogates and synthetic graphs are separate blocks, and originals are compared with their surrogates as pairs.
- **LLM answers:** only answers whose final output is one valid JSON profile are scored. Nothing is repaired.

Full method: [protocol](docs/PROTOCOL_PANEL888_20260921.md).

## Results

MAE_2 over the 12 real sources (lower is better). The reference estimator of each design is in bold; ExtraTrees is the production fit (11-fit mean ± SD in [variability](docs/results/final_20260928/VARIABILITY.md)).

| Design | Plugin | Median | MLE | ExtraTrees | Qwen 3.6 (thinking) | DeepSeek Flash | GPT-6 Sol | GPT-6 Sol + Python |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| R | **0.027** | 0.232 | 0.037 | 0.028 | 0.027 | 0.027 | 0.027 | 0.027 |
| S | 0.301 | 0.232 | **0.069** | 0.050 | 0.181 | 0.087 | 0.084 | 0.081 |
| H | 0.064 | 0.232 | **0.071** | 0.039 | 0.160 | 0.118 | 0.054 | 0.061 |
| B | 0.164 | 0.232 | **0.076** | 0.079 | 0.220 | 0.195 | 0.108 | 0.151 |

Model repeats: Qwen 3, DeepSeek 1, GPT-6 Sol 3, GPT-6 Sol + Python 3. Recorded API spend (upper bounds, all runs): DeepSeek USD 9.44, GPT-6 Sol USD 28.13, GPT-6 Sol + Python USD 95.76. Earlier results (8 real sources, S_obs ablation, design estimator) remain under `docs/results/` as history.

**Where the details are:**
- [Main results](docs/results/final_20260928/MAIN_RESULTS.md): all blocks, paired original–surrogate comparisons, per-source tables
- [Run report](docs/results/final_20260928/REPORT.md): what ran, checks, diagnostics, API extension
- [API results](docs/results/final_20260928/api/API_RESULTS.md): all API models per block, paired tests, spend
- [Variability](docs/results/final_20260928/VARIABILITY.md), [history truncation](docs/results/final_20260928/HISTORY.md), [walk diagnostics](docs/results/final_20260928/WALK.md)
- Historical: [v11 main results](docs/results/panel888_v11_main_20260923/MAIN_RESULTS.md), [v11 API results](docs/results/api_v11_20260923/API_RESULTS.md), [hidden panel size](docs/results/panel888_v10_RH_panel_release/REPORT.md), [walk gate](docs/results/panel888_v10_walk_gate_20260923/WALK_GATE.md)

## Repository

| Path | Contents |
|---|---|
| `config/` | Study settings, dataset registry and the exact LLM prompts |
| `src/main_experiment/` | Sampling, observations, estimators, ExtraTrees features, evaluation |
| `src/census.py`, `src/dataset_census.py` | Audited raw-data parsers |
| `src/v11_ext/` | Added sources and surrogates, ExtraTrees replicates, diagnostics, final report, SLURM orchestration |
| `scripts/` | Pipeline entry points (see below) |
| `cluster/` | Slurm jobs for bwUniCluster (CPU stages, Qwen on H100) |
| `tests/` | Design invariants, estimators, evaluation, API runner, extension |
| `docs/` | Protocol, API runbook and all committed result tables |

Pipeline, in order (cluster jobs for the CPU and GPU stages are in `cluster/`):

1. `prepare_study.py`: graphs, budgets, sampler draws and prompts (`fetch_tokenizers.py` first)
2. `build_v10_pool.py`: synthetic training pool; `build_v10_et.py cache|select|train`: ExtraTrees
3. `run_qwen_engine.py`: Qwen answers on H100 (via `cluster/submit_production.sh`)
4. `rh_panel_sensitivity.py`: v11 release of the R/H panel size (observations, ET, Qwen, comparison)
5. `build_v10_results.py`: v10 tables and the v11 composition; `api_runner.py`, `evaluate_api.py`: paid API runs ([runbook](docs/RUNBOOK_PANEL888.md))
6. `v11_ext.py --dry-run | --status | --submit`: added sources, surrogates, ET replicates, diagnostics and final tables, resumable on uc3
7. `audit_v10_walk.py`, `confirm_v10_walk.py`, `report_v10_walk_gate.py`: walk-gate check; `verify_*.py`: archive and ET verification
8. `api_cycle.sh`: drives one paid API run (GPT Batch, DeepSeek off-peak); commands in the [run report](docs/results/final_20260928/REPORT.md)

Raw data, generated artifacts and raw model answers stay local; their checksums are committed with the results. See [third-party material](docs/THIRD_PARTY.md).
