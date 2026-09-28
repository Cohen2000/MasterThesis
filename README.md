# How persistent is a network you only partly see?

Temporal networks (who interacts with whom, and when) are rarely observed completely. This project asks how well the **persistence** of a network can be recovered from a small sample. Persistence here means the share of interacting pairs that are active in at least *k* of five time windows (`rho_k`, with `rho_2` as the main target). Classical estimators, a supervised model (ExtraTrees) and language models (Qwen, DeepSeek, GPT) all get the same sampled data.

## Setup

- **24 graphs:** 8 real interaction datasets, 8 matched surrogates and 8 synthetic graphs.
- **4 sampling designs, each seeing about 10% of the activity:**
  - **R:** a random panel of nodes
  - **S:** an interaction-following random walk, with its crawl log
  - **H:** a node panel that only sees the last 60% of history
  - **B:** random thinning of individual events
- Each design is drawn 3 times per graph, and estimators and models receive only the released design information (e.g. panel size, crawl log, thinning probability).
- **Metric:** absolute error of `rho_2`, averaged per graph, then equally over graphs (MAE_2). Real graphs are the primary group.
- **LLM answers:** only answers whose final output is one valid JSON profile are scored. Nothing is repaired. All 2,016 paid API answers were valid.

Full method: [protocol](docs/PROTOCOL_PANEL888_20260921.md).

## Results

MAE_2 over all 24 graphs (lower is better). The reference is the design's standard estimator: plugin for R, design-based for S, and MLE for H and B.

| Design | Reference | ExtraTrees | Qwen 3.6 (thinking) | DeepSeek Flash | GPT-6 Sol | GPT-6 Sol + Python |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.019 | 0.020 | 0.019 | 0.019 | 0.019 | 0.020 |
| S | 0.042 | 0.036 | 0.128 | 0.046 | 0.041 | 0.039 |
| H | 0.049 | 0.035 | 0.154 | 0.093 | 0.048 | 0.052 |
| B | 0.086 | 0.070 | 0.286 | 0.218 | 0.118 | 0.119 |

The same table for the 8 real graphs only:

| Design | Reference | ExtraTrees | Qwen 3.6 (thinking) | DeepSeek Flash | GPT-6 Sol | GPT-6 Sol + Python |
|---|---:|---:|---:|---:|---:|---:|
| R | 0.021 | 0.023 | 0.021 | 0.021 | 0.021 | 0.022 |
| S | 0.043 | 0.042 | 0.151 | 0.043 | 0.045 | 0.043 |
| H | 0.052 | 0.028 | 0.149 | 0.096 | 0.042 | 0.055 |
| B | 0.055 | 0.055 | 0.206 | 0.167 | 0.091 | 0.115 |

| Model | Repeats | Recorded API spend (upper bound) |
|---|---:|---:|
| Qwen | 3 | none (bwUniCluster H100) |
| DeepSeek Flash | 1 (budget) | USD 6.92 |
| GPT-6 Sol | 3 | USD 20.25 |
| GPT-6 Sol + Python | 3 | USD 71.43 |

### Extension: 12 real sources and ExtraTrees replicates

Added on 2026-09-28 without changing anything above: four more real test sources (Reality Mining, LKML replies, SocioPatterns Malawi, and nr_radoslaw_email promoted from training) run through the same offline pipeline and Qwen, plus ten ExtraTrees replicates (new training draws, new seeds, rerun selection) as the counterpart to the LLM repeats. MAE_2 over the 12 real sources:

| Design | Reference | ExtraTrees (production) | ExtraTrees (11 replicates, mean ± SD) | Qwen 3.6 (thinking) |
|---|---:|---:|---:|---:|
| R | 0.027 | 0.028 | 0.027 ± 0.001 | 0.027 |
| S | 0.088 | 0.075 | 0.074 ± 0.003 | 0.181 |
| H | 0.071 | 0.039 | 0.042 ± 0.002 | 0.160 |
| B | 0.076 | 0.079 | 0.080 ± 0.001 | 0.220 |

Paid API models were not run on the new sources.

### Where the details are

- Extension: [results](docs/results/v11_ext_20260928/MAIN_RESULTS.md), [ET replicates](docs/results/v11_ext_20260928/ET_REPLICATES.md), [run report](docs/results/v11_ext_20260928/REPORT.md)
- Per group of eight graphs, paired sign-flip tests and token use: [API results](docs/results/api_v11_20260923/API_RESULTS.md)
- All offline methods, Qwen, the `S_obs` ablation and MCSEs: [main results](docs/results/panel888_v11_main_20260923/MAIN_RESULTS.md)
- Sensitivity checks: [hidden panel size](docs/results/panel888_v10_RH_panel_release/REPORT.md), [walk gate](docs/results/panel888_v10_walk_gate_20260923/WALK_GATE.md)

## Repository

| Path | Contents |
|---|---|
| `config/` | Study settings, dataset registry and the exact LLM prompts |
| `src/main_experiment/` | Sampling, observations, estimators, ExtraTrees features, evaluation |
| `src/census.py`, `src/dataset_census.py` | Audited raw-data parsers |
| `src/v11_ext/` | Extension: new real sources, ExtraTrees replicates, SLURM orchestration |
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
6. `v11_ext.py --dry-run | --status | --submit`: the extension, resumable on uc3
7. `audit_v10_walk.py`, `confirm_v10_walk.py`, `report_v10_walk_gate.py`: walk-gate check; `verify_*.py`: archive and ET verification

Raw data, generated artifacts and raw model answers stay local; their checksums are committed with the results. See [third-party material](docs/THIRD_PARTY.md).
