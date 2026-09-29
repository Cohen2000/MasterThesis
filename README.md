# Network persistence from partial temporal observations

Many networks change over time: people meet, email or reply to each other on some days and not on others. This study measures how well different methods can tell, from a small sample of such a network, how persistent its connections are. The target, `rho_2`, is the share of interacting pairs that are active in at least two of five equal time windows.

The comparison covers 12 real networks, a time-shuffled copy of each, and 8 synthetic networks with known answers. Each network is sampled in four ways (sampling arms): R takes random nodes, S follows a random walk along interactions, H takes random nodes but sees only the last 60% of the time span, and B keeps each interaction with a fixed probability. Every network has three samples per arm, 384 samples in total. Four methods without a language model and five language-model configurations estimate `rho_2` from exactly the same samples. The [study design](docs/DESIGN.md) explains every method and term in plain words.

## Main result

Average error in `rho_2` (MAE_2, lower is better) over the 12 real networks; 0.03 means the estimate is off by 3 percentage points on average. The observed share is the standard method for arm R, the statistical model for S, H and B.

| Arm | Observed share (plug-in) | Training median | Statistical model (MLE) | ExtraTrees | Qwen, thinking | Qwen, no thinking | DeepSeek Flash | GPT-6 Sol | GPT-6 Sol + Python |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 0.027 | 0.232 | 0.037 | 0.028 | 0.027 | 0.411 | 0.026 | 0.027 | 0.027 |
| S | 0.301 | 0.232 | 0.069 | 0.050 | 0.181 | 0.475 | 0.090 | 0.084 | 0.081 |
| H | 0.064 | 0.232 | 0.071 | 0.039 | 0.160 | 0.252 | 0.129 | 0.054 | 0.061 |
| B | 0.164 | 0.232 | 0.076 | 0.079 | 0.220 | 0.527 | 0.176 | 0.108 | 0.151 |

Every language model answered each sample three times; only answers in the required format are scored. ExtraTrees output is limited to valid values (between 0 and 1, never increasing), which changes no `rho_2` value. Recorded API spend: DeepSeek Flash USD 23.50, GPT-6 Sol USD 27.59, GPT-6 Sol + Python USD 94.50. All tables, validity counts, variability and checks are in [docs/results/final](docs/results/final/REPORT.md).

DeepSeek Flash: 1,022 of 1,152 answers are collected so far; the tables are updated when the remaining answers are in.

## Repository layout

| Path | Contents |
|---|---|
| `src/main_experiment/` | The building blocks: loading networks (`data.py`), sampling arms (`sampling.py`, `walk.py`), the sample text shown to every method (`observation.py`), the offline estimators (`baselines.py`, `shared_mle.py`, `mixtures.py`), the answer format check (`evaluation.py`), synthetic networks and time-shuffled copies |
| `src/pipeline/` | The cluster pipeline: one task per step (`dag.py` plans them), the real networks prepared in the pipeline (`sources.py`), ExtraTrees (`et.py`, `replicates.py`), time-shuffled copies and checks (`surrogates_and_checks.py`), Qwen (`qwen.py`) and all result tables (`report.py`) |
| `scripts/` | Programs to run: preparation, training networks, ExtraTrees, the API models (`api_runner.py`), evaluation (`evaluate_api.py`) |
| `cluster/` | Job scripts for the computing cluster (SLURM) |
| `config/` | Prompts, data sources and settings |
| `tests/` | Automatic tests |
| `docs/` | Study design with glossary, and the final results |

## Reproduce the evaluation

Install the Python requirements, provide the raw data (`data/raw/`) and the frozen observation and answer directories, and run:

```bash
PYTHONPATH=src .venv/bin/pytest -q
M=~/.local/share/masterthesis
.venv/bin/python scripts/evaluate_api.py --observations $M/api_observations \
  --deepseek $M/api_runs/deepseek --openai $M/api_runs/openai --openai-tools $M/api_runs/openai_tools \
  --out results/api_evaluation
PYTHONPATH=src .venv/bin/python -c "from pipeline.report import finalize; finalize('docs/results/final', 'results/api_evaluation')"
```

The evaluation reads the frozen predictions and answers; it does not regenerate observations or model predictions. Raw data and provider answers stay outside Git; their hashes are in [`CHECKSUMS.json`](docs/results/final/CHECKSUMS.json). See [third-party material](docs/THIRD_PARTY.md) for data sources.
