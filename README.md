# Network persistence from partial temporal observations

Many networks change over time: people meet, email or reply to each other on some days and not on others. This study measures how well different methods can tell, from a small sample of such a network, how persistent its connections are. The target, `rho_2`, is the share of interacting pairs that are active in at least two of five equal time windows.

The comparison covers 12 real networks, a time-shuffled copy of each, and 8 synthetic networks with known answers. Each network is sampled in four ways (sampling arms): R takes random nodes, S follows a random walk along interactions, H takes random nodes but sees only the last 60% of the time span, and B keeps each interaction with a fixed probability. Every network has three samples per arm, 384 samples in total. Four methods without a language model and five language-model configurations estimate `rho_2` from exactly the same samples. The [study design](docs/DESIGN.md) explains every method and term in plain words.

## Main result

Average error in `rho_2` (MAE_2, lower is better) over the 12 real networks; 0.03 means the estimate is off by 3 percentage points on average. The observed share is the standard method for arm R, the statistical model for S, H and B.

| Arm | Observed share (plug-in) | Training median | Statistical model (MLE) | ExtraTrees | Qwen, thinking | Qwen, no thinking | DeepSeek Flash | GPT-6 Sol | GPT-6 Sol + Python |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 0.027 | 0.232 | 0.037 | 0.028 | 0.027 | 0.411 | 0.027 | 0.027 | 0.027 |
| S | 0.301 | 0.232 | 0.069 | 0.050 | 0.181 | 0.475 | 0.090 | 0.084 | 0.081 |
| H | 0.064 | 0.232 | 0.071 | 0.039 | 0.160 | 0.252 | 0.133 | 0.054 | 0.061 |
| B | 0.164 | 0.232 | 0.076 | 0.079 | 0.220 | 0.527 | 0.172 | 0.108 | 0.151 |

Every language model answered each sample three times; only answers in the required format are scored. ExtraTrees output is limited to valid values (between 0 and 1, never increasing), which changes no `rho_2` value. Recorded API spend: DeepSeek Flash USD 26.51, GPT-6 Sol USD 27.59, GPT-6 Sol + Python USD 94.50. All tables, validity counts, variability and checks are in [docs/results/final](docs/results/final/REPORT.md). The [written analysis](docs/analysis/ANALYSIS.md) explains the results with figures.

## Repository layout

| Path | Contents |
|---|---|
| `docs/` | [Study design](docs/DESIGN.md) with glossary and sources, the [final results](docs/results/final/REPORT.md) and the [written analysis](docs/analysis/ANALYSIS.md) |
| `src/study/` | The method: networks (`data.py`, `surrogates.py`, `synthetic.py`), sampling arms (`sampling.py`, `walk.py`), the sample text every method sees (`observation.py`), estimators (`estimators.py`, `mle.py`, `thinning_model.py`), the training networks of ExtraTrees (`training_pool.py`), the answer check (`answer_format.py`) and the result tables (`tables.py`) |
| `scripts/` | What can be run again: the result tables (`evaluate.py`, `window_sensitivity.py`) and the figures and tables of the analysis (`analysis_figures.py`) |
| `production/` | Record of the runs on the computing cluster and against the paid APIs that produced the frozen predictions ([overview](production/README.md)) |
| `config/` | Prompts and data sources |
| `tests/` | Automatic tests |

## Reproduce the results

Every estimate of the study is frozen in [`PREDICTIONS.csv`](docs/results/final/PREDICTIONS.csv). The tables and the analysis are rebuilt from it:

```bash
pip install -r requirements.txt
python -m pytest -q                            # tests
python scripts/evaluate.py                     # all tables in docs/results/final
python scripts/window_sensitivity.py           # the tables on the number of time windows
python scripts/analysis_figures.py --inputs    # data, figures and tables in docs/analysis
```

The evaluation first checks the frozen predictions: every API answer is scored again from its raw text, and every stored error is recomputed from the truth. It calls no model. Without `--inputs`, the last command only redraws the figures and needs nothing outside the repository.

Not in the repository: the raw networks (`data/raw/`, sources in [third-party material](docs/THIRD_PARTY.md)) and the frozen samples and answers of the language models (`~/.local/share/masterthesis`). Their hashes are in [`CHECKSUMS.json`](docs/results/final/CHECKSUMS.json); the evaluation stops if a sample or an answer file differs.
