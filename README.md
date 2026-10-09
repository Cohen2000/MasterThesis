# Persistence under sampling bias

**How does sampling bias affect persistence estimates in temporal networks, and how do large language models compare with conventional methods?**

A temporal network is a list of events "node A with node B at time t". Its persistence `rho_2` is the share of interacting pairs that are active in at least two of five equal time windows. The study estimates `rho_2` from a small sample of a network. It uses 12 real networks, a time-shuffled twin of each and 8 synthetic networks; each is sampled three times in four ways (384 samples), and nine methods get exactly the same samples.

| Sampler | How the sample is taken |
|---|---|
| R | random nodes |
| S | a random walk along the events |
| H | random nodes, but only the last 60% of the time span |
| B | every event is kept with a fixed probability |

## Results

Mean error of `rho_2` on the 12 real networks (0.03 = 3 percentage points; lower is better):

| Sampler | Observed share | Training median | MLE | ExtraTrees | Qwen, thinking | Qwen, no thinking | DeepSeek Flash | GPT-6 Sol | GPT-6 Sol + Python |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R | 0.027 | 0.232 | 0.037 | 0.028 | 0.027 | 0.411 | 0.027 | 0.027 | 0.027 |
| S | 0.301 | 0.232 | 0.069 | 0.050 | 0.181 | 0.475 | 0.090 | 0.084 | 0.081 |
| H | 0.064 | 0.232 | 0.071 | 0.039 | 0.160 | 0.252 | 0.133 | 0.054 | 0.061 |
| B | 0.164 | 0.232 | 0.076 | 0.079 | 0.220 | 0.527 | 0.172 | 0.108 | 0.151 |

![Mean absolute error by method and sampler](docs/analysis/figures/fig2_ranking.png)

Key findings of the [analysis](docs/analysis/ANALYSIS.md):

1. **The bias depends on what the sampler hides.** Random nodes: no selection bias. Walks overstate persistence (+28 pp). Missing time and missing events understate it (−5 and −16 pp).
2. **What is hidden decides how well it can be corrected.** Walk weights are known, so a formula removes most of the bias (30 → 5–9 pp). Hidden time and lost events must be modelled; 4–8 pp remain even for the best method.
3. **What makes a network hard depends on the sampler:** few nodes (R), many events per pair (S), early windows unlike the late ones (H), high persistence (B).
4. **Where a formula exists (R, S), the best LLM's answers match it** and come close to the conventional methods.
5. **Where none exists (H, B), only GPT keeps up.** DeepSeek and Qwen fall behind. Longer reasoning or Python brings no gain.
6. **In short:** the best LLM gets close to the conventional methods, but it is noisier and never clearly better than the best of them.

All tables: [docs/results/final](docs/results/final/MAIN_RESULTS.md). Method, glossary and sources: [docs/DESIGN.md](docs/DESIGN.md).

Every language model answered each sample three times (1,152 answers per configuration). Only answers in the required format are scored: all were valid except 1 (Qwen, thinking), 2 (Qwen, no thinking) and 1 (DeepSeek). API cost: DeepSeek USD 26.51, GPT USD 27.59, GPT + Python USD 94.50.

## Repository

| Path | Contents |
|---|---|
| `src/study/` | The method: networks (`data.py`, `synthetic.py`), samplers (`sampling.py`, `walk.py`), sample text and prompt (`sample.py`), estimators (`estimators.py`, `extratrees.py`), answer check (`answers.py`), tables (`tables.py`) |
| `scripts/` | The steps below |
| `config/` | The prompts and the list of real networks with their sources |
| `docs/` | Design, result tables, written analysis |
| `tests/` | Tests (`python -m pytest -q`) |

## Steps

```bash
pip install -r requirements.txt
python scripts/samples.py                    # the 384 samples from the raw networks
python scripts/estimates.py --fits 11        # observed share, training median, MLE, ExtraTrees
python scripts/qwen.py ...                   # Qwen answers (GPU)
python scripts/api.py ...                    # DeepSeek and GPT answers (paid)
python scripts/checks.py                     # checks of the random walk and of the time cut
python scripts/evaluate.py                   # all result tables
python scripts/window_sensitivity.py         # tables on the number of time windows
python scripts/analysis_figures.py --inputs  # data, figures and tables of the analysis
```

Every estimate of the study is frozen in [`PREDICTIONS.csv`](docs/results/final/PREDICTIONS.csv), because the answers of the language models cannot be repeated. What a new run gives:

- **Samples, observed share, training median:** exactly the frozen values. The samples are checked against [`CHECKSUMS.json`](docs/results/final/CHECKSUMS.json).
- **MLE and ExtraTrees:** the frozen values up to small numerical differences between machines. The MLE is fitted by a numerical optimiser and differs by at most 0.0002. ExtraTrees starts from the MLE; from the same start values it reproduces the frozen fit exactly.
- **Tables and analysis:** exactly the files in `docs/`. The evaluation first scores every language-model answer again from its raw text and stops if a row of `PREDICTIONS.csv` differs.

Not in the repository: the raw networks (`data/raw/`; files, sources and fingerprints in [`config/networks.yaml`](config/networks.yaml)) and the raw answers of the language models (`~/.local/share/masterthesis`). Their rights remain with the providers.
