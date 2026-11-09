# Persistence under sampling: code and data of the Master's thesis

Code and frozen results of the Master's thesis *Evaluating Large Language Models for Edge Persistence
Estimation in Temporal Graphs under Partial Observation* (Albert Borchardt, University of Tübingen, 2026).

**Question.** How many pairs of users that interacted once interact again later? The thesis calls this
*persistence*: the share of interacting pairs that are active in at least two of five equal time windows.
Outsiders rarely see a whole platform, only a sample. The study draws samples from 12 fully recorded
networks with four samplers of equal budget and asks three LLMs, a statistical model and a learned
reference to estimate the persistence of the whole network.

| Sampler | Like | Problem it causes |
|---|---|---|
| node sampling (`R`) | a user panel | none (control) |
| random walk (`S`) | a crawler | sampling bias: busy pairs are seen more often |
| time truncation (`H`) | a search API for recent posts | lost history: windows 1 and 2 are hidden |
| event sampling (`B`) | a 1 % stream | both |

## Check every number of the thesis (one command, no raw data, no model)

```bash
pip install -r requirements.txt
python scripts/thesis.py       # every number and table of the thesis, from data/ -> thesis/numbers.md, thesis/tables/
python scripts/figures.py      # every figure of the thesis -> thesis/figures/
python -m pytest -q            # quick checks (samples, prompts, estimators, all numbers)
```

`thesis/numbers.md` lists each number of the thesis next to the value recomputed from `data/` (79 numbers,
all matching). The script stops with an error if one differs.

| In the thesis | File |
|---|---|
| Table 3.1 (networks) | `thesis/tables/networks.csv` |
| Section 3.5 (the hospital sample), Appendix A (full prompt) | `data/samples/sp_hospital__R-...__s1.json` |
| Table 5.1 (all methods) | `thesis/tables/results.csv` |
| Table 5.2 (random walk) | `thesis/tables/walk.csv` |
| Table 5.3 (lost history) | `thesis/tables/history.csv` |
| Tables 5.4 to 5.6 (robustness) | `thesis/tables/windows.csv`, `thresholds.csv`, `copies.csv` |
| Appendix B (per network) | `thesis/tables/per_network.csv` |
| Figures 1.1 and 5.1 to 5.5 | `thesis/figures/*.png` |

## What is in the repository

| Path | Contents |
|---|---|
| `data/samples/` | the 288 samples (12 networks and their 12 time-shuffled copies, 4 samplers, 3 draws), each with the exact prompt the LLMs received |
| `data/predictions.csv` | every estimate of the study (4,536 rows): method, sample, the four estimates `rho_2 ... rho_5`, valid or not |
| `data/truth.json` | the true persistence of every network and copy |
| `data/networks.csv` | facts per network (size, window length, one-off pairs, what the samplers hide) |
| `data/network_properties.csv`, `data/sample_properties.csv` | the 99 properties of Section 5.6 |
| `data/window_sensitivity.csv` | the true persistence for 2 to 20 windows (Section 5.7) |
| `data/checksums.json` | fingerprints of the 288 sample files |
| `src/study/` | the method: networks (`data.py`, `synthetic.py`), samplers (`sampling.py`, `walk.py`, `walk.cpp`), sample text and prompt (`sample.py`), estimators (`estimators.py`, `extratrees.py`), answer check (`answers.py`) |
| `config/` | the prompts (`prompts/`) and the raw networks with source and fingerprint (`networks.yaml`) |
| `scripts/` | the steps below |
| `tests/` | the quick checks |

The method names in `data/predictions.csv`:

| Name | In the thesis |
|---|---|
| `observed` | observed persistence (Section 4.1) |
| `statistical_model` | statistical model, beta-binomial (Section 4.3) |
| `learned_reference` | learned reference, ExtraTrees (Section 4.5) |
| `gpt_6_sol`, `deepseek_v4_1_flash`, `qwen3_6` | the three LLMs (Section 4.4) |
| `gpt_6_sol_python` | GPT-6 Sol with a Python interpreter |
| `gpt_6_sol_no_walk_log` | GPT-6 Sol under the random walk without the walk's log |

The textbook correction of the random walk (Section 4.2) is not a row of its own; `scripts/thesis.py`
computes it from each walk sample (`study.estimators.ratio`).

## How data/ was made

The LLM answers cannot be repeated exactly, so all estimates are frozen in `data/predictions.csv`.
These steps made `data/`. All except the LLM steps run on a laptop.

```bash
python scripts/samples.py                 # the 288 samples from the raw networks (needs data/raw)
python scripts/estimates.py [--learned]   # observed persistence, statistical model [, learned reference]
python scripts/api.py deepseek|openai ... # GPT-6 Sol and DeepSeek V4.1 Flash (paid APIs; dry run without --execute)
python scripts/qwen.py ...                # Qwen3.6-35B-A3B on one GPU (vLLM 0.29.0)
python scripts/ablation_walk_log.py       # GPT-6 Sol without the walk's log
python scripts/score.py                   # raw answers -> rows of data/predictions.csv
python scripts/networks.py                # data/networks.csv
python scripts/properties.py              # data/network_properties.csv, data/sample_properties.csv
python scripts/window_sensitivity.py      # data/window_sensitivity.csv
```

Each script writes to `results/` and compares its output with the frozen file in `data/`. A new run gives
the same samples byte for byte and the same observed persistence exactly. The statistical model and the
learned reference agree up to small numerical differences between machines, because both are fitted
numerically.

**Models.** GPT-6 Sol (`gpt-6-sol`, OpenAI API, reasoning effort high) and DeepSeek V4.1 Flash
(`deepseek-flash`, DeepSeek API, high effort). All main answers were collected by 29 September 2026,
before DeepSeek moved `deepseek-flash` to a new version on 17 October 2026. The run without the walk's
log took place on 21 October 2026. Qwen3.6-35B-A3B ran locally with thinking on (vLLM 0.29.0).

**Not in the repository:** the raw networks (`data/raw/`; files, sources and fingerprints in
`config/networks.yaml`; the rights stay with the providers) and the raw LLM answers (only their scores
are in `data/predictions.csv`).

## License

MIT, see `LICENSE`. The data sets keep the licenses of their providers.
