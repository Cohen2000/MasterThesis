# Final results

This folder holds every result of the study. The [study design](../../DESIGN.md) explains the networks, the four sampling arms (R, S, H, B), the methods and the scoring; terms are explained in its [glossary](../../DESIGN.md#glossary). In short: 32 networks (12 real, a time-shuffled copy of each, 8 synthetic), each sampled three times in each arm, give 384 samples. Every method estimates `rho_2`, the share of interacting pairs that are active in at least two of five time windows, from exactly the same samples.

## Files

- [`MAIN_RESULTS.md`](MAIN_RESULTS.md): the main tables. For each group of networks, sampling arm and method: the average error (MAE_2 and ProfileMAE), its direction and the share of valid answers, plus method-against-method comparisons. The same numbers as CSV: `SUMMARY.csv` (per group), `PER_SOURCE.csv` (per network), `PAIRED_METHODS.csv` and `PAIRED_FAMILIES.csv` (comparisons).
- `PREDICTIONS.csv`: every single estimate, one row per sample and method (ExtraTrees: per fit; language models: per repeat). 29 rows per sample, 11,136 rows. `TRUTH.json`: the true rho_2..rho_5 of every network.
- [`VARIABILITY.md`](VARIABILITY.md): how much results change between ExtraTrees fits, between the three samples of a network, and between the three answers of a language model.
- [`HISTORY.md`](HISTORY.md): why seeing only the last 60% of the time span (arm H) biases the estimates.
- [`WALK.md`](WALK.md): a check of the random walk (arm S) on all 32 networks, with 1000 simulated walks each.
- [`W_SENSITIVITY.md`](W_SENSITIVITY.md): how the true answer and the ordering of the networks change with 2 to 20 instead of 5 time windows (`scripts/window_sensitivity.py`; all values in `W_SENSITIVITY.csv` and `W_SENSITIVITY_NETWORKS.csv`).
- `S_DESIGN_APPENDIX.csv`: a simple re-weighted estimate for arm S, shown for comparison only.
- `CHECKSUMS.json`: SHA-256 fingerprints of the raw data files, of the 384 frozen samples given to the API models, and of every raw answer file, so anyone can verify they have the same data.

## When an answer counts

A language-model answer is valid when its final text is exactly one JSON object with the four values rho_2..rho_5, each between 0 and 1 and never increasing (one surrounding code block is allowed), and the model did not stop at its length limit. Invalid answers are counted and left out of the error; language-model answers are never corrected or asked again. ExtraTrees output is always turned into a valid profile: each value is limited to [0, 1] and each rho_k to at most rho_(k-1).

| Model | Answers | Valid |
|---|---:|---:|
| Qwen, thinking | 1,152 | 1,151 |
| Qwen, no thinking | 1,152 | 1,150 |
| DeepSeek Flash | 1,152 | 1,151 |
| GPT-6 Sol | 1,152 | 1,152 |
| GPT-6 Sol + Python | 1,152 | 1,152 |

## Checks

- `PREDICTIONS.csv` has exactly 29 rows for each of the 384 samples: observed share, training median, statistical model, 11 ExtraTrees fits and 3 answers from each of the five language-model configurations.
- The evaluation recomputes every stored error from `TRUTH.json` and stops if one differs.
- Before any request, the 384 samples are checked against the fingerprint in `CHECKSUMS.json`, and every prompt is checked against its sample.
- Leakage check over all 528 ExtraTrees fits: no network being predicted (or its time-shuffled copy) appears in the training data or in the tuning of the fit that predicts it.
- ExtraTrees output: turning it into valid profiles changed 88 of 4,224 predictions (12 of 384 in the reported fit) slightly and no rho_2 value.
- Random-walk check: networks where the walk's bias cannot be corrected at the 10% budget are marked in `WALK.md`; no network is excluded.

## API spend

Spend computed from the token counts the providers reported, at list prices, including a few short test requests made before the main runs:

| Provider | Model | Spend (USD) |
|---|---|---:|
| DeepSeek | DeepSeek Flash | 26.51 |
| OpenAI | GPT-6 Sol | 27.59 |
| OpenAI | GPT-6 Sol + Python | 94.50 |
| OpenAI | total | 122.10 |
