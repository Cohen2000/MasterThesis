# Variability

How to read this: a result can change for three separate reasons, reported separately.
SD is the standard deviation (typical size of the variation) of the rho_2 estimate. Terms are explained in the [glossary](../../DESIGN.md#glossary).

1. **Training** (ExtraTrees only, 11 fits): fit 0 is the one reported everywhere; fits 1-10 are
   trained on newly drawn training observations with new random seeds. The test observations stay the same.
2. **Sampling**: each network is observed three times per arm; how much does the estimate change
   between these three observations?
3. **Answers**: each language model answered each observation three times; how much do the answers differ?

## 1. Training (ExtraTrees, 11 fits)

| Networks | Sampling arm | MAE_2 mean | MAE_2 SD | MAE_2 min | MAE_2 max | Median SD per observation |
|---|---|---:|---:|---:|---:|---:|
| real | R (random nodes) | 0.0268 | 0.0010 | 0.0250 | 0.0282 | 0.0044 |
| real | S (random walk) | 0.0554 | 0.0028 | 0.0504 | 0.0591 | 0.0063 |
| real | H (random nodes, last 60% of time) | 0.0419 | 0.0021 | 0.0390 | 0.0463 | 0.0057 |
| real | B (random event loss) | 0.0801 | 0.0014 | 0.0777 | 0.0821 | 0.0051 |
| time-shuffled copy | R (random nodes) | 0.0175 | 0.0013 | 0.0155 | 0.0202 | 0.0066 |
| time-shuffled copy | S (random walk) | 0.0587 | 0.0017 | 0.0566 | 0.0618 | 0.0094 |
| time-shuffled copy | H (random nodes, last 60% of time) | 0.0448 | 0.0025 | 0.0423 | 0.0499 | 0.0072 |
| time-shuffled copy | B (random event loss) | 0.1161 | 0.0029 | 0.1116 | 0.1195 | 0.0058 |
| synthetic | R (random nodes) | 0.0225 | 0.0006 | 0.0217 | 0.0240 | 0.0020 |
| synthetic | S (random walk) | 0.0177 | 0.0009 | 0.0165 | 0.0200 | 0.0033 |
| synthetic | H (random nodes, last 60% of time) | 0.0286 | 0.0008 | 0.0272 | 0.0299 | 0.0033 |
| synthetic | B (random event loss) | 0.0370 | 0.0011 | 0.0355 | 0.0388 | 0.0063 |

## 2. Sampling

SD of the rho_2 estimate across the three observations of a network (language models: mean of their
valid answers per observation; ExtraTrees: fit 0), median over networks. The simple estimators give
the same number for the same observation, so this is their only source of variation. Networks whose
H arm covers every node have a single observation there and are left out.

| Networks | Sampling arm | Method | Networks used | Median SD across observations |
|---|---|---:|---:|---:|
| real | R (random nodes) | Observed share (plug-in) | 12/12 | 0.0188 |
| real | R (random nodes) | Training median | 12/12 | 0.0000 |
| real | R (random nodes) | Statistical model (MLE) | 12/12 | 0.0161 |
| real | R (random nodes) | ExtraTrees (trained model) | 12/12 | 0.0177 |
| real | R (random nodes) | Qwen, thinking | 12/12 | 0.0188 |
| real | R (random nodes) | Qwen, no thinking | 12/12 | 0.0375 |
| real | R (random nodes) | DeepSeek Flash | 12/12 | 0.0188 |
| real | R (random nodes) | GPT-6 Sol | 12/12 | 0.0188 |
| real | R (random nodes) | GPT-6 Sol + Python | 12/12 | 0.0188 |
| real | S (random walk) | Observed share (plug-in) | 12/12 | 0.0115 |
| real | S (random walk) | Training median | 12/12 | 0.0000 |
| real | S (random walk) | Statistical model (MLE) | 12/12 | 0.0374 |
| real | S (random walk) | ExtraTrees (trained model) | 12/12 | 0.0245 |
| real | S (random walk) | Qwen, thinking | 12/12 | 0.0782 |
| real | S (random walk) | Qwen, no thinking | 12/12 | 0.0688 |
| real | S (random walk) | DeepSeek Flash | 12/12 | 0.0678 |
| real | S (random walk) | GPT-6 Sol | 12/12 | 0.0544 |
| real | S (random walk) | GPT-6 Sol + Python | 12/12 | 0.0470 |
| real | H (random nodes, last 60% of time) | Observed share (plug-in) | 12/12 | 0.0087 |
| real | H (random nodes, last 60% of time) | Training median | 12/12 | 0.0000 |
| real | H (random nodes, last 60% of time) | Statistical model (MLE) | 12/12 | 0.0132 |
| real | H (random nodes, last 60% of time) | ExtraTrees (trained model) | 12/12 | 0.0090 |
| real | H (random nodes, last 60% of time) | Qwen, thinking | 12/12 | 0.0920 |
| real | H (random nodes, last 60% of time) | Qwen, no thinking | 12/12 | 0.1209 |
| real | H (random nodes, last 60% of time) | DeepSeek Flash | 12/12 | 0.0856 |
| real | H (random nodes, last 60% of time) | GPT-6 Sol | 12/12 | 0.0152 |
| real | H (random nodes, last 60% of time) | GPT-6 Sol + Python | 12/12 | 0.0225 |
| real | B (random event loss) | Observed share (plug-in) | 12/12 | 0.0064 |
| real | B (random event loss) | Training median | 12/12 | 0.0000 |
| real | B (random event loss) | Statistical model (MLE) | 12/12 | 0.0088 |
| real | B (random event loss) | ExtraTrees (trained model) | 12/12 | 0.0075 |
| real | B (random event loss) | Qwen, thinking | 12/12 | 0.1176 |
| real | B (random event loss) | Qwen, no thinking | 12/12 | 0.0303 |
| real | B (random event loss) | DeepSeek Flash | 12/12 | 0.1370 |
| real | B (random event loss) | GPT-6 Sol | 12/12 | 0.0388 |
| real | B (random event loss) | GPT-6 Sol + Python | 12/12 | 0.0587 |
| time-shuffled copy | R (random nodes) | Observed share (plug-in) | 12/12 | 0.0160 |
| time-shuffled copy | R (random nodes) | Training median | 12/12 | 0.0000 |
| time-shuffled copy | R (random nodes) | Statistical model (MLE) | 12/12 | 0.0147 |
| time-shuffled copy | R (random nodes) | ExtraTrees (trained model) | 12/12 | 0.0168 |
| time-shuffled copy | R (random nodes) | Qwen, thinking | 12/12 | 0.0167 |
| time-shuffled copy | R (random nodes) | Qwen, no thinking | 12/12 | 0.0543 |
| time-shuffled copy | R (random nodes) | DeepSeek Flash | 12/12 | 0.0160 |
| time-shuffled copy | R (random nodes) | GPT-6 Sol | 12/12 | 0.0160 |
| time-shuffled copy | R (random nodes) | GPT-6 Sol + Python | 12/12 | 0.0160 |
| time-shuffled copy | S (random walk) | Observed share (plug-in) | 12/12 | 0.0034 |
| time-shuffled copy | S (random walk) | Training median | 12/12 | 0.0000 |
| time-shuffled copy | S (random walk) | Statistical model (MLE) | 12/12 | 0.0452 |
| time-shuffled copy | S (random walk) | ExtraTrees (trained model) | 12/12 | 0.0462 |
| time-shuffled copy | S (random walk) | Qwen, thinking | 12/12 | 0.0444 |
| time-shuffled copy | S (random walk) | Qwen, no thinking | 12/12 | 0.1808 |
| time-shuffled copy | S (random walk) | DeepSeek Flash | 12/12 | 0.0527 |
| time-shuffled copy | S (random walk) | GPT-6 Sol | 12/12 | 0.0408 |
| time-shuffled copy | S (random walk) | GPT-6 Sol + Python | 12/12 | 0.0417 |
| time-shuffled copy | H (random nodes, last 60% of time) | Observed share (plug-in) | 12/12 | 0.0082 |
| time-shuffled copy | H (random nodes, last 60% of time) | Training median | 12/12 | 0.0000 |
| time-shuffled copy | H (random nodes, last 60% of time) | Statistical model (MLE) | 12/12 | 0.0056 |
| time-shuffled copy | H (random nodes, last 60% of time) | ExtraTrees (trained model) | 12/12 | 0.0063 |
| time-shuffled copy | H (random nodes, last 60% of time) | Qwen, thinking | 12/12 | 0.0852 |
| time-shuffled copy | H (random nodes, last 60% of time) | Qwen, no thinking | 12/12 | 0.1615 |
| time-shuffled copy | H (random nodes, last 60% of time) | DeepSeek Flash | 12/12 | 0.0568 |
| time-shuffled copy | H (random nodes, last 60% of time) | GPT-6 Sol | 12/12 | 0.0304 |
| time-shuffled copy | H (random nodes, last 60% of time) | GPT-6 Sol + Python | 12/12 | 0.0190 |
| time-shuffled copy | B (random event loss) | Observed share (plug-in) | 12/12 | 0.0087 |
| time-shuffled copy | B (random event loss) | Training median | 12/12 | 0.0000 |
| time-shuffled copy | B (random event loss) | Statistical model (MLE) | 12/12 | 0.0097 |
| time-shuffled copy | B (random event loss) | ExtraTrees (trained model) | 12/12 | 0.0095 |
| time-shuffled copy | B (random event loss) | Qwen, thinking | 12/12 | 0.1490 |
| time-shuffled copy | B (random event loss) | Qwen, no thinking | 12/12 | 0.0384 |
| time-shuffled copy | B (random event loss) | DeepSeek Flash | 12/12 | 0.1193 |
| time-shuffled copy | B (random event loss) | GPT-6 Sol | 12/12 | 0.0717 |
| time-shuffled copy | B (random event loss) | GPT-6 Sol + Python | 12/12 | 0.0571 |
| synthetic | R (random nodes) | Observed share (plug-in) | 8/8 | 0.0270 |
| synthetic | R (random nodes) | Training median | 8/8 | 0.0000 |
| synthetic | R (random nodes) | Statistical model (MLE) | 8/8 | 0.0248 |
| synthetic | R (random nodes) | ExtraTrees (trained model) | 8/8 | 0.0251 |
| synthetic | R (random nodes) | Qwen, thinking | 8/8 | 0.0274 |
| synthetic | R (random nodes) | Qwen, no thinking | 8/8 | 0.0626 |
| synthetic | R (random nodes) | DeepSeek Flash | 8/8 | 0.0270 |
| synthetic | R (random nodes) | GPT-6 Sol | 8/8 | 0.0271 |
| synthetic | R (random nodes) | GPT-6 Sol + Python | 8/8 | 0.0270 |
| synthetic | S (random walk) | Observed share (plug-in) | 8/8 | 0.0177 |
| synthetic | S (random walk) | Training median | 8/8 | 0.0000 |
| synthetic | S (random walk) | Statistical model (MLE) | 8/8 | 0.0194 |
| synthetic | S (random walk) | ExtraTrees (trained model) | 8/8 | 0.0166 |
| synthetic | S (random walk) | Qwen, thinking | 8/8 | 0.0525 |
| synthetic | S (random walk) | Qwen, no thinking | 8/8 | 0.0378 |
| synthetic | S (random walk) | DeepSeek Flash | 8/8 | 0.0316 |
| synthetic | S (random walk) | GPT-6 Sol | 8/8 | 0.0207 |
| synthetic | S (random walk) | GPT-6 Sol + Python | 8/8 | 0.0187 |
| synthetic | H (random nodes, last 60% of time) | Observed share (plug-in) | 8/8 | 0.0258 |
| synthetic | H (random nodes, last 60% of time) | Training median | 8/8 | 0.0000 |
| synthetic | H (random nodes, last 60% of time) | Statistical model (MLE) | 8/8 | 0.0337 |
| synthetic | H (random nodes, last 60% of time) | ExtraTrees (trained model) | 8/8 | 0.0257 |
| synthetic | H (random nodes, last 60% of time) | Qwen, thinking | 8/8 | 0.0555 |
| synthetic | H (random nodes, last 60% of time) | Qwen, no thinking | 8/8 | 0.0962 |
| synthetic | H (random nodes, last 60% of time) | DeepSeek Flash | 8/8 | 0.0261 |
| synthetic | H (random nodes, last 60% of time) | GPT-6 Sol | 8/8 | 0.0337 |
| synthetic | H (random nodes, last 60% of time) | GPT-6 Sol + Python | 8/8 | 0.0385 |
| synthetic | B (random event loss) | Observed share (plug-in) | 8/8 | 0.0112 |
| synthetic | B (random event loss) | Training median | 8/8 | 0.0000 |
| synthetic | B (random event loss) | Statistical model (MLE) | 8/8 | 0.0904 |
| synthetic | B (random event loss) | ExtraTrees (trained model) | 8/8 | 0.0253 |
| synthetic | B (random event loss) | Qwen, thinking | 8/8 | 0.1183 |
| synthetic | B (random event loss) | Qwen, no thinking | 8/8 | 0.1934 |
| synthetic | B (random event loss) | DeepSeek Flash | 8/8 | 0.2734 |
| synthetic | B (random event loss) | GPT-6 Sol | 8/8 | 0.1580 |
| synthetic | B (random event loss) | GPT-6 Sol + Python | 8/8 | 0.1082 |

## 3. Answers of the language models

SD of rho_2 across the three answers to the same observation, median over observations with three
valid answers.

| Networks | Sampling arm | Model | Observations | Median SD across answers |
|---|---|---:|---:|---:|
| real | R (random nodes) | Qwen, thinking | 36 | 0.0000 |
| real | R (random nodes) | Qwen, no thinking | 36 | 0.0777 |
| real | R (random nodes) | DeepSeek Flash | 36 | 0.0000 |
| real | R (random nodes) | GPT-6 Sol | 36 | 0.0000 |
| real | R (random nodes) | GPT-6 Sol + Python | 36 | 0.0000 |
| real | S (random walk) | Qwen, thinking | 36 | 0.0794 |
| real | S (random walk) | Qwen, no thinking | 35 | 0.0681 |
| real | S (random walk) | DeepSeek Flash | 35 | 0.0087 |
| real | S (random walk) | GPT-6 Sol | 36 | 0.0001 |
| real | S (random walk) | GPT-6 Sol + Python | 36 | 0.0000 |
| real | H (random nodes, last 60% of time) | Qwen, thinking | 36 | 0.2187 |
| real | H (random nodes, last 60% of time) | Qwen, no thinking | 36 | 0.1577 |
| real | H (random nodes, last 60% of time) | DeepSeek Flash | 36 | 0.1032 |
| real | H (random nodes, last 60% of time) | GPT-6 Sol | 36 | 0.0167 |
| real | H (random nodes, last 60% of time) | GPT-6 Sol + Python | 36 | 0.0216 |
| real | B (random event loss) | Qwen, thinking | 36 | 0.1628 |
| real | B (random event loss) | Qwen, no thinking | 36 | 0.0706 |
| real | B (random event loss) | DeepSeek Flash | 36 | 0.1186 |
| real | B (random event loss) | GPT-6 Sol | 36 | 0.0509 |
| real | B (random event loss) | GPT-6 Sol + Python | 36 | 0.1220 |
| time-shuffled copy | R (random nodes) | Qwen, thinking | 36 | 0.0000 |
| time-shuffled copy | R (random nodes) | Qwen, no thinking | 36 | 0.0611 |
| time-shuffled copy | R (random nodes) | DeepSeek Flash | 36 | 0.0000 |
| time-shuffled copy | R (random nodes) | GPT-6 Sol | 36 | 0.0000 |
| time-shuffled copy | R (random nodes) | GPT-6 Sol + Python | 36 | 0.0000 |
| time-shuffled copy | S (random walk) | Qwen, thinking | 35 | 0.0068 |
| time-shuffled copy | S (random walk) | Qwen, no thinking | 36 | 0.0786 |
| time-shuffled copy | S (random walk) | DeepSeek Flash | 36 | 0.0042 |
| time-shuffled copy | S (random walk) | GPT-6 Sol | 36 | 0.0001 |
| time-shuffled copy | S (random walk) | GPT-6 Sol + Python | 36 | 0.0001 |
| time-shuffled copy | H (random nodes, last 60% of time) | Qwen, thinking | 36 | 0.0771 |
| time-shuffled copy | H (random nodes, last 60% of time) | Qwen, no thinking | 36 | 0.1400 |
| time-shuffled copy | H (random nodes, last 60% of time) | DeepSeek Flash | 36 | 0.0853 |
| time-shuffled copy | H (random nodes, last 60% of time) | GPT-6 Sol | 36 | 0.0364 |
| time-shuffled copy | H (random nodes, last 60% of time) | GPT-6 Sol + Python | 36 | 0.0410 |
| time-shuffled copy | B (random event loss) | Qwen, thinking | 36 | 0.1850 |
| time-shuffled copy | B (random event loss) | Qwen, no thinking | 36 | 0.0764 |
| time-shuffled copy | B (random event loss) | DeepSeek Flash | 36 | 0.2526 |
| time-shuffled copy | B (random event loss) | GPT-6 Sol | 36 | 0.1370 |
| time-shuffled copy | B (random event loss) | GPT-6 Sol + Python | 36 | 0.0992 |
| synthetic | R (random nodes) | Qwen, thinking | 24 | 0.0000 |
| synthetic | R (random nodes) | Qwen, no thinking | 24 | 0.1020 |
| synthetic | R (random nodes) | DeepSeek Flash | 24 | 0.0000 |
| synthetic | R (random nodes) | GPT-6 Sol | 24 | 0.0000 |
| synthetic | R (random nodes) | GPT-6 Sol + Python | 24 | 0.0000 |
| synthetic | S (random walk) | Qwen, thinking | 24 | 0.0378 |
| synthetic | S (random walk) | Qwen, no thinking | 24 | 0.0617 |
| synthetic | S (random walk) | DeepSeek Flash | 24 | 0.0071 |
| synthetic | S (random walk) | GPT-6 Sol | 24 | 0.0022 |
| synthetic | S (random walk) | GPT-6 Sol + Python | 24 | 0.0003 |
| synthetic | H (random nodes, last 60% of time) | Qwen, thinking | 24 | 0.1831 |
| synthetic | H (random nodes, last 60% of time) | Qwen, no thinking | 24 | 0.1529 |
| synthetic | H (random nodes, last 60% of time) | DeepSeek Flash | 24 | 0.0211 |
| synthetic | H (random nodes, last 60% of time) | GPT-6 Sol | 24 | 0.0033 |
| synthetic | H (random nodes, last 60% of time) | GPT-6 Sol + Python | 24 | 0.0073 |
| synthetic | B (random event loss) | Qwen, thinking | 24 | 0.2311 |
| synthetic | B (random event loss) | Qwen, no thinking | 24 | 0.2328 |
| synthetic | B (random event loss) | DeepSeek Flash | 24 | 0.0603 |
| synthetic | B (random event loss) | GPT-6 Sol | 24 | 0.0127 |
| synthetic | B (random event loss) | GPT-6 Sol + Python | 24 | 0.0168 |

## Same input, three outputs (real networks)

For a fair comparison with the three answers of a language model, ExtraTrees is also looked at
through three of its fits (200 random choices of three out of 11). The MLE always gives the same
number, so its SD is 0.

| Sampling arm | Method | Observations | Median SD |
|---|---|---:|---:|
| R (random nodes) | Statistical model (MLE) | 36 | 0.0000 |
| R (random nodes) | ExtraTrees (trained model) | 36 | 0.0041 |
| R (random nodes) | Qwen, thinking | 36 | 0.0000 |
| R (random nodes) | DeepSeek Flash | 36 | 0.0000 |
| R (random nodes) | GPT-6 Sol | 36 | 0.0000 |
| R (random nodes) | GPT-6 Sol + Python | 36 | 0.0000 |
| S (random walk) | Statistical model (MLE) | 36 | 0.0000 |
| S (random walk) | ExtraTrees (trained model) | 36 | 0.0060 |
| S (random walk) | Qwen, thinking | 36 | 0.0794 |
| S (random walk) | DeepSeek Flash | 35 | 0.0087 |
| S (random walk) | GPT-6 Sol | 36 | 0.0001 |
| S (random walk) | GPT-6 Sol + Python | 36 | 0.0000 |
| H (random nodes, last 60% of time) | Statistical model (MLE) | 36 | 0.0000 |
| H (random nodes, last 60% of time) | ExtraTrees (trained model) | 36 | 0.0051 |
| H (random nodes, last 60% of time) | Qwen, thinking | 36 | 0.2187 |
| H (random nodes, last 60% of time) | DeepSeek Flash | 36 | 0.1032 |
| H (random nodes, last 60% of time) | GPT-6 Sol | 36 | 0.0167 |
| H (random nodes, last 60% of time) | GPT-6 Sol + Python | 36 | 0.0216 |
| B (random event loss) | Statistical model (MLE) | 36 | 0.0000 |
| B (random event loss) | ExtraTrees (trained model) | 36 | 0.0048 |
| B (random event loss) | Qwen, thinking | 36 | 0.1628 |
| B (random event loss) | DeepSeek Flash | 36 | 0.1186 |
| B (random event loss) | GPT-6 Sol | 36 | 0.0509 |
| B (random event loss) | GPT-6 Sol + Python | 36 | 0.1220 |
