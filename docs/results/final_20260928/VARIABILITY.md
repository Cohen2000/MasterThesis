# Variability

Training, sampling and answer variability are reported below; the fixed-input table compares
training and answer variation at three predictions per observation.

## Training variability (ExtraTrees, 11 fits)

Fit 0 is the production fit; fits 1-10 redraw the real training and pool observations, reseed the
forests and rerun the nested selection. Test observations are fixed.

| Group | Arm | MAE_2 mean | MAE_2 SD | MAE_2 min | MAE_2 max | Median obs. SD |
|---|---|---:|---:|---:|---:|---:|
| real | R | 0.0268 | 0.0010 | 0.0250 | 0.0282 | 0.0044 |
| real | S | 0.0554 | 0.0028 | 0.0504 | 0.0591 | 0.0063 |
| real | H | 0.0419 | 0.0021 | 0.0390 | 0.0463 | 0.0057 |
| real | B | 0.0801 | 0.0014 | 0.0777 | 0.0821 | 0.0051 |
| surrogate | R | 0.0175 | 0.0013 | 0.0155 | 0.0202 | 0.0066 |
| surrogate | S | 0.0587 | 0.0017 | 0.0566 | 0.0618 | 0.0094 |
| surrogate | H | 0.0448 | 0.0025 | 0.0423 | 0.0499 | 0.0072 |
| surrogate | B | 0.1161 | 0.0029 | 0.1116 | 0.1195 | 0.0058 |
| synthetic | R | 0.0225 | 0.0006 | 0.0217 | 0.0240 | 0.0020 |
| synthetic | S | 0.0177 | 0.0009 | 0.0165 | 0.0200 | 0.0033 |
| synthetic | H | 0.0286 | 0.0008 | 0.0272 | 0.0299 | 0.0033 |
| synthetic | B | 0.0370 | 0.0011 | 0.0355 | 0.0388 | 0.0063 |

## Sampling variability

SD of the rho_2 estimate across the three sampler draws of a graph (LLM: mean of the valid answers per
draw; ET: production fit), median over graphs. Plugin, median and MLE are deterministic given an
observation; their uncertainty is this sampling variability, which is not zero. Graphs with a
saturated (single-draw) H panel are excluded. API methods: v11 graphs only.

| Group | Arm | Method | Graphs | Median SD across draws |
|---|---|---:|---:|---:|
| real | R | plugin | 12/12 | 0.0188 |
| real | R | median | 12/12 | 0.0000 |
| real | R | mle | 12/12 | 0.0161 |
| real | R | et | 12/12 | 0.0177 |
| real | R | qwen_thinking | 12/12 | 0.0188 |
| real | R | qwen_nonthinking | 12/12 | 0.0375 |
| real | R | deepseek_flash | 12/12 | 0.0188 |
| real | R | gpt_6_sol | 12/12 | 0.0188 |
| real | R | gpt_6_sol_tools | 12/12 | 0.0188 |
| real | S | plugin | 12/12 | 0.0115 |
| real | S | median | 12/12 | 0.0000 |
| real | S | mle | 12/12 | 0.0374 |
| real | S | et | 12/12 | 0.0245 |
| real | S | qwen_thinking | 12/12 | 0.0782 |
| real | S | qwen_nonthinking | 12/12 | 0.0688 |
| real | S | deepseek_flash | 12/12 | 0.0605 |
| real | S | gpt_6_sol | 12/12 | 0.0544 |
| real | S | gpt_6_sol_tools | 12/12 | 0.0470 |
| real | H | plugin | 12/12 | 0.0087 |
| real | H | median | 12/12 | 0.0000 |
| real | H | mle | 12/12 | 0.0132 |
| real | H | et | 12/12 | 0.0090 |
| real | H | qwen_thinking | 12/12 | 0.0920 |
| real | H | qwen_nonthinking | 12/12 | 0.1209 |
| real | H | deepseek_flash | 12/12 | 0.1070 |
| real | H | gpt_6_sol | 12/12 | 0.0152 |
| real | H | gpt_6_sol_tools | 12/12 | 0.0225 |
| real | B | plugin | 12/12 | 0.0064 |
| real | B | median | 12/12 | 0.0000 |
| real | B | mle | 12/12 | 0.0088 |
| real | B | et | 12/12 | 0.0075 |
| real | B | qwen_thinking | 12/12 | 0.1176 |
| real | B | qwen_nonthinking | 12/12 | 0.0303 |
| real | B | deepseek_flash | 12/12 | 0.1512 |
| real | B | gpt_6_sol | 12/12 | 0.0388 |
| real | B | gpt_6_sol_tools | 12/12 | 0.0587 |
| surrogate | R | plugin | 12/12 | 0.0160 |
| surrogate | R | median | 12/12 | 0.0000 |
| surrogate | R | mle | 12/12 | 0.0147 |
| surrogate | R | et | 12/12 | 0.0168 |
| surrogate | R | qwen_thinking | 12/12 | 0.0167 |
| surrogate | R | qwen_nonthinking | 12/12 | 0.0543 |
| surrogate | R | deepseek_flash | 12/12 | 0.0160 |
| surrogate | R | gpt_6_sol | 12/12 | 0.0160 |
| surrogate | R | gpt_6_sol_tools | 12/12 | 0.0160 |
| surrogate | S | plugin | 12/12 | 0.0034 |
| surrogate | S | median | 12/12 | 0.0000 |
| surrogate | S | mle | 12/12 | 0.0452 |
| surrogate | S | et | 12/12 | 0.0462 |
| surrogate | S | qwen_thinking | 12/12 | 0.0444 |
| surrogate | S | qwen_nonthinking | 12/12 | 0.1808 |
| surrogate | S | deepseek_flash | 12/12 | 0.0632 |
| surrogate | S | gpt_6_sol | 12/12 | 0.0408 |
| surrogate | S | gpt_6_sol_tools | 12/12 | 0.0417 |
| surrogate | H | plugin | 12/12 | 0.0082 |
| surrogate | H | median | 12/12 | 0.0000 |
| surrogate | H | mle | 12/12 | 0.0056 |
| surrogate | H | et | 12/12 | 0.0063 |
| surrogate | H | qwen_thinking | 12/12 | 0.0852 |
| surrogate | H | qwen_nonthinking | 12/12 | 0.1615 |
| surrogate | H | deepseek_flash | 12/12 | 0.1257 |
| surrogate | H | gpt_6_sol | 12/12 | 0.0304 |
| surrogate | H | gpt_6_sol_tools | 12/12 | 0.0190 |
| surrogate | B | plugin | 12/12 | 0.0087 |
| surrogate | B | median | 12/12 | 0.0000 |
| surrogate | B | mle | 12/12 | 0.0097 |
| surrogate | B | et | 12/12 | 0.0095 |
| surrogate | B | qwen_thinking | 12/12 | 0.1490 |
| surrogate | B | qwen_nonthinking | 12/12 | 0.0384 |
| surrogate | B | deepseek_flash | 12/12 | 0.2886 |
| surrogate | B | gpt_6_sol | 12/12 | 0.0717 |
| surrogate | B | gpt_6_sol_tools | 12/12 | 0.0571 |
| synthetic | R | plugin | 8/8 | 0.0270 |
| synthetic | R | median | 8/8 | 0.0000 |
| synthetic | R | mle | 8/8 | 0.0248 |
| synthetic | R | et | 8/8 | 0.0251 |
| synthetic | R | qwen_thinking | 8/8 | 0.0274 |
| synthetic | R | qwen_nonthinking | 8/8 | 0.0626 |
| synthetic | R | deepseek_flash | 8/8 | 0.0270 |
| synthetic | R | gpt_6_sol | 8/8 | 0.0271 |
| synthetic | R | gpt_6_sol_tools | 8/8 | 0.0270 |
| synthetic | S | plugin | 8/8 | 0.0177 |
| synthetic | S | median | 8/8 | 0.0000 |
| synthetic | S | mle | 8/8 | 0.0194 |
| synthetic | S | et | 8/8 | 0.0166 |
| synthetic | S | qwen_thinking | 8/8 | 0.0525 |
| synthetic | S | qwen_nonthinking | 8/8 | 0.0378 |
| synthetic | S | deepseek_flash | 8/8 | 0.0262 |
| synthetic | S | gpt_6_sol | 8/8 | 0.0207 |
| synthetic | S | gpt_6_sol_tools | 8/8 | 0.0187 |
| synthetic | H | plugin | 8/8 | 0.0258 |
| synthetic | H | median | 8/8 | 0.0000 |
| synthetic | H | mle | 8/8 | 0.0337 |
| synthetic | H | et | 8/8 | 0.0257 |
| synthetic | H | qwen_thinking | 8/8 | 0.0555 |
| synthetic | H | qwen_nonthinking | 8/8 | 0.0962 |
| synthetic | H | deepseek_flash | 8/8 | 0.0270 |
| synthetic | H | gpt_6_sol | 8/8 | 0.0337 |
| synthetic | H | gpt_6_sol_tools | 8/8 | 0.0385 |
| synthetic | B | plugin | 8/8 | 0.0112 |
| synthetic | B | median | 8/8 | 0.0000 |
| synthetic | B | mle | 8/8 | 0.0904 |
| synthetic | B | et | 8/8 | 0.0253 |
| synthetic | B | qwen_thinking | 8/8 | 0.1183 |
| synthetic | B | qwen_nonthinking | 8/8 | 0.1934 |
| synthetic | B | deepseek_flash | 8/8 | 0.4309 |
| synthetic | B | gpt_6_sol | 8/8 | 0.1580 |
| synthetic | B | gpt_6_sol_tools | 8/8 | 0.1082 |

## LLM answer variability

SD of the rho_2 answer across the three repeats of one observation, median over observations with
three valid answers. DeepSeek has one repeat.

| Group | Arm | Method | Observations | Median SD across repeats |
|---|---|---:|---:|---:|
| real | R | qwen_thinking | 36 | 0.0000 |
| real | R | qwen_nonthinking | 36 | 0.0777 |
| real | R | gpt_6_sol | 36 | 0.0000 |
| real | R | gpt_6_sol_tools | 36 | 0.0000 |
| real | S | qwen_thinking | 36 | 0.0794 |
| real | S | qwen_nonthinking | 35 | 0.0681 |
| real | S | gpt_6_sol | 36 | 0.0001 |
| real | S | gpt_6_sol_tools | 36 | 0.0000 |
| real | H | qwen_thinking | 36 | 0.2187 |
| real | H | qwen_nonthinking | 36 | 0.1577 |
| real | H | gpt_6_sol | 36 | 0.0167 |
| real | H | gpt_6_sol_tools | 36 | 0.0216 |
| real | B | qwen_thinking | 36 | 0.1628 |
| real | B | qwen_nonthinking | 36 | 0.0706 |
| real | B | gpt_6_sol | 36 | 0.0509 |
| real | B | gpt_6_sol_tools | 36 | 0.1220 |
| surrogate | R | qwen_thinking | 36 | 0.0000 |
| surrogate | R | qwen_nonthinking | 36 | 0.0611 |
| surrogate | R | gpt_6_sol | 36 | 0.0000 |
| surrogate | R | gpt_6_sol_tools | 36 | 0.0000 |
| surrogate | S | qwen_thinking | 35 | 0.0068 |
| surrogate | S | qwen_nonthinking | 36 | 0.0786 |
| surrogate | S | gpt_6_sol | 36 | 0.0001 |
| surrogate | S | gpt_6_sol_tools | 36 | 0.0001 |
| surrogate | H | qwen_thinking | 36 | 0.0771 |
| surrogate | H | qwen_nonthinking | 36 | 0.1400 |
| surrogate | H | gpt_6_sol | 36 | 0.0364 |
| surrogate | H | gpt_6_sol_tools | 36 | 0.0410 |
| surrogate | B | qwen_thinking | 36 | 0.1850 |
| surrogate | B | qwen_nonthinking | 36 | 0.0764 |
| surrogate | B | gpt_6_sol | 36 | 0.1370 |
| surrogate | B | gpt_6_sol_tools | 36 | 0.0992 |
| synthetic | R | qwen_thinking | 24 | 0.0000 |
| synthetic | R | qwen_nonthinking | 24 | 0.1020 |
| synthetic | R | gpt_6_sol | 24 | 0.0000 |
| synthetic | R | gpt_6_sol_tools | 24 | 0.0000 |
| synthetic | S | qwen_thinking | 24 | 0.0378 |
| synthetic | S | qwen_nonthinking | 24 | 0.0617 |
| synthetic | S | gpt_6_sol | 24 | 0.0022 |
| synthetic | S | gpt_6_sol_tools | 24 | 0.0003 |
| synthetic | H | qwen_thinking | 24 | 0.1831 |
| synthetic | H | qwen_nonthinking | 24 | 0.1529 |
| synthetic | H | gpt_6_sol | 24 | 0.0033 |
| synthetic | H | gpt_6_sol_tools | 24 | 0.0073 |
| synthetic | B | qwen_thinking | 24 | 0.2311 |
| synthetic | B | qwen_nonthinking | 24 | 0.2328 |
| synthetic | B | gpt_6_sol | 24 | 0.0127 |
| synthetic | B | gpt_6_sol_tools | 24 | 0.0168 |

## Fixed-input comparison (real sources)

Median per-observation SD of rho_2. MLE is deterministic; ET uses 200 seeded random
three-fit subsets of its 11 fits, averaging each observation’s SD over subsets.
LLMs use observations with three valid repeats.

| Arm | Method | Observations | Median SD |
|---|---|---:|---:|
| R | mle | 36 | 0.0000 |
| R | et | 36 | 0.0041 |
| R | qwen_thinking | 36 | 0.0000 |
| R | gpt_6_sol | 36 | 0.0000 |
| R | gpt_6_sol_tools | 36 | 0.0000 |
| S | mle | 36 | 0.0000 |
| S | et | 36 | 0.0060 |
| S | qwen_thinking | 36 | 0.0794 |
| S | gpt_6_sol | 36 | 0.0001 |
| S | gpt_6_sol_tools | 36 | 0.0000 |
| H | mle | 36 | 0.0000 |
| H | et | 36 | 0.0051 |
| H | qwen_thinking | 36 | 0.2187 |
| H | gpt_6_sol | 36 | 0.0167 |
| H | gpt_6_sol_tools | 36 | 0.0216 |
| B | mle | 36 | 0.0000 |
| B | et | 36 | 0.0048 |
| B | qwen_thinking | 36 | 0.1628 |
| B | gpt_6_sol | 36 | 0.0509 |
| B | gpt_6_sol_tools | 36 | 0.1220 |
