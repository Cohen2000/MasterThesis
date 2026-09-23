# v10 main results

Primary metric: equal-source MAE_2 across eight real sources. ProfileMAE is secondary.
LLM accuracy uses valid final answers only; no estimate is clipped or repaired.
ET selection used only the synthetic development pool; synthetic ET evaluation is in-distribution.
sp_hospital__pwt remains flagged as not correctable at the 10% S budget.

Qwen complete: False.

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Validity | Fallback | ET profile validity | Skill vs plugin | S flag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| R | plugin | 0.0209 | 0.0130 | -0.0044 | 1.0000 |  |  | 0.0000 |  |
| R | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  | -10.4768 |  |
| R | mle | 0.0234 | 0.0139 | 0.0074 | 1.0000 | 0.0417 |  | -0.1218 |  |
| R | et | 0.0251 | 0.0167 | -0.0031 | 1.0000 |  | 0.7083 | -0.2007 |  |
| S | plugin | 0.2816 | 0.2110 | 0.2816 | 1.0000 |  |  | 0.0000 |  |
| S | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  | 0.1489 |  |
| S | design | 0.0426 | 0.0183 | 0.0190 | 1.0000 |  |  | 0.8489 |  |
| S | mle | 0.0359 | 0.0204 | 0.0283 | 1.0000 | 0.0000 |  | 0.8726 |  |
| S | et | 0.0299 | 0.0370 | 0.0235 | 1.0000 |  | 0.5833 | 0.8938 |  |
| S_obs | plugin | 0.2816 | 0.2110 | 0.2816 | 1.0000 |  |  | 0.0000 |  |
| S_obs | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  | 0.1489 |  |
| S_obs | design | 0.0447 | 0.0215 | 0.0023 | 1.0000 |  |  | 0.8413 |  |
| S_obs | mle | 0.0347 | 0.0225 | 0.0106 | 1.0000 | 0.0000 |  | 0.8766 |  |
| S_obs | et | 0.0349 | 0.0376 | 0.0264 | 1.0000 |  | 0.5000 | 0.8762 |  |
| H | plugin | 0.0666 | 0.0573 | -0.0619 | 1.0000 |  |  | 0.0000 |  |
| H | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  | -2.5988 |  |
| H | mle | 0.0674 | 0.0349 | 0.0668 | 1.0000 | 0.1667 |  | -0.0113 |  |
| H | et | 0.0372 | 0.0266 | 0.0271 | 1.0000 |  | 1.0000 | 0.4414 |  |
| B | plugin | 0.1525 | 0.0786 | -0.1525 | 1.0000 |  |  | 0.0000 |  |
| B | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  | -0.5716 |  |
| B | mle | 0.0551 | 0.0347 | -0.0266 | 1.0000 | 0.0833 |  | 0.6389 |  |
| B | et | 0.1049 | 0.0517 | 0.0932 | 1.0000 |  | 1.0000 | 0.3119 |  |

## Source-level paired inference

Difference is first method minus second method in source-level MAE_2.

| Arm | Comparison | Mean difference | Wins / 8 | Exact sign-flip p | LOSO range |
|---|---|---:|---:|---:|---:|
| R | qwen_thinking vs plugin |  |  |  |  |
| R | qwen_thinking vs plugin |  |  |  |  |
| R | plugin vs plugin | 0.0000 | 0 | 1.0000 | 0.0000 to 0.0000 |
| R | et vs plugin | 0.0042 | 1 | 0.1953 | 0.0023 to 0.0064 |
| S | qwen_thinking vs plugin |  |  |  |  |
| S | qwen_thinking vs design |  |  |  |  |
| S | design vs plugin | -0.2391 | 8 | 0.0078 | -0.2726 to -0.2148 |
| S | et vs design | -0.0126 | 2 | 0.7891 | -0.0206 to 0.0109 |
| S_obs | qwen_thinking vs plugin |  |  |  |  |
| S_obs | qwen_thinking vs design |  |  |  |  |
| S_obs | design vs plugin | -0.2369 | 8 | 0.0078 | -0.2702 to -0.2144 |
| S_obs | et vs design | -0.0098 | 2 | 0.6641 | -0.0177 to 0.0074 |
| H | qwen_thinking vs plugin |  |  |  |  |
| H | qwen_thinking vs mle |  |  |  |  |
| H | mle vs plugin | 0.0008 | 4 | 0.9766 | -0.0122 to 0.0163 |
| H | et vs mle | -0.0301 | 6 | 0.0938 | -0.0363 to -0.0131 |
| B | qwen_thinking vs plugin |  |  |  |  |
| B | qwen_thinking vs mle |  |  |  |  |
| B | mle vs plugin | -0.0974 | 6 | 0.0312 | -0.1176 to -0.0846 |
| B | et vs mle | 0.0499 | 2 | 0.0938 | 0.0373 to 0.0649 |
