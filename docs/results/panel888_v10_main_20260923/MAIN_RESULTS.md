# v10 main results

Primary metric: equal-source MAE_2 across eight real sources. ProfileMAE is secondary.
LLM accuracy uses valid final answers only; no estimate is clipped or repaired.
ET hyperparameters were selected by nested leave-one-real-training-source-out CV.
sp_hospital__pwt remains flagged as not correctable at the 10% S budget.

Qwen complete: True.

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Validity | Fallback | Old-rule MLE MAE_2 | Old-rule ProfileMAE | Old-rule signed rho_2 | Old-rule fallback | ET profile validity | Skill vs plugin | S flag |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R | plugin | 0.0209 | 0.0130 | -0.0044 | 1.0000 |  |  |  |  |  |  | 0.0000 |  |
| R | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  |  |  |  |  | -10.4768 |  |
| R | mle | 0.0234 | 0.0139 | 0.0074 | 1.0000 | 0.0000 | 0.0234 | 0.0139 | 0.0074 | 0.0417 |  | -0.1218 |  |
| R | et | 0.0229 | 0.0140 | -0.0028 | 1.0000 |  |  |  |  |  | 0.9583 | -0.0975 |  |
| R | qwen_thinking | 0.0213 | 0.0133 | -0.0056 | 1.0000 |  |  |  |  |  |  | -0.0194 |  |
| R | qwen_nonthinking | 0.4382 | 0.3403 | 0.4380 | 1.0000 |  |  |  |  |  |  | -19.9819 |  |
| S | plugin | 0.2816 | 0.2110 | 0.2816 | 1.0000 |  |  |  |  |  |  | 0.0000 |  |
| S | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  |  |  |  |  | 0.1489 |  |
| S | design | 0.0426 | 0.0183 | 0.0190 | 1.0000 |  |  |  |  |  |  | 0.8489 |  |
| S | mle | 0.0359 | 0.0204 | 0.0283 | 1.0000 | 0.0000 | 0.0359 | 0.0204 | 0.0283 | 0.0000 |  | 0.8726 |  |
| S | et | 0.0421 | 0.0192 | 0.0239 | 1.0000 |  |  |  |  |  | 0.9583 | 0.8506 |  |
| S | qwen_thinking | 0.1513 | 0.1169 | 0.1311 | 1.0000 |  |  |  |  |  |  | 0.4629 |  |
| S | qwen_nonthinking | 0.5563 | 0.4720 | 0.5563 | 1.0000 |  |  |  |  |  |  | -0.9751 |  |
| S_obs | plugin | 0.2816 | 0.2110 | 0.2816 | 1.0000 |  |  |  |  |  |  | 0.0000 |  |
| S_obs | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  |  |  |  |  | 0.1489 |  |
| S_obs | design | 0.0447 | 0.0215 | 0.0023 | 1.0000 |  |  |  |  |  |  | 0.8413 |  |
| S_obs | mle | 0.0347 | 0.0225 | 0.0106 | 1.0000 | 0.0000 | 0.0347 | 0.0225 | 0.0106 | 0.0000 |  | 0.8766 |  |
| S_obs | et | 0.0486 | 0.0254 | 0.0346 | 1.0000 |  |  |  |  |  | 1.0000 | 0.8274 |  |
| S_obs | qwen_thinking | 0.1212 | 0.0797 | 0.0948 | 1.0000 |  |  |  |  |  |  | 0.5695 |  |
| S_obs | qwen_nonthinking | 0.5028 | 0.4464 | 0.5027 | 1.0000 |  |  |  |  |  |  | -0.7852 |  |
| H | plugin | 0.0666 | 0.0573 | -0.0619 | 1.0000 |  |  |  |  |  |  | 0.0000 |  |
| H | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  |  |  |  |  | -2.5988 |  |
| H | mle | 0.0516 | 0.0289 | 0.0511 | 1.0000 | 0.0000 | 0.0674 | 0.0349 | 0.0668 | 0.1667 |  | 0.2246 |  |
| H | et | 0.0304 | 0.0253 | 0.0118 | 1.0000 |  |  |  |  |  | 1.0000 | 0.5440 |  |
| H | qwen_thinking | 0.1434 | 0.0805 | 0.0585 | 0.9861 |  |  |  |  |  |  | -1.1536 |  |
| H | qwen_nonthinking | 0.2606 | 0.1568 | 0.0925 | 1.0000 |  |  |  |  |  |  | -2.9120 |  |
| B | plugin | 0.1525 | 0.0786 | -0.1525 | 1.0000 |  |  |  |  |  |  | 0.0000 |  |
| B | median | 0.2397 | 0.1102 | 0.0590 | 1.0000 |  |  |  |  |  |  | -0.5716 |  |
| B | mle | 0.0551 | 0.0347 | -0.0266 | 1.0000 | 0.0000 | 0.0551 | 0.0347 | -0.0266 | 0.0833 |  | 0.6389 |  |
| B | et | 0.0549 | 0.0322 | 0.0045 | 1.0000 |  |  |  |  |  | 1.0000 | 0.6398 |  |
| B | qwen_thinking | 0.2062 | 0.1186 | 0.0109 | 1.0000 |  |  |  |  |  |  | -0.3517 |  |
| B | qwen_nonthinking | 0.5630 | 0.5482 | 0.5462 | 1.0000 |  |  |  |  |  |  | -2.6910 |  |

## Source-level paired inference

Difference is first method minus second method in source-level MAE_2.

| Arm | Comparison | Mean difference | Wins / 8 | Exact sign-flip p | LOSO range |
|---|---|---:|---:|---:|---:|
| R | qwen_thinking vs plugin | 0.0004 | 4 | 0.8281 | -0.0003 to 0.0006 |
| R | et vs plugin | 0.0020 | 2 | 0.2812 | 0.0005 to 0.0030 |
| S | qwen_thinking vs plugin | -0.1304 | 8 | 0.0078 | -0.1486 to -0.1059 |
| S | qwen_thinking vs design | 0.1087 | 0 | 0.0078 | 0.0855 to 0.1240 |
| S | design vs plugin | -0.2391 | 8 | 0.0078 | -0.2726 to -0.2148 |
| S | et vs design | -0.0005 | 5 | 0.8047 | -0.0019 to 0.0014 |
| S_obs | qwen_thinking vs plugin | -0.1604 | 8 | 0.0078 | -0.1829 to -0.1428 |
| S_obs | qwen_thinking vs design | 0.0765 | 0 | 0.0078 | 0.0644 to 0.0873 |
| S_obs | design vs plugin | -0.2369 | 8 | 0.0078 | -0.2702 to -0.2144 |
| S_obs | et vs design | 0.0039 | 4 | 0.5625 | -0.0008 to 0.0062 |
| H | qwen_thinking vs plugin | 0.0768 | 1 | 0.0156 | 0.0612 to 0.0877 |
| H | qwen_thinking vs mle | 0.0916 | 1 | 0.0234 | 0.0687 to 0.1085 |
| H | mle vs plugin | -0.0150 | 5 | 0.5156 | -0.0301 to -0.0016 |
| H | et vs mle | -0.0213 | 6 | 0.0625 | -0.0267 to -0.0166 |
| B | qwen_thinking vs plugin | 0.0536 | 1 | 0.0625 | 0.0368 to 0.0691 |
| B | qwen_thinking vs mle | 0.1511 | 1 | 0.0156 | 0.1383 to 0.1759 |
| B | mle vs plugin | -0.0974 | 6 | 0.0312 | -0.1176 to -0.0846 |
| B | et vs mle | -0.0001 | 4 | 0.9922 | -0.0050 to 0.0092 |
