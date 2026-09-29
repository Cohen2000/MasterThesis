# Main results

How to read this: each number is an average error in estimating rho_2, the share of interacting
pairs that are active in at least two of five time windows. Lower is better; 0.03 means the estimate
is off by 3 percentage points on average. Errors are averaged per network first, then over networks,
so every network counts equally. The 12 real networks are the main analysis. Terms are explained in the [glossary](../../DESIGN.md#glossary).

- **MAE_2**: mean absolute error of rho_2 (main measure). **ProfileMAE**: the same over rho_2 to rho_5.
- **Signed error**: average error with its sign; positive means the method overestimates.
- **Valid**: share of answers that follow the answer format; only valid answers are scored.
- **(reference)**: the standard estimator for that sampling arm, which the others are compared with.
- Language-model answers are never corrected. ExtraTrees output is limited to a valid profile
  (values between 0 and 1, never increasing from rho_2 to rho_5).

## Real networks (main analysis, 12 networks)

| Sampling arm | Method | MAE_2 | ProfileMAE | Signed error | Valid | Networks |
|---|---|---:|---:|---:|---:|---:|
| R (random nodes) | Observed share (plug-in) (reference) | 0.0267 | 0.0220 | -0.0087 | 1.000 | 12/12 |
| R (random nodes) | Training median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| R (random nodes) | Statistical model (MLE) | 0.0372 | 0.0270 | 0.0225 | 1.000 | 12/12 |
| R (random nodes) | ExtraTrees (trained model) | 0.0282 | 0.0230 | -0.0085 | 1.000 | 12/12 |
| R (random nodes) | Qwen, thinking | 0.0269 | 0.0234 | -0.0095 | 1.000 | 12/12 |
| R (random nodes) | Qwen, no thinking | 0.4114 | 0.3280 | 0.4086 | 1.000 | 12/12 |
| R (random nodes) | DeepSeek Flash | 0.0267 | 0.0220 | -0.0087 | 1.000 | 12/12 |
| R (random nodes) | GPT-6 Sol | 0.0267 | 0.0220 | -0.0087 | 1.000 | 12/12 |
| R (random nodes) | GPT-6 Sol + Python | 0.0273 | 0.0221 | -0.0081 | 1.000 | 12/12 |
| S (random walk) | Observed share (plug-in) | 0.3006 | 0.2522 | 0.2836 | 1.000 | 12/12 |
| S (random walk) | Training median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| S (random walk) | Statistical model (MLE) (reference) | 0.0685 | 0.0415 | 0.0433 | 1.000 | 12/12 |
| S (random walk) | ExtraTrees (trained model) | 0.0504 | 0.0334 | 0.0233 | 1.000 | 12/12 |
| S (random walk) | Qwen, thinking | 0.1814 | 0.1489 | 0.1416 | 1.000 | 12/12 |
| S (random walk) | Qwen, no thinking | 0.4752 | 0.4210 | 0.4587 | 0.981 | 12/12 |
| S (random walk) | DeepSeek Flash | 0.0897 | 0.0481 | 0.0407 | 0.991 | 12/12 |
| S (random walk) | GPT-6 Sol | 0.0841 | 0.0402 | 0.0406 | 1.000 | 12/12 |
| S (random walk) | GPT-6 Sol + Python | 0.0813 | 0.0404 | 0.0380 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Observed share (plug-in) | 0.0639 | 0.0720 | -0.0533 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Training median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Statistical model (MLE) (reference) | 0.0706 | 0.0444 | 0.0676 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | ExtraTrees (trained model) | 0.0390 | 0.0316 | 0.0146 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Qwen, thinking | 0.1598 | 0.1021 | 0.0568 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Qwen, no thinking | 0.2516 | 0.1654 | 0.0577 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | DeepSeek Flash | 0.1326 | 0.0728 | 0.0948 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | GPT-6 Sol | 0.0537 | 0.0328 | 0.0305 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | GPT-6 Sol + Python | 0.0615 | 0.0384 | 0.0405 | 1.000 | 12/12 |
| B (random event loss) | Observed share (plug-in) | 0.1643 | 0.1049 | -0.1643 | 1.000 | 12/12 |
| B (random event loss) | Training median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| B (random event loss) | Statistical model (MLE) (reference) | 0.0757 | 0.0493 | -0.0085 | 1.000 | 12/12 |
| B (random event loss) | ExtraTrees (trained model) | 0.0789 | 0.0457 | 0.0230 | 1.000 | 12/12 |
| B (random event loss) | Qwen, thinking | 0.2196 | 0.1429 | -0.0152 | 1.000 | 12/12 |
| B (random event loss) | Qwen, no thinking | 0.5267 | 0.4988 | 0.5156 | 1.000 | 12/12 |
| B (random event loss) | DeepSeek Flash | 0.1724 | 0.1044 | -0.0262 | 1.000 | 12/12 |
| B (random event loss) | GPT-6 Sol | 0.1077 | 0.0777 | 0.0050 | 1.000 | 12/12 |
| B (random event loss) | GPT-6 Sol + Python | 0.1512 | 0.1102 | 0.0645 | 1.000 | 12/12 |

## Time-shuffled copies of the real networks (12, separate block)

| Sampling arm | Method | MAE_2 | ProfileMAE | Signed error | Valid | Networks |
|---|---|---:|---:|---:|---:|---:|
| R (random nodes) | Observed share (plug-in) (reference) | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| R (random nodes) | Training median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| R (random nodes) | Statistical model (MLE) | 0.0365 | 0.0335 | 0.0227 | 1.000 | 12/12 |
| R (random nodes) | ExtraTrees (trained model) | 0.0171 | 0.0212 | -0.0063 | 1.000 | 12/12 |
| R (random nodes) | Qwen, thinking | 0.0172 | 0.0233 | 0.0008 | 1.000 | 12/12 |
| R (random nodes) | Qwen, no thinking | 0.2604 | 0.2506 | 0.1723 | 1.000 | 12/12 |
| R (random nodes) | DeepSeek Flash | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| R (random nodes) | GPT-6 Sol | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| R (random nodes) | GPT-6 Sol + Python | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| S (random walk) | Observed share (plug-in) | 0.2456 | 0.3152 | 0.1945 | 1.000 | 12/12 |
| S (random walk) | Training median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| S (random walk) | Statistical model (MLE) (reference) | 0.0861 | 0.0902 | 0.0283 | 1.000 | 12/12 |
| S (random walk) | ExtraTrees (trained model) | 0.0618 | 0.0631 | -0.0546 | 1.000 | 12/12 |
| S (random walk) | Qwen, thinking | 0.1742 | 0.2004 | 0.1077 | 0.991 | 12/12 |
| S (random walk) | Qwen, no thinking | 0.4426 | 0.4427 | 0.0768 | 1.000 | 12/12 |
| S (random walk) | DeepSeek Flash | 0.1101 | 0.0902 | 0.0013 | 1.000 | 12/12 |
| S (random walk) | GPT-6 Sol | 0.0923 | 0.0745 | 0.0084 | 1.000 | 12/12 |
| S (random walk) | GPT-6 Sol + Python | 0.0878 | 0.0701 | 0.0005 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Observed share (plug-in) | 0.0945 | 0.2008 | -0.0934 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Training median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Statistical model (MLE) (reference) | 0.0491 | 0.0471 | 0.0150 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | ExtraTrees (trained model) | 0.0435 | 0.0597 | -0.0389 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Qwen, thinking | 0.1382 | 0.1880 | -0.0150 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | Qwen, no thinking | 0.3627 | 0.3015 | -0.2744 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | DeepSeek Flash | 0.0969 | 0.0746 | 0.0350 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | GPT-6 Sol | 0.0575 | 0.0428 | -0.0113 | 1.000 | 12/12 |
| H (random nodes, last 60% of time) | GPT-6 Sol + Python | 0.0504 | 0.0430 | -0.0104 | 1.000 | 12/12 |
| B (random event loss) | Observed share (plug-in) | 0.3251 | 0.2827 | -0.3251 | 1.000 | 12/12 |
| B (random event loss) | Training median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| B (random event loss) | Statistical model (MLE) (reference) | 0.1321 | 0.1516 | -0.1063 | 1.000 | 12/12 |
| B (random event loss) | ExtraTrees (trained model) | 0.1186 | 0.1317 | -0.0764 | 1.000 | 12/12 |
| B (random event loss) | Qwen, thinking | 0.3353 | 0.2975 | -0.1974 | 1.000 | 12/12 |
| B (random event loss) | Qwen, no thinking | 0.2669 | 0.2887 | 0.2388 | 1.000 | 12/12 |
| B (random event loss) | DeepSeek Flash | 0.2925 | 0.2632 | -0.1338 | 1.000 | 12/12 |
| B (random event loss) | GPT-6 Sol | 0.1329 | 0.1221 | -0.0181 | 1.000 | 12/12 |
| B (random event loss) | GPT-6 Sol + Python | 0.1279 | 0.1211 | 0.0108 | 1.000 | 12/12 |

## Synthetic networks (8, separate block)

| Sampling arm | Method | MAE_2 | ProfileMAE | Signed error | Valid | Networks |
|---|---|---:|---:|---:|---:|---:|
| R (random nodes) | Observed share (plug-in) (reference) | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| R (random nodes) | Training median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| R (random nodes) | Statistical model (MLE) | 0.0219 | 0.0152 | -0.0077 | 1.000 | 8/8 |
| R (random nodes) | ExtraTrees (trained model) | 0.0226 | 0.0167 | -0.0034 | 1.000 | 8/8 |
| R (random nodes) | Qwen, thinking | 0.0235 | 0.0173 | -0.0033 | 1.000 | 8/8 |
| R (random nodes) | Qwen, no thinking | 0.2194 | 0.1734 | 0.1827 | 1.000 | 8/8 |
| R (random nodes) | DeepSeek Flash | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| R (random nodes) | GPT-6 Sol | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| R (random nodes) | GPT-6 Sol + Python | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| S (random walk) | Observed share (plug-in) | 0.1271 | 0.1083 | 0.1271 | 1.000 | 8/8 |
| S (random walk) | Training median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| S (random walk) | Statistical model (MLE) (reference) | 0.0250 | 0.0201 | 0.0085 | 1.000 | 8/8 |
| S (random walk) | ExtraTrees (trained model) | 0.0184 | 0.0164 | 0.0032 | 1.000 | 8/8 |
| S (random walk) | Qwen, thinking | 0.0786 | 0.0627 | 0.0614 | 1.000 | 8/8 |
| S (random walk) | Qwen, no thinking | 0.2807 | 0.2544 | 0.2750 | 1.000 | 8/8 |
| S (random walk) | DeepSeek Flash | 0.0310 | 0.0227 | 0.0028 | 1.000 | 8/8 |
| S (random walk) | GPT-6 Sol | 0.0295 | 0.0210 | 0.0081 | 1.000 | 8/8 |
| S (random walk) | GPT-6 Sol + Python | 0.0271 | 0.0199 | 0.0083 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | Observed share (plug-in) | 0.0948 | 0.1214 | -0.0902 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | Training median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | Statistical model (MLE) (reference) | 0.0375 | 0.0442 | 0.0217 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | ExtraTrees (trained model) | 0.0293 | 0.0191 | -0.0134 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | Qwen, thinking | 0.1631 | 0.1348 | 0.0426 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | Qwen, no thinking | 0.2435 | 0.1822 | -0.0826 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | DeepSeek Flash | 0.0578 | 0.0520 | 0.0355 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | GPT-6 Sol | 0.0361 | 0.0248 | 0.0057 | 1.000 | 8/8 |
| H (random nodes, last 60% of time) | GPT-6 Sol + Python | 0.0412 | 0.0279 | 0.0077 | 1.000 | 8/8 |
| B (random event loss) | Observed share (plug-in) | 0.4255 | 0.2543 | -0.4255 | 1.000 | 8/8 |
| B (random event loss) | Training median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| B (random event loss) | Statistical model (MLE) (reference) | 0.0754 | 0.0630 | -0.0206 | 1.000 | 8/8 |
| B (random event loss) | ExtraTrees (trained model) | 0.0365 | 0.0443 | -0.0190 | 1.000 | 8/8 |
| B (random event loss) | Qwen, thinking | 0.3303 | 0.2518 | -0.1291 | 1.000 | 8/8 |
| B (random event loss) | Qwen, no thinking | 0.3382 | 0.3083 | 0.1636 | 1.000 | 8/8 |
| B (random event loss) | DeepSeek Flash | 0.1970 | 0.1612 | -0.0148 | 1.000 | 8/8 |
| B (random event loss) | GPT-6 Sol | 0.1295 | 0.1037 | 0.0578 | 1.000 | 8/8 |
| B (random event loss) | GPT-6 Sol + Python | 0.1067 | 0.0917 | 0.0458 | 1.000 | 8/8 |

The eight synthetic networks come from four generators, two networks each, so these results
describe the generators and are not tested for significance.

## Real network versus its time-shuffled copy

For each of the 12 real networks: the error on the real network minus the error on its
time-shuffled copy (negative: the real network is easier). "p" is an exact sign-flip test over
the 12 pairs: the chance of a difference at least this large if real and copy were alike.

| Sampling arm | Method | Real | Shuffled copy | Difference | Real better | p |
|---|---|---:|---:|---:|---:|---:|
| R (random nodes) | Observed share (plug-in) | 0.0267 | 0.0160 | 0.0107 | 3/12 | 0.1465 |
| R (random nodes) | Training median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| R (random nodes) | Statistical model (MLE) | 0.0372 | 0.0365 | 0.0008 | 7/12 | 0.9438 |
| R (random nodes) | ExtraTrees (trained model) | 0.0282 | 0.0171 | 0.0112 | 4/12 | 0.0566 |
| R (random nodes) | Qwen, thinking | 0.0269 | 0.0172 | 0.0097 | 3/12 | 0.1558 |
| R (random nodes) | Qwen, no thinking | 0.4114 | 0.2604 | 0.1510 | 2/12 | 0.0151 |
| S (random walk) | Observed share (plug-in) | 0.3006 | 0.2456 | 0.0550 | 4/12 | 0.2339 |
| S (random walk) | Training median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| S (random walk) | Statistical model (MLE) | 0.0685 | 0.0861 | -0.0175 | 9/12 | 0.4141 |
| S (random walk) | ExtraTrees (trained model) | 0.0504 | 0.0618 | -0.0114 | 7/12 | 0.6328 |
| S (random walk) | Qwen, thinking | 0.1814 | 0.1742 | 0.0073 | 5/12 | 0.8394 |
| S (random walk) | Qwen, no thinking | 0.4752 | 0.4426 | 0.0326 | 3/12 | 0.5884 |
| H (random nodes, last 60% of time) | Observed share (plug-in) | 0.0639 | 0.0945 | -0.0305 | 7/12 | 0.2881 |
| H (random nodes, last 60% of time) | Training median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| H (random nodes, last 60% of time) | Statistical model (MLE) | 0.0706 | 0.0491 | 0.0214 | 5/12 | 0.1787 |
| H (random nodes, last 60% of time) | ExtraTrees (trained model) | 0.0390 | 0.0435 | -0.0045 | 4/12 | 0.7886 |
| H (random nodes, last 60% of time) | Qwen, thinking | 0.1598 | 0.1382 | 0.0216 | 6/12 | 0.4258 |
| H (random nodes, last 60% of time) | Qwen, no thinking | 0.2516 | 0.3627 | -0.1111 | 8/12 | 0.1196 |
| B (random event loss) | Observed share (plug-in) | 0.1643 | 0.3251 | -0.1608 | 12/12 | 0.0005 |
| B (random event loss) | Training median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| B (random event loss) | Statistical model (MLE) | 0.0757 | 0.1321 | -0.0564 | 9/12 | 0.0508 |
| B (random event loss) | ExtraTrees (trained model) | 0.0789 | 0.1186 | -0.0398 | 8/12 | 0.1704 |
| B (random event loss) | Qwen, thinking | 0.2196 | 0.3353 | -0.1156 | 12/12 | 0.0005 |
| B (random event loss) | Qwen, no thinking | 0.5267 | 0.2669 | 0.2598 | 0/12 | 0.0005 |

## Method against method

Error of the first method minus error of the second, per network and then averaged
(negative: the first method is better). "First better" counts the networks it wins; p is the
exact sign-flip test over networks.

| Networks | Sampling arm | Comparison | Difference | First better | p |
|---|---|---:|---:|---:|---:|
| real | R (random nodes) | ExtraTrees (trained model) vs Observed share (plug-in) | 0.0015 | 3/12 | 0.2212 |
| real | R (random nodes) | Qwen, thinking vs Observed share (plug-in) | 0.0002 | 5/12 | 0.1729 |
| real | S (random walk) | ExtraTrees (trained model) vs Statistical model (MLE) | -0.0181 | 7/12 | 0.0781 |
| real | S (random walk) | Qwen, thinking vs Statistical model (MLE) | 0.1129 | 0/12 | 0.0005 |
| real | S (random walk) | Statistical model (MLE) vs Observed share (plug-in) | -0.2321 | 12/12 | 0.0005 |
| real | S (random walk) | ExtraTrees (trained model) vs Observed share (plug-in) | -0.2502 | 11/12 | 0.0010 |
| real | S (random walk) | Qwen, thinking vs Observed share (plug-in) | -0.1192 | 12/12 | 0.0005 |
| real | H (random nodes, last 60% of time) | ExtraTrees (trained model) vs Statistical model (MLE) | -0.0316 | 9/12 | 0.0151 |
| real | H (random nodes, last 60% of time) | Qwen, thinking vs Statistical model (MLE) | 0.0892 | 2/12 | 0.0049 |
| real | H (random nodes, last 60% of time) | Statistical model (MLE) vs Observed share (plug-in) | 0.0066 | 6/12 | 0.7695 |
| real | H (random nodes, last 60% of time) | ExtraTrees (trained model) vs Observed share (plug-in) | -0.0250 | 8/12 | 0.1660 |
| real | H (random nodes, last 60% of time) | Qwen, thinking vs Observed share (plug-in) | 0.0958 | 1/12 | 0.0010 |
| real | B (random event loss) | ExtraTrees (trained model) vs Statistical model (MLE) | 0.0032 | 5/12 | 0.7314 |
| real | B (random event loss) | Qwen, thinking vs Statistical model (MLE) | 0.1439 | 1/12 | 0.0015 |
| real | B (random event loss) | Statistical model (MLE) vs Observed share (plug-in) | -0.0885 | 8/12 | 0.0210 |
| real | B (random event loss) | ExtraTrees (trained model) vs Observed share (plug-in) | -0.0854 | 8/12 | 0.0435 |
| real | B (random event loss) | Qwen, thinking vs Observed share (plug-in) | 0.0554 | 2/12 | 0.0171 |
| time-shuffled copy | R (random nodes) | ExtraTrees (trained model) vs Observed share (plug-in) | 0.0011 | 5/12 | 0.5767 |
| time-shuffled copy | R (random nodes) | Qwen, thinking vs Observed share (plug-in) | 0.0012 | 6/12 | 0.0991 |
| time-shuffled copy | S (random walk) | ExtraTrees (trained model) vs Statistical model (MLE) | -0.0243 | 8/12 | 0.1670 |
| time-shuffled copy | S (random walk) | Qwen, thinking vs Statistical model (MLE) | 0.0881 | 0/12 | 0.0005 |
| time-shuffled copy | S (random walk) | Statistical model (MLE) vs Observed share (plug-in) | -0.1595 | 12/12 | 0.0005 |
| time-shuffled copy | S (random walk) | ExtraTrees (trained model) vs Observed share (plug-in) | -0.1838 | 12/12 | 0.0005 |
| time-shuffled copy | S (random walk) | Qwen, thinking vs Observed share (plug-in) | -0.0714 | 10/12 | 0.0020 |
| time-shuffled copy | H (random nodes, last 60% of time) | ExtraTrees (trained model) vs Statistical model (MLE) | -0.0056 | 6/12 | 0.6211 |
| time-shuffled copy | H (random nodes, last 60% of time) | Qwen, thinking vs Statistical model (MLE) | 0.0891 | 0/12 | 0.0005 |
| time-shuffled copy | H (random nodes, last 60% of time) | Statistical model (MLE) vs Observed share (plug-in) | -0.0454 | 9/12 | 0.0234 |
| time-shuffled copy | H (random nodes, last 60% of time) | ExtraTrees (trained model) vs Observed share (plug-in) | -0.0510 | 11/12 | 0.0015 |
| time-shuffled copy | H (random nodes, last 60% of time) | Qwen, thinking vs Observed share (plug-in) | 0.0437 | 0/12 | 0.0005 |
| time-shuffled copy | B (random event loss) | ExtraTrees (trained model) vs Statistical model (MLE) | -0.0135 | 9/12 | 0.2358 |
| time-shuffled copy | B (random event loss) | Qwen, thinking vs Statistical model (MLE) | 0.2031 | 0/12 | 0.0005 |
| time-shuffled copy | B (random event loss) | Statistical model (MLE) vs Observed share (plug-in) | -0.1930 | 11/12 | 0.0010 |
| time-shuffled copy | B (random event loss) | ExtraTrees (trained model) vs Observed share (plug-in) | -0.2064 | 11/12 | 0.0010 |
| time-shuffled copy | B (random event loss) | Qwen, thinking vs Observed share (plug-in) | 0.0102 | 7/12 | 0.6089 |

## Error per real network (MAE_2)

| Network | Sampling arm | Observed share (plug-in) | Training median | Statistical model (MLE) | ExtraTrees (trained model) | Qwen, thinking | Qwen, no thinking |
|---|---|---:|---:|---:|---:|---:|---:|
| copenhagen_bluetooth | R | 0.0121 | 0.1568 | 0.0096 | 0.0131 | 0.0121 | 0.5312 |
| copenhagen_bluetooth | S | 0.4230 | 0.1568 | 0.0102 | 0.0149 | 0.2053 | 0.5056 |
| copenhagen_bluetooth | H | 0.0955 | 0.1568 | 0.0298 | 0.0061 | 0.1715 | 0.2377 |
| copenhagen_bluetooth | B | 0.1925 | 0.1568 | 0.0049 | 0.0288 | 0.2363 | 0.5161 |
| lkml_reply | R | 0.0099 | 0.2016 | 0.0517 | 0.0169 | 0.0099 | 0.6956 |
| lkml_reply | S | 0.2121 | 0.2016 | 0.1071 | 0.0652 | 0.1772 | 0.3400 |
| lkml_reply | H | 0.0069 | 0.2016 | 0.1136 | 0.0441 | 0.3036 | 0.4584 |
| lkml_reply | B | 0.0454 | 0.2016 | 0.0863 | 0.1049 | 0.1969 | 0.8066 |
| nr_digg_reply | R | 0.0007 | 0.4158 | 0.0008 | 0.0076 | 0.0007 | 0.2082 |
| nr_digg_reply | S | 0.0049 | 0.4158 | 0.0006 | 0.0070 | 0.0019 | 0.6697 |
| nr_digg_reply | H | 0.0016 | 0.4158 | 0.0003 | 0.0133 | 0.0630 | 0.0401 |
| nr_digg_reply | B | 0.0022 | 0.4158 | 0.0058 | 0.0300 | 0.0473 | 0.7352 |
| nr_radoslaw_email | R | 0.0409 | 0.2610 | 0.0684 | 0.0422 | 0.0409 | 0.2639 |
| nr_radoslaw_email | S | 0.3843 | 0.2610 | 0.0651 | 0.0267 | 0.1673 | 0.3636 |
| nr_radoslaw_email | H | 0.0438 | 0.2610 | 0.1561 | 0.1204 | 0.1228 | 0.3795 |
| nr_radoslaw_email | B | 0.2619 | 0.2610 | 0.0188 | 0.0374 | 0.2896 | 0.3024 |
| reality_mining | R | 0.0213 | 0.2521 | 0.0266 | 0.0240 | 0.0213 | 0.2819 |
| reality_mining | S | 0.3280 | 0.2521 | 0.1045 | 0.0869 | 0.2531 | 0.2912 |
| reality_mining | H | 0.1281 | 0.2521 | 0.0255 | 0.0525 | 0.2040 | 0.1518 |
| reality_mining | B | 0.3263 | 0.2521 | 0.1614 | 0.1253 | 0.2939 | 0.2881 |
| snap_collegemsg | R | 0.0030 | 0.3158 | 0.0083 | 0.0068 | 0.0031 | 0.2244 |
| snap_collegemsg | S | 0.1452 | 0.3158 | 0.0033 | 0.0033 | 0.0312 | 0.6453 |
| snap_collegemsg | H | 0.0187 | 0.3158 | 0.1098 | 0.0645 | 0.1769 | 0.3183 |
| snap_collegemsg | B | 0.0718 | 0.3158 | 0.0081 | 0.0421 | 0.2436 | 0.4473 |
| snap_email_eu | R | 0.0283 | 0.2156 | 0.0319 | 0.0256 | 0.0283 | 0.3535 |
| snap_email_eu | S | 0.4097 | 0.2156 | 0.0157 | 0.0231 | 0.3044 | 0.4357 |
| snap_email_eu | H | 0.1244 | 0.2156 | 0.0632 | 0.0476 | 0.1435 | 0.1950 |
| snap_email_eu | B | 0.2412 | 0.2156 | 0.0776 | 0.0660 | 0.2869 | 0.4641 |
| snap_mathoverflow | R | 0.0054 | 0.3345 | 0.0094 | 0.0043 | 0.0054 | 0.7143 |
| snap_mathoverflow | S | 0.1529 | 0.3345 | 0.0146 | 0.0119 | 0.0570 | 0.7476 |
| snap_mathoverflow | H | 0.0198 | 0.3345 | 0.0449 | 0.0125 | 0.1031 | 0.1283 |
| snap_mathoverflow | B | 0.0538 | 0.3345 | 0.0972 | 0.1124 | 0.0748 | 0.7938 |
| sp_highschool2013 | R | 0.0137 | 0.1954 | 0.0031 | 0.0169 | 0.0137 | 0.4288 |
| sp_highschool2013 | S | 0.3976 | 0.1954 | 0.0555 | 0.0647 | 0.2950 | 0.4341 |
| sp_highschool2013 | H | 0.1165 | 0.1954 | 0.0083 | 0.0198 | 0.2227 | 0.1514 |
| sp_highschool2013 | B | 0.2267 | 0.1954 | 0.1023 | 0.0964 | 0.1723 | 0.4737 |
| sp_hospital | R | 0.0778 | 0.1550 | 0.0833 | 0.0790 | 0.0786 | 0.4309 |
| sp_hospital | S | 0.3395 | 0.1550 | 0.1563 | 0.1434 | 0.2369 | 0.4817 |
| sp_hospital | H | 0.0687 | 0.1550 | 0.0957 | 0.0448 | 0.2239 | 0.2810 |
| sp_hospital | B | 0.2862 | 0.1550 | 0.1053 | 0.0401 | 0.3081 | 0.5123 |
| sp_malawi | R | 0.0814 | 0.1527 | 0.1126 | 0.0739 | 0.0827 | 0.2954 |
| sp_malawi | S | 0.4300 | 0.1527 | 0.2583 | 0.1357 | 0.3695 | 0.2571 |
| sp_malawi | H | 0.0555 | 0.1527 | 0.1383 | 0.0299 | 0.0946 | 0.3087 |
| sp_malawi | B | 0.1172 | 0.1527 | 0.2013 | 0.2392 | 0.2058 | 0.4198 |
| sp_workplace | R | 0.0262 | 0.1287 | 0.0410 | 0.0281 | 0.0262 | 0.5088 |
| sp_workplace | S | 0.3803 | 0.1287 | 0.0309 | 0.0218 | 0.0785 | 0.5304 |
| sp_workplace | H | 0.0877 | 0.1287 | 0.0612 | 0.0123 | 0.0877 | 0.3690 |
| sp_workplace | B | 0.1459 | 0.1287 | 0.0393 | 0.0236 | 0.2799 | 0.5611 |

All networks, the API models and all measures: `PER_SOURCE.csv`.

## Appendix: re-weighted walk estimate (arm S)

The random walk visits busy pairs more often. Re-weighting each observed pair by the inverse of its
number of events gives a simple corrected share (the "design ratio"). It is shown for comparison
only and is not one of the four estimators. The second table shows how often the valid answers of
each language model lie within 0.005 of the observed share, the design ratio or the MLE.

| Networks | Design ratio MAE_2 |
|---|---|
| real | 0.0880 |
| time-shuffled copy | 0.0996 |
| synthetic | 0.0284 |

| Model (real networks, arm S) | Valid answers | Near observed share | Near design ratio | Near MLE |
|---|---|---:|---:|---:|
| Qwen, thinking | 108 | 0.500 | 0.204 | 0.130 |
| Qwen, no thinking | 106 | 0.066 | 0.047 | 0.047 |
| DeepSeek Flash | 107 | 0.159 | 0.682 | 0.243 |
| GPT-6 Sol | 108 | 0.120 | 0.898 | 0.241 |
| GPT-6 Sol + Python | 108 | 0.120 | 0.926 | 0.231 |

Per group and model: `S_DESIGN_APPENDIX.csv`.
