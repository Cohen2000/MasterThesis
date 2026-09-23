# Analysis snapshot: final v9, Main + coverage grid

Five main methods: plugin, shared_mle, extratrees (mixed-budget), qwen_thinking, qwen_nonthinking.
MAE2 = equal-graph mean absolute error of rho_2 (each real source counts once; surrogates and
synthetic instances are separate blocks, never extra real sources). LLM: formally valid answers only.
The mechanism-aware secondary reference is in MAIN_SECONDARY_REFERENCE.csv, not among the five.

## A) Main, real sources — MAE2

| arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R | 0.0209 | 0.0234 | 0.0303 | 0.0152 | 0.4442 |
| S1 | 0.0607 | 0.0671 | 0.0709 | 0.0595 | 0.4613 |
| H | 0.0666 | 0.0750 | 0.0411 | 0.1646 | 0.2417 |
| B | 0.1525 | 0.0580 | 0.0554 | 0.1935 | 0.4934 |

ProfileMAE:

| arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R | 0.0130 | 0.0139 | 0.0192 | 0.0116 | 0.3251 |
| S1 | 0.0286 | 0.0295 | 0.0383 | 0.0296 | 0.3546 |
| H | 0.0573 | 0.0378 | 0.0308 | 0.0936 | 0.1442 |
| B | 0.0786 | 0.0354 | 0.0411 | 0.1029 | 0.4305 |

## C) Main, surrogates — MAE2

| arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R | 0.0135 | 0.0337 | 0.0267 | 0.0157 | 0.2905 |
| S1 | 0.0744 | 0.1060 | 0.0460 | 0.0578 | 0.2921 |
| H | 0.1012 | 0.0661 | 0.0698 | 0.1554 | 0.3229 |
| B | 0.3058 | 0.1594 | 0.1067 | 0.3062 | 0.3032 |

ProfileMAE:

| arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R | 0.0163 | 0.0290 | 0.0316 | 0.0129 | 0.2538 |
| S1 | 0.0640 | 0.0823 | 0.0427 | 0.0462 | 0.2597 |
| H | 0.1736 | 0.0608 | 0.0786 | 0.1691 | 0.2506 |
| B | 0.2404 | 0.1474 | 0.1153 | 0.2577 | 0.3121 |

## D) Main, synthetic — MAE2

| arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R | 0.0235 | 0.0219 | 0.0209 | 0.0199 | 0.2076 |
| S1 | 0.0332 | 0.0305 | 0.0213 | 0.0281 | 0.2223 |
| H | 0.0948 | 0.0375 | 0.0255 | 0.1572 | 0.2575 |
| B | 0.4255 | 0.0754 | 0.0386 | 0.3780 | 0.3147 |

ProfileMAE:

| arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R | 0.0168 | 0.0152 | 0.0152 | 0.0147 | 0.1584 |
| S1 | 0.0224 | 0.0230 | 0.0164 | 0.0204 | 0.1829 |
| H | 0.1214 | 0.0442 | 0.0178 | 0.1343 | 0.1873 |
| B | 0.2543 | 0.0630 | 0.0446 | 0.2651 | 0.2808 |

## B) Coverage grid, real sources — MAE2

| budget | arm | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|---|
| 0.025 | R | 0.0245 | 0.0256 | 0.0355 | 0.0287 | 0.4242 |
| 0.025 | S1 | 0.0790 | 0.0854 | 0.0805 | 0.0807 | 0.3744 |
| 0.025 | H | 0.0840 | 0.0758 | 0.0513 | 0.2078 | 0.2377 |
| 0.025 | B | 0.2269 | 0.0906 | 0.1218 | 0.2980 | 0.4444 |
| 0.050 | R | 0.0137 | 0.0129 | 0.0274 | 0.0260 | 0.3746 |
| 0.050 | S1 | 0.0727 | 0.0792 | 0.0677 | 0.0680 | 0.4416 |
| 0.050 | H | 0.0630 | 0.0745 | 0.0465 | 0.1773 | 0.2093 |
| 0.050 | B | 0.1951 | 0.0634 | 0.0735 | 0.2358 | 0.4095 |
| 0.100 | R | 0.0209 | 0.0234 | 0.0303 | 0.0152 | 0.4442 |
| 0.100 | S1 | 0.0607 | 0.0671 | 0.0709 | 0.0595 | 0.4613 |
| 0.100 | H | 0.0666 | 0.0750 | 0.0411 | 0.1646 | 0.2417 |
| 0.100 | B | 0.1525 | 0.0580 | 0.0554 | 0.1935 | 0.4934 |
| 0.200 | R | 0.0108 | 0.0158 | 0.0191 | 0.0121 | 0.4265 |
| 0.200 | S1 | 0.0575 | 0.0683 | 0.0520 | 0.0534 | 0.4139 |
| 0.200 | H | 0.0718 | 0.0614 | 0.0382 | 0.1637 | 0.2333 |
| 0.200 | B | 0.1095 | 0.0540 | 0.0521 | 0.1918 | 0.5472 |
| 0.300 | R | 0.0114 | 0.0167 | 0.0237 | 0.0123 | 0.4450 |
| 0.300 | S1 | 0.0448 | 0.0618 | 0.0395 | 0.0462 | 0.4165 |
| 0.300 | H | 0.0641 | 0.0890 | 0.0414 | 0.1969 | 0.2465 |
| 0.300 | B | 0.0796 | 0.0493 | 0.0459 | 0.1448 | 0.5340 |
| 0.400 | R | 0.0129 | 0.0217 | 0.0235 | 0.0085 | 0.4352 |
| 0.400 | S1 | 0.0356 | 0.0604 | 0.0311 | 0.0383 | 0.4493 |
| 0.400 | H | 0.0706 | 0.0634 | 0.0384 | 0.1854 | 0.2351 |
| 0.400 | B | 0.0635 | 0.0407 | 0.0421 | 0.1772 | 0.5672 |
| 0.500 | R | 0.0068 | 0.0128 | 0.0182 | 0.0061 | 0.4484 |
| 0.500 | S1 | 0.0319 | 0.0613 | 0.0296 | 0.0312 | 0.4185 |
| 0.500 | H | 0.0705 | 0.0708 | 0.0366 | 0.1607 | 0.3063 |
| 0.500 | B | 0.0482 | 0.0321 | 0.0374 | 0.1223 | 0.5422 |

## LLM validity (share of formally valid answers), all blocks

| coverage | qwen_thinking | qwen_nonthinking |
|---|---|---|
| 0.025 | 0.9988 (1 invalid) | 1.0000 (0 invalid) |
| 0.050 | 1.0000 (0 invalid) | 1.0000 (0 invalid) |
| 0.100 | 1.0000 (0 invalid) | 1.0000 (0 invalid) |
| 0.200 | 0.9988 (1 invalid) | 1.0000 (0 invalid) |
| 0.300 | 0.9977 (2 invalid) | 1.0000 (0 invalid) |
| 0.400 | 0.9988 (1 invalid) | 1.0000 (0 invalid) |
| 0.500 | 0.9976 (2 invalid) | 1.0000 (0 invalid) |

## E) shared_mle fallback and adequacy

| coverage | observations | fallbacks | rate |
|---|---|---|---|
| 0.025 | 288 | 16 | 0.056 |
| 0.050 | 288 | 24 | 0.083 |
| 0.100 | 288 | 20 | 0.069 |
| 0.200 | 286 | 25 | 0.087 |
| 0.300 | 284 | 28 | 0.099 |
| 0.400 | 284 | 23 | 0.081 |
| 0.500 | 276 | 23 | 0.083 |

Adequacy (development/training material only; model not changed on these numbers):

- full_data: n=116, MAE2=0.0073, fallback 5/116
- h_like: n=116, MAE2=0.0345, fallback 14/116
- b_like: n=500, MAE2=0.1120, fallback 62/500

## F) extratrees profile validity (prediction in [0,1] and non-increasing)

| coverage | predictions | valid profiles | rate |
|---|---|---|---|
| 0.025 | 288 | 269 | 0.934 |
| 0.050 | 288 | 267 | 0.927 |
| 0.100 | 288 | 271 | 0.941 |
| 0.200 | 286 | 268 | 0.937 |
| 0.300 | 284 | 265 | 0.933 |
| 0.400 | 284 | 265 | 0.933 |
| 0.500 | 276 | 259 | 0.938 |
