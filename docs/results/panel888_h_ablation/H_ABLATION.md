# H history-fraction ablation (arm H only; h=0.60 is the final v9 main arm)

h=0.40 and higher coverage levels are stress tests, not design choices. h=0.50 is not run: its cutoff
falls inside window 3, which the 0/1 Temporal_access contract cannot express (see CURRENT_STATE.md).
MAE2 = equal-graph mean |error of rho_2|; LLM on formally valid answers only; the mechanism-aware
reference (homogeneous ZT-Binomial on the visible windows) is shown separately.
shared_mle at h=0.40 is NOT identified: with m=2 visible windows the zero-truncated Beta-Binomial has one
free cell share but two parameters, so all starts reach the same likelihood with different W=5
extrapolations (rho_2 spread up to ~0.13 on one observation); its h=0.40 numbers are an optimizer artefact.
The model is used unchanged, as specified. Qwen h=0.40 columns fill in once its chain is archived.

## A) Main (coverage 0.10), real — MAE2

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | 0.1407 | 0.1475 | 0.0978 | — | — | 0.1826 |
| 0.600 | 0.0666 | 0.0750 | 0.0411 | 0.1646 | 0.2417 | 0.1343 |

## A) Main (coverage 0.10), real — ProfileMAE

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | 0.0896 | 0.0659 | 0.0531 | — | — | 0.0788 |
| 0.600 | 0.0573 | 0.0378 | 0.0308 | 0.0936 | 0.1442 | 0.0557 |

## A) Main (coverage 0.10), real — signed_rho2

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | -0.1407 | +0.1382 | +0.0101 | — | — | +0.1786 |
| 0.600 | -0.0619 | +0.0744 | +0.0188 | +0.0694 | +0.0935 | +0.1343 |

## C1) Main, surrogate — MAE2

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | 0.1955 | 0.1381 | 0.0968 | — | — | 0.1701 |
| 0.600 | 0.1012 | 0.0661 | 0.0698 | 0.1554 | 0.3229 | 0.1257 |

## C1) Main, surrogate — ProfileMAE

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | 0.2512 | 0.1147 | 0.0896 | — | — | 0.1446 |
| 0.600 | 0.1736 | 0.0608 | 0.0786 | 0.1691 | 0.2506 | 0.1051 |

## C1) Main, surrogate — signed_rho2

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | -0.1955 | +0.0945 | -0.0800 | — | — | +0.1357 |
| 0.600 | -0.0996 | +0.0310 | -0.0643 | +0.0248 | -0.2608 | +0.1206 |

## C2) Main, synthetic — MAE2

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | 0.1823 | 0.0772 | 0.0416 | — | — | 0.1040 |
| 0.600 | 0.0948 | 0.0375 | 0.0255 | 0.1572 | 0.2575 | 0.0972 |

## C2) Main, synthetic — ProfileMAE

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | 0.1954 | 0.0756 | 0.0213 | — | — | 0.0963 |
| 0.600 | 0.1214 | 0.0442 | 0.0178 | 0.1343 | 0.1873 | 0.0768 |

## C2) Main, synthetic — signed_rho2

| h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|
| 0.400 | -0.1823 | +0.0550 | -0.0281 | — | — | +0.0946 |
| 0.600 | -0.0902 | +0.0217 | -0.0044 | +0.0206 | -0.0781 | +0.0849 |

## B) Coverage x h, real — MAE2

| budget | h | plugin | shared_mle | extratrees | qwen_thinking | qwen_nonthinking | mechanism_aware_reference |
|---|---|---|---|---|---|---|---|
| 0.025 | 0.400 | 0.1475 | 0.1426 | 0.1054 | — | — | 0.1830 |
| 0.025 | 0.600 | 0.0840 | 0.0758 | 0.0513 | 0.2078 | 0.2377 | 0.1285 |
| 0.050 | 0.400 | 0.1359 | 0.1406 | 0.1174 | — | — | 0.1943 |
| 0.050 | 0.600 | 0.0630 | 0.0745 | 0.0465 | 0.1773 | 0.2093 | 0.1290 |
| 0.100 | 0.400 | 0.1407 | 0.1475 | 0.0978 | — | — | 0.1826 |
| 0.100 | 0.600 | 0.0666 | 0.0750 | 0.0411 | 0.1646 | 0.2417 | 0.1343 |
| 0.200 | 0.400 | 0.1423 | 0.1651 | 0.0948 | — | — | 0.1850 |
| 0.200 | 0.600 | 0.0718 | 0.0614 | 0.0382 | 0.1637 | 0.2333 | 0.1291 |
| 0.300 | 0.400 | 0.1427 | 0.1666 | 0.0918 | — | — | 0.1848 |
| 0.300 | 0.600 | 0.0641 | 0.0890 | 0.0414 | 0.1969 | 0.2465 | 0.1357 |
| 0.400 | 0.400 | 0.1437 | 0.1593 | 0.0926 | — | — | 0.1830 |
| 0.400 | 0.600 | 0.0706 | 0.0634 | 0.0384 | 0.1854 | 0.2351 | 0.1259 |
| 0.500 | 0.400 | 0.1441 | 0.1465 | 0.0908 | — | — | 0.1821 |
| 0.500 | 0.600 | 0.0705 | 0.0708 | 0.0366 | 0.1607 | 0.3063 | 0.1268 |

## D) Feasibility (24 main graphs per cell)

| h | coverage | target unreachable | outside ±5% | saturated panel |
|---|---|---|---|---|
| 0.40 | 0.025 | 0 | 0 | 0 |
| 0.40 | 0.050 | 0 | 0 | 0 |
| 0.40 | 0.100 | 1 | 1 | 1 |
| 0.40 | 0.200 | 2 | 2 | 2 |
| 0.40 | 0.300 | 4 | 4 | 4 |
| 0.40 | 0.400 | 12 | 7 | 14 |
| 0.40 | 0.500 | 24 | 23 | 24 |
| 0.60 | 0.025 | 0 | 0 | 0 |
| 0.60 | 0.050 | 0 | 0 | 0 |
| 0.60 | 0.100 | 0 | 0 | 0 |
| 0.60 | 0.200 | 1 | 1 | 1 |
| 0.60 | 0.300 | 2 | 2 | 2 |
| 0.60 | 0.400 | 2 | 2 | 2 |
| 0.60 | 0.500 | 6 | 3 | 6 |

## E) Plugin signed rho_2 error (real / surrogate / synthetic)

| coverage | h | real | surrogate | synthetic |
|---|---|---|---|---|
| 0.025 | 0.40 | -0.1473 | -0.1909 | -0.1793 |
| 0.025 | 0.60 | -0.0825 | -0.0979 | -0.0976 |
| 0.050 | 0.40 | -0.1351 | -0.1968 | -0.1861 |
| 0.050 | 0.60 | -0.0607 | -0.1069 | -0.0976 |
| 0.100 | 0.40 | -0.1407 | -0.1955 | -0.1823 |
| 0.100 | 0.60 | -0.0619 | -0.0996 | -0.0902 |
| 0.200 | 0.40 | -0.1423 | -0.1961 | -0.1846 |
| 0.200 | 0.60 | -0.0669 | -0.1010 | -0.0901 |
| 0.300 | 0.40 | -0.1427 | -0.1958 | -0.1817 |
| 0.300 | 0.60 | -0.0593 | -0.1022 | -0.0902 |
| 0.400 | 0.40 | -0.1437 | -0.1972 | -0.1833 |
| 0.400 | 0.60 | -0.0658 | -0.1023 | -0.0913 |
| 0.500 | 0.40 | -0.1441 | -0.1975 | -0.1833 |
| 0.500 | 0.60 | -0.0657 | -0.1027 | -0.0923 |

## Validity

| coverage | h | method | evaluable | validity / fallback / valid-profile rate |
|---|---|---|---|---|
| 0.025 | 0.40 | shared_mle | 72 | 0.0000 |
| 0.025 | 0.40 | extratrees | 72 | 1.0000 |
| 0.025 | 0.60 | shared_mle | 72 | 0.0139 |
| 0.025 | 0.60 | extratrees | 72 | 1.0000 |
| 0.025 | 0.60 | qwen_thinking | 215 | 0.9954 |
| 0.025 | 0.60 | qwen_nonthinking | 216 | 1.0000 |
| 0.050 | 0.40 | shared_mle | 72 | 0.0000 |
| 0.050 | 0.40 | extratrees | 72 | 1.0000 |
| 0.050 | 0.60 | shared_mle | 72 | 0.0694 |
| 0.050 | 0.60 | extratrees | 72 | 1.0000 |
| 0.050 | 0.60 | qwen_thinking | 216 | 1.0000 |
| 0.050 | 0.60 | qwen_nonthinking | 216 | 1.0000 |
| 0.100 | 0.40 | shared_mle | 70 | 0.0000 |
| 0.100 | 0.40 | extratrees | 70 | 1.0000 |
| 0.100 | 0.60 | shared_mle | 72 | 0.0833 |
| 0.100 | 0.60 | extratrees | 72 | 1.0000 |
| 0.100 | 0.60 | qwen_thinking | 216 | 1.0000 |
| 0.100 | 0.60 | qwen_nonthinking | 216 | 1.0000 |
| 0.200 | 0.40 | shared_mle | 68 | 0.0000 |
| 0.200 | 0.40 | extratrees | 68 | 1.0000 |
| 0.200 | 0.60 | shared_mle | 70 | 0.1429 |
| 0.200 | 0.60 | extratrees | 70 | 1.0000 |
| 0.200 | 0.60 | qwen_thinking | 210 | 1.0000 |
| 0.200 | 0.60 | qwen_nonthinking | 210 | 1.0000 |
| 0.300 | 0.40 | shared_mle | 64 | 0.0000 |
| 0.300 | 0.40 | extratrees | 64 | 1.0000 |
| 0.300 | 0.60 | shared_mle | 68 | 0.2206 |
| 0.300 | 0.60 | extratrees | 68 | 1.0000 |
| 0.300 | 0.60 | qwen_thinking | 204 | 1.0000 |
| 0.300 | 0.60 | qwen_nonthinking | 204 | 1.0000 |
| 0.400 | 0.40 | shared_mle | 44 | 0.0000 |
| 0.400 | 0.40 | extratrees | 44 | 1.0000 |
| 0.400 | 0.60 | shared_mle | 68 | 0.1618 |
| 0.400 | 0.60 | extratrees | 68 | 1.0000 |
| 0.400 | 0.60 | qwen_thinking | 204 | 1.0000 |
| 0.400 | 0.60 | qwen_nonthinking | 204 | 1.0000 |
| 0.500 | 0.40 | shared_mle | 24 | 0.0000 |
| 0.500 | 0.40 | extratrees | 24 | 1.0000 |
| 0.500 | 0.60 | shared_mle | 60 | 0.1833 |
| 0.500 | 0.60 | extratrees | 60 | 1.0000 |
| 0.500 | 0.60 | qwen_thinking | 180 | 1.0000 |
| 0.500 | 0.60 | qwen_nonthinking | 180 | 1.0000 |
