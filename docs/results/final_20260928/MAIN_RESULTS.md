# Main results

Arms R, S, H and B; estimators plugin, median, MLE and ExtraTrees (ET, production fit), plus Qwen and
the paid APIs. Reference (ref.): plugin for R, MLE for S, H and B. Equal-source MAE_2 is primary,
ProfileMAE secondary; LLM scores use valid answers and ET scores all raw profiles, including invalid ones.
Nothing is clipped or repaired. The twelve real sources are
the main analysis and are weighted equally. A method missing any source of a block is marked
**pending** and not ranked. S_obs is a historical ablation and is not reported here.

## Real sources (main analysis, 12 sources)

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Validity | Sources |
|---|---|---:|---:|---:|---:|---:|
| R | plugin (ref.) | 0.0267 | 0.0220 | -0.0087 | 1.000 | 12/12 |
| R | median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| R | mle | 0.0372 | 0.0270 | 0.0225 | 1.000 | 12/12 |
| R | et | 0.0282 | 0.0230 | -0.0085 | 0.944 | 12/12 |
| R | qwen_thinking | 0.0269 | 0.0234 | -0.0095 | 1.000 | 12/12 |
| R | qwen_nonthinking | 0.4114 | 0.3280 | 0.4086 | 1.000 | 12/12 |
| R | deepseek_flash | 0.0267 | 0.0220 | -0.0087 | 1.000 | 12/12 |
| R | gpt_6_sol | 0.0267 | 0.0220 | -0.0087 | 1.000 | 12/12 |
| R | gpt_6_sol_tools | 0.0273 | 0.0221 | -0.0081 | 1.000 | 12/12 |
| S | plugin | 0.3006 | 0.2522 | 0.2836 | 1.000 | 12/12 |
| S | median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| S | mle (ref.) | 0.0685 | 0.0415 | 0.0433 | 1.000 | 12/12 |
| S | et | 0.0504 | 0.0334 | 0.0233 | 0.972 | 12/12 |
| S | qwen_thinking | 0.1814 | 0.1489 | 0.1416 | 1.000 | 12/12 |
| S | qwen_nonthinking | 0.4752 | 0.4210 | 0.4587 | 0.981 | 12/12 |
| S | deepseek_flash | 0.0873 | 0.0508 | 0.0341 | 0.972 | 12/12 |
| S | gpt_6_sol | 0.0841 | 0.0402 | 0.0406 | 1.000 | 12/12 |
| S | gpt_6_sol_tools | 0.0813 | 0.0404 | 0.0380 | 1.000 | 12/12 |
| H | plugin | 0.0639 | 0.0720 | -0.0533 | 1.000 | 12/12 |
| H | median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| H | mle (ref.) | 0.0706 | 0.0444 | 0.0676 | 1.000 | 12/12 |
| H | et | 0.0390 | 0.0316 | 0.0146 | 1.000 | 12/12 |
| H | qwen_thinking | 0.1598 | 0.1021 | 0.0568 | 1.000 | 12/12 |
| H | qwen_nonthinking | 0.2516 | 0.1654 | 0.0577 | 1.000 | 12/12 |
| H | deepseek_flash | 0.1184 | 0.0654 | 0.0772 | 1.000 | 12/12 |
| H | gpt_6_sol | 0.0537 | 0.0328 | 0.0305 | 1.000 | 12/12 |
| H | gpt_6_sol_tools | 0.0615 | 0.0384 | 0.0405 | 1.000 | 12/12 |
| B | plugin | 0.1643 | 0.1049 | -0.1643 | 1.000 | 12/12 |
| B | median | 0.2321 | 0.1287 | 0.0006 | 1.000 | 12/12 |
| B | mle (ref.) | 0.0757 | 0.0493 | -0.0085 | 1.000 | 12/12 |
| B | et | 0.0789 | 0.0457 | 0.0230 | 1.000 | 12/12 |
| B | qwen_thinking | 0.2196 | 0.1429 | -0.0152 | 1.000 | 12/12 |
| B | qwen_nonthinking | 0.5267 | 0.4988 | 0.5156 | 1.000 | 12/12 |
| B | deepseek_flash | 0.1949 | 0.1183 | 0.0344 | 1.000 | 12/12 |
| B | gpt_6_sol | 0.1077 | 0.0777 | 0.0050 | 1.000 | 12/12 |
| B | gpt_6_sol_tools | 0.1512 | 0.1102 | 0.0645 | 1.000 | 12/12 |

## Surrogates (12, separate block)

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Validity | Sources |
|---|---|---:|---:|---:|---:|---:|
| R | plugin (ref.) | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| R | median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| R | mle | 0.0365 | 0.0335 | 0.0227 | 1.000 | 12/12 |
| R | et | 0.0171 | 0.0214 | -0.0063 | 0.861 | 12/12 |
| R | qwen_thinking | 0.0172 | 0.0233 | 0.0008 | 1.000 | 12/12 |
| R | qwen_nonthinking | 0.2604 | 0.2506 | 0.1723 | 1.000 | 12/12 |
| R | deepseek_flash | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| R | gpt_6_sol | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| R | gpt_6_sol_tools | 0.0160 | 0.0211 | 0.0008 | 1.000 | 12/12 |
| S | plugin | 0.2456 | 0.3152 | 0.1945 | 1.000 | 12/12 |
| S | median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| S | mle (ref.) | 0.0861 | 0.0902 | 0.0283 | 1.000 | 12/12 |
| S | et | 0.0618 | 0.0631 | -0.0546 | 1.000 | 12/12 |
| S | qwen_thinking | 0.1742 | 0.2004 | 0.1077 | 0.991 | 12/12 |
| S | qwen_nonthinking | 0.4426 | 0.4427 | 0.0768 | 1.000 | 12/12 |
| S | deepseek_flash | 0.1251 | 0.1013 | -0.0031 | 1.000 | 12/12 |
| S | gpt_6_sol | 0.0923 | 0.0745 | 0.0084 | 1.000 | 12/12 |
| S | gpt_6_sol_tools | 0.0878 | 0.0701 | 0.0005 | 1.000 | 12/12 |
| H | plugin | 0.0945 | 0.2008 | -0.0934 | 1.000 | 12/12 |
| H | median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| H | mle (ref.) | 0.0491 | 0.0471 | 0.0150 | 1.000 | 12/12 |
| H | et | 0.0435 | 0.0597 | -0.0389 | 1.000 | 12/12 |
| H | qwen_thinking | 0.1382 | 0.1880 | -0.0150 | 1.000 | 12/12 |
| H | qwen_nonthinking | 0.3627 | 0.3015 | -0.2744 | 1.000 | 12/12 |
| H | deepseek_flash | 0.1049 | 0.0786 | 0.0285 | 1.000 | 12/12 |
| H | gpt_6_sol | 0.0575 | 0.0428 | -0.0113 | 1.000 | 12/12 |
| H | gpt_6_sol_tools | 0.0504 | 0.0430 | -0.0104 | 1.000 | 12/12 |
| B | plugin | 0.3251 | 0.2827 | -0.3251 | 1.000 | 12/12 |
| B | median | 0.3453 | 0.2994 | -0.2655 | 1.000 | 12/12 |
| B | mle (ref.) | 0.1321 | 0.1516 | -0.1063 | 1.000 | 12/12 |
| B | et | 0.1186 | 0.1317 | -0.0764 | 1.000 | 12/12 |
| B | qwen_thinking | 0.3353 | 0.2975 | -0.1974 | 1.000 | 12/12 |
| B | qwen_nonthinking | 0.2669 | 0.2887 | 0.2388 | 1.000 | 12/12 |
| B | deepseek_flash | 0.2663 | 0.2437 | -0.1314 | 1.000 | 12/12 |
| B | gpt_6_sol | 0.1329 | 0.1221 | -0.0181 | 1.000 | 12/12 |
| B | gpt_6_sol_tools | 0.1279 | 0.1211 | 0.0108 | 1.000 | 12/12 |

## Synthetic graphs (8, separate block)

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Validity | Sources |
|---|---|---:|---:|---:|---:|---:|
| R | plugin (ref.) | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| R | median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| R | mle | 0.0219 | 0.0152 | -0.0077 | 1.000 | 8/8 |
| R | et | 0.0226 | 0.0167 | -0.0034 | 0.875 | 8/8 |
| R | qwen_thinking | 0.0235 | 0.0173 | -0.0033 | 1.000 | 8/8 |
| R | qwen_nonthinking | 0.2194 | 0.1734 | 0.1827 | 1.000 | 8/8 |
| R | deepseek_flash | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| R | gpt_6_sol | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| R | gpt_6_sol_tools | 0.0235 | 0.0168 | -0.0032 | 1.000 | 8/8 |
| S | plugin | 0.1271 | 0.1083 | 0.1271 | 1.000 | 8/8 |
| S | median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| S | mle (ref.) | 0.0250 | 0.0201 | 0.0085 | 1.000 | 8/8 |
| S | et | 0.0184 | 0.0164 | 0.0032 | 0.958 | 8/8 |
| S | qwen_thinking | 0.0786 | 0.0627 | 0.0614 | 1.000 | 8/8 |
| S | qwen_nonthinking | 0.2807 | 0.2544 | 0.2750 | 1.000 | 8/8 |
| S | deepseek_flash | 0.0278 | 0.0196 | 0.0010 | 1.000 | 8/8 |
| S | gpt_6_sol | 0.0295 | 0.0210 | 0.0081 | 1.000 | 8/8 |
| S | gpt_6_sol_tools | 0.0271 | 0.0199 | 0.0083 | 1.000 | 8/8 |
| H | plugin | 0.0948 | 0.1214 | -0.0902 | 1.000 | 8/8 |
| H | median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| H | mle (ref.) | 0.0375 | 0.0442 | 0.0217 | 1.000 | 8/8 |
| H | et | 0.0293 | 0.0191 | -0.0134 | 1.000 | 8/8 |
| H | qwen_thinking | 0.1631 | 0.1348 | 0.0426 | 1.000 | 8/8 |
| H | qwen_nonthinking | 0.2435 | 0.1822 | -0.0826 | 1.000 | 8/8 |
| H | deepseek_flash | 0.0617 | 0.0567 | 0.0434 | 1.000 | 8/8 |
| H | gpt_6_sol | 0.0361 | 0.0248 | 0.0057 | 1.000 | 8/8 |
| H | gpt_6_sol_tools | 0.0412 | 0.0279 | 0.0077 | 1.000 | 8/8 |
| B | plugin | 0.4255 | 0.2543 | -0.4255 | 1.000 | 8/8 |
| B | median | 0.2987 | 0.2119 | -0.1506 | 1.000 | 8/8 |
| B | mle (ref.) | 0.0754 | 0.0630 | -0.0206 | 1.000 | 8/8 |
| B | et | 0.0365 | 0.0443 | -0.0190 | 1.000 | 8/8 |
| B | qwen_thinking | 0.3303 | 0.2518 | -0.1291 | 1.000 | 8/8 |
| B | qwen_nonthinking | 0.3382 | 0.3083 | 0.1636 | 1.000 | 8/8 |
| B | deepseek_flash | 0.2452 | 0.1794 | -0.0439 | 1.000 | 8/8 |
| B | gpt_6_sol | 0.1295 | 0.1037 | 0.0578 | 1.000 | 8/8 |
| B | gpt_6_sol_tools | 0.1067 | 0.0917 | 0.0458 | 1.000 | 8/8 |

These eight graphs form four generator pairs with shared random numbers; their MAE tables are descriptive,
since a four-block two-sided sign-flip test has minimum p = 0.125.

## Paired original minus surrogate (12 source families)

Source-level MAE_2 of the original minus that of its surrogate; each family is one pair.

| Arm | Method | Original | Surrogate | Mean difference | Original better | Sign-flip p |
|---|---|---:|---:|---:|---:|---:|
| R | plugin | 0.0267 | 0.0160 | 0.0107 | 3/12 | 0.1465 |
| R | median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| R | mle | 0.0372 | 0.0365 | 0.0008 | 7/12 | 0.9438 |
| R | et | 0.0282 | 0.0171 | 0.0112 | 4/12 | 0.0566 |
| R | qwen_thinking | 0.0269 | 0.0172 | 0.0097 | 3/12 | 0.1558 |
| R | qwen_nonthinking | 0.4114 | 0.2604 | 0.1510 | 2/12 | 0.0151 |
| S | plugin | 0.3006 | 0.2456 | 0.0550 | 4/12 | 0.2339 |
| S | median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| S | mle | 0.0685 | 0.0861 | -0.0175 | 9/12 | 0.4141 |
| S | et | 0.0504 | 0.0618 | -0.0114 | 7/12 | 0.6328 |
| S | qwen_thinking | 0.1814 | 0.1742 | 0.0073 | 5/12 | 0.8394 |
| S | qwen_nonthinking | 0.4752 | 0.4426 | 0.0326 | 3/12 | 0.5884 |
| H | plugin | 0.0639 | 0.0945 | -0.0305 | 7/12 | 0.2881 |
| H | median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| H | mle | 0.0706 | 0.0491 | 0.0214 | 5/12 | 0.1787 |
| H | et | 0.0390 | 0.0435 | -0.0045 | 4/12 | 0.7886 |
| H | qwen_thinking | 0.1598 | 0.1382 | 0.0216 | 6/12 | 0.4258 |
| H | qwen_nonthinking | 0.2516 | 0.3627 | -0.1111 | 8/12 | 0.1196 |
| B | plugin | 0.1643 | 0.3251 | -0.1608 | 12/12 | 0.0005 |
| B | median | 0.2321 | 0.3453 | -0.1132 | 8/12 | 0.0854 |
| B | mle | 0.0757 | 0.1321 | -0.0564 | 9/12 | 0.0508 |
| B | et | 0.0789 | 0.1186 | -0.0398 | 8/12 | 0.1704 |
| B | qwen_thinking | 0.2196 | 0.3353 | -0.1156 | 12/12 | 0.0005 |
| B | qwen_nonthinking | 0.5267 | 0.2669 | 0.2598 | 0/12 | 0.0005 |

## Paired methods within a block

First minus second source-level MAE_2 (negative: first better); exact sign-flip over sources.

| Group | Arm | Comparison | Mean difference | First better | Sign-flip p |
|---|---|---:|---:|---:|---:|
| real | R | et vs plugin | 0.0015 | 3/12 | 0.2212 |
| real | R | qwen_thinking vs plugin | 0.0002 | 5/12 | 0.1729 |
| real | S | et vs mle | -0.0181 | 7/12 | 0.0781 |
| real | S | qwen_thinking vs mle | 0.1129 | 0/12 | 0.0005 |
| real | S | mle vs plugin | -0.2321 | 12/12 | 0.0005 |
| real | S | et vs plugin | -0.2502 | 11/12 | 0.0010 |
| real | S | qwen_thinking vs plugin | -0.1192 | 12/12 | 0.0005 |
| real | H | et vs mle | -0.0316 | 9/12 | 0.0151 |
| real | H | qwen_thinking vs mle | 0.0892 | 2/12 | 0.0049 |
| real | H | mle vs plugin | 0.0066 | 6/12 | 0.7695 |
| real | H | et vs plugin | -0.0250 | 8/12 | 0.1660 |
| real | H | qwen_thinking vs plugin | 0.0958 | 1/12 | 0.0010 |
| real | B | et vs mle | 0.0032 | 5/12 | 0.7314 |
| real | B | qwen_thinking vs mle | 0.1439 | 1/12 | 0.0015 |
| real | B | mle vs plugin | -0.0885 | 8/12 | 0.0210 |
| real | B | et vs plugin | -0.0854 | 8/12 | 0.0435 |
| real | B | qwen_thinking vs plugin | 0.0554 | 2/12 | 0.0171 |
| surrogate | R | et vs plugin | 0.0011 | 5/12 | 0.5767 |
| surrogate | R | qwen_thinking vs plugin | 0.0012 | 6/12 | 0.0991 |
| surrogate | S | et vs mle | -0.0243 | 8/12 | 0.1670 |
| surrogate | S | qwen_thinking vs mle | 0.0881 | 0/12 | 0.0005 |
| surrogate | S | mle vs plugin | -0.1595 | 12/12 | 0.0005 |
| surrogate | S | et vs plugin | -0.1838 | 12/12 | 0.0005 |
| surrogate | S | qwen_thinking vs plugin | -0.0714 | 10/12 | 0.0020 |
| surrogate | H | et vs mle | -0.0056 | 6/12 | 0.6211 |
| surrogate | H | qwen_thinking vs mle | 0.0891 | 0/12 | 0.0005 |
| surrogate | H | mle vs plugin | -0.0454 | 9/12 | 0.0234 |
| surrogate | H | et vs plugin | -0.0510 | 11/12 | 0.0015 |
| surrogate | H | qwen_thinking vs plugin | 0.0437 | 0/12 | 0.0005 |
| surrogate | B | et vs mle | -0.0135 | 9/12 | 0.2358 |
| surrogate | B | qwen_thinking vs mle | 0.2031 | 0/12 | 0.0005 |
| surrogate | B | mle vs plugin | -0.1930 | 11/12 | 0.0010 |
| surrogate | B | et vs plugin | -0.2064 | 11/12 | 0.0010 |
| surrogate | B | qwen_thinking vs plugin | 0.0102 | 7/12 | 0.6089 |

## Real sources (MAE_2 per source)

| Source | Arm | plugin | median | mle | et | qwen_thinking | qwen_nonthinking |
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

All groups, API methods and metrics: `PER_SOURCE.csv`.

## Appendix: S design ratio

The plain protocol ratio sums traversals_per_event for patterns with at least k active windows
and divides by the sum over all patterns. It is a descriptive appendix, not a reference.
MAE_2 is equal-source weighted; answer shares use valid LLM answers and tolerance 0.005.

| Group | S design MAE_2 |
|---|---|
| real | 0.0880 |
| surrogate | 0.0996 |
| synthetic | 0.0284 |

| Real S model | Valid answers | Near plugin | Near design | Near MLE |
|---|---|---:|---:|---:|
| qwen_thinking | 108 | 0.500 | 0.204 | 0.130 |
| qwen_nonthinking | 106 | 0.066 | 0.047 | 0.047 |
| deepseek_flash | 35 | 0.171 | 0.657 | 0.286 |
| gpt_6_sol | 108 | 0.120 | 0.898 | 0.241 |
| gpt_6_sol_tools | 108 | 0.120 | 0.926 | 0.231 |

Per-group and per-model values: `S_DESIGN_APPENDIX.csv`.
