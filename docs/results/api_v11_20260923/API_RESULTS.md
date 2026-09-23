# API results (v11)

DeepSeek Flash (reasoning high, off-peak, one repeat), GPT-6 Sol (reasoning high, Batch, three
repeats) and GPT-6 Sol with the hosted Python tool (`gpt_6_sol_tools`, otherwise identical) on the
288 frozen R/S/H/B observations. Offline methods and Qwen (three repeats) are the committed
v11 predictions. MAE_2 is the equal-source mean over valid answers; `all24` weights all 24 graphs
equally. Nothing is clipped, repaired or imputed.

Recorded spend charges all DeepSeek input at the cache-miss price and all GPT input at the
cache-write price, so it is an upper bound on the provider bill.

- deepseek_flash: 288 main answers, recorded spend USD 6.9177 (incl. smoke/pilot), generation-limit hits 0, mean output tokens 38204.
- gpt_6_sol: 864 main answers, recorded spend USD 20.2535 (incl. smoke/pilot), generation-limit hits 0, mean output tokens 4311.
- gpt_6_sol_tools: 864 main answers, recorded spend USD 71.4307 (incl. smoke/pilot), generation-limit hits 0, mean output tokens 3595, answers using the tool 802.

## real

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources valid | MCSE_2 | Skill vs plugin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0209 | 0.0130 | -0.0044 | 24/24 | 8/8 | 0.0049 | 0.0000 |
| R | median | 0.2397 | 0.1102 | 0.0590 | 24/24 | 8/8 | 0.0000 | -10.4768 |
| R | mle | 0.0234 | 0.0139 | 0.0074 | 24/24 | 8/8 | 0.0063 | -0.1218 |
| R | et | 0.0227 | 0.0141 | -0.0036 | 24/24 | 8/8 | 0.0047 | -0.0853 |
| R | qwen_thinking | 0.0210 | 0.0131 | -0.0045 | 72/72 | 8/8 | 0.0049 | -0.0053 |
| R | qwen_nonthinking | 0.4250 | 0.3427 | 0.4248 | 72/72 | 8/8 | 0.0113 | -19.3497 |
| R | deepseek_flash | 0.0209 | 0.0130 | -0.0044 | 24/24 | 8/8 | 0.0049 | 0.0000 |
| R | gpt_6_sol | 0.0209 | 0.0131 | -0.0044 | 72/72 | 8/8 | 0.0049 | 0.0001 |
| R | gpt_6_sol_tools | 0.0218 | 0.0132 | -0.0035 | 72/72 | 8/8 | 0.0051 | -0.0437 |
| S | plugin | 0.2816 | 0.2110 | 0.2816 | 24/24 | 8/8 | 0.0040 | 0.0000 |
| S | median | 0.2397 | 0.1102 | 0.0590 | 24/24 | 8/8 | 0.0000 | 0.1489 |
| S | design | 0.0426 | 0.0183 | 0.0190 | 24/24 | 8/8 | 0.0099 | 0.8489 |
| S | mle | 0.0359 | 0.0204 | 0.0283 | 24/24 | 8/8 | 0.0073 | 0.8726 |
| S | et | 0.0421 | 0.0192 | 0.0239 | 24/24 | 8/8 | 0.0075 | 0.8506 |
| S | qwen_thinking | 0.1513 | 0.1169 | 0.1311 | 72/72 | 8/8 | 0.0172 | 0.4629 |
| S | qwen_nonthinking | 0.5563 | 0.4720 | 0.5563 | 72/72 | 8/8 | 0.0203 | -0.9751 |
| S | deepseek_flash | 0.0426 | 0.0235 | 0.0129 | 24/24 | 8/8 | 0.0107 | 0.8487 |
| S | gpt_6_sol | 0.0455 | 0.0198 | 0.0218 | 72/72 | 8/8 | 0.0099 | 0.8386 |
| S | gpt_6_sol_tools | 0.0428 | 0.0184 | 0.0194 | 72/72 | 8/8 | 0.0099 | 0.8481 |
| H | plugin | 0.0666 | 0.0573 | -0.0619 | 24/24 | 8/8 | 0.0022 | 0.0000 |
| H | median | 0.2397 | 0.1102 | 0.0590 | 24/24 | 8/8 | 0.0000 | -2.5988 |
| H | mle | 0.0516 | 0.0289 | 0.0511 | 24/24 | 8/8 | 0.0047 | 0.2246 |
| H | et | 0.0276 | 0.0239 | 0.0097 | 24/24 | 8/8 | 0.0024 | 0.5853 |
| H | qwen_thinking | 0.1490 | 0.0833 | 0.0431 | 72/72 | 8/8 | 0.0168 | -1.2376 |
| H | qwen_nonthinking | 0.2151 | 0.1276 | 0.0879 | 72/72 | 8/8 | 0.0207 | -2.2292 |
| H | deepseek_flash | 0.0963 | 0.0447 | 0.0616 | 24/24 | 8/8 | 0.0177 | -0.4456 |
| H | gpt_6_sol | 0.0421 | 0.0229 | 0.0328 | 72/72 | 8/8 | 0.0059 | 0.3675 |
| H | gpt_6_sol_tools | 0.0553 | 0.0290 | 0.0422 | 72/72 | 8/8 | 0.0065 | 0.1696 |
| B | plugin | 0.1525 | 0.0786 | -0.1525 | 24/24 | 8/8 | 0.0026 | 0.0000 |
| B | median | 0.2397 | 0.1102 | 0.0590 | 24/24 | 8/8 | 0.0000 | -0.5716 |
| B | mle | 0.0551 | 0.0347 | -0.0266 | 24/24 | 8/8 | 0.0048 | 0.6389 |
| B | et | 0.0549 | 0.0322 | 0.0045 | 24/24 | 8/8 | 0.0045 | 0.6398 |
| B | qwen_thinking | 0.2062 | 0.1186 | 0.0109 | 72/72 | 8/8 | 0.0224 | -0.3517 |
| B | qwen_nonthinking | 0.5630 | 0.5482 | 0.5462 | 72/72 | 8/8 | 0.0270 | -2.6910 |
| B | deepseek_flash | 0.1667 | 0.0947 | 0.0246 | 24/24 | 8/8 | 0.0390 | -0.0933 |
| B | gpt_6_sol | 0.0906 | 0.0605 | -0.0029 | 72/72 | 8/8 | 0.0077 | 0.4060 |
| B | gpt_6_sol_tools | 0.1154 | 0.0724 | 0.0332 | 72/72 | 8/8 | 0.0100 | 0.2432 |

## surrogate

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources valid | MCSE_2 | Skill vs plugin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0135 | 0.0163 | 0.0029 | 24/24 | 8/8 | 0.0025 | 0.0000 |
| R | median | 0.3153 | 0.2477 | -0.1957 | 24/24 | 8/8 | 0.0000 | -22.3454 |
| R | mle | 0.0337 | 0.0290 | 0.0286 | 24/24 | 8/8 | 0.0025 | -1.4969 |
| R | et | 0.0151 | 0.0171 | -0.0031 | 24/24 | 8/8 | 0.0019 | -0.1170 |
| R | qwen_thinking | 0.0140 | 0.0184 | 0.0024 | 72/72 | 8/8 | 0.0025 | -0.0345 |
| R | qwen_nonthinking | 0.2480 | 0.2211 | 0.2414 | 72/72 | 8/8 | 0.0313 | -17.3586 |
| R | deepseek_flash | 0.0135 | 0.0163 | 0.0029 | 24/24 | 8/8 | 0.0025 | 0.0000 |
| R | gpt_6_sol | 0.0135 | 0.0163 | 0.0029 | 72/72 | 8/8 | 0.0025 | -0.0001 |
| R | gpt_6_sol_tools | 0.0135 | 0.0163 | 0.0029 | 72/72 | 8/8 | 0.0025 | 0.0003 |
| S | plugin | 0.2463 | 0.3006 | 0.2463 | 24/24 | 8/8 | 0.0021 | 0.0000 |
| S | median | 0.3153 | 0.2477 | -0.1957 | 24/24 | 8/8 | 0.0000 | -0.2802 |
| S | design | 0.0540 | 0.0381 | 0.0141 | 24/24 | 8/8 | 0.0048 | 0.7809 |
| S | mle | 0.0438 | 0.0499 | 0.0354 | 24/24 | 8/8 | 0.0069 | 0.8223 |
| S | et | 0.0474 | 0.0380 | -0.0314 | 24/24 | 8/8 | 0.0062 | 0.8074 |
| S | qwen_thinking | 0.1544 | 0.1734 | 0.1335 | 71/72 | 8/8 | 0.0091 | 0.3732 |
| S | qwen_nonthinking | 0.4110 | 0.4331 | 0.2699 | 72/72 | 8/8 | 0.0328 | -0.6684 |
| S | deepseek_flash | 0.0691 | 0.0488 | 0.0213 | 24/24 | 8/8 | 0.0118 | 0.7196 |
| S | gpt_6_sol | 0.0485 | 0.0342 | 0.0086 | 72/72 | 8/8 | 0.0050 | 0.8030 |
| S | gpt_6_sol_tools | 0.0464 | 0.0320 | 0.0046 | 72/72 | 8/8 | 0.0051 | 0.8116 |
| H | plugin | 0.1012 | 0.1736 | -0.0996 | 24/24 | 8/8 | 0.0015 | 0.0000 |
| H | median | 0.3153 | 0.2477 | -0.1957 | 24/24 | 8/8 | 0.0000 | -2.1170 |
| H | mle | 0.0573 | 0.0541 | 0.0222 | 24/24 | 8/8 | 0.0031 | 0.4341 |
| H | et | 0.0491 | 0.0603 | -0.0424 | 24/24 | 8/8 | 0.0019 | 0.5142 |
| H | qwen_thinking | 0.1487 | 0.1681 | -0.0233 | 72/72 | 8/8 | 0.0111 | -0.4699 |
| H | qwen_nonthinking | 0.2808 | 0.2394 | -0.1720 | 72/72 | 8/8 | 0.0229 | -1.7755 |
| H | deepseek_flash | 0.1220 | 0.0912 | 0.0197 | 24/24 | 8/8 | 0.0232 | -0.2058 |
| H | gpt_6_sol | 0.0670 | 0.0458 | -0.0168 | 72/72 | 8/8 | 0.0066 | 0.3378 |
| H | gpt_6_sol_tools | 0.0593 | 0.0480 | -0.0141 | 72/72 | 8/8 | 0.0044 | 0.4138 |
| B | plugin | 0.3058 | 0.2404 | -0.3058 | 24/24 | 8/8 | 0.0022 | 0.0000 |
| B | median | 0.3153 | 0.2477 | -0.1957 | 24/24 | 8/8 | 0.0000 | -0.0311 |
| B | mle | 0.1265 | 0.1363 | -0.0878 | 24/24 | 8/8 | 0.0024 | 0.5863 |
| B | et | 0.1185 | 0.1189 | -0.0551 | 24/24 | 8/8 | 0.0025 | 0.6127 |
| B | qwen_thinking | 0.3207 | 0.2706 | -0.1343 | 72/72 | 8/8 | 0.0219 | -0.0486 |
| B | qwen_nonthinking | 0.3082 | 0.3092 | 0.2759 | 72/72 | 8/8 | 0.0140 | -0.0076 |
| B | deepseek_flash | 0.2409 | 0.1938 | -0.0652 | 24/24 | 8/8 | 0.0314 | 0.2124 |
| B | gpt_6_sol | 0.1325 | 0.1094 | -0.0218 | 72/72 | 8/8 | 0.0096 | 0.5666 |
| B | gpt_6_sol_tools | 0.1336 | 0.1031 | 0.0077 | 72/72 | 8/8 | 0.0145 | 0.5630 |

## synthetic

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources valid | MCSE_2 | Skill vs plugin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0235 | 0.0168 | -0.0032 | 24/24 | 8/8 | 0.0026 | 0.0000 |
| R | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8 | 0.0000 | -11.7198 |
| R | mle | 0.0219 | 0.0152 | -0.0077 | 24/24 | 8/8 | 0.0031 | 0.0664 |
| R | et | 0.0226 | 0.0167 | -0.0034 | 24/24 | 8/8 | 0.0025 | 0.0374 |
| R | qwen_thinking | 0.0235 | 0.0173 | -0.0033 | 72/72 | 8/8 | 0.0027 | 0.0003 |
| R | qwen_nonthinking | 0.2194 | 0.1734 | 0.1827 | 72/72 | 8/8 | 0.0223 | -8.3404 |
| R | deepseek_flash | 0.0235 | 0.0168 | -0.0032 | 24/24 | 8/8 | 0.0026 | -0.0000 |
| R | gpt_6_sol | 0.0235 | 0.0168 | -0.0032 | 72/72 | 8/8 | 0.0026 | 0.0000 |
| R | gpt_6_sol_tools | 0.0235 | 0.0168 | -0.0032 | 72/72 | 8/8 | 0.0026 | -0.0001 |
| S | plugin | 0.1271 | 0.1083 | 0.1271 | 24/24 | 8/8 | 0.0040 | 0.0000 |
| S | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8 | 0.0000 | -1.3504 |
| S | design | 0.0284 | 0.0207 | 0.0076 | 24/24 | 8/8 | 0.0045 | 0.7768 |
| S | mle | 0.0250 | 0.0201 | 0.0085 | 24/24 | 8/8 | 0.0044 | 0.8033 |
| S | et | 0.0184 | 0.0169 | 0.0035 | 24/24 | 8/8 | 0.0035 | 0.8549 |
| S | qwen_thinking | 0.0786 | 0.0627 | 0.0614 | 72/72 | 8/8 | 0.0084 | 0.3812 |
| S | qwen_nonthinking | 0.2807 | 0.2544 | 0.2750 | 72/72 | 8/8 | 0.0222 | -1.2082 |
| S | deepseek_flash | 0.0278 | 0.0196 | 0.0010 | 24/24 | 8/8 | 0.0050 | 0.7816 |
| S | gpt_6_sol | 0.0295 | 0.0210 | 0.0081 | 72/72 | 8/8 | 0.0048 | 0.7681 |
| S | gpt_6_sol_tools | 0.0271 | 0.0199 | 0.0083 | 72/72 | 8/8 | 0.0047 | 0.7868 |
| H | plugin | 0.0948 | 0.1214 | -0.0902 | 24/24 | 8/8 | 0.0051 | 0.0000 |
| H | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8 | 0.0000 | -2.1525 |
| H | mle | 0.0375 | 0.0442 | 0.0217 | 24/24 | 8/8 | 0.0061 | 0.6044 |
| H | et | 0.0293 | 0.0191 | -0.0134 | 24/24 | 8/8 | 0.0041 | 0.6911 |
| H | qwen_thinking | 0.1631 | 0.1348 | 0.0426 | 72/72 | 8/8 | 0.0065 | -0.7207 |
| H | qwen_nonthinking | 0.2435 | 0.1822 | -0.0826 | 72/72 | 8/8 | 0.0223 | -1.5699 |
| H | deepseek_flash | 0.0617 | 0.0567 | 0.0434 | 24/24 | 8/8 | 0.0070 | 0.3484 |
| H | gpt_6_sol | 0.0361 | 0.0248 | 0.0057 | 72/72 | 8/8 | 0.0049 | 0.6195 |
| H | gpt_6_sol_tools | 0.0412 | 0.0279 | 0.0077 | 72/72 | 8/8 | 0.0067 | 0.5655 |
| B | plugin | 0.4255 | 0.2543 | -0.4255 | 24/24 | 8/8 | 0.0028 | 0.0000 |
| B | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8 | 0.0000 | 0.2979 |
| B | mle | 0.0754 | 0.0630 | -0.0206 | 24/24 | 8/8 | 0.0134 | 0.8228 |
| B | et | 0.0365 | 0.0443 | -0.0190 | 24/24 | 8/8 | 0.0075 | 0.9143 |
| B | qwen_thinking | 0.3303 | 0.2518 | -0.1291 | 72/72 | 8/8 | 0.0196 | 0.2236 |
| B | qwen_nonthinking | 0.3382 | 0.3083 | 0.1636 | 72/72 | 8/8 | 0.0321 | 0.2051 |
| B | deepseek_flash | 0.2452 | 0.1794 | -0.0439 | 24/24 | 8/8 | 0.0482 | 0.4238 |
| B | gpt_6_sol | 0.1295 | 0.1037 | 0.0578 | 72/72 | 8/8 | 0.0135 | 0.6956 |
| B | gpt_6_sol_tools | 0.1067 | 0.0917 | 0.0458 | 72/72 | 8/8 | 0.0076 | 0.7493 |

## all24

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources valid | MCSE_2 | Skill vs plugin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0193 | 0.0154 | -0.0016 | 72/72 | 24/24 | 0.0020 | 0.0000 |
| R | median | 0.2846 | 0.1899 | -0.0958 | 72/72 | 24/24 | 0.0000 | -13.7511 |
| R | mle | 0.0264 | 0.0194 | 0.0094 | 72/72 | 24/24 | 0.0025 | -0.3664 |
| R | et | 0.0201 | 0.0160 | -0.0034 | 72/72 | 24/24 | 0.0019 | -0.0429 |
| R | qwen_thinking | 0.0195 | 0.0163 | -0.0018 | 216/216 | 24/24 | 0.0020 | -0.0098 |
| R | qwen_nonthinking | 0.2975 | 0.2457 | 0.2829 | 216/216 | 24/24 | 0.0133 | -14.4177 |
| R | deepseek_flash | 0.0193 | 0.0154 | -0.0016 | 72/72 | 24/24 | 0.0020 | -0.0000 |
| R | gpt_6_sol | 0.0193 | 0.0154 | -0.0016 | 216/216 | 24/24 | 0.0020 | 0.0000 |
| R | gpt_6_sol_tools | 0.0196 | 0.0154 | -0.0013 | 216/216 | 24/24 | 0.0021 | -0.0157 |
| S | plugin | 0.2184 | 0.2067 | 0.2184 | 72/72 | 24/24 | 0.0020 | 0.0000 |
| S | median | 0.2846 | 0.1899 | -0.0958 | 72/72 | 24/24 | 0.0000 | -0.3034 |
| S | design | 0.0416 | 0.0257 | 0.0136 | 72/72 | 24/24 | 0.0040 | 0.8093 |
| S | mle | 0.0349 | 0.0301 | 0.0241 | 72/72 | 24/24 | 0.0037 | 0.8402 |
| S | et | 0.0360 | 0.0247 | -0.0013 | 72/72 | 24/24 | 0.0035 | 0.8352 |
| S | qwen_thinking | 0.1281 | 0.1177 | 0.1086 | 215/216 | 24/24 | 0.0071 | 0.4133 |
| S | qwen_nonthinking | 0.4160 | 0.3865 | 0.3671 | 216/216 | 24/24 | 0.0148 | -0.9050 |
| S | deepseek_flash | 0.0465 | 0.0306 | 0.0117 | 72/72 | 24/24 | 0.0056 | 0.7871 |
| S | gpt_6_sol | 0.0411 | 0.0250 | 0.0129 | 216/216 | 24/24 | 0.0040 | 0.8116 |
| S | gpt_6_sol_tools | 0.0388 | 0.0234 | 0.0108 | 216/216 | 24/24 | 0.0040 | 0.8225 |
| H | plugin | 0.0875 | 0.1174 | -0.0839 | 72/72 | 24/24 | 0.0019 | 0.0000 |
| H | median | 0.2846 | 0.1899 | -0.0958 | 72/72 | 24/24 | 0.0000 | -2.2520 |
| H | mle | 0.0488 | 0.0424 | 0.0317 | 72/72 | 24/24 | 0.0028 | 0.4424 |
| H | et | 0.0353 | 0.0345 | -0.0154 | 72/72 | 24/24 | 0.0017 | 0.5961 |
| H | qwen_thinking | 0.1536 | 0.1287 | 0.0208 | 216/216 | 24/24 | 0.0070 | -0.7552 |
| H | qwen_nonthinking | 0.2465 | 0.1830 | -0.0556 | 216/216 | 24/24 | 0.0127 | -1.8164 |
| H | deepseek_flash | 0.0933 | 0.0642 | 0.0415 | 72/72 | 24/24 | 0.0100 | -0.0666 |
| H | gpt_6_sol | 0.0484 | 0.0312 | 0.0073 | 216/216 | 24/24 | 0.0034 | 0.4470 |
| H | gpt_6_sol_tools | 0.0519 | 0.0350 | 0.0119 | 216/216 | 24/24 | 0.0034 | 0.4066 |
| B | plugin | 0.2946 | 0.1911 | -0.2946 | 72/72 | 24/24 | 0.0015 | 0.0000 |
| B | median | 0.2846 | 0.1899 | -0.0958 | 72/72 | 24/24 | 0.0000 | 0.0340 |
| B | mle | 0.0857 | 0.0780 | -0.0450 | 72/72 | 24/24 | 0.0048 | 0.7092 |
| B | et | 0.0700 | 0.0651 | -0.0232 | 72/72 | 24/24 | 0.0030 | 0.7626 |
| B | qwen_thinking | 0.2857 | 0.2137 | -0.0842 | 216/216 | 24/24 | 0.0123 | 0.0302 |
| B | qwen_nonthinking | 0.4031 | 0.3886 | 0.3286 | 216/216 | 24/24 | 0.0147 | -0.3683 |
| B | deepseek_flash | 0.2176 | 0.1560 | -0.0282 | 72/72 | 24/24 | 0.0232 | 0.2614 |
| B | gpt_6_sol | 0.1175 | 0.0912 | 0.0111 | 216/216 | 24/24 | 0.0061 | 0.6010 |
| B | gpt_6_sol_tools | 0.1186 | 0.0891 | 0.0289 | 216/216 | 24/24 | 0.0064 | 0.5975 |

## Source-level paired differences

First minus second source-level MAE_2; exact two-sided sign-flip over sources (8 or 24).

| Group | Arm | Comparison | Mean difference | First better | Exact sign-flip p |
|---|---|---|---:|---:|---:|
| real | R | deepseek_flash vs plugin | -0.0000 | 7/8 | 0.5078 |
| real | R | deepseek_flash vs et | -0.0018 | 6/8 | 0.1250 |
| real | R | deepseek_flash vs qwen_thinking | -0.0001 | 4/8 | 0.4375 |
| real | R | gpt_6_sol vs plugin | -0.0000 | 4/8 | 0.8359 |
| real | R | gpt_6_sol vs et | -0.0018 | 6/8 | 0.1250 |
| real | R | gpt_6_sol vs qwen_thinking | -0.0001 | 5/8 | 0.2422 |
| real | R | gpt_6_sol_tools vs plugin | 0.0009 | 3/8 | 0.6016 |
| real | R | gpt_6_sol_tools vs et | -0.0009 | 6/8 | 0.6328 |
| real | R | gpt_6_sol_tools vs qwen_thinking | 0.0008 | 4/8 | 0.8047 |
| real | R | deepseek_flash vs gpt_6_sol | 0.0000 | 4/8 | 0.8359 |
| real | R | gpt_6_sol_tools vs gpt_6_sol | 0.0009 | 2/8 | 0.1719 |
| real | S | deepseek_flash vs plugin | -0.2390 | 8/8 | 0.0078 |
| real | S | deepseek_flash vs design | 0.0001 | 3/8 | 0.9688 |
| real | S | deepseek_flash vs et | 0.0005 | 3/8 | 0.8594 |
| real | S | deepseek_flash vs qwen_thinking | -0.1087 | 8/8 | 0.0078 |
| real | S | gpt_6_sol vs plugin | -0.2362 | 8/8 | 0.0078 |
| real | S | gpt_6_sol vs design | 0.0029 | 2/8 | 0.0781 |
| real | S | gpt_6_sol vs et | 0.0034 | 3/8 | 0.3828 |
| real | S | gpt_6_sol vs qwen_thinking | -0.1058 | 8/8 | 0.0078 |
| real | S | gpt_6_sol_tools vs plugin | -0.2388 | 8/8 | 0.0078 |
| real | S | gpt_6_sol_tools vs design | 0.0002 | 3/8 | 0.2969 |
| real | S | gpt_6_sol_tools vs et | 0.0007 | 3/8 | 0.7500 |
| real | S | gpt_6_sol_tools vs qwen_thinking | -0.1085 | 8/8 | 0.0078 |
| real | S | deepseek_flash vs gpt_6_sol | -0.0028 | 4/8 | 0.5156 |
| real | S | gpt_6_sol_tools vs gpt_6_sol | -0.0027 | 7/8 | 0.1172 |
| real | H | deepseek_flash vs plugin | 0.0297 | 2/8 | 0.2812 |
| real | H | deepseek_flash vs mle | 0.0446 | 1/8 | 0.0703 |
| real | H | deepseek_flash vs et | 0.0687 | 3/8 | 0.0625 |
| real | H | deepseek_flash vs qwen_thinking | -0.0528 | 5/8 | 0.1406 |
| real | H | gpt_6_sol vs plugin | -0.0245 | 6/8 | 0.1562 |
| real | H | gpt_6_sol vs mle | -0.0095 | 4/8 | 0.4688 |
| real | H | gpt_6_sol vs et | 0.0145 | 3/8 | 0.2969 |
| real | H | gpt_6_sol vs qwen_thinking | -0.1069 | 7/8 | 0.0156 |
| real | H | gpt_6_sol_tools vs plugin | -0.0113 | 6/8 | 0.5625 |
| real | H | gpt_6_sol_tools vs mle | 0.0037 | 4/8 | 0.8828 |
| real | H | gpt_6_sol_tools vs et | 0.0277 | 2/8 | 0.1016 |
| real | H | gpt_6_sol_tools vs qwen_thinking | -0.0937 | 7/8 | 0.0234 |
| real | H | deepseek_flash vs gpt_6_sol | 0.0542 | 3/8 | 0.1172 |
| real | H | gpt_6_sol_tools vs gpt_6_sol | 0.0132 | 3/8 | 0.0938 |
| real | B | deepseek_flash vs plugin | 0.0142 | 4/8 | 0.8125 |
| real | B | deepseek_flash vs mle | 0.1117 | 2/8 | 0.0469 |
| real | B | deepseek_flash vs et | 0.1118 | 2/8 | 0.0469 |
| real | B | deepseek_flash vs qwen_thinking | -0.0394 | 5/8 | 0.3750 |
| real | B | gpt_6_sol vs plugin | -0.0619 | 7/8 | 0.0156 |
| real | B | gpt_6_sol vs mle | 0.0355 | 3/8 | 0.1719 |
| real | B | gpt_6_sol vs et | 0.0357 | 3/8 | 0.1875 |
| real | B | gpt_6_sol vs qwen_thinking | -0.1156 | 8/8 | 0.0078 |
| real | B | gpt_6_sol_tools vs plugin | -0.0371 | 7/8 | 0.0234 |
| real | B | gpt_6_sol_tools vs mle | 0.0604 | 3/8 | 0.1094 |
| real | B | gpt_6_sol_tools vs et | 0.0605 | 2/8 | 0.1094 |
| real | B | gpt_6_sol_tools vs qwen_thinking | -0.0907 | 7/8 | 0.0156 |
| real | B | deepseek_flash vs gpt_6_sol | 0.0762 | 1/8 | 0.0156 |
| real | B | gpt_6_sol_tools vs gpt_6_sol | 0.0248 | 3/8 | 0.0625 |
| surrogate | R | deepseek_flash vs plugin | -0.0000 | 4/8 | 0.3594 |
| surrogate | R | deepseek_flash vs et | -0.0016 | 5/8 | 0.5234 |
| surrogate | R | deepseek_flash vs qwen_thinking | -0.0005 | 3/8 | 0.7891 |
| surrogate | R | gpt_6_sol vs plugin | 0.0000 | 4/8 | 0.7969 |
| surrogate | R | gpt_6_sol vs et | -0.0016 | 5/8 | 0.5234 |
| surrogate | R | gpt_6_sol vs qwen_thinking | -0.0005 | 3/8 | 0.8672 |
| surrogate | R | gpt_6_sol_tools vs plugin | -0.0000 | 7/8 | 0.0156 |
| surrogate | R | gpt_6_sol_tools vs et | -0.0016 | 5/8 | 0.5234 |
| surrogate | R | gpt_6_sol_tools vs qwen_thinking | -0.0005 | 5/8 | 0.6172 |
| surrogate | R | deepseek_flash vs gpt_6_sol | -0.0000 | 4/8 | 0.7969 |
| surrogate | R | gpt_6_sol_tools vs gpt_6_sol | -0.0000 | 7/8 | 0.0312 |
| surrogate | S | deepseek_flash vs plugin | -0.1773 | 7/8 | 0.0156 |
| surrogate | S | deepseek_flash vs design | 0.0151 | 2/8 | 0.0625 |
| surrogate | S | deepseek_flash vs et | 0.0216 | 2/8 | 0.3203 |
| surrogate | S | deepseek_flash vs qwen_thinking | -0.0853 | 7/8 | 0.0156 |
| surrogate | S | gpt_6_sol vs plugin | -0.1978 | 8/8 | 0.0078 |
| surrogate | S | gpt_6_sol vs design | -0.0054 | 6/8 | 0.6328 |
| surrogate | S | gpt_6_sol vs et | 0.0011 | 3/8 | 0.8906 |
| surrogate | S | gpt_6_sol vs qwen_thinking | -0.1059 | 8/8 | 0.0078 |
| surrogate | S | gpt_6_sol_tools vs plugin | -0.1999 | 8/8 | 0.0078 |
| surrogate | S | gpt_6_sol_tools vs design | -0.0076 | 3/8 | 0.8047 |
| surrogate | S | gpt_6_sol_tools vs et | -0.0010 | 3/8 | 0.8984 |
| surrogate | S | gpt_6_sol_tools vs qwen_thinking | -0.1080 | 8/8 | 0.0078 |
| surrogate | S | deepseek_flash vs gpt_6_sol | 0.0205 | 2/8 | 0.0312 |
| surrogate | S | gpt_6_sol_tools vs gpt_6_sol | -0.0021 | 3/8 | 0.5000 |
| surrogate | H | deepseek_flash vs plugin | 0.0208 | 3/8 | 0.4219 |
| surrogate | H | deepseek_flash vs mle | 0.0647 | 2/8 | 0.0391 |
| surrogate | H | deepseek_flash vs et | 0.0728 | 2/8 | 0.0391 |
| surrogate | H | deepseek_flash vs qwen_thinking | -0.0267 | 6/8 | 0.3438 |
| surrogate | H | gpt_6_sol vs plugin | -0.0342 | 4/8 | 0.2734 |
| surrogate | H | gpt_6_sol vs mle | 0.0097 | 3/8 | 0.4609 |
| surrogate | H | gpt_6_sol vs et | 0.0178 | 4/8 | 0.5078 |
| surrogate | H | gpt_6_sol vs qwen_thinking | -0.0817 | 7/8 | 0.0312 |
| surrogate | H | gpt_6_sol_tools vs plugin | -0.0419 | 6/8 | 0.1953 |
| surrogate | H | gpt_6_sol_tools vs mle | 0.0021 | 4/8 | 0.8828 |
| surrogate | H | gpt_6_sol_tools vs et | 0.0102 | 4/8 | 0.7031 |
| surrogate | H | gpt_6_sol_tools vs qwen_thinking | -0.0894 | 7/8 | 0.0234 |
| surrogate | H | deepseek_flash vs gpt_6_sol | 0.0550 | 2/8 | 0.0312 |
| surrogate | H | gpt_6_sol_tools vs gpt_6_sol | -0.0077 | 5/8 | 0.4375 |
| surrogate | B | deepseek_flash vs plugin | -0.0650 | 7/8 | 0.0234 |
| surrogate | B | deepseek_flash vs mle | 0.1144 | 2/8 | 0.0312 |
| surrogate | B | deepseek_flash vs et | 0.1224 | 1/8 | 0.0234 |
| surrogate | B | deepseek_flash vs qwen_thinking | -0.0798 | 7/8 | 0.0156 |
| surrogate | B | gpt_6_sol vs plugin | -0.1733 | 8/8 | 0.0078 |
| surrogate | B | gpt_6_sol vs mle | 0.0060 | 4/8 | 0.8047 |
| surrogate | B | gpt_6_sol vs et | 0.0141 | 4/8 | 0.6953 |
| surrogate | B | gpt_6_sol vs qwen_thinking | -0.1881 | 8/8 | 0.0078 |
| surrogate | B | gpt_6_sol_tools vs plugin | -0.1722 | 8/8 | 0.0078 |
| surrogate | B | gpt_6_sol_tools vs mle | 0.0071 | 4/8 | 0.8750 |
| surrogate | B | gpt_6_sol_tools vs et | 0.0152 | 4/8 | 0.7266 |
| surrogate | B | gpt_6_sol_tools vs qwen_thinking | -0.1871 | 8/8 | 0.0078 |
| surrogate | B | deepseek_flash vs gpt_6_sol | 0.1083 | 0/8 | 0.0078 |
| surrogate | B | gpt_6_sol_tools vs gpt_6_sol | 0.0011 | 3/8 | 0.9844 |
| synthetic | R | deepseek_flash vs plugin | 0.0000 | 2/8 | 0.5000 |
| synthetic | R | deepseek_flash vs et | 0.0009 | 4/8 | 0.2500 |
| synthetic | R | deepseek_flash vs qwen_thinking | 0.0000 | 5/8 | 0.9922 |
| synthetic | R | gpt_6_sol vs plugin | -0.0000 | 4/8 | 0.8594 |
| synthetic | R | gpt_6_sol vs et | 0.0009 | 4/8 | 0.2578 |
| synthetic | R | gpt_6_sol vs qwen_thinking | 0.0000 | 4/8 | 0.9297 |
| synthetic | R | gpt_6_sol_tools vs plugin | 0.0000 | 2/8 | 0.2500 |
| synthetic | R | gpt_6_sol_tools vs et | 0.0009 | 4/8 | 0.2500 |
| synthetic | R | gpt_6_sol_tools vs qwen_thinking | 0.0000 | 4/8 | 0.9531 |
| synthetic | R | deepseek_flash vs gpt_6_sol | 0.0000 | 4/8 | 0.8516 |
| synthetic | R | gpt_6_sol_tools vs gpt_6_sol | 0.0000 | 3/8 | 0.4609 |
| synthetic | S | deepseek_flash vs plugin | -0.0993 | 8/8 | 0.0078 |
| synthetic | S | deepseek_flash vs design | -0.0006 | 3/8 | 0.8750 |
| synthetic | S | deepseek_flash vs et | 0.0093 | 2/8 | 0.0625 |
| synthetic | S | deepseek_flash vs qwen_thinking | -0.0509 | 8/8 | 0.0078 |
| synthetic | S | gpt_6_sol vs plugin | -0.0976 | 8/8 | 0.0078 |
| synthetic | S | gpt_6_sol vs design | 0.0011 | 1/8 | 0.4766 |
| synthetic | S | gpt_6_sol vs et | 0.0110 | 2/8 | 0.0781 |
| synthetic | S | gpt_6_sol vs qwen_thinking | -0.0492 | 7/8 | 0.0156 |
| synthetic | S | gpt_6_sol_tools vs plugin | -0.1000 | 8/8 | 0.0078 |
| synthetic | S | gpt_6_sol_tools vs design | -0.0013 | 4/8 | 0.3047 |
| synthetic | S | gpt_6_sol_tools vs et | 0.0087 | 3/8 | 0.1250 |
| synthetic | S | gpt_6_sol_tools vs qwen_thinking | -0.0515 | 7/8 | 0.0156 |
| synthetic | S | deepseek_flash vs gpt_6_sol | -0.0017 | 6/8 | 0.2969 |
| synthetic | S | gpt_6_sol_tools vs gpt_6_sol | -0.0024 | 5/8 | 0.0625 |
| synthetic | H | deepseek_flash vs plugin | -0.0330 | 7/8 | 0.2578 |
| synthetic | H | deepseek_flash vs mle | 0.0243 | 1/8 | 0.0469 |
| synthetic | H | deepseek_flash vs et | 0.0325 | 1/8 | 0.1328 |
| synthetic | H | deepseek_flash vs qwen_thinking | -0.1013 | 6/8 | 0.0312 |
| synthetic | H | gpt_6_sol vs plugin | -0.0587 | 8/8 | 0.0078 |
| synthetic | H | gpt_6_sol vs mle | -0.0014 | 5/8 | 0.8750 |
| synthetic | H | gpt_6_sol vs et | 0.0068 | 3/8 | 0.5703 |
| synthetic | H | gpt_6_sol vs qwen_thinking | -0.1270 | 8/8 | 0.0078 |
| synthetic | H | gpt_6_sol_tools vs plugin | -0.0536 | 7/8 | 0.0156 |
| synthetic | H | gpt_6_sol_tools vs mle | 0.0037 | 2/8 | 0.6016 |
| synthetic | H | gpt_6_sol_tools vs et | 0.0119 | 2/8 | 0.2500 |
| synthetic | H | gpt_6_sol_tools vs qwen_thinking | -0.1219 | 8/8 | 0.0078 |
| synthetic | H | deepseek_flash vs gpt_6_sol | 0.0257 | 2/8 | 0.1484 |
| synthetic | H | gpt_6_sol_tools vs gpt_6_sol | 0.0051 | 3/8 | 0.1641 |
| synthetic | B | deepseek_flash vs plugin | -0.1803 | 8/8 | 0.0078 |
| synthetic | B | deepseek_flash vs mle | 0.1698 | 1/8 | 0.0156 |
| synthetic | B | deepseek_flash vs et | 0.2087 | 0/8 | 0.0078 |
| synthetic | B | deepseek_flash vs qwen_thinking | -0.0851 | 6/8 | 0.0625 |
| synthetic | B | gpt_6_sol vs plugin | -0.2960 | 8/8 | 0.0078 |
| synthetic | B | gpt_6_sol vs mle | 0.0541 | 1/8 | 0.0234 |
| synthetic | B | gpt_6_sol vs et | 0.0930 | 1/8 | 0.0156 |
| synthetic | B | gpt_6_sol vs qwen_thinking | -0.2008 | 8/8 | 0.0078 |
| synthetic | B | gpt_6_sol_tools vs plugin | -0.3188 | 8/8 | 0.0078 |
| synthetic | B | gpt_6_sol_tools vs mle | 0.0313 | 0/8 | 0.0078 |
| synthetic | B | gpt_6_sol_tools vs et | 0.0702 | 0/8 | 0.0078 |
| synthetic | B | gpt_6_sol_tools vs qwen_thinking | -0.2236 | 8/8 | 0.0078 |
| synthetic | B | deepseek_flash vs gpt_6_sol | 0.1157 | 1/8 | 0.0156 |
| synthetic | B | gpt_6_sol_tools vs gpt_6_sol | -0.0228 | 6/8 | 0.0703 |
| all24 | R | deepseek_flash vs plugin | 0.0000 | 13/24 | 0.8287 |
| all24 | R | deepseek_flash vs et | -0.0008 | 15/24 | 0.3700 |
| all24 | R | deepseek_flash vs qwen_thinking | -0.0002 | 12/24 | 0.3375 |
| all24 | R | gpt_6_sol vs plugin | -0.0000 | 12/24 | 0.8087 |
| all24 | R | gpt_6_sol vs et | -0.0008 | 15/24 | 0.3696 |
| all24 | R | gpt_6_sol vs qwen_thinking | -0.0002 | 12/24 | 0.3310 |
| all24 | R | gpt_6_sol_tools vs plugin | 0.0003 | 12/24 | 0.7328 |
| all24 | R | gpt_6_sol_tools vs et | -0.0005 | 15/24 | 0.6052 |
| all24 | R | gpt_6_sol_tools vs qwen_thinking | 0.0001 | 13/24 | 0.8343 |
| all24 | R | deepseek_flash vs gpt_6_sol | 0.0000 | 12/24 | 0.8058 |
| all24 | R | gpt_6_sol_tools vs gpt_6_sol | 0.0003 | 12/24 | 0.5667 |
| all24 | S | deepseek_flash vs plugin | -0.1719 | 23/24 | 0.0000 |
| all24 | S | deepseek_flash vs design | 0.0049 | 8/24 | 0.2777 |
| all24 | S | deepseek_flash vs et | 0.0105 | 7/24 | 0.2044 |
| all24 | S | deepseek_flash vs qwen_thinking | -0.0816 | 23/24 | 0.0000 |
| all24 | S | gpt_6_sol vs plugin | -0.1772 | 24/24 | 0.0000 |
| all24 | S | gpt_6_sol vs design | -0.0005 | 9/24 | 0.9276 |
| all24 | S | gpt_6_sol vs et | 0.0052 | 8/24 | 0.4130 |
| all24 | S | gpt_6_sol vs qwen_thinking | -0.0870 | 23/24 | 0.0000 |
| all24 | S | gpt_6_sol_tools vs plugin | -0.1796 | 24/24 | 0.0000 |
| all24 | S | gpt_6_sol_tools vs design | -0.0029 | 10/24 | 0.3013 |
| all24 | S | gpt_6_sol_tools vs et | 0.0028 | 9/24 | 0.6296 |
| all24 | S | gpt_6_sol_tools vs qwen_thinking | -0.0893 | 23/24 | 0.0000 |
| all24 | S | deepseek_flash vs gpt_6_sol | 0.0053 | 12/24 | 0.3085 |
| all24 | S | gpt_6_sol_tools vs gpt_6_sol | -0.0024 | 15/24 | 0.0076 |
| all24 | H | deepseek_flash vs plugin | 0.0058 | 12/24 | 0.7092 |
| all24 | H | deepseek_flash vs mle | 0.0445 | 4/24 | 0.0007 |
| all24 | H | deepseek_flash vs et | 0.0580 | 6/24 | 0.0005 |
| all24 | H | deepseek_flash vs qwen_thinking | -0.0603 | 17/24 | 0.0029 |
| all24 | H | gpt_6_sol vs plugin | -0.0391 | 18/24 | 0.0033 |
| all24 | H | gpt_6_sol vs mle | -0.0004 | 12/24 | 0.9469 |
| all24 | H | gpt_6_sol vs et | 0.0130 | 10/24 | 0.1719 |
| all24 | H | gpt_6_sol vs qwen_thinking | -0.1052 | 22/24 | 0.0000 |
| all24 | H | gpt_6_sol_tools vs plugin | -0.0356 | 19/24 | 0.0111 |
| all24 | H | gpt_6_sol_tools vs mle | 0.0031 | 10/24 | 0.6765 |
| all24 | H | gpt_6_sol_tools vs et | 0.0166 | 8/24 | 0.1342 |
| all24 | H | gpt_6_sol_tools vs qwen_thinking | -0.1017 | 22/24 | 0.0000 |
| all24 | H | deepseek_flash vs gpt_6_sol | 0.0449 | 7/24 | 0.0007 |
| all24 | H | gpt_6_sol_tools vs gpt_6_sol | 0.0035 | 11/24 | 0.4237 |
| all24 | B | deepseek_flash vs plugin | -0.0770 | 19/24 | 0.0125 |
| all24 | B | deepseek_flash vs mle | 0.1319 | 5/24 | 0.0000 |
| all24 | B | deepseek_flash vs et | 0.1476 | 3/24 | 0.0000 |
| all24 | B | deepseek_flash vs qwen_thinking | -0.0681 | 18/24 | 0.0030 |
| all24 | B | gpt_6_sol vs plugin | -0.1771 | 23/24 | 0.0000 |
| all24 | B | gpt_6_sol vs mle | 0.0319 | 8/24 | 0.0169 |
| all24 | B | gpt_6_sol vs et | 0.0476 | 8/24 | 0.0101 |
| all24 | B | gpt_6_sol vs qwen_thinking | -0.1682 | 24/24 | 0.0000 |
| all24 | B | gpt_6_sol_tools vs plugin | -0.1760 | 23/24 | 0.0000 |
| all24 | B | gpt_6_sol_tools vs mle | 0.0329 | 7/24 | 0.0446 |
| all24 | B | gpt_6_sol_tools vs et | 0.0486 | 6/24 | 0.0152 |
| all24 | B | gpt_6_sol_tools vs qwen_thinking | -0.1671 | 23/24 | 0.0000 |
| all24 | B | deepseek_flash vs gpt_6_sol | 0.1000 | 2/24 | 0.0000 |
| all24 | B | gpt_6_sol_tools vs gpt_6_sol | 0.0010 | 12/24 | 0.8963 |
