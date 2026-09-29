# API results

DeepSeek Flash (reasoning high, off-peak, one repeat), GPT-6 Sol (reasoning high, Batch, three
repeats) and GPT-6 Sol with the hosted Python tool (`gpt_6_sol_tools`, otherwise identical) on the
frozen R/S/H/B observations. Offline methods (plugin, median, MLE, ExtraTrees; reference plugin for R
and the MLE otherwise) and Qwen are the final predictions. MAE_2 is the equal-source mean over valid
LLM answers and all raw ET profiles; a block missing any source is **pending** and not ranked. Nothing is
clipped, repaired or imputed. Recorded spend is an upper bound on the provider bill.

- deepseek_flash:v11: 288 main answers, recorded spend USD 6.9177 (incl. smoke/pilot), generation-limit hits 0.
- deepseek_flash:ext: 96 main answers, recorded spend USD 2.5177 (incl. smoke/pilot), generation-limit hits 0.
- gpt_6_sol:v11: 864 main answers, recorded spend USD 20.2535 (incl. smoke/pilot), generation-limit hits 0.
- gpt_6_sol:ext: 288 main answers, recorded spend USD 7.8769 (incl. smoke/pilot), generation-limit hits 0.
- gpt_6_sol_tools:v11: 864 main answers, recorded spend USD 71.4307 (incl. smoke/pilot), generation-limit hits 0.
- gpt_6_sol_tools:ext: 288 main answers, recorded spend USD 24.3327 (incl. smoke/pilot), generation-limit hits 0.

## real

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources | MCSE_2 |
|---|---|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0267 | 0.0220 | -0.0087 | 36/36 | 12/12  | 0.0039 |
| R | median | 0.2321 | 0.1287 | 0.0006 | 36/36 | 12/12  | 0.0000 |
| R | mle | 0.0372 | 0.0270 | 0.0225 | 36/36 | 12/12  | 0.0071 |
| R | et | 0.0282 | 0.0230 | -0.0085 | 34/36 | 12/12  | 0.0037 |
| R | qwen_thinking | 0.0269 | 0.0234 | -0.0095 | 108/108 | 12/12  | 0.0038 |
| R | qwen_nonthinking | 0.4114 | 0.3280 | 0.4086 | 108/108 | 12/12  | 0.0117 |
| R | deepseek_flash | 0.0267 | 0.0220 | -0.0087 | 36/36 | 12/12  | 0.0039 |
| R | gpt_6_sol | 0.0267 | 0.0220 | -0.0087 | 108/108 | 12/12  | 0.0039 |
| R | gpt_6_sol_tools | 0.0273 | 0.0221 | -0.0081 | 108/108 | 12/12  | 0.0040 |
| S | plugin | 0.3006 | 0.2522 | 0.2836 | 36/36 | 12/12  | 0.0068 |
| S | median | 0.2321 | 0.1287 | 0.0006 | 36/36 | 12/12  | 0.0000 |
| S | mle | 0.0685 | 0.0415 | 0.0433 | 36/36 | 12/12  | 0.0080 |
| S | et | 0.0504 | 0.0334 | 0.0233 | 35/36 | 12/12  | 0.0054 |
| S | qwen_thinking | 0.1814 | 0.1489 | 0.1416 | 108/108 | 12/12  | 0.0131 |
| S | qwen_nonthinking | 0.4752 | 0.4210 | 0.4587 | 106/108 | 12/12  | 0.0228 |
| S | deepseek_flash | 0.0873 | 0.0508 | 0.0341 | 35/36 | 12/12  | 0.0102 |
| S | gpt_6_sol | 0.0841 | 0.0402 | 0.0406 | 108/108 | 12/12  | 0.0084 |
| S | gpt_6_sol_tools | 0.0813 | 0.0404 | 0.0380 | 108/108 | 12/12  | 0.0086 |
| H | plugin | 0.0639 | 0.0720 | -0.0533 | 36/36 | 12/12  | 0.0032 |
| H | median | 0.2321 | 0.1287 | 0.0006 | 36/36 | 12/12  | 0.0000 |
| H | mle | 0.0706 | 0.0444 | 0.0676 | 36/36 | 12/12  | 0.0039 |
| H | et | 0.0390 | 0.0316 | 0.0146 | 36/36 | 12/12  | 0.0021 |
| H | qwen_thinking | 0.1598 | 0.1021 | 0.0568 | 108/108 | 12/12  | 0.0157 |
| H | qwen_nonthinking | 0.2516 | 0.1654 | 0.0577 | 108/108 | 12/12  | 0.0177 |
| H | deepseek_flash | 0.1184 | 0.0654 | 0.0772 | 36/36 | 12/12  | 0.0188 |
| H | gpt_6_sol | 0.0537 | 0.0328 | 0.0305 | 108/108 | 12/12  | 0.0043 |
| H | gpt_6_sol_tools | 0.0615 | 0.0384 | 0.0405 | 108/108 | 12/12  | 0.0049 |
| B | plugin | 0.1643 | 0.1049 | -0.1643 | 36/36 | 12/12  | 0.0035 |
| B | median | 0.2321 | 0.1287 | 0.0006 | 36/36 | 12/12  | 0.0000 |
| B | mle | 0.0757 | 0.0493 | -0.0085 | 36/36 | 12/12  | 0.0050 |
| B | et | 0.0789 | 0.0457 | 0.0230 | 36/36 | 12/12  | 0.0039 |
| B | qwen_thinking | 0.2196 | 0.1429 | -0.0152 | 108/108 | 12/12  | 0.0178 |
| B | qwen_nonthinking | 0.5267 | 0.4988 | 0.5156 | 108/108 | 12/12  | 0.0186 |
| B | deepseek_flash | 0.1949 | 0.1183 | 0.0344 | 36/36 | 12/12  | 0.0325 |
| B | gpt_6_sol | 0.1077 | 0.0777 | 0.0050 | 108/108 | 12/12  | 0.0073 |
| B | gpt_6_sol_tools | 0.1512 | 0.1102 | 0.0645 | 108/108 | 12/12  | 0.0082 |

## surrogate

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources | MCSE_2 |
|---|---|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0160 | 0.0211 | 0.0008 | 36/36 | 12/12  | 0.0023 |
| R | median | 0.3453 | 0.2994 | -0.2655 | 36/36 | 12/12  | 0.0000 |
| R | mle | 0.0365 | 0.0335 | 0.0227 | 36/36 | 12/12  | 0.0024 |
| R | et | 0.0171 | 0.0214 | -0.0063 | 31/36 | 12/12  | 0.0024 |
| R | qwen_thinking | 0.0172 | 0.0233 | 0.0008 | 108/108 | 12/12  | 0.0024 |
| R | qwen_nonthinking | 0.2604 | 0.2506 | 0.1723 | 108/108 | 12/12  | 0.0228 |
| R | deepseek_flash | 0.0160 | 0.0211 | 0.0008 | 36/36 | 12/12  | 0.0023 |
| R | gpt_6_sol | 0.0160 | 0.0211 | 0.0008 | 108/108 | 12/12  | 0.0023 |
| R | gpt_6_sol_tools | 0.0160 | 0.0211 | 0.0008 | 108/108 | 12/12  | 0.0023 |
| S | plugin | 0.2456 | 0.3152 | 0.1945 | 36/36 | 12/12  | 0.0053 |
| S | median | 0.3453 | 0.2994 | -0.2655 | 36/36 | 12/12  | 0.0000 |
| S | mle | 0.0861 | 0.0902 | 0.0283 | 36/36 | 12/12  | 0.0141 |
| S | et | 0.0618 | 0.0631 | -0.0546 | 36/36 | 12/12  | 0.0092 |
| S | qwen_thinking | 0.1742 | 0.2004 | 0.1077 | 107/108 | 12/12  | 0.0119 |
| S | qwen_nonthinking | 0.4426 | 0.4427 | 0.0768 | 108/108 | 12/12  | 0.0264 |
| S | deepseek_flash | 0.1251 | 0.1013 | -0.0031 | 36/36 | 12/12  | 0.0194 |
| S | gpt_6_sol | 0.0923 | 0.0745 | 0.0084 | 108/108 | 12/12  | 0.0128 |
| S | gpt_6_sol_tools | 0.0878 | 0.0701 | 0.0005 | 108/108 | 12/12  | 0.0131 |
| H | plugin | 0.0945 | 0.2008 | -0.0934 | 36/36 | 12/12  | 0.0018 |
| H | median | 0.3453 | 0.2994 | -0.2655 | 36/36 | 12/12  | 0.0000 |
| H | mle | 0.0491 | 0.0471 | 0.0150 | 36/36 | 12/12  | 0.0024 |
| H | et | 0.0435 | 0.0597 | -0.0389 | 36/36 | 12/12  | 0.0017 |
| H | qwen_thinking | 0.1382 | 0.1880 | -0.0150 | 108/108 | 12/12  | 0.0086 |
| H | qwen_nonthinking | 0.3627 | 0.3015 | -0.2744 | 108/108 | 12/12  | 0.0213 |
| H | deepseek_flash | 0.1049 | 0.0786 | 0.0285 | 36/36 | 12/12  | 0.0161 |
| H | gpt_6_sol | 0.0575 | 0.0428 | -0.0113 | 108/108 | 12/12  | 0.0052 |
| H | gpt_6_sol_tools | 0.0504 | 0.0430 | -0.0104 | 108/108 | 12/12  | 0.0031 |
| B | plugin | 0.3251 | 0.2827 | -0.3251 | 36/36 | 12/12  | 0.0031 |
| B | median | 0.3453 | 0.2994 | -0.2655 | 36/36 | 12/12  | 0.0000 |
| B | mle | 0.1321 | 0.1516 | -0.1063 | 36/36 | 12/12  | 0.0036 |
| B | et | 0.1186 | 0.1317 | -0.0764 | 36/36 | 12/12  | 0.0026 |
| B | qwen_thinking | 0.3353 | 0.2975 | -0.1974 | 108/108 | 12/12  | 0.0161 |
| B | qwen_nonthinking | 0.2669 | 0.2887 | 0.2388 | 108/108 | 12/12  | 0.0098 |
| B | deepseek_flash | 0.2663 | 0.2437 | -0.1314 | 36/36 | 12/12  | 0.0285 |
| B | gpt_6_sol | 0.1329 | 0.1221 | -0.0181 | 108/108 | 12/12  | 0.0102 |
| B | gpt_6_sol_tools | 0.1279 | 0.1211 | 0.0108 | 108/108 | 12/12  | 0.0115 |

## synthetic

| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Valid/planned | Sources | MCSE_2 |
|---|---|---:|---:|---:|---:|---:|---:|
| R | plugin | 0.0235 | 0.0168 | -0.0032 | 24/24 | 8/8  | 0.0026 |
| R | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8  | 0.0000 |
| R | mle | 0.0219 | 0.0152 | -0.0077 | 24/24 | 8/8  | 0.0031 |
| R | et | 0.0226 | 0.0167 | -0.0034 | 21/24 | 8/8  | 0.0025 |
| R | qwen_thinking | 0.0235 | 0.0173 | -0.0033 | 72/72 | 8/8  | 0.0027 |
| R | qwen_nonthinking | 0.2194 | 0.1734 | 0.1827 | 72/72 | 8/8  | 0.0223 |
| R | deepseek_flash | 0.0235 | 0.0168 | -0.0032 | 24/24 | 8/8  | 0.0026 |
| R | gpt_6_sol | 0.0235 | 0.0168 | -0.0032 | 72/72 | 8/8  | 0.0026 |
| R | gpt_6_sol_tools | 0.0235 | 0.0168 | -0.0032 | 72/72 | 8/8  | 0.0026 |
| S | plugin | 0.1271 | 0.1083 | 0.1271 | 24/24 | 8/8  | 0.0040 |
| S | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8  | 0.0000 |
| S | mle | 0.0250 | 0.0201 | 0.0085 | 24/24 | 8/8  | 0.0044 |
| S | et | 0.0184 | 0.0164 | 0.0032 | 23/24 | 8/8  | 0.0038 |
| S | qwen_thinking | 0.0786 | 0.0627 | 0.0614 | 72/72 | 8/8  | 0.0084 |
| S | qwen_nonthinking | 0.2807 | 0.2544 | 0.2750 | 72/72 | 8/8  | 0.0222 |
| S | deepseek_flash | 0.0278 | 0.0196 | 0.0010 | 24/24 | 8/8  | 0.0050 |
| S | gpt_6_sol | 0.0295 | 0.0210 | 0.0081 | 72/72 | 8/8  | 0.0048 |
| S | gpt_6_sol_tools | 0.0271 | 0.0199 | 0.0083 | 72/72 | 8/8  | 0.0047 |
| H | plugin | 0.0948 | 0.1214 | -0.0902 | 24/24 | 8/8  | 0.0051 |
| H | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8  | 0.0000 |
| H | mle | 0.0375 | 0.0442 | 0.0217 | 24/24 | 8/8  | 0.0061 |
| H | et | 0.0293 | 0.0191 | -0.0134 | 24/24 | 8/8  | 0.0041 |
| H | qwen_thinking | 0.1631 | 0.1348 | 0.0426 | 72/72 | 8/8  | 0.0065 |
| H | qwen_nonthinking | 0.2435 | 0.1822 | -0.0826 | 72/72 | 8/8  | 0.0223 |
| H | deepseek_flash | 0.0617 | 0.0567 | 0.0434 | 24/24 | 8/8  | 0.0070 |
| H | gpt_6_sol | 0.0361 | 0.0248 | 0.0057 | 72/72 | 8/8  | 0.0049 |
| H | gpt_6_sol_tools | 0.0412 | 0.0279 | 0.0077 | 72/72 | 8/8  | 0.0067 |
| B | plugin | 0.4255 | 0.2543 | -0.4255 | 24/24 | 8/8  | 0.0028 |
| B | median | 0.2987 | 0.2119 | -0.1506 | 24/24 | 8/8  | 0.0000 |
| B | mle | 0.0754 | 0.0630 | -0.0206 | 24/24 | 8/8  | 0.0134 |
| B | et | 0.0365 | 0.0443 | -0.0190 | 24/24 | 8/8  | 0.0075 |
| B | qwen_thinking | 0.3303 | 0.2518 | -0.1291 | 72/72 | 8/8  | 0.0196 |
| B | qwen_nonthinking | 0.3382 | 0.3083 | 0.1636 | 72/72 | 8/8  | 0.0321 |
| B | deepseek_flash | 0.2452 | 0.1794 | -0.0439 | 24/24 | 8/8  | 0.0482 |
| B | gpt_6_sol | 0.1295 | 0.1037 | 0.0578 | 72/72 | 8/8  | 0.0135 |
| B | gpt_6_sol_tools | 0.1067 | 0.0917 | 0.0458 | 72/72 | 8/8  | 0.0076 |

The eight synthetic graphs form four generator pairs with shared random numbers;
these MAE results are descriptive (minimum two-sided sign-flip p with four blocks: 0.125).

## Paired original minus surrogate

| Arm | Method | Families | Mean difference | Original better | Exact sign-flip p |
|---|---|---:|---:|---:|---:|
| R | plugin | 12 | 0.0107 | 3/12 | 0.1465 |
| R | median | 12 | -0.1132 | 8/12 | 0.0854 |
| R | mle | 12 | 0.0008 | 7/12 | 0.9438 |
| R | et | 12 | 0.0112 | 4/12 | 0.0566 |
| R | qwen_thinking | 12 | 0.0097 | 3/12 | 0.1558 |
| R | qwen_nonthinking | 12 | 0.1510 | 2/12 | 0.0151 |
| R | deepseek_flash | 12 | 0.0107 | 3/12 | 0.1465 |
| R | gpt_6_sol | 12 | 0.0107 | 3/12 | 0.1470 |
| R | gpt_6_sol_tools | 12 | 0.0113 | 3/12 | 0.1074 |
| S | plugin | 12 | 0.0550 | 4/12 | 0.2339 |
| S | median | 12 | -0.1132 | 8/12 | 0.0854 |
| S | mle | 12 | -0.0175 | 9/12 | 0.4141 |
| S | et | 12 | -0.0114 | 7/12 | 0.6328 |
| S | qwen_thinking | 12 | 0.0073 | 5/12 | 0.8394 |
| S | qwen_nonthinking | 12 | 0.0326 | 3/12 | 0.5884 |
| S | deepseek_flash | 12 | -0.0378 | 9/12 | 0.2637 |
| S | gpt_6_sol | 12 | -0.0082 | 9/12 | 0.6992 |
| S | gpt_6_sol_tools | 12 | -0.0065 | 8/12 | 0.7588 |
| H | plugin | 12 | -0.0305 | 7/12 | 0.2881 |
| H | median | 12 | -0.1132 | 8/12 | 0.0854 |
| H | mle | 12 | 0.0214 | 5/12 | 0.1787 |
| H | et | 12 | -0.0045 | 4/12 | 0.7886 |
| H | qwen_thinking | 12 | 0.0216 | 6/12 | 0.4258 |
| H | qwen_nonthinking | 12 | -0.1111 | 8/12 | 0.1196 |
| H | deepseek_flash | 12 | 0.0136 | 6/12 | 0.6685 |
| H | gpt_6_sol | 12 | -0.0038 | 8/12 | 0.8306 |
| H | gpt_6_sol_tools | 12 | 0.0110 | 7/12 | 0.5596 |
| B | plugin | 12 | -0.1608 | 12/12 | 0.0005 |
| B | median | 12 | -0.1132 | 8/12 | 0.0854 |
| B | mle | 12 | -0.0564 | 9/12 | 0.0508 |
| B | et | 12 | -0.0398 | 8/12 | 0.1704 |
| B | qwen_thinking | 12 | -0.1156 | 12/12 | 0.0005 |
| B | qwen_nonthinking | 12 | 0.2598 | 0/12 | 0.0005 |
| B | deepseek_flash | 12 | -0.0714 | 11/12 | 0.0107 |
| B | gpt_6_sol | 12 | -0.0252 | 7/12 | 0.2959 |
| B | gpt_6_sol_tools | 12 | 0.0234 | 5/12 | 0.5312 |

## Source-level paired differences within a block

| Group | Arm | Comparison | Mean difference | First better | Exact sign-flip p |
|---|---|---|---:|---:|---:|
| real | R | deepseek_flash vs plugin | -0.0000 | 10/12 | 0.2520 |
| real | R | deepseek_flash vs et | -0.0015 | 9/12 | 0.2212 |
| real | R | deepseek_flash vs qwen_thinking | -0.0002 | 7/12 | 0.1709 |
| real | R | gpt_6_sol vs plugin | -0.0000 | 6/12 | 0.4917 |
| real | R | gpt_6_sol vs et | -0.0015 | 9/12 | 0.2197 |
| real | R | gpt_6_sol vs qwen_thinking | -0.0002 | 8/12 | 0.0757 |
| real | R | gpt_6_sol_tools vs plugin | 0.0006 | 5/12 | 0.6567 |
| real | R | gpt_6_sol_tools vs et | -0.0009 | 9/12 | 0.5596 |
| real | R | gpt_6_sol_tools vs qwen_thinking | 0.0004 | 8/12 | 0.9424 |
| real | R | deepseek_flash vs gpt_6_sol | 0.0000 | 6/12 | 0.5010 |
| real | R | gpt_6_sol_tools vs gpt_6_sol | 0.0006 | 4/12 | 0.1602 |
| real | S | deepseek_flash vs plugin | -0.2133 | 12/12 | 0.0005 |
| real | S | deepseek_flash vs mle | 0.0188 | 2/12 | 0.0347 |
| real | S | deepseek_flash vs et | 0.0369 | 3/12 | 0.0078 |
| real | S | deepseek_flash vs qwen_thinking | -0.0941 | 11/12 | 0.0015 |
| real | S | gpt_6_sol vs plugin | -0.2166 | 12/12 | 0.0005 |
| real | S | gpt_6_sol vs mle | 0.0156 | 3/12 | 0.0479 |
| real | S | gpt_6_sol vs et | 0.0337 | 4/12 | 0.0195 |
| real | S | gpt_6_sol vs qwen_thinking | -0.0974 | 12/12 | 0.0005 |
| real | S | gpt_6_sol_tools vs plugin | -0.2193 | 12/12 | 0.0005 |
| real | S | gpt_6_sol_tools vs mle | 0.0128 | 4/12 | 0.0781 |
| real | S | gpt_6_sol_tools vs et | 0.0310 | 4/12 | 0.0254 |
| real | S | gpt_6_sol_tools vs qwen_thinking | -0.1001 | 12/12 | 0.0005 |
| real | S | deepseek_flash vs gpt_6_sol | 0.0032 | 5/12 | 0.7856 |
| real | S | gpt_6_sol_tools vs gpt_6_sol | -0.0027 | 10/12 | 0.1528 |
| real | H | deepseek_flash vs plugin | 0.0545 | 3/12 | 0.0645 |
| real | H | deepseek_flash vs mle | 0.0479 | 2/12 | 0.0181 |
| real | H | deepseek_flash vs et | 0.0795 | 3/12 | 0.0054 |
| real | H | deepseek_flash vs qwen_thinking | -0.0413 | 7/12 | 0.1602 |
| real | H | gpt_6_sol vs plugin | -0.0102 | 7/12 | 0.5454 |
| real | H | gpt_6_sol vs mle | -0.0168 | 6/12 | 0.1958 |
| real | H | gpt_6_sol vs et | 0.0147 | 5/12 | 0.1533 |
| real | H | gpt_6_sol vs qwen_thinking | -0.1061 | 10/12 | 0.0029 |
| real | H | gpt_6_sol_tools vs plugin | -0.0025 | 7/12 | 0.9053 |
| real | H | gpt_6_sol_tools vs mle | -0.0091 | 7/12 | 0.5439 |
| real | H | gpt_6_sol_tools vs et | 0.0225 | 4/12 | 0.1216 |
| real | H | gpt_6_sol_tools vs qwen_thinking | -0.0983 | 10/12 | 0.0078 |
| real | H | deepseek_flash vs gpt_6_sol | 0.0647 | 3/12 | 0.0103 |
| real | H | gpt_6_sol_tools vs gpt_6_sol | 0.0077 | 5/12 | 0.2568 |
| real | B | deepseek_flash vs plugin | 0.0307 | 5/12 | 0.4351 |
| real | B | deepseek_flash vs mle | 0.1192 | 2/12 | 0.0059 |
| real | B | deepseek_flash vs et | 0.1161 | 3/12 | 0.0078 |
| real | B | deepseek_flash vs qwen_thinking | -0.0247 | 7/12 | 0.4268 |
| real | B | gpt_6_sol vs plugin | -0.0566 | 9/12 | 0.0195 |
| real | B | gpt_6_sol vs mle | 0.0320 | 6/12 | 0.1592 |
| real | B | gpt_6_sol vs et | 0.0288 | 5/12 | 0.1973 |
| real | B | gpt_6_sol vs qwen_thinking | -0.1120 | 12/12 | 0.0005 |
| real | B | gpt_6_sol_tools vs plugin | -0.0130 | 9/12 | 0.7642 |
| real | B | gpt_6_sol_tools vs mle | 0.0755 | 4/12 | 0.0254 |
| real | B | gpt_6_sol_tools vs et | 0.0724 | 3/12 | 0.0229 |
| real | B | gpt_6_sol_tools vs qwen_thinking | -0.0684 | 10/12 | 0.0615 |
| real | B | deepseek_flash vs gpt_6_sol | 0.0873 | 1/12 | 0.0010 |
| real | B | gpt_6_sol_tools vs gpt_6_sol | 0.0436 | 4/12 | 0.0078 |
| surrogate | R | deepseek_flash vs plugin | -0.0000 | 5/12 | 0.8691 |
| surrogate | R | deepseek_flash vs et | -0.0011 | 7/12 | 0.5767 |
| surrogate | R | deepseek_flash vs qwen_thinking | -0.0012 | 6/12 | 0.0991 |
| surrogate | R | gpt_6_sol vs plugin | 0.0000 | 5/12 | 0.3921 |
| surrogate | R | gpt_6_sol vs et | -0.0011 | 7/12 | 0.5786 |
| surrogate | R | gpt_6_sol vs qwen_thinking | -0.0012 | 6/12 | 0.1138 |
| surrogate | R | gpt_6_sol_tools vs plugin | -0.0000 | 9/12 | 0.0479 |
| surrogate | R | gpt_6_sol_tools vs et | -0.0011 | 7/12 | 0.5762 |
| surrogate | R | gpt_6_sol_tools vs qwen_thinking | -0.0012 | 8/12 | 0.0806 |
| surrogate | R | deepseek_flash vs gpt_6_sol | -0.0000 | 7/12 | 0.3916 |
| surrogate | R | gpt_6_sol_tools vs gpt_6_sol | -0.0000 | 10/12 | 0.0874 |
| surrogate | S | deepseek_flash vs plugin | -0.1205 | 9/12 | 0.0137 |
| surrogate | S | deepseek_flash vs mle | 0.0391 | 2/12 | 0.0161 |
| surrogate | S | deepseek_flash vs et | 0.0634 | 2/12 | 0.0239 |
| surrogate | S | deepseek_flash vs qwen_thinking | -0.0490 | 8/12 | 0.0859 |
| surrogate | S | gpt_6_sol vs plugin | -0.1533 | 12/12 | 0.0005 |
| surrogate | S | gpt_6_sol vs mle | 0.0063 | 5/12 | 0.4648 |
| surrogate | S | gpt_6_sol vs et | 0.0305 | 4/12 | 0.1094 |
| surrogate | S | gpt_6_sol vs qwen_thinking | -0.0819 | 12/12 | 0.0005 |
| surrogate | S | gpt_6_sol_tools vs plugin | -0.1578 | 12/12 | 0.0005 |
| surrogate | S | gpt_6_sol_tools vs mle | 0.0018 | 6/12 | 0.8403 |
| surrogate | S | gpt_6_sol_tools vs et | 0.0261 | 4/12 | 0.1367 |
| surrogate | S | gpt_6_sol_tools vs qwen_thinking | -0.0863 | 12/12 | 0.0005 |
| surrogate | S | deepseek_flash vs gpt_6_sol | 0.0328 | 3/12 | 0.0044 |
| surrogate | S | gpt_6_sol_tools vs gpt_6_sol | -0.0045 | 7/12 | 0.0918 |
| surrogate | H | deepseek_flash vs plugin | 0.0104 | 4/12 | 0.6040 |
| surrogate | H | deepseek_flash vs mle | 0.0557 | 3/12 | 0.0068 |
| surrogate | H | deepseek_flash vs et | 0.0614 | 3/12 | 0.0122 |
| surrogate | H | deepseek_flash vs qwen_thinking | -0.0334 | 9/12 | 0.1104 |
| surrogate | H | gpt_6_sol vs plugin | -0.0370 | 7/12 | 0.1060 |
| surrogate | H | gpt_6_sol vs mle | 0.0084 | 4/12 | 0.3662 |
| surrogate | H | gpt_6_sol vs et | 0.0140 | 5/12 | 0.4341 |
| surrogate | H | gpt_6_sol vs qwen_thinking | -0.0807 | 11/12 | 0.0029 |
| surrogate | H | gpt_6_sol_tools vs plugin | -0.0440 | 10/12 | 0.0493 |
| surrogate | H | gpt_6_sol_tools vs mle | 0.0013 | 6/12 | 0.8926 |
| surrogate | H | gpt_6_sol_tools vs et | 0.0069 | 5/12 | 0.6870 |
| surrogate | H | gpt_6_sol_tools vs qwen_thinking | -0.0878 | 11/12 | 0.0020 |
| surrogate | H | deepseek_flash vs gpt_6_sol | 0.0474 | 2/12 | 0.0020 |
| surrogate | H | gpt_6_sol_tools vs gpt_6_sol | -0.0071 | 8/12 | 0.2803 |
| surrogate | B | deepseek_flash vs plugin | -0.0588 | 10/12 | 0.0034 |
| surrogate | B | deepseek_flash vs mle | 0.1342 | 2/12 | 0.0020 |
| surrogate | B | deepseek_flash vs et | 0.1477 | 1/12 | 0.0015 |
| surrogate | B | deepseek_flash vs qwen_thinking | -0.0689 | 10/12 | 0.0029 |
| surrogate | B | gpt_6_sol vs plugin | -0.1922 | 12/12 | 0.0005 |
| surrogate | B | gpt_6_sol vs mle | 0.0008 | 6/12 | 0.9751 |
| surrogate | B | gpt_6_sol vs et | 0.0143 | 6/12 | 0.6089 |
| surrogate | B | gpt_6_sol vs qwen_thinking | -0.2024 | 12/12 | 0.0005 |
| surrogate | B | gpt_6_sol_tools vs plugin | -0.1972 | 12/12 | 0.0005 |
| surrogate | B | gpt_6_sol_tools vs mle | -0.0042 | 6/12 | 0.8706 |
| surrogate | B | gpt_6_sol_tools vs et | 0.0093 | 6/12 | 0.7461 |
| surrogate | B | gpt_6_sol_tools vs qwen_thinking | -0.2074 | 12/12 | 0.0005 |
| surrogate | B | deepseek_flash vs gpt_6_sol | 0.1334 | 0/12 | 0.0005 |
| surrogate | B | gpt_6_sol_tools vs gpt_6_sol | -0.0050 | 5/12 | 0.7861 |
