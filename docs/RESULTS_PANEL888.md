# Results: panel888-pwt-srw-20260921 (Qwen)

One final panel: 8 real sources, 8 matched P[w,t] surrogates, 8 synthetic instances;
five arms R, S1, S2, H, B. Qwen answers were generated from the frozen commit
`9db974e` in one production chain (`panel888_final_main`, jobs
7116955/7116956/7116959, archive 7117295). Where a request's ID, configuration,
repeat, prompt hash, payload hash and seed were byte-identical to one already
answered in an earlier (superseded) workspace, that answer was reused
(`cluster/reuse_answers.py`; 749 of 2,160 reused, all R/S1/B); the remaining
1,411 were generated fresh. Sol and DeepSeek were not run.

- Answers: 2,160 of 2,160 (1,080 thinking, 1,080 non-thinking); no duplicates,
  no missing, 0 technical errors, 0 output-limit hits.
- Strictly valid: 2,158; two thinking answers invalid. No imputation.
- Independent re-parse (`scripts/audit_qwen.py`) agrees with the evaluation on every answer.

Accuracy is conditional on valid answers, equal-source means with draw-clustered MCSE
([summary.csv](results/panel888_qwen/summary.csv), per source
[source_results.csv](results/panel888_qwen/source_results.csv)). Paired temporal control
(AE_Delta_2 primary): [paired_summary.csv](results/panel888_qwen/paired_summary.csv).
Synthetic mode contrasts within r1/r2:
[synthetic_within_replicate_contrasts.csv](results/panel888_qwen/synthetic_within_replicate_contrasts.csv).
Hashes of all evidence: [RESULT_FREEZE.json](results/panel888_qwen/RESULT_FREEZE.json);
offline freeze: [FREEZE.json](results/panel888_offline/FREEZE.json).

S1 and S2 are the identical degree-biased walk draw; S1 shows only the observed
dyad table (like R/H), S2 additionally shows the per-pattern traversal counts and
inverse-degree weights the design-aware correction needs. "ref" below is each
arm's primary same-information reference (R/S1 plug-in, S2 the design-aware
inverse-traversal-weight estimate, H the homogeneous ZT-Binomial, B the beta-ZTP
mixture) computed from the identical observation the model sees.

Real sources, MAE2 (MCSE) of valid answers:

| Arm | Qwen thinking | Qwen non-thinking | same-info reference | plug-in | ExtraTrees (pooled) |
|---|---|---|---|---|---|
| R  | 0.015 (0.003) | 0.505 (0.018) | 0.015 | 0.015 | 0.043 |
| S1 | 0.060 (0.002) | 0.386 (0.019) | 0.060 | 0.060 | 0.068 |
| S2 | 0.053 (0.005) | 0.500 (0.019) | 0.014 | 0.060 | 0.045 |
| H  | 0.227 (0.019) | 0.253 (0.024) | 0.126 | 0.070 | 0.051 |
| B  | 0.205 (0.023) | 0.559 (0.016) | 0.065 | 0.156 | 0.063 |

Surrogates, MAE2 (MCSE) of valid answers:

| Arm | Qwen thinking | Qwen non-thinking | same-info reference | plug-in | ExtraTrees (pooled) |
|---|---|---|---|---|---|
| R  | 0.012 (0.002) | 0.307 (0.014) | 0.012 | 0.012 | 0.026 |
| S1 | 0.057 (0.002) | 0.245 (0.010) | 0.056 | 0.056 | 0.044 |
| S2 | 0.075 (0.013) | 0.275 (0.010) | 0.023 | 0.056 | 0.030 |
| H  | 0.185 (0.011) | 0.284 (0.024) | 0.125 | 0.105 | 0.079 |
| B  | 0.309 (0.019) | 0.310 (0.018) | 0.161 | 0.304 | 0.078 |

**S1 vs. S2**: Qwen-thinking on S1 tracks its plug-in baseline almost exactly
(0.060 vs. 0.060 real; 0.057 vs. 0.056 surrogate) — expected, since S1 shows no
information beyond what the plug-in already uses. On S2, Qwen-thinking's error
is somewhat lower than on S1 (0.053 real; but 0.075 on surrogates, higher) while
the design-aware same-information estimator computed from the identical S2
observation achieves 0.014 (real) / 0.023 (surrogate) — four to five times
smaller. The model uses some of the walker information shown in S2 but comes
nowhere near the statistically optimal use of it; see BUDGET_SENSITIVITY.md for
the same pattern across the full 2.5–50% coverage range.
