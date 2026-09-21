# Results: panel888-pwt-srw-20260921 (Qwen)

One final panel: 8 real sources, 8 matched P[w,t] surrogates, 8 synthetic instances.
All Qwen answers were newly generated from the frozen commit 54d5832 in one production
chain (jobs 7093897–7093900); no earlier answer was reused. Sol and DeepSeek were not run.

- Answers: 1,728 of 1,728 (864 thinking, 864 non-thinking); no duplicates, no missing,
  0 technical errors, 0 output-limit hits.
- Strictly valid: 1,727; one thinking answer invalid. No imputation.
- Independent re-parse (`scripts/audit_qwen.py`) agrees with the evaluation on every answer.

Accuracy is conditional on valid answers, equal-source means with draw-clustered MCSE
([summary.csv](results/panel888_qwen/summary.csv), per source
[source_results.csv](results/panel888_qwen/source_results.csv)). Paired temporal control
(AE_Delta_2 primary): [paired_summary.csv](results/panel888_qwen/paired_summary.csv).
Synthetic mode contrasts within r1/r2:
[synthetic_within_replicate_contrasts.csv](results/panel888_qwen/synthetic_within_replicate_contrasts.csv).
Hashes of all evidence: [RESULT_FREEZE.json](results/panel888_qwen/RESULT_FREEZE.json);
offline freeze: [FREEZE.json](results/panel888_offline/FREEZE.json).

Real sources, MAE2 (MCSE) of valid answers vs. the primary reference of each arm:

| Arm | Qwen thinking | Qwen non-thinking | primary reference | plug-in | ExtraTrees (pooled) |
|---|---|---|---|---|---|
| R | 0.015 (0.003) | 0.505 (0.018) | 0.015 | 0.015 | 0.039 |
| S | 0.014 (0.003) | 0.403 (0.032) | 0.009 | 0.010 | 0.056 |
| H | 0.213 (0.021) | 0.357 (0.021) | 0.126 | 0.070 | 0.048 |
| B | 0.215 (0.021) | 0.554 (0.024) | 0.064 | 0.156 | 0.065 |
