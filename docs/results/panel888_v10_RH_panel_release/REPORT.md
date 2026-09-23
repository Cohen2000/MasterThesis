# R/H panel-size release sensitivity

Separate from the frozen v10 main results. The same 144 R/H draws release only
`n_panel` and the corresponding sampling-rule text. R+N retains complete retrieved
dyad histories; H+N retains the original truncated history and Temporal_access.
Neither releases full-archive sizes, a sampling fraction, calibration target or truth.
Qwen uses the sealed v10 model settings and the original paired generation seeds.
Negative delta means lower error.
Real-source equal-source MAE_2 is primary; ProfileMAE is secondary. Qwen accuracy
is conditional on strict valid final JSON answers. No outputs are repaired.

| Stratum | Arm | Method | Hidden MAE_2 | Released MAE_2 | Delta | Hidden ProfileMAE | Released ProfileMAE | Delta | Validity hidden → released |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| real | R | plugin | 0.0209 | 0.0209 | 0.0000 | 0.0130 | 0.0130 | 0.0000 | 1.0000 → 1.0000 |
| real | R | median | 0.2397 | 0.2397 | 0.0000 | 0.1102 | 0.1102 | 0.0000 | 1.0000 → 1.0000 |
| real | R | mle | 0.0234 | 0.0234 | 0.0000 | 0.0139 | 0.0139 | 0.0000 | 1.0000 → 1.0000 |
| real | R | et | 0.0229 | 0.0227 | -0.0003 | 0.0140 | 0.0141 | 0.0000 | 1.0000 → 1.0000 |
| real | R | qwen_thinking | 0.0213 | 0.0210 | -0.0003 | 0.0133 | 0.0131 | -0.0002 | 1.0000 → 1.0000 |
| real | R | qwen_nonthinking | 0.4382 | 0.4250 | -0.0132 | 0.3403 | 0.3427 | 0.0024 | 1.0000 → 1.0000 |
| real | H | plugin | 0.0666 | 0.0666 | 0.0000 | 0.0573 | 0.0573 | 0.0000 | 1.0000 → 1.0000 |
| real | H | median | 0.2397 | 0.2397 | 0.0000 | 0.1102 | 0.1102 | 0.0000 | 1.0000 → 1.0000 |
| real | H | mle | 0.0516 | 0.0516 | 0.0000 | 0.0289 | 0.0289 | 0.0000 | 1.0000 → 1.0000 |
| real | H | et | 0.0304 | 0.0276 | -0.0028 | 0.0253 | 0.0239 | -0.0014 | 1.0000 → 1.0000 |
| real | H | qwen_thinking | 0.1434 | 0.1490 | 0.0056 | 0.0805 | 0.0833 | 0.0028 | 0.9861 → 1.0000 |
| real | H | qwen_nonthinking | 0.2606 | 0.2151 | -0.0455 | 0.1568 | 0.1276 | -0.0292 | 1.0000 → 1.0000 |
| surrogate | R | plugin | 0.0135 | 0.0135 | 0.0000 | 0.0163 | 0.0163 | 0.0000 | 1.0000 → 1.0000 |
| surrogate | R | median | 0.3153 | 0.3153 | 0.0000 | 0.2477 | 0.2477 | 0.0000 | 1.0000 → 1.0000 |
| surrogate | R | mle | 0.0337 | 0.0337 | 0.0000 | 0.0290 | 0.0290 | 0.0000 | 1.0000 → 1.0000 |
| surrogate | R | et | 0.0155 | 0.0151 | -0.0005 | 0.0174 | 0.0171 | -0.0003 | 1.0000 → 1.0000 |
| surrogate | R | qwen_thinking | 0.0153 | 0.0140 | -0.0013 | 0.0188 | 0.0184 | -0.0004 | 1.0000 → 1.0000 |
| surrogate | R | qwen_nonthinking | 0.2967 | 0.2480 | -0.0487 | 0.2759 | 0.2211 | -0.0548 | 1.0000 → 1.0000 |
| surrogate | H | plugin | 0.1012 | 0.1012 | 0.0000 | 0.1736 | 0.1736 | 0.0000 | 1.0000 → 1.0000 |
| surrogate | H | median | 0.3153 | 0.3153 | 0.0000 | 0.2477 | 0.2477 | 0.0000 | 1.0000 → 1.0000 |
| surrogate | H | mle | 0.0573 | 0.0573 | 0.0000 | 0.0541 | 0.0541 | 0.0000 | 1.0000 → 1.0000 |
| surrogate | H | et | 0.0479 | 0.0491 | 0.0012 | 0.0605 | 0.0603 | -0.0001 | 1.0000 → 1.0000 |
| surrogate | H | qwen_thinking | 0.1637 | 0.1487 | -0.0150 | 0.1606 | 0.1681 | 0.0075 | 1.0000 → 1.0000 |
| surrogate | H | qwen_nonthinking | 0.3130 | 0.2808 | -0.0322 | 0.2471 | 0.2394 | -0.0078 | 1.0000 → 1.0000 |
| synthetic | R | plugin | 0.0235 | 0.0235 | 0.0000 | 0.0168 | 0.0168 | 0.0000 | 1.0000 → 1.0000 |
| synthetic | R | median | 0.2987 | 0.2987 | 0.0000 | 0.2119 | 0.2119 | 0.0000 | 1.0000 → 1.0000 |
| synthetic | R | mle | 0.0219 | 0.0219 | 0.0000 | 0.0152 | 0.0152 | 0.0000 | 1.0000 → 1.0000 |
| synthetic | R | et | 0.0225 | 0.0226 | 0.0001 | 0.0168 | 0.0167 | -0.0000 | 1.0000 → 1.0000 |
| synthetic | R | qwen_thinking | 0.0242 | 0.0235 | -0.0007 | 0.0171 | 0.0173 | 0.0002 | 1.0000 → 1.0000 |
| synthetic | R | qwen_nonthinking | 0.1694 | 0.2194 | 0.0500 | 0.1353 | 0.1734 | 0.0381 | 1.0000 → 1.0000 |
| synthetic | H | plugin | 0.0948 | 0.0948 | 0.0000 | 0.1214 | 0.1214 | 0.0000 | 1.0000 → 1.0000 |
| synthetic | H | median | 0.2987 | 0.2987 | 0.0000 | 0.2119 | 0.2119 | 0.0000 | 1.0000 → 1.0000 |
| synthetic | H | mle | 0.0375 | 0.0375 | 0.0000 | 0.0442 | 0.0442 | 0.0000 | 1.0000 → 1.0000 |
| synthetic | H | et | 0.0291 | 0.0293 | 0.0001 | 0.0191 | 0.0191 | -0.0000 | 1.0000 → 1.0000 |
| synthetic | H | qwen_thinking | 0.1680 | 0.1631 | -0.0049 | 0.1317 | 0.1348 | 0.0031 | 1.0000 → 1.0000 |
| synthetic | H | qwen_nonthinking | 0.2288 | 0.2435 | 0.0148 | 0.1705 | 0.1822 | 0.0117 | 1.0000 → 1.0000 |

## Panel-size-only diagnostic

Leave-one-real-source-out linear regression on `log1p(n_panel)` versus the
training-median-only baseline; this is diagnostic, not a new main estimator.

| Arm | Median-only MAE_2 | Panel-size-only MAE_2 |
|---|---:|---:|
| R | 0.2397 | 0.1226 |
| H | 0.2397 | 0.1168 |

## Real graph-level deltas

Counts use the eight real graph means; the full paired dataset is in
`GRAPH_COMPARISON.csv`.

| Arm | Method | Improved / 8 | Largest improvement | Largest worsening |
|---|---|---:|---|---|
| R | et | 4 | nr_digg_reply (-0.0057) | sp_highschool2013 (+0.0023) |
| R | qwen_thinking | 5 | snap_email_eu (-0.0051) | sp_hospital (+0.0017) |
| R | qwen_nonthinking | 3 | nr_digg_reply (-0.1432) | sp_highschool2013 (+0.0566) |
| H | et | 5 | snap_collegemsg (-0.0213) | snap_email_eu (+0.0052) |
| H | qwen_thinking | 3 | copenhagen_bluetooth (-0.1099) | snap_collegemsg (+0.0922) |
| H | qwen_nonthinking | 5 | nr_digg_reply (-0.2566) | sp_hospital (+0.0661) |

The ExtraTrees comparison uses one additional `log1p_n_panel` feature in
the same training rows, nested LOSO folds, grid, anchors, weights and seeds.
Plugin and median were re-evaluated and are identical. MLE was fitted to both
paired blocks locally and is identical by construction; the sealed v10
numerical predictions are retained to avoid platform-level optimizer drift.
See `GRAPH_COMPARISON.csv` for all 288 graph/arm/method paired deltas and
`ET_CHOICES.csv` for the 18 selected ET configurations.

## Provenance

Raw Qwen responses and rendered prompts are preserved in the cluster archive at
`/pfs/work9/workspace/scratch/tu_zxokn55-llm_pilot/panel888_v10_RH_panel_release/mainexp/archive`.
Qwen jobs: 7148186 and recovery 7148187; redundant job 7148188 was canceled
after all 864 responses were complete. Archive job: 7148189. The archive
readback found zero mismatches; its `CHECKSUMS.json` SHA-256 is recorded in
`VERIFICATION.json`. Its count of 865 JSON files under `answers` includes
`engine_inputs.json`; the 864 response files match all requested IDs.
