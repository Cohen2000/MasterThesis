# Results

Generated experiment artifacts are ignored by default and remain local unless a
small result or provenance artifact is intentionally selected for version control.

Current local experiment directory:
`results/main_experiment/cells10_final_20260920/`

Final-only raw archive: `results/imported_final_20260920/`.
Final evaluation: `results/main_experiment/cells10_final_20260920_qwen/`.
The SRW run/reference directory and H/S imports remain historical dependencies;
their model responses are never part of the final study. Fixed SRW reference
models are reused only after exact parsed-observation/feature verification.

Previous local runs and fitted models were moved, without modification, to
`archive/pre_srw_20260920/results/` and `archive/pre_h_time_20260920/results/`.

Historical tracked results are retained under `../archive/`.

Raw model responses, generated graphs, fitted models and other bulk artifacts
should not be committed directly.
