# Project state (supersedes HANDOFF_v2.md for the current direction)

HANDOFF_v2.md predates the results below and is now outdated on the thesis spine.
Keep it for history; this file is the current state.

## Thesis spine (as of this sync)

Estimating edge persistence rho under forward-time sampling bias is a label-free
ESTIMATOR problem, not a representation problem:

1. The compact (n,w) walk summary retains the rho signal: a supervised oracle
   recovers rho across the whole coverage range (down to coverage ~1e-4).
2. The standard uniform-occupancy MLE is INCONSISTENT under forward-time
   sampling: it falls below the information-free floor at essentially all
   coverage (not merely inefficient).
3. The bias is not correctable from the biased sample alone. Adding a bias
   parameter beta and re-maximising the same likelihood does not help: the
   likelihood is maximised at beta=0 (the no-correction value). beta is not
   identifiable from the sample.
4. A richer input (per-window counts) makes the bias DETECTABLE but still not
   CORRECTABLE: a forward crawl observes early windows with probability ~0.4%,
   so per-edge early activity is structurally unsampled.
5. Correction requires externally supplied bias strength (calibration on labels,
   or disclosure). The beta-calibrated bias-aware MLE reaches the oracle at
   mid/high coverage, proving the external information is sufficient.

This does not depend on LLMs "winning". mean_occupancy is co-primary (collinear
with rho, threshold-robustness only). C is redundant on synthetic data.

## What is done

- Twin generator, 4 walk strategies, occupancy MLE (uniform): existing src/.
- Coverage backbone on bwUniCluster (sizes 400..50k, continuous rho): outputs in
  results/coverage/ (summaries.csv.gz, grid_results.csv, manifest.csv,
  grid_mae_vs_coverage.png, band_per_walk_rho.png).
- Bias identifiability (beta-fit vs beta-calibrated vs oracle) and per-window
  bias test: src/bias_identifiability.py, src/per_window_bias_test.py; findings
  in results/bias_identifiability/RESULTS.md.

## What is NOT done yet

- Phase 3, the LLM experiment: zero-shot vs disclosed-mechanism, over the API,
  ~80 continuous-rho cases across coverage bands, median of >=3 runs, scored into
  the same floor / uniform / beta-calibrated / oracle band. The sharp question:
  does an LLM told about the forward bias use it like the beta-calibrated MLE, or
  revert to beta=0? The earlier 5-model A/B (web UIs, n=5, degenerate test set)
  is NOT this experiment and should be redone properly via the API.
  - `experiments/disclosure/` holds the prompt templates and answer keys.
  - There is currently NO `score_disclosure.py` and NO systematic `model_outputs/`;
    both are to be built for Phase 3.

## File provenance notes

- `src/corrected_estimator.py` is RECONSTRUCTED (occupancy_table + rho_mle,
  returns (rho, pi)). If an original with more functions exists, merge.
- `src/run_pilot_walks.py` and `src/pilot_eval.py` are PATCHED: run_pilot_walks
  emits rho_mle/occ_mle per checkpoint; pilot_eval adds the `mle` estimator tier
  and a floor->mle->oracle band line. The analysis scripts depend on these.
