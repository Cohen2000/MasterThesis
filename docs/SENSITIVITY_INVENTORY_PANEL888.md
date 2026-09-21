# Offline diagnostics of the final panel

All grids were inherited from the development history (inventory of the source
commit b95fa3e in `archive/pre_panel888_20260921/`) or fixed by the design
requirements before any new Qwen answer existed. None selects a primary
parameter, a surrogate or a production walk length. Outputs are in
`results/panel888/diagnostics/`; compact copies in `docs/results/panel888_offline/`.

| Analysis | Script | Fixed grid / treatment |
|---|---|---|
| H history fraction and P/Q/C/T oracle decomposition | `diagnose_history.py`, `history_diagnostics.py` | h = .40/.60/.80; primary .60; ExtraTrees trained at .60 (cross-h) |
| Selection vs. lost-history split of the plug-in error | `diagnose_decomposition.py` | all arms; 100 development pool graphs and the 24 main graphs |
| SRW components, reachability, coverage, revisits, traversal and hub concentration, finite-walk vs. stationary | `diagnose_srw.py` | 32 paths; L and min(4L, 10^6); analytic component profiles |
| P[w,t] null distribution | `diagnose_null_model.py` | 99 further shuffles per parent; rank of the productive surrogate only |
| Window count and target definition | `diagnose_windows.py` | W = 2..20 incl. historical {4,5,8}; relative thresholds .2/.4/.6/.8/1; occupancy; one-step persistence |
| Census descriptors and event-count feasibility bounds | `diagnose_windows.py` (`census.py`, `dataset_census.py`) | half-window shift, equal-event rank windows, event weighting, lifetimes, inter-event burstiness, censoring; bounds for .15/.55 |
| B mixture bound sensitivity | `diagnose_mixture_bounds.py`, `mixtures.bound_sensitivity` | bounds widened by two decades |
| Budget calibration and MCSE | `sampling.calibrate` | 256 calibration walks; 1024 validation, 4096 if relative MCSE > .01; tolerance .05 |
| Development check, fit status, B fallback, budget-matched subset | `references.development_check` | 100 development graphs; fixed fallback rule |
| Paired temporal control of the references | `evaluate_paired_controls.py` | eight parent–surrogate pairs; AE_Delta_2 primary |

Historical prompt ablations, bootstrap imputation, trailing-JSON extraction,
weighted-walk kernels, earlier H definitions, the target-driven timing allocator
and earlier instance panels belong to superseded methods; they are documented in
the archive only and are not sensitivity arms of the final study.
