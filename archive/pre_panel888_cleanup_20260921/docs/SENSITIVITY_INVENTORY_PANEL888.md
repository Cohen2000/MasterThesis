# Inventory before revision

Source commit b95fa3e; exact pre-edit code/config/docs and search transcript are
in archive/pre_panel888_20260921. ARCHIVE_MANIFEST.json inventories preserved files.

| Analysis | Source of truth | Fixed grid / treatment |
|---|---|---|
| H history and P/Q/C/T oracle decomposition | scripts/analyze_htime.py; history_diagnostics.py | .40/.60/.80; main .60, cross-h ExtraTrees |
| SRW components/reachability/revisits/finite mixing | scripts/analyze_srw.py; PROTOCOL_SRW_20260920.md | 32 paths; L and min(4L,1e6); analytic component profiles |
| Calibration/MCSE | sampling.py | 256 calibration; 1024 validation; extend 4096 above .01; tolerance .05 |
| Main/development selection-history decomposition | run_baseline_revision.py stages decompose/decompose_main | all arms, existing dev pool and final panel |
| Budget-matched subset / fit status and B fallback | run_baseline_revision.py stages dev/main; evaluator | fixed matching and fit flags |
| W/window count | src/dataset_census.py | W=2..20; relative thresholds .2/.4/.6/.8/1 (verify executable constants) |
| Older truth W robustness | archive/legacy_pre_current_design/src/analyze_target_robustness.py | {4,5,8}, rho, occupancy and one-step persistence; repeat on final panel |
| Mixture bound sensitivity | mixtures.py::bound_sensitivity | two log10 decades, diagnostic only |
| Retained census boundary/time diagnostics | src/census.py | W=5 half-window shift, equal-event rank windows, event weighting, lifetime, interevent burstiness, censoring |
| Event-count feasibility bounds | src/dataset_census.py | thresholds .15/.55 and distinct-time bound; no target-driven allocator |
| Additional P[w,t] null diagnostics | new user requirement | 99 seeds per parent, no selection |

Traversal concentration and hub concentration are recorded on the inherited
32-path L/4L grid. No new result-dependent grid is introduced.

Historical prompt ablations, bootstrap imputation, trailing-JSON extraction,
weighted-walk kernels, old H definitions and old 32-instance panels belong to
superseded methods. They are inventoried in archived docs/scripts but are not
current thesis sensitivity arms: resurrecting them would violate the fixed
prompt, strict parser and unchanged final mechanisms. Existing unit tests for
historical helpers are retained as regression checks, clearly historical.

The old optional target-driven timing allocator (rho targets .15/.55) is not a P[w,t] shuffle and belongs to earlier controlled-timing generator development. Its necessary event-count bounds are rerun, but no such outcome-constrained graph is introduced. W diagnostics use current source-time cutpoints and fixed horizons, without the superseded census epsilon shift. The inherited half-window and equal-event grids are descriptive only.
