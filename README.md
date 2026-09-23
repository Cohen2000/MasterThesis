# Temporal persistence from sampled interaction data

**Current status (2026-09-23):** the proposed v10 interaction-walk design failed
its prespecified offline gate on three real or surrogate sources. The v10
offline study and Qwen runs were stopped. See the [generated gate table](docs/results/panel888_v10_walk_gate_20260923/WALK_GATE.md)
and [current state](docs/CURRENT_STATE.md). The pipeline below remains the
sealed v9 study until a new design is approved.

Can a language model estimate how persistent the dyads of a temporal network are
from a small sampled observation of it? The study (design identifier
`panel888-pwt-srw-20260921`) uses:

- 8 real human–human sources, 8 matched P[w,t] timestamp-shuffled surrogates
  (one per real source) and 8 synthetic instances;
- W = 5 windows, estimand rho_2..rho_5 (primary MAE2, secondary ProfileMAE);
- four observation mechanisms matched to 10% of the active dyad-windows:
  R node panel, a degree-biased random walk, H node panel with elapsed-time
  history (h = .60), B Bernoulli event thinning; the walk is shown in two
  information conditions on the identical draws: S1 (observed dyads only) and
  S2 (plus the walker information needed for a design-aware correction);
- five arms R, S1, S2, H, B x 3 test draws x 3 model repeats (360 observations,
  1,080 requests per model configuration), 5 training draws for the learned
  references;
- an ancillary budget sensitivity at 2.5–50% coverage.

Documents: [protocol](docs/PROTOCOL_PANEL888_20260921.md) (design),
[runbook](docs/RUNBOOK_PANEL888.md) (how to run), [current state](docs/CURRENT_STATE.md),
[sensitivity inventory](docs/SENSITIVITY_INVENTORY_PANEL888.md),
[methodological notes](docs/METHODOLOGICAL_AUDIT_PANEL888.md).

## Code map

`src/main_experiment/` — the scientific modules:

| Module | Content |
|---|---|
| `common.py` | panel, design constants, seed rule, I/O helpers |
| `data.py` | canonical graphs, windows, truth rho_k, real-source loading |
| `synthetic.py` | DAR and activity-driven generators |
| `surrogates.py` | P[w,t] timestamp shuffles and their invariant audit |
| `sampling.py`, `walk_kernel.cpp` | the four arms, budget calibration, degree-biased random walk |
| `observation.py` | observation blocks, prompts, 192 features |
| `baselines.py`, `mixtures.py` | same-information baselines, S1/S2 design-aware oracle, B Beta-mixture |
| `pool.py`, `training.py` | synthetic training pool, LOSO ExtraTrees |
| `prepare.py`, `references.py` | the offline stages |
| `requests.py`, `evaluation.py` | request manifest, strict answer parsing, MCSE |
| `history_diagnostics.py`, `token_sizes.py` | H oracle decomposition, prompt sizes |
| `execution.py`, `integrity.py` | Sol/DeepSeek transport (disabled), resume guards |

`scripts/` — entry points: `run_offline.sh` (whole offline study), `diagnose_*.py`
(offline diagnostics), `audit_offline.py`, `seal_offline.py`, `cluster_bundle.sh`
and `run_qwen_engine.py` (Qwen production), `integrate_qwen.sh` (after inference),
`budget_sensitivity.py` and `integrate_budget_sensitivity.sh` (budget sensitivity).
`cluster/` — SLURM submission, status and archive jobs. `tests/` — unit tests.

`src/census.py` and `src/dataset_census.py` hold the audited parsers of the raw
sources and the inherited dataset-census grids.

Earlier development states are kept under `archive/` and are not part of the
study. Sol and DeepSeek are prepared only.
