# Temporal persistence from sampled interaction data

Can a language model estimate how persistent the dyads of a temporal network are
from a small sampled observation of it? The single final study is
**panel888-pwt-srw-20260921**:

- 8 real human–human sources, 8 matched P[w,t] timestamp-shuffled surrogates
  (one per real source) and 8 synthetic instances;
- W = 5 windows, estimand rho_2..rho_5 (primary MAE2, secondary ProfileMAE);
- four observation mechanisms matched to 10% of the active dyad-windows:
  R node panel, S simple random walk, H node panel with elapsed-time history
  (h = .60), B Bernoulli event thinning;
- 3 test sampler draws x 3 model repeats (288 observations, 864 requests per
  model configuration), 5 training draws for the learned references.

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
| `sampling.py`, `walk_kernel.cpp` | the four arms, budget calibration, simple random walk |
| `observation.py` | observation blocks, prompts, 134 features |
| `baselines.py`, `mixtures.py` | plug-in, arm-specific correctors, B Beta-mixture |
| `pool.py`, `training.py` | synthetic training pool, LOSO ExtraTrees |
| `prepare.py`, `references.py` | the offline stages |
| `requests.py`, `evaluation.py` | request manifest, strict answer parsing, MCSE |
| `history_diagnostics.py`, `token_sizes.py` | H oracle decomposition, prompt sizes |
| `execution.py`, `integrity.py` | Sol/DeepSeek transport (disabled), resume guards |

`scripts/` — entry points: `run_offline.sh` (whole offline study), `diagnose_*.py`
(offline diagnostics), `audit_offline.py`, `seal_offline.py`, `cluster_bundle.sh`
and `run_qwen_engine.py` (Qwen production), `integrate_qwen.sh` (after inference).
`cluster/` — SLURM submission, status and archive jobs. `tests/` — unit tests.

`src/census.py` and `src/dataset_census.py` hold the audited parsers of the raw
sources and the inherited dataset-census grids.

Earlier designs, runs and answers are development provenance under `archive/`;
they are not part of the final panel. Sol and DeepSeek are prepared only.
