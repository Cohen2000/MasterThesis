# Persistence estimation in partially observed temporal graphs

This Master's thesis estimates the full-archive persistence profile
`rho_k = |{e: K_e >= k}| / |E_full|`, k=2,3,4,5, from partial observations.
The current design is **`cells10-20260917`**, implemented in
[src/main_experiment](src/main_experiment) and specified in the last section of
[the implementation document](docs/MAIN_EXPERIMENT_IMPLEMENTATION.md). It revises
the 16 September 2026 freeze three times: the budget became a share of the full
archive (`budget10-20261001`), arm H became a uniform sample of active dyads with
their five most recent events (`budget10-hrecent5-20260917`), and the arms are now
matched on the quantity the target is made of, the expected number of observed
active dyad-windows (ten percent of all of them), while Qwen's final answers are
constrained to the JSON format (`cells10-20260917`). Each revision was specified
after earlier results had been seen; they are documented revisions, not
retroactive preregistrations. Superseded designs and their artifacts are kept.

The main experiment has six real sources, eight synthetic instances and four
observation mechanisms:

| Arm | Mechanism |
| --- | --- |
| R | uniform node panel, full histories of the induced dyads |
| S | event-weighted random walk, full histories of traversed dyads |
| H | uniform sample of active dyads, each with its min(5, m_e) most recent events |
| B | independent Bernoulli event sampling |

Each graph gets five sampler draws per arm (a saturated, deterministic H sample
would be carried once), three answers per configuration, and at most four LLM
configurations. The primary metric is MAE₂ with the fixed replacement rule;
ProfileMAE is secondary. References: plug-in, the fold's real training median,
one fixed corrector per arm (walk ratio for S, bound midpoint for H, Beta–ZTP
mixture for B) and a pooled ExtraTrees model with 195 features trained on 16 real
sources and 400 synthetic graphs.

```bash
bash scripts/run_cells10_offline.sh      # resumable offline chain, see docs/HANDOFF_cells10.md
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

Execution state, cluster steps and how to continue are in
[docs/HANDOFF_cells10.md](docs/HANDOFF_cells10.md). Qwen3.6-35B-A3B is the only
model run so far; GPT-5.6 Sol and DeepSeek remain disabled. Results of the
previous designs: [H revision](results/baseline_revision_hrecent5_20260917/REVISION_REPORT.md),
[ten-percent event budget](results/baseline_revision_20261001/REVISION_REPORT.md).
Raw empirical data, model files and bulk artifacts stay local and are ignored by Git.

Earlier entry points (`census_datasets.py`, `pre_main_freeze_checks.py`) are
historical diagnostics; they do not implement the main experiment.
[archive/](archive/README.md) preserves older designs and experiments.
