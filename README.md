# Persistence estimation in partially observed temporal graphs

This Master's thesis estimates the full-archive persistence profile
`rho_k = |{e: K_e >= k}| / |E_full|`, k=2,3,4,5, from partial observations.
The authoritative design is the **16 September 2026 freeze**, implemented in
[src/main_experiment](src/main_experiment). Earlier experimental code and
historical documentation are retained for traceability.

The main experiment has six real sources, eight synthetic instances, and four
observation mechanisms: fixed uniform node panels, event-weighted walks with
full histories, a fixed final-three-window suffix, and independent Bernoulli
event sampling. The budget matches **expected observed event volume**. There
are 224 observations and at most 2,688 future LLM requests. The primary metric
is MAE₂ with the fixed replacement rule; ProfileMAE is secondary.

The offline pipeline prepares data and provenance, calibrates and independently
validates walks, generates the eight synthetic instances, trains seven
source-separated ExtraTrees models with 88 features, evaluates the four offline
baselines, constructs exact production prompts, checks token sizes, and writes
request manifests. **It contains no inference transport and submits no jobs.**

```bash
.venv/bin/python scripts/run_main_offline.py --out results/main_experiment/frozen_20260916
# Repeat exactly the same command to resume; changed dependencies are rejected.
.venv/bin/python scripts/verify_main_offline.py --run results/main_experiment/frozen_20260916
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

See [the runbook](docs/MAIN_EXPERIMENT_RUNBOOK.md),
[specification/code/evidence mapping and generator algorithms](docs/MAIN_EXPERIMENT_IMPLEMENTATION.md),
[extracted freeze](docs/MAIN_FREEZE_SOURCE.txt), and
[offline acceptance report](results/main_experiment/ACCEPTANCE.md).
Raw empirical data and downloaded tokenizers remain local and ignored by Git.
The exact environment lock, model files and full manifests are in the run directory.

Prior entry points (`census_datasets.py`, `pre_main_freeze_checks.py`) are
historical diagnostics; they do not implement this frozen main experiment.
[archive/](archive/README.md) preserves older designs and experiments.
