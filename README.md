# Persistence estimation in partially observed temporal graphs

This Master's thesis estimates the full-archive persistence profile
`rho_k = |{e: K_e >= k}| / |E_full|`, k=2,3,4,5, from partial observations.
The current design is **`budget10-hrecent5-20260917`**, implemented in
[src/main_experiment](src/main_experiment) and specified in
[the implementation document](docs/MAIN_EXPERIMENT_IMPLEMENTATION.md). It revises
the 16 September 2026 freeze twice: the observation budget became ten percent of
the full event archive (`budget10-20261001`), and arm H became a uniform sample of
active dyads with their five most recent events (`budget10-hrecent5-20260917`).
The H revision was specified after earlier results had been seen; it is a
documented revision, not a retroactive preregistration. Superseded designs and
their artifacts are kept for traceability.

The main experiment has six real sources, eight synthetic instances and four
observation mechanisms:

| Arm | Mechanism |
| --- | --- |
| R | uniform node panel, full histories of the induced dyads |
| S | event-weighted random walk, full histories of traversed dyads |
| H | uniform sample of active dyads, each with its min(5, m_e) most recent events |
| B | independent Bernoulli event sampling, p = 0.10 |

Each arm is calibrated to the same expected observed event volume. Each graph
gets five sampler draws per arm, except that a saturated H sample (every active
dyad drawn) is deterministic and carried once; currently that is
`sp_highschool2013`. This gives 276 main observations and at most 3 312 LLM
requests. The primary metric is MAE₂ with the fixed replacement rule; ProfileMAE
is secondary. References: plug-in, the fold's real training median, one fixed
corrector per arm (walk ratio for S, bound midpoint for H, Beta–ZTP mixture for B)
and a pooled ExtraTrees model with 195 features trained on 16 real sources and
400 synthetic graphs.

```bash
.venv/bin/python scripts/run_main_offline.py --out results/main_experiment/hrecent5_20260917
.venv/bin/python scripts/verify_main_offline.py --run results/main_experiment/hrecent5_20260917
MAIN_RUN=results/main_experiment/hrecent5_20260917 \
  .venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_hrecent5_20260917 --stage all
.venv/bin/python scripts/check_h_recent.py          # 20-draw offline check of arm H
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests/frozen_main -v
```

Qwen3.6-35B-A3B is the only model run so far; GPT-5.6 Sol and DeepSeek remain
disabled. See [the runbook](docs/MAIN_EXPERIMENT_RUNBOOK.md) for cluster
execution, [the H revision report](results/baseline_revision_hrecent5_20260917/REVISION_REPORT.md)
for current results, and
[the previous revision report](results/baseline_revision_20261001/REVISION_REPORT.md)
for the design it replaces. Raw empirical data, model files and bulk artifacts
stay local and are ignored by Git.

Earlier entry points (`census_datasets.py`, `pre_main_freeze_checks.py`) are
historical diagnostics; they do not implement the main experiment.
[archive/](archive/README.md) preserves older designs and experiments.
