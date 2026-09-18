# Persistence estimation in partially observed temporal graphs

This repository contains the current Master's thesis experiment for estimating the
full-archive persistence profile rho_k for k = 2,3,4,5 from partially observed
temporal graphs.

## Current experiment

Design: `cells10-json-20260918`

- 6 real temporal-network sources
- 8 synthetic main-test instances
- observation mechanisms R, S, H and B
- 5 sampler draws per arm where applicable
- 3 LLM responses per configuration
- primary target: `rho_2`
- secondary profile: `rho_2` to `rho_5`
- matched observation budget: 10% of full active dyad-windows

The executable implementation is in `src/main_experiment/`.

## Main entry points

Offline preparation: `bash scripts/run_json_revision_offline.sh`

Qwen production:
- `scripts/cluster_bundle.sh`
- `cluster/qwen_engine.sbatch`
- `cluster/submit_production.sh`
- `cluster/status.py`
- `cluster/build_archive.py`

Evaluation:
- `scripts/collect_qwen_answers.py`
- `scripts/evaluate_main_responses.py`

Current documentation:
- `docs/PROTOCOL_REVISION_20260918.md`
- `docs/RUNBOOK_JSON_REVISION_20260918.md`
- `docs/MAIN_EXPERIMENT_IMPLEMENTATION.md`

Historical designs, diagnostics and results are retained under `archive/`.
Raw datasets, generated graphs, fitted models and other bulk artifacts remain local
and are ignored by Git.
