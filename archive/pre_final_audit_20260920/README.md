# Persistence estimation in partially observed temporal graphs

This repository contains the current Master's thesis experiment for estimating the
full-archive persistence profile rho_k for k = 2,3,4,5 from partially observed
temporal graphs.

## Current experiment

Design: `cells10-srw-htime60-20260920`

S is a simple random walk: uniform start vertex and uniform current neighbor,
with a fixed calibrated length and full histories for traversed dyads. Its raw
traversal-frequency reference is stationary/asymptotic, not finite-walk unbiased.
H uses uniform node sampling and the most recent 60% of archive time, calibrated
to the same 10% active-dyad-window coverage as R/S/B. Offline sensitivity uses
h=0.40/0.60/0.80. Its primary reference is a homogeneous zero-truncated Binomial
extrapolator, treated as a working model. See [protocol](docs/PROTOCOL_SRW_20260920.md).
New Qwen S generation is authorized; H/R/B are reused only after identity checks.
GPT/Sol and DeepSeek remain unstarted.

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

Offline preparation: `bash scripts/run_srw_offline.sh`

Qwen production:
- `scripts/cluster_bundle.sh`
- `cluster/qwen_engine.sbatch`
- `cluster/submit_production.sh`
- `cluster/status.py`
- `cluster/build_archive.py`

Evaluation:

- `scripts/integrate_srw_qwen.py`
- `scripts/collect_qwen_answers.py`
- `scripts/evaluate_main_responses.py`
- `scripts/audit_srw_results.py`

Current documentation:

- `docs/PROTOCOL_SRW_20260920.md`
- `docs/RUNBOOK_SRW_20260920.md`
- `docs/RESULTS_SRW_20260920.md`

Historical designs, diagnostics and results are retained under `archive/`.
Raw datasets, generated graphs, fitted models and other bulk artifacts remain local
and are ignored by Git.
