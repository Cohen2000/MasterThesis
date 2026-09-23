# Temporal persistence from sampled interaction data

The v10 study (`panel888-access-v10-20260923`) estimates the five-window persistence profile rho_2..rho_5 from partial temporal-network observations. It compares Qwen thinking and non-thinking, plug-in, a leave-one-source-out training median, arm-specific design or model estimates, and pooled ExtraTrees. The primary outcome is equal-source MAE_2 over eight real sources.

The panel has eight real networks, eight matched timestamp-shuffled P[w,t] surrogates, and eight synthetic instances. Its five arms are R (node panel), S (interaction-following walk with crawl log), S_obs (the same walk without the log), H (partial history), and B (event thinning). All arms calibrate to 10% of full active dyad-windows. S and S_obs use the same draws.

- [Protocol](docs/PROTOCOL_PANEL888_20260921.md)
- [Runbook](docs/RUNBOOK_PANEL888.md)
- [Current state](docs/CURRENT_STATE.md)
- [Generated walk gate](docs/results/panel888_v10_walk_gate_20260923/WALK_GATE.md)
- [Generated main tables](docs/results/panel888_v10_main_20260923/MAIN_RESULTS.md)

`src/main_experiment/` contains graph preparation, the generic weighted-walk kernel, block construction and parsing, estimators, and request generation. `scripts/audit_v10_walk.py`, `scripts/build_v10_pool.py`, `scripts/build_v10_et.py`, and `scripts/build_v10_results.py` are the active offline stages. `cluster/` holds their SLURM jobs and the frozen Qwen production chain. Retired v9 production and diagnostics are in `archive/pre_v10_20260923/`; sealed v9 evidence used for comparisons remains in `docs/results/`.
