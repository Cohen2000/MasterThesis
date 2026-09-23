# Temporal persistence from sampled interaction data

The final `panel888-access-v11-20260923` study estimates five-window temporal-network persistence, with `rho_2` primary. Eight real sources, eight matched P[w,t] surrogates and eight synthetic instances are evaluated separately. R and H release sampled panel size; S releases its crawl log; B releases thinning probability. The Qwen-only `S_obs` arm tests the value of the crawl log.

The [v11 main results](docs/results/panel888_v11_main_20260923/MAIN_RESULTS.md) compose completed released-panel R/H Qwen and ExtraTrees results with unchanged sealed v10 S/S_obs/B results. The [hidden-panel-size comparison](docs/results/panel888_v10_RH_panel_release/REPORT.md) is sensitivity evidence. The [walk gate](docs/results/panel888_v10_walk_gate_20260923/WALK_GATE.md) and original [v10 evidence](docs/results/panel888_v10_main_20260923/MAIN_RESULTS.md) remain sealed.

See [current state](docs/CURRENT_STATE.md), [protocol](docs/PROTOCOL_PANEL888_20260921.md) and [runbook](docs/RUNBOOK_PANEL888.md). The laptop API comparison uses the 288 locally frozen v11 R/S/H/B observations through the single `scripts/api_runner.py`; every provider request requires `--execute` and an explicit budget.
