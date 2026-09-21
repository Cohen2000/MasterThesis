# Current state

Design panel888-pwt-srw-20260921 (the single final panel).

- Offline freeze: commit 54d5832ce450f8bad9ac557070df5324fcd7c8b1 (pushed). Full clean
  `scripts/run_offline.sh` run, independent audit and 104 tests green; evidence in
  `docs/results/panel888_offline/`. Code cleanup verified equivalent to the pre-cleanup
  implementation (`archive/pre_panel888_cleanup_20260921/CLEANUP_EQUIVALENCE.json`).
- Qwen main production: complete (one chain, jobs 7093897–7093900, never resubmitted).
  1,728/1,728 answers, 1,727 valid; integrated and audited, see RESULTS_PANEL888.md.
- Budget sensitivity (ancillary, grid .025–.50, 10% = main study): complete. One Qwen chain
  (jobs 7094845–7094848, commit 2ac8373), 10,236/10,236 answers collected; results and
  plots in docs/results/panel888_budget_sensitivity, see BUDGET_SENSITIVITY.md.
- Sol / DeepSeek: prepared only (864 requests each), not started.
