# Current state

Design panel888-pwt-srw-20260921 (the single final panel).

- Offline study complete and sealed: `bash scripts/run_offline.sh` from scratch,
  independent audit and 104 unit tests green; evidence in `docs/results/panel888_offline/`.
  288 main / 320 training observations; 864 requests per configuration.
- Code cleanup verified scientifically equivalent to the pre-cleanup implementation
  (`archive/pre_panel888_cleanup_20260921/CLEANUP_EQUIVALENCE.json`).
- Qwen production: not yet submitted (see RUNBOOK section 3).
- Sol / DeepSeek: prepared only, not started.
