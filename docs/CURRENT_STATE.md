# Current state

Design identifier panel888-pwt-srw-20260921; arms R, S1, S2, H, B.

- Offline study: sealed, commit 9db974e (docs/results/panel888_offline). Independent
  audit and 109 tests green.
- Qwen main study: complete. Chain `panel888_final_main` (jobs 7116955/56/59,
  archive 7117295); 2,160/2,160 answers, 2,158 valid, 0 technical errors. See
  RESULTS_PANEL888.md.
- Qwen budget sensitivity: complete. Chain `panel888_final_budget` (jobs
  7117176/77/78, archive 7117179); 12,828/12,828 answers, 1 technical failure
  (job-deadline in-flight, not regenerated). See BUDGET_SENSITIVITY.md.
- Sol / DeepSeek: prepared only, not started.
- No cluster jobs of this project running. Worktree clean.
