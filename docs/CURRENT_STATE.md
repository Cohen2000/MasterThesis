# Current state

Design identifier panel888-pwt-srw-20260921; arms R, S1, S2, H, B.

- Offline study: sealed, commit 9db974e (docs/results/panel888_offline). Independent
  audit and 109 tests green.
- Qwen main study: complete. One chain (panel888_final_main; jobs 7116955/56/59,
  archive 7117295, from cancelled superseded panel888_main answers reused by strict
  identity where possible). 2,160/2,160 answers, 2,158 valid. See RESULTS_PANEL888.md.
- Qwen budget sensitivity: running (panel888_final_budget; jobs 7117176/77/78,
  archive 7117179). Answers reused by strict identity from the cancelled
  panel888_budget workspace where possible; the rest generated fresh.
- Sol / DeepSeek: prepared only, not started.
