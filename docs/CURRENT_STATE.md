# Current state

Design identifier panel888-pwt-srw-20260921; S is the degree-biased random walk.

- Offline study sealed (`docs/results/panel888_offline/`, sources of commit b5702e3):
  full `run_offline.sh` on uc3 (job 7098486), audit and tests green; laptop reproduces
  all R/H/B parameters and observations exactly. Budget sensitivity prepared
  (job 7098487), feasibility and audit in `docs/results/panel888_budget_sensitivity/`.
- Qwen: R and H answers are reused by byte-identical request; S and B answers are
  being generated (workspaces `panel888_main`, `panel888_budget`).
- Sol / DeepSeek: prepared only, not started.
