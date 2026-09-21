# Current state

Design identifier panel888-pwt-srw-20260921; S is the degree-biased random walk.

- Offline study sealed (`docs/results/panel888_offline/`, sources of commit b5702e3):
  full `run_offline.sh` on uc3 (job 7098486), audit and tests green; laptop reproduces
  all R/H/B parameters and observations exactly. Budget sensitivity prepared
  (job 7098487), feasibility and audit in `docs/results/panel888_budget_sensitivity/`.
- Qwen: R and H answers are reused by byte-identical request (listed in
  answers/REUSED_ANSWERS.json); S and B are generated from commit 4c82d48 in one chain
  each, never to be resubmitted: `panel888_main` 7103120/7103121/7103122, archive 7103123;
  `panel888_budget` (36 shards) 7103125/7103126/7103127, archive 7103128.
- Sol / DeepSeek: prepared only, not started.
