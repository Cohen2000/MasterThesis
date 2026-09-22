# Current state

Design identifier panel888-access-v9-20260922; active arms R, S1, H, B (S2 retired).

## Final v9 results: complete for all five main methods

`docs/results/panel888_analysis_snapshot/` (ANALYSIS_SNAPSHOT.md plus
MAIN_METHOD_COMPARISON, MAIN_REAL_BY_SOURCE, BUDGET_METHOD_COMPARISON,
SURROGATE_COMPARISON, SYNTHETIC_COMPARISON, MAIN_SECONDARY_REFERENCE,
FIT_DIAGNOSTICS) covers Main (0.10) and the full coverage grid
(0.025/0.05/0.10/0.20/0.30/0.40/0.50) x R/S1/H/B x real/surrogate/synthetic for
plugin, shared_mle, mixed-budget extratrees, qwen_thinking, qwen_nonthinking.
Regenerate with `scripts/build_analysis_snapshot.py` (reads the gitignored
`results/` tree; the committed copies live in `docs/results/`).

- plugin: computed from each level's observations (no model).
- shared_mle: `scripts/build_shared_mle.py`, run locally on the sealed
  observations; `docs/results/panel888_shared_mle/` (1994 observations,
  fallback 5.6-9.9 % per level; adequacy on dev/training only).
- extratrees (mixed-budget): `scripts/build_mixed_budget_et.py`, job 7128339
  (48 cores, ~25 min): 36 arm x fold forests (PARAMETERS unchanged, 58,211
  training rows pooled over all seven levels, budget not a feature), scored in
  the same run on all 1994 test observations; `docs/results/panel888_et_mixed/`
  (evaluation.csv, report.json). The 20 GB of forests and their manifests stay
  on the cluster: `$WS/panel888_access_v9_cpu/results/panel888_et_mixed/models/`.
- Qwen thinking/non-thinking: final v9 archives of `panel888_access_v9_{main,
  b025,...,b500}` collected with `collect_qwen_answers.py` (all expected answers
  found, 0 missing), scored on formally valid answers only (validity >= 0.9976).

Changes to the ET script, none methodological: features/anchors computed once
per row instead of twice per fold (arm B alone would otherwise have needed
~300k Beta-mixture fits, ~3.5 h); fits and scoring parallel; empty draws
(D_obs=0, only at 2.5 % coverage) excluded, as in `training.fold_rows`; and
the 0.10 level's training pool is now included -- the old script looked for
it under `prepared/pool`, which does not exist (it is `references/pool`, train
partition), so the main level's pool was silently missing.

## Cluster state

- Offline study sealed (job 7126631, `OFFLINE_FREEZE_SEALED`, 1364 artifacts);
  budget-sensitivity prepare 7126706 (1-5) + 7126781 (index 0) completed.
- H-known ablation (fixed request builder, resubmitted this session): b050,
  b200, b300, b500 archived with all answers; b025 and b400 have all answers,
  archive jobs pending/running; main 421/432 answers, round 3 running. Not
  integrated yet.
- Dead jobs still listed in `squeue` (DependencyNeverSatisfied, harmless; the
  user can `scancel` them): 7118941, 7118957, 7119026, 7119028, 7119235,
  7119236, 7124537, 7124542, 7126149, 7126643, 7126713, 7126790, 7126792.
- No Sol/DeepSeek jobs.

## Open

- Integrate H-known once its seven archives are complete.
- Mechanism-aware secondary reference exists for Main only (budget levels
  would need `main_references` output per level; not needed for the five-method
  comparison).

Sealed pre-v9 reference point: commit 9db974e, docs/results/panel888_offline.
