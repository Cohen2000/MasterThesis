# Current state

Design identifier panel888-access-v9-20260922; active arms R, S1, H, B (S2 retired).

## Qwen production (final v9)

- Main + all six budget-sensitivity levels (b025/b050/b200/b300/b400/b500): every
  chain's round1/round2/round3/archive completed successfully (jobs
  7118786-7118817, archives 7118789/7118797/7118801/7118805/7118809/7118813/7118817).
  Untouched. Main integrated locally this session (see below).
- H-known ablation (h=.60 spelled out in the prompt): the original round1-3
  (jobs 7119193-7119221) all failed with a `design_version` mismatch between
  the ablation's own `-hknown` label and the deployed contract. Fixed
  (`scripts/build_hknown_requests.py`: keep the real `DESIGN_VERSION`,
  distinguish only the request `id`; also fixed a second real bug this session
  found -- the paired observation's stored `messages` must carry the h=0.60
  wording too, or `run_qwen_engine.load_requests` rejects every request with
  `prompt hash mismatch`) and regenerated + resubmitted for all seven
  workspaces: main 7126667/7126672/7126673/7126674, b025
  7126675/7126676/7126677/7126678, b050 7126679/7126680/7126681/7126682, b200
  7126683/7126684/7126685/7126686, b300 7126687/7126688/7126689/7126690, b400
  7126691/7126692/7126693/7126694, b500 7126695/7126696/7126697/7126698
  (round1/round2/round3/archive). round1 (and, for main, round2) ran against
  the pre-fix data and failed as expected; round3 onward runs against the
  corrected `mainexp/run/` and should succeed -- not confirmed complete at
  session end (queued behind the GPU node backlog; not polled further, per
  instructions). `tests/test_hknown_requests.py` (5 tests) pins both bugs and
  the H-known/H-unknown identity requirement.
- No Sol/DeepSeek jobs exist or were started.

## Final-v9 CPU chain (offline + budget-sensitivity prep + mixed-budget ET)

Sealed successfully this session: **offline 7126631** (after 10 earlier
resubmissions to fix, in order: `record_environment` crashing on git in the
rsynced CPU workspace; a stray root-level `main_experiment/` duplicate on that
workspace shadowing `src/main_experiment` during `-m unittest` test discovery
[moved aside, not deleted: `_stray_root_main_experiment_dup_20260922`]; and,
in `scripts/audit_offline.py`, five separate v9-migration staleness bugs --
hardcoded `(360,400,4320)`/`192` from the pre-S2 design, a stale
`'p888-20260921'` id substring, ExtraTrees' anchor+residual scheme not added
back before comparison, an unconditional `[0,1]`/monotone check applied to
that same unconstrained residual, and `rule_R.txt`/`rule_S1.txt`/`rule_H.txt`
checked against pre-v9 wording that the v9 access contract deliberately
changed [n_panel/L/degree hidden] -- plus two local-only archived files
(`archive/pre_panel888_20260921/.../requests.jsonl` and `.../seed_manifest.json`,
gitignored, needed by the audit's "no ID/seed reuse across generations"
checks) that existed on this laptop but never reached the cluster checkout).
`OFFLINE_FREEZE_SEALED, 1364 artifacts`; 122 tests green on the cluster too.

Then **budget-sensitivity prepare 7126706** (array 1-5) + **7126781** (index 0
retry): found and fixed one more real bug, in `src/main_experiment/training.py`
`fold_rows` -- at low budget-sensitivity coverage (2.5%) a training draw can
have zero observed dyads, and `fit_folds` called `baselines.anchor_profile`
on it unconditionally, which raises (`plugin` refuses an empty sample).
`observation.features()` already had the equivalent zero-fallback; `fold_rows`
now excludes empty draws the same way. `tests/test_training.py` gained
`test_empty_draws_are_excluded_from_training`. Both prepare jobs completed;
chained to **mixed-budget ET 7126790** and **shared-MLE offline evaluation
7126792** (`afterok:7126706:7126790`... i.e. both prepare jobs) -- running at
session end, not polled to completion (see below).

Three earlier dead attempts from before this session's fixes
(7118905/7118941, 7118954/7118957, 7119020/7119026/7119028, plus the
once-thought-fixed 7119233/7119235/7119236 and 7123951/7124537/7124542,
7126124/7126126/7126149, 7126612/7126637/7126643) are permanently stuck
`DependencyNeverSatisfied`/dead in `squeue`; cancelling was blocked by this
session's tool permissions (`scancel`), so they are still listed and harmless
-- the user can `scancel` them in bulk.

## Shared statistical working-model baseline

- `src/main_experiment/shared_mle.py`: shared zero-truncated Beta-Binomial MLE
  across R/S1/H (visible-window likelihood, W=5 target from the same fitted
  alpha/beta -- H at m=5 is bit-identical to R/S1, test-checked) and B (reuses
  `mixtures.fit_events`, same activity/ZTP/thinning model, not duplicated).
  Deterministic homogeneous-binomial fallback (`fit_profile_from_counts`),
  reported and counted, shared between the fit path and the offline adequacy
  diagnostics. `tests/test_shared_mle.py`: 13 tests (input-contract leakage,
  validity, likelihood/synthetic-recovery, fallback wiring).
- `scripts/build_shared_mle.py`: offline runner for Main + every budget level
  (`results/panel888_shared_mle/`), plus three fixed adequacy diagnostics
  (full-data representation, H-like extrapolation, B-like observation) on
  development/training material only -- fully local, no cluster artifact
  needed (real graphs rebuilt from `data/raw`, synthetic dev graphs
  regenerated from their seed).
- `scripts/build_shared_mle_main_quicklook.py`: fast Main-only quicklook
  (`results/panel888_shared_mle_main_quicklook/`) reading only the already
  -prepared main observations + existing plugin/corrector baselines; no Slurm
  wait needed once those two stages exist.
- `scripts/build_qwen_main_quicklook.py`: Qwen main validity/MAE2 by arm x
  config from an already-collected `responses.jsonl`, no re-inference.
- `scripts/build_analysis_snapshot.py`: `results/panel888_analysis_snapshot/`
  -- MAIN_METHOD_COMPARISON.csv, MAIN_REAL_BY_SOURCE.csv,
  SURROGATE/SYNTHETIC_COMPARISON.csv, FIT_DIAGNOSTICS.csv,
  ANALYSIS_SNAPSHOT.md. Main only so far; budget columns and the mixed-budget
  ET column are added once `results/panel888_shared_mle/main_and_budget_observations.csv`
  and the ET job exist (script already checks for and skips them cleanly).
- `docs/PROTOCOL_PANEL888_20260921.md`: new section describing the baseline,
  its assumptions and its limitations (S1 selection left uncorrected, H
  extrapolation is a working-model assumption).
- No new Qwen/Sol/DeepSeek jobs; no ExtraTrees redesign (only the
  empty-draw-exclusion bugfix above, not a design change); no shared-MLE
  hyperparameter tuning; existing plugin/corrector/mixture/ExtraTrees
  references untouched.

## Local quicklook results (Main, 288 observations; Qwen main pulled and
integrated from the cluster archive; MAE2 = conditional mean absolute error
of rho_2)

| arm | plugin | shared_mle | mechanism-aware ref | qwen_thinking | qwen_nonthinking |
|---|---|---|---|---|---|
| R  | 0.019 | 0.026 | 0.019 | 0.017 | 0.314 |
| S1 | 0.056 | 0.068 | 0.056 | 0.049 | 0.325 |
| H  | 0.088 | 0.060 | 0.119 | 0.159 | 0.274 |
| B  | 0.295 | 0.098 | 0.102 | 0.293 | 0.370 |

Qwen validity (formally valid final answers): thinking 1.000, non-thinking
1.000. Shared-MLE adequacy (development/training only): full-data
representation MAE2=0.007 (fallback 5/116); H-like extrapolation MAE2=0.035
(fallback 14/116); B-like observation MAE2=0.112 (fallback 62/500). No model
change was made based on any of this.

Sealed offline reference point (pre-v9, unaffected by this session):
commit 9db974e, results/panel888 archived under docs/results/panel888_offline.
