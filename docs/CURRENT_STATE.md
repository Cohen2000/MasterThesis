# Current state

Design identifier panel888-access-v9-20260922; active arms R, S1, H, B (S2 retired).

## Qwen production (final v9)

- Main + all six budget-sensitivity levels (b025/b050/b200/b300/b400/b500): every
  chain's round1/round2/round3/archive completed successfully (jobs
  7118786-7118817, archives 7118789/7118797/7118801/7118805/7118809/7118813/7118817).
  Left untouched, per instructions.
- H-known ablation (h=.60 spelled out in the prompt), all seven workspaces
  (main + six budgets): round1/2/3 (jobs 7119193-7119221) all FAILED with
  `ValueError: request protocol mismatch`. Cause: the ablation's requests were
  built with `design_version = panel888-access-v9-hknown-20260922`, but the
  deployed `src/main_experiment/common.py` in those workspaces defines
  `DESIGN_VERSION = panel888-access-v9-20260922` (no `-hknown` suffix), so
  every request fails `validate_request`. The archive jobs (7119196 etc.)
  still report COMPLETED, but archived zero answers -- the whole ablation
  currently has 0/expected valid answers. NOT fixed or rerun this session
  (instructions explicitly rule out new H-known runs without confirmation);
  flagged to the user. Fix would be either regenerating the ablation's requests
  under the deployed DESIGN_VERSION, or updating that workspace's `common.py`
  to accept the `-hknown` suffix, then rerunning round1-3 + archive for all
  seven workspaces.
- No Sol/DeepSeek jobs exist or were started (unchanged).

## Final-v9 CPU chain (offline + budget-sensitivity prep + mixed-budget ET)

- Original chain (7119233 offline / 7119235 budget array / 7119236 ET) FAILED:
  7119233 crashed in `record_environment` because `panel888_access_v9_cpu` on
  the cluster is an rsynced copy with no `.git`, so `git rev-parse HEAD`
  raised instead of returning a commit. Fixed in
  `src/main_experiment/prepare.py` (git calls now degrade to `None` instead of
  raising) and synced to the cluster. Three earlier same-cause failed attempts
  (7118905/7118941, 7118954/7118957, 7119020/7119026/7119028) are permanently
  stuck `DependencyNeverSatisfied`; cancelling them was blocked by this
  session's tool permissions (`scancel`), so they are still visible in
  `squeue` and harmless -- the user can `scancel` them.
- Resubmitted chain, same commands/resources as the original:
  offline **7123951** -> budget-sensitivity prepare **7124537** (array 0-5)
  -> mixed-budget ET **7124542**. Running at session end (offline stage was
  past `pool`/`train`, into `references`, ~28 min in).
- Also fixed while resubmitting, because `run_offline.sh`'s own `tests` step
  (`set -euo pipefail`) would otherwise abort the chain before sealing: 8
  pre-existing test failures from the S2-retirement / features-v9 migration
  (509a0b2/fa6fecf/3e3ac27) that never updated the test suite --
  `config/study.yaml` had a YAML indentation bug and a stale
  `features-v8`/`192` pair; `tests/test_baselines.py`,
  `tests/test_observation.py`, `tests/test_sampling.py`, `tests/test_data.py`
  asserted the old S2 arm, the old 192/85/107 feature scheme, a leaked
  `n_panel_history` in H's block, and the old design-version-based IDs/sizes.
  All 122 tests pass locally now; synced to the cluster workspace.

## Shared statistical working-model baseline (this session)

- `src/main_experiment/shared_mle.py`: shared zero-truncated Beta-Binomial MLE
  across R/S1/H (visible-window likelihood, W=5 target from the same fitted
  alpha/beta) and B (reuses `mixtures.fit_events`, same activity/ZTP/thinning
  model, not duplicated). Deterministic homogeneous-binomial fallback,
  reported and counted. `tests/test_shared_mle.py`: 13 tests (input-contract
  leakage, validity, likelihood/synthetic-recovery, fallback wiring) -- all
  pass.
- `scripts/build_shared_mle.py`: offline runner -- shared_mle on every main +
  budget-level observation (results/panel888_shared_mle/), plus the three
  fixed adequacy diagnostics (full-data representation, H-like extrapolation,
  B-like observation) on development/training material only.
- `docs/PROTOCOL_PANEL888_20260921.md`: new section describing the baseline
  and its assumptions/limitations.
- Cluster run: synced to `panel888_access_v9_cpu`; not yet submitted at
  session end (waits on the offline+budget chain above: needs
  `afterok:7123951:7124537`). Command to submit once that chain is healthy:
  `sbatch --job-name=access_v9_shared_mle --partition=cpu --time=06:00:00
  --cpus-per-task=4 --mem=32G --dependency=afterok:7123951:7124537
  --output=.../logs/shared_mle_%j.out --wrap="module load
  devel/python/3.12.3-gnu-14.2; source $WS/venv_offline/bin/activate; cd
  $WS/panel888_access_v9_cpu; export PYTHONPATH=.../src;
  python scripts/build_shared_mle.py"`.
- No new Qwen/Sol/DeepSeek jobs; no ExtraTrees redesign; existing
  plugin/corrector/mixture/ExtraTrees references untouched.

Sealed offline reference point (pre-v9, unaffected by this session):
commit 9db974e, results/panel888 archived under docs/results/panel888_offline.
