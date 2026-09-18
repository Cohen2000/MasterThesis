# Baseline revision 1 — development report

Revision id: `baseline-revision-1-20260916` · Date: 2026-09-16 · Branch: `experiment/offline-freeze-20260916`
Machine-readable results: `development_summary.csv` / `development_summary.json`;
frozen pool: `pool_definition.json`; freeze proof: `freeze_unchanged.json`.

**Reason for the revision.** The learned baseline was previously fitted on sixteen
real sources only. On the frozen main run it behaved like a model applied outside
its training range — on the high-persistence synthetic strata it was worse than
simply predicting the frozen training median. This revision gives it a diverse
synthetic training pool and adds the derived features that were computable from
the observation input all along. It also checks two Beta-mixture extensions of
the statistical correctors on separate development data.

**No LLM inference, no API call and no paid job was run at any point.** The frozen
run `results/main_experiment/frozen_20260916` was never written to.

---

## 1. What is fixed and what changed

Unchanged: test panel, the four samplers, the budget definition, the estimand,
the LLM prompts, the four LLM configurations, the repeat counts, the ExtraTrees
hyperparameters, the single shared multi-output regressor, the real
leave-one-source-out split, and the fold-specific real training median including
the empty-sample replacement.

Changed: the synthetic training pool (new), the training weights, the feature
block (88 → 133), and two additional development-only correctors.

**Proof that the freeze is untouched** (`freeze_unchanged.json`): re-serialising
and re-hashing all 480 frozen observations with the revision code reproduces every
`block_sha256`; rebuilding all 224 prompts reproduces every `prompt_sha256`
byte-for-byte; the request manifest still holds 2 688 rows with 0 started calls;
and the eight synthetic main-test instances regenerate with identical
`N/D/M/B`, ground truth, shared-latent hashes and state hashes
(`tests/frozen_main/test_pool_and_features.py::FrozenGeneratorTests`).
Parameterising the generators required carrying the activity-driven tail as
`tail = gamma - 1` with the default exactly `1.8`, because `2.8 - 1.0` is not the
float `1.8` and would have perturbed every later draw.

## 2. The pool

500 graphs, generated with the current samplers and the current budget definition,
in 442 s on one local core with no failures. No cluster was used and no compute ran
on a login node; the measured worst-case corner is 14 s per graph, so a scheduler
round trip would have bought nothing.

| | DAR | Activity-driven | total |
| --- | --- | --- | --- |
| training | 200 | 200 | 400 |
| development | 50 | 50 | 100 |

Stratified NumPy draw with a fixed stream: DAR over chi × alpha (5 × 5 cells,
8 training + 2 development each), activity-driven over mode × eta (2 × 5 cells,
20 + 5 each). Sizes, and for activity-driven the round count, are cycled through
their grid inside every cell so no cell depends on the draw for coverage. The
admissible ranges are derived in `src/main_experiment/pool.py` from generator
semantics, together with the reason each constant is held fixed — including that
`m` stays at one contact per activation, because extending the memory rule to
m > 1 needs an arbitrary choice about within-round memory updates that the source
model does not fix.

Coverage: true rho_2 spans **0.008 to 0.915** (median 0.624), N from 114 to 1 200,
D from 79 to 193 491, M from 375 to 244 105. The main-test instances sit at
rho_2 in [0.043, 0.789], so the pool brackets them on both sides. No parameter was
chosen by looking at a resulting rho_2.

Independence: each pool graph is drawn on its own stream keyed by its id under the
domain `pool`, disjoint from the main-test domain `graph`; pool graphs are
generated one at a time, so no latent quantity is shared between any two graphs and
the "paired variants stay in one partition" rule holds trivially; training and
development ids and seeds are disjoint. A held-out real source is excluded with all
its rows. These are asserted, not asserted-in-prose:
`tests/frozen_main/test_pool_and_features.py::PoolDefinitionTests`.

**12 of 500 graphs (3 of 100 development graphs) are not budget-matched** — all DAR
with very thin activity, where the event-weighted walk hits the calibration cap.
The rule was fixed before any result was computed: they stay in, because removing
graphs on a quantity correlated with the mechanism would be a selection, and the
development table is additionally reported restricted to matched graphs. The
restriction moves nothing (§5).

## 3. Training weights and features

Weights are exactly 50 % real / 25 % DAR / 25 % activity-driven, and inside a block
every graph carries the same weight, inside a graph every arm, inside an arm every
observation. The synthetic rows outnumber the real ones roughly 25:1, so without
this the real block would be outvoted by volume alone; the tests assert both facts.
A real-only fold reproduces the original `1/(15·4·n_a)` weights exactly.

The 45 new features are the 31 pattern dyad shares, the 5 window event shares,
`M_obs/D_obs`, the 4 plug-in profile values and the 4 values of the existing
homogeneous corrector — the last used as a fixed transform regardless of the
candidate outcome. All 45 are computed from the serialized observation input only.
No full-graph size, ground truth, realised coverage, source name, generator
parameter or generator family enters as a feature. An empty sample codes every
derived entry as a plain zero instead of dividing by zero, and missing windows stay
distinguishable through `access_1..access_5`.

The median baseline and the empty-sample replacement are computed from the real
fold sources only. The pooled and the real-only model report the identical median
`[0.3545, 0.1416, 0.0515, 0.0067]`, which is also the frozen run's value.

**This is a training advantage, stated openly.** The learned reference is fitted on
500 graphs from the same two families the synthetic development graphs come from,
with features the prompt does not contain. It is a task-specific reference, not a
demonstrated performance ceiling.

## 4. The two candidates

Both are Beta mixtures over a per-dyad activity q_e ~ Beta(a,b) with conditionally
independent windows, targeting the full five-window profile conditioned on at least
one truly active window — not the observed profile. The likelihoods, the sufficiency
arguments and the identifiability limits are derived in the module docstring of
`src/main_experiment/mixtures.py`. The model family follows Dorazio & Royle (2003);
the coupling to our samplers and to the persistence target is our own derivation and
carries no literature guarantee about MAE.

**Suffix (arm H).** J | J>0 is a zero-truncated Beta-Binomial on the three observed
windows, with no thinning. The pattern table alone is sufficient: the event layer is
independent of q given activity, so the event counts carry no information about
(a,b) and are correctly unused. Identifiability is the honest weak point — two
parameters against two free cell probabilities is exactly saturated, so there are
zero degrees of freedom, no goodness-of-fit is possible, and the extrapolation from
three windows to a five-window profile cannot be checked from the data.

**Bernoulli events (arm B).** For a truly active window with N ~ ZTP(lambda) and
independent retention p, summing the thinned ZTP gives
Pr{K=k} = Poisson(p·lambda; k)/(1 − e^−lambda), hence
d(lambda,p) = (1 − e^−p·lambda)/(1 − e^−lambda) and, conditioned on K ≥ 1, the
retained counts are exactly ZTP(p·lambda). Its sufficient statistic is the sample
mean, so **M_obs/S is sufficient for lambda given S** and the table's event sums
suffice — per-dyad counts are never needed. Because d depends on lambda, the window
part and the event part are coupled, which is exactly why lambda may not be fitted
separately and substituted; (a,b,lambda) are optimised jointly and the event-only
lambda is reported alongside in `development_observations.csv`.

Numerics. Parameters are carried as (logit mu, log kappa, log lambda) with bounds
fixed before any performance check: |logit mu| ≤ 12, kappa in [1e−3, 1e6],
lambda in [1e−6, 1e3]; kappa = 1e6 is the homogeneous boundary and is reported as
such. Three deterministic starts, L-BFGS-B, no MCMC and no new heavy dependency.
Statuses distinguish `converged`, `boundary_homogeneous`, `boundary_other`,
`starts_disagree`, `weakly_identified` and `not_converged`; weak identifiability is
measured, not guessed, by re-optimising every other coordinate with log kappa
displaced by a decade and reporting the smaller profile rise.

One real defect was found and fixed by the exactness test rather than by inspection:
the first implementation expanded the cell probabilities in the plain moments E[q^r],
an alternating sum that loses every significant digit when q·d concentrates near one
(at a = 1e5, b = 1e−2, d = 1 it returned 6.7e−16 where the true value is 2.5e−26).
Substituting 1 − qd = (1 − q) + q(1 − d) gives an all-positive form; the worst
relative error against exact rational arithmetic over a wide grid is now 8.3e−16.
No quadrature, no regularisation and no clipping is used anywhere.

## 5. Development results

100 held-out development graphs, 1 600 observations. Errors are averaged per graph
first, so every graph counts once; the standard error is across graphs.
Full table in `development_summary.csv`.

MAE2 by arm, all development graphs:

| arm | plug-in | existing corrector | candidate | ExtraTrees pooled | ExtraTrees real-only | median |
| --- | --- | --- | --- | --- | --- | --- |
| R | 0.00541 | = plug-in | — | 0.00937 | 0.14891 | 0.30550 |
| S | 0.10310 | **0.01095** | — | 0.01165 | 0.14905 | 0.30550 |
| H | 0.09929 | 0.12015 | **0.02592** | 0.01521 | 0.15742 | 0.30550 |
| B | 0.06586 | 0.08069 | **0.01023** | 0.01029 | 0.15523 | 0.30550 |

Paired differences in MAE2 against the two references, per generator family
(negative is better; standard error across graphs):

| family | arm | candidate − plug-in | candidate − corrector |
| --- | --- | --- | --- |
| DAR | H | −0.0822 ± 0.0099 | −0.1407 |
| DAR | B | −0.0473 ± 0.0045 | −0.0914 |
| AD | H | −0.0645 ± 0.0091 | −0.0478 |
| AD | B | −0.0639 ± 0.0053 | −0.0495 |

Both candidates beat both references on both families, by seven to twelve standard
errors. Note the references disagree with each other by family: the existing
homogeneous corrector is *worse* than the raw plug-in on DAR (H +0.058, B +0.044)
and better on activity-driven (H −0.017, B −0.014), which is what the mechanisms
predict — DAR has homogeneous per-edge activity with serial copy dependence, while
activity-driven has genuinely heterogeneous dyad activity. The Beta mixture helps on
DAR as well, which it does by absorbing over-dispersion in K whatever its source,
not because DAR is a Beta mixture.

Stability, runtime and boundaries:

| arm | fits | converged | boundary / disagree / weak | mean s | max s |
| --- | --- | --- | --- | --- | --- |
| H | 100 | 85 | 15 | 0.024 | 0.105 |
| B | 500 | 453 | 47 | 0.052 | 0.283 |

There were no numerical failures: `not_converged` never occurred. The non-converged
statuses are **not** a degradation — those fits are more accurate, not less
(H 0.0174 vs 0.0274; B 0.0038 vs 0.0109), because they occur where the data are close
to homogeneous and the boundary solution is the correct answer. The pre-registered
restriction to budget-matched development graphs moves every number by at most
0.0008 (H candidate 0.02592 → 0.02554, B candidate 0.01023 → 0.01023).

ExtraTrees: the pooled fit is an order of magnitude better than the real-only fit on
every arm and both families (e.g. arm B, DAR 0.0126 vs 0.1565; AD 0.0080 vs 0.1540).
**This is within-family generalisation and nothing more**: the development graphs come
from the same two families and the same parameter pool design as the 400 training
graphs, which is in-distribution for the pooled model and far out-of-distribution for
the real-only one. It does not show that the pooled model is better on real sources,
and the six real main-test sources were deliberately not evaluated.

## 6. Adoption recommendation

**Arm B — adopt the Beta-mixture candidate as the primary corrector.** It is
over-identified (four free cell probabilities plus one event statistic against three
parameters), so the model is testable rather than merely fitted; it recovers the
existing hurdle-Poisson corrector in the homogeneous limit; it improves on both
references on both families by more than ten standard errors; it costs 0.05 s per
observation; and its boundary cases are benign. The one required follow-up is a
single pre-registered evaluation on the six real main-test sources, run once and not
iterated on.

**Arm H — do not adopt as primary; keep it as a reported secondary estimate.** Its
development performance is good, but the fit is exactly saturated, so nothing in the
data can detect a failed three-to-five-window extrapolation, and the main experiment
draws only one suffix sample per graph, so there is no replicate that would expose a
silent failure either. Good numbers without any available diagnostic are not enough
to make it primary.

**ExtraTrees — adopt the pooled training as the learned reference**, labelled as a
task-specific reference trained with a feature and data advantage over the prompt,
not as a performance ceiling. Its real-source behaviour remains open by design.

No further candidate search was run and none is proposed.

## 7. Commands

```bash
.venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_20260916 --stage pool
.venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_20260916 --stage train
.venv/bin/python scripts/run_baseline_revision.py --out results/baseline_revision_20260916 --stage dev
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Each stage resumes: the pool skips finished graphs, training reuses hash-verified
model manifests, and the development stage reuses the per-observation record file.
Bulk artifacts (pool graphs, calibration checkpoints, fitted models, the
per-observation record table) stay local and out of git.
