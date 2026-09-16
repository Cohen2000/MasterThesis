# Design revision `budget10-20261001` — report

Branch `experiment/offline-freeze-20260916`. Superseded designs are kept:
`frozen_20260916` (suffix budget, 88 features) and `baseline_revision_20260916`
(pooled training at that budget) remain on disk as the development history.

## 1. What changed, and why

**The budget.** Previously every arm was calibrated to an expected observed volume
equal to the suffix event count — about 56 % of the archive. At that level three of
the four arms saw more than half of all active dyad-windows, so the task was closer
to counting than to inference. The budget is now a fixed **10 % of the full event
archive**. W, the target population and the persistence definitions are unchanged.

**Arm H.** It changes from "every event in windows 3–5" to "a uniform node panel,
then every event in windows 3–5 inside that panel". This is what makes the lower
budget possible at all: the old H had no free parameter — it *was* the budget — so
the only way to lower it was to shorten the suffix, and at two observed windows the
zero-truncated model has one free cell probability for two parameters and stops
being identifiable. The panel size follows the same rule as R, from
E[M_obs] = n(n−1)/(N(N−1)) · M_suffix, so H needs the larger panel by the factor
M_full/M_suffix.

*Why the panel does not disturb the model.* The panel is drawn over nodes without
reference to any event, so a dyad is included with probability n(n−1)/(N(N−1))
independently of its activity. Conditional on inclusion its window pattern is
exactly what it was, so J | q ~ Bin(3, q) truncated at J ≥ 1 continues to hold and
both the homogeneous suffix corrector and the Beta-Binomial candidate remain valid
unchanged. Checked empirically: over 30 draws the observed window-count
distribution moves by at most 0.005 from the full-suffix reference
(`test_panel_leaves_the_window_count_distribution_alone`). Dyads sharing a node are
included together, so dyads are **not** independent; that inflates the variance of
the cell counts and therefore any standard error computed as if they were, but it
does not affect the correctness of the likelihood.

**Replication.** All four arms are now stochastic, so all four get five samples:
280 main observations, 3 360 planned calls, 1 680 of them Qwen. These sizes are
derived from the replication scheme in `common.py`, not written as literals.
Running the chain end to end exposed three places that still hard-coded the old
5/5/1/5 scheme — the evaluator's cell shape, the mock check and the structural
verification — all now derived.

## 2. How much information is actually observed

Median over the six real main sources; the middle column is the one that determines
ρ, and absolute counts are given because shares alone hide how thin arm S is.

| Arm | dyads | **active dyad-windows** | events | D_obs median (min) |
| --- | --- | --- | --- | --- |
| R | 9.7 % | 9.8 % | 9.6 % | 1 472 (96) |
| S | 1.6 % | **2.4 %** | 10.2 % | 276 (19) |
| H | 12.7 % | 10.7 % | 10.4 % | 1 882 (110) |
| B | 43.4 % | 37.1 % | 10.0 % | 5 371 (607) |

Against the superseded design (57 / 20 / 55 / 78 % of active dyad-windows) this is
a five- to eightfold reduction. Every arm observes about 10 % of events by
construction; they differ in how those events are spread. Arm B still finds many
dyads, because one retained event reveals a dyad — what it loses is *which* windows
they were active in.

**Budget attainment.** Arm B keeps each event with probability exactly 0.10. The
two panel arms round to an integer panel and report the resulting error: across the
24 graphs the worst is 0.63 % for R and 2.58 % for H, and the worst walk-validation
deviation is 1.87 %, all inside the unchanged 5 % tolerance. All 24 graphs report
`budget_matched: true`, so unlike the previous pool run no case had to be carried
as unmatched.
No observation is empty (`empty_observations: 0`), so the frozen-median replacement
path is implemented and tested but never exercised.

## 3. Where the error comes from

An offline decomposition splits the plug-in error into the part caused by *which*
dyads a mechanism reaches and the part caused by what it loses about them. Full
histories are used for this evaluation only and never feed an estimator.

| Arm | selection | lost history | share of absolute error from lost history |
| --- | --- | --- | --- |
| R | −0.0002 | 0 | 0 % |
| S | **+0.134** | 0 | 0 % |
| H | +0.060 | **−0.161** | 72 % |
| B | +0.133 | **−0.517** | 80 % |

R and S retrieve complete histories, so their entire error is selection; for H and
B the lost windows dominate. This is the evidence for the split in the design:
simple methods on R and S, mixture models that address history loss on H and B.

## 4. Which baselines work, and under which conditions

### Development graphs (100 held-out synthetic, 2 000 observations)

| Arm | plug-in | homogeneous corrector | candidate | ExtraTrees pooled |
| --- | --- | --- | --- | --- |
| R | **0.0176** | = plug-in | — | 0.0185 |
| S | 0.1369 | 0.0318 | — | **0.0213** |
| H | 0.1010 | 0.1199 | 0.0380 | **0.0246** |
| B | 0.3839 | 0.1164 | 0.0719 | **0.0305** |

### Six real main sources (MAE₂, and the paired difference with its standard error)

| Arm | plug-in | corrector | candidate | ET pooled | what is actually distinguishable |
| --- | --- | --- | --- | --- | --- |
| R | **0.0197** | 0.0197 | — | 0.0299 | nothing beats the plug-in |
| S | 0.3222 | 0.0972 | — | 0.0615 | corrector −0.225 ± 0.063 (t 3.6): clearly needed |
| H | 0.0721 | 0.1705 | 0.0779 | 0.0590 | homogeneous corrector **+0.098 ± 0.023 (t 4.3): clearly harmful**; candidate repairs that (−0.093 ± 0.027, t 3.4) but only reaches parity with the plug-in (+0.006 ± 0.027, t 0.2) |
| B | 0.0823 | 0.0555 | 0.0585 | 0.0524 | everything within noise of everything else (all t < 2) |

**The honest reading.** On synthetic development data both candidates beat both
references by large margins. On the six real sources that advantage largely
disappears: the H candidate does **not** beat the plug-in, and on arm B no method
is distinguishable from any other at n = 6. What the real sources do show clearly is
that the *homogeneous* suffix corrector is actively harmful and that the walk
corrector on arm S is essential. Six sources is the binding constraint on
statistical power, and no amount of modelling fixes that.

The corrector choice was written down in `CORRECTOR_DECISION.md` **before** the real
sources were evaluated, and it is not revised in the light of the table above.

### Numerical behaviour of the candidates

`not_converged` never occurred in 1 000 development fits. Identifiability is
visibly weaker at this budget than at the old one: only 179/500 arm-B fits and
344/500 arm-H fits are `converged`, with 184 and 56 flagged `weakly_identified`.
The pre-registered fallback (unreliable fit → homogeneous corrector) fired on 3 %
of H and 16 % of B fits. Widening every parameter bound by two decades moves the
objective by ~1e−12 and the predicted profile by ~2e−8, so no fit is pinned by an
artificial bound.

Three documentation-versus-code defects were found and fixed: the optimiser
accepted any finite objective including the invalid-region penalty; the thinned
positive-Poisson closed form was stated for k ≥ 0 when it holds only for k ≥ 1
(the k = 0 term differs by 1/(e^λ − 1), and no computed quantity used it); and the
claim of "no clipping anywhere" contradicted a clamp in `predict_profile`, which
now raises beyond last-bit noise.

## 5. Limits

* Six real test sources. Most differences on them are not statistically resolvable.
* The development graphs come from the same two generator families as the training
  pool, so ExtraTrees' large margin there is within-family generalisation. Its real-
  source margin is much smaller and mostly inside the noise.
* Arm H's fit remains exactly saturated on the aggregated window counts; the
  extrapolation into the two unobserved windows is a model assumption the data
  cannot check. The seven visible time patterns support descriptive exchangeability
  diagnostics, but because dyads sharing a node are dependent these are not turned
  into independent multinomial significance tests.
* The walk corrector on arm S is not claimed to be unbiased at finite, short walk
  lengths; at this budget the walk is short (median 276 discovered dyads).
