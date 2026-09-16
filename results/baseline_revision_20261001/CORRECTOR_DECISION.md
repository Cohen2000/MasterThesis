# Corrector decision, fixed before the real sources were evaluated

Date: 2026-10-01 · Design: `budget10-20261001` · Evidence: development graphs only
(`development_summary.json`, 100 held-out synthetic graphs, 2 000 observations).

## Decision

| Arm | Primary corrector | Reason |
| --- | --- | --- |
| R | plug-in (the existing corrector equals it by construction) | The decomposition shows selection error −0.0002 and **zero** history loss: a uniform node panel with full histories is unbiased, and the residual 0.0176 is sampling noise. There is nothing for a corrector to remove. |
| S | existing walk ratio corrector | Again zero history loss; the whole error is selection (+0.134). The ratio corrector addresses exactly that and takes MAE₂ from 0.137 to 0.032. It is **not** claimed to be unbiased at finite, short walk lengths. |
| H | **heterogeneous zero-truncated Beta-Binomial** | History loss dominates (72 % of the absolute error). MAE₂ 0.038 against 0.101 for the plug-in and 0.120 for the homogeneous corrector. |
| B | **heterogeneous Beta mixture with the ZTP event layer** | History loss dominates even more (80 %). MAE₂ 0.072 against 0.384 plug-in and 0.116 homogeneous. |

## Numerical acceptance, checked before this decision

* `not_converged` never occurred on any of the 1 000 development fits.
* Acceptance now requires the optimiser to report success and the objective to be
  outside the invalid-region penalty; a finite objective alone is not accepted.
* Pre-registered fallback: a fit whose status is `not_converged` or
  `starts_disagree` does not contribute a mixture prediction and falls back to the
  homogeneous corrector. It fired on 14/500 (3 %) H fits and 80/500 (16 %) B fits.
* Widening every parameter bound by two decades moves the objective by ~1e-12 and
  the predicted profile by ~2e-8, so no fit is pinned by an artificial bound.
* Identifiability is visibly weaker at this budget than at the superseded one:
  only 179/500 B fits and 344/500 H fits are `converged`, with 184 and 56
  respectively flagged `weakly_identified`. This is reported, not smoothed over.

## Limits carried into the main evaluation

The development graphs are synthetic and come from the same two generator families
as the training pool. Both families violate the model's assumption of conditionally
independent, exchangeable windows — DAR through serial copy dependence, activity-
driven through activity that grows over time — and the candidates still win, which
is evidence of robustness but not of correctness. For arm H the fit remains exactly
saturated on the aggregated window counts, so the extrapolation into the two
unobserved windows stays a model assumption that the data cannot check. The seven
visible time patterns do allow descriptive exchangeability diagnostics; because
dyads sharing a node are not independent, those are reported descriptively and are
not turned into independent multinomial significance tests.

The real main sources are evaluated **after** this file was written. Their ranking
does not feed back into the choice above.
