# Numbers cleared for slides

*2026-09-02, updated 2026-09-03. For the Thursday idea presentation. Every number that may appear
on a slide is here, with its source table, its n, its interval and its model
role. **Nothing that is not in this file goes on a slide.** Numbers still
moving are marked PROVISIONAL with the reason.*

Model roles are freeze (f): `qwen36-27b_think` and `qwen36-27b_nothink` are
**confirmatory** (version-pinned weights, recorded sampling parameters, fully
reproducible). Codex `gpt-5.6-sol` is **exploratory** — its harness injects
instructions outside the frozen prompt and cannot be version-pinned. A
Codex-only finding carries that label on the slide itself, not in a footnote.

Unless stated otherwise: intervals are 95% cluster percentile bootstrap over
the 12 graph groups, 4000 draws, seed 20260901; the case set is `seed_slot 0`;
generations are averaged before any contrast.

---

## 1. The primary slope

`results_summary/g4/primary_slope.csv`. Response is the paired contrast
`rho2(mechanism) − rho2(hidden)`; predictor is the required correction
`delta_i = rho2_true − rho2_naive`. A slope of 1 is a perfect case-specific
corrector.

| model | role | scope | n | slope | 95% CI | R² |
|---|---|---|---|---|---|---|
| `qwen36-27b_think` | confirmatory | clean arms | 95 | **0.826** | [0.725, 0.989] | 0.75 |
| `qwen36-27b_nothink` | confirmatory | clean arms | 95 | **0.685** | [0.570, 0.846] | 0.64 |
| `codex-gpt-5.6-sol` | exploratory | clean arms | 81 | **0.830** | [0.747, 0.960] | 0.82 |

**The four nulls, same scope.** These belong on the same slide as the slope;
quoting a slope without them is the thing the external review caught.

| null | what it is | clean-arm value |
|---|---|---|
| N0 | both legs move together | 0.000 |
| N1 | hidden leg pinned to the plug-in, mechanism leg a constant | 0.888 (Qwen), 0.902 (Codex) |
| N2 lookup | knows its arm's typical bias, nothing case-specific | 0.617 (Qwen), 0.623 (Codex) |
| permutation | predictor permuted within arm, 4000 draws | 0.516 [0.408, 0.625] (think) |

**How to say it, and this wording matters:**

- Think and Codex clear the lookup null and the permutation null; their CIs sit
  entirely above both.
- **Nothing here clears N1 pooled.** 0.888 lies inside the think CI
  [0.725, 0.989]. The pooled slope alone does not separate the model from a
  prior fallback. Do not put "the model corrects case by case" on a slide
  supported by the pooled number.
- `qwen36-27b_nothink` at 0.685 [0.570, 0.846] **contains** the lookup null
  0.617. Pooled, the non-thinking model is consistent with an arm lookup table.

## 2. Where it does separate: per arm

| model | arm | n | slope | 95% CI | N1 | N2 |
|---|---|---|---|---|---|---|
| `qwen36-27b_think` | `time_agnostic_t` | 32 | **1.058** | [0.867, 1.244] | 0.210 | 0.000 |
| `qwen36-27b_nothink` | `time_agnostic_t` | 32 | 0.539 | [0.327, 0.753] | 0.210 | 0.000 |
| `codex-gpt-5.6-sol` | `time_agnostic_t` | 32 | 0.852 | [0.711, 1.005] | 0.210 | 0.000 |
| `qwen36-27b_think` | `event_sample_then_full_history` | 32 | 0.452 | [0.211, 0.767] | 1.588 | 0.000 |

This is the slide the claim rests on: on `time_agnostic_t` the thinking model
is at 1.058 against a prior-fallback null of 0.210, and the interval is nowhere
near it.

`node_panel_full_history` has **no slope on any slide.** Freeze (b) addendum:
its Var(delta) is 0.0026 and the within-arm slope is not a point estimate.

**Opposed arms (extension, not the claim):** think −0.060 [−0.134, −0.009],
nothink −0.013 [−0.089, +0.074]. Where the sign of the correction is a net
quantity nobody could derive from the text, the models do not derive it.

## 3. Skill score — and the arm reversal

`results_summary/g4/skill_scores.csv`. `1 − MSE(model)/MSE(best constant)`;
the best constant is the in-sample mean of the truth, so N1 scores exactly 0
and the constant-output artefact cannot enter. All n = 32, seed slot 0.

**Both clean arms belong on the slide. Showing only the first is selective.**

`time_agnostic_t` — censoring-driven, required correction +0.280:

| condition | think | nothink | Codex (exploratory) |
|---|---|---|---|
| `hidden` | −1.216 | −1.149 | −1.208 |
| `direction_only` | −0.655 | −0.324 | −1.164 (n=18) |
| `mechanism` | **+0.811** | **+0.627** | +0.831 |
| `mechanism_direction` | +0.783 | +0.683 | +0.773 (n=18) |

`event_sample_then_full_history` — single selection term, required
correction −0.163:

| condition | think | nothink |
|---|---|---|
| `hidden` | +0.390 | +0.467 |
| `direction_only` | **+0.760** | **+0.840** |
| `mechanism` | +0.613 | +0.601 |
| `mechanism_direction` | +0.740 | +0.683 |

**The ordering reverses.** On `time_agnostic_t` the mechanism description is
worth 1.47 skill points over naming the direction; on
`event_sample_then_full_history` naming the direction beats the description by
0.15 (think) and 0.24 (nothink), in both models.

**The sentence for the slide** — this replaces "naming the direction alone does
not help; the gain comes from the process", which was true only of the arm it
was measured on:

> What a mechanism description buys depends on the channel structure. Where the
> bias is censoring-driven and the required correction is large, the
> description is worth a great deal and a bare direction sentence is worth less
> than nothing. Where the bias is one modest selection term, naming the
> direction is enough and the description adds nothing.

Codex `metadata_only` (n=7) and `mismatched` (n=7) are too small for a slide
claim; quote them only as "not estimated at this n" if asked.

## 3b. It is a `rho_2` result, not a profile result

`results_summary/g4/profile_component_slopes.csv` and
`profile_mae_by_condition.csv`, freeze (l). The frozen target is the joint
profile with ProfileMAE as its loss; `rho_2` is the headline component only.

Paired slope by component, clean arms:

| model | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|
| `qwen36-27b_think` | 0.826 | 0.641 | 0.558 | 0.421 |
| `qwen36-27b_nothink` | 0.685 | 0.506 | 0.366 | 0.383 |
| `codex-gpt-5.6-sol` *(expl.)* | 0.830 | 0.846 | 0.764 | 0.775 |

**Any slide showing 0.826 must say "`rho_2`".** Say in the same breath that the
mean required correction also collapses across k (+0.040, +0.001, −0.009,
−0.001), so there is progressively less to correct — the decay is not purely a
failure of the model.

Frozen group-macro ProfileMAE, clean arms:

| condition | think | nothink |
|---|---|---|
| `hidden` | 0.0892 | 0.0920 |
| `direction_only` | 0.0768 | **0.0648** |
| `mechanism` | **0.0509** | **0.0652** |

On the frozen loss, "mechanism beats direction" holds for the thinking model
and **not** for the non-thinking one (0.0652 against 0.0648).

## 4. Properness: the two operationalizations disagree

Freeze (h) with its 2026-09-03 correction. Both halves appear together; either
alone misleads.

**Stated intervals**, nominal 90%, `results_summary/g4/stated_intervals.csv`.
Grouped **by condition** — the earlier pooled-by-arm numbers were a mixture and
are withdrawn. `metadata_only` is excluded: no sample was shown, so a sample
coverage rate for it is meaningless.

`qwen36-27b_think`, `time_agnostic_t`:

| condition | n | median width | empirical coverage |
|---|---|---|---|
| `hidden` | 96 | 0.016 | **0.031** |
| `direction_only` | 96 | 0.058 | 0.062 |
| `mechanism` | 96 | 0.160 | **0.625** |
| `mechanism_direction` | 96 | 0.170 | 0.625 |
| `mismatched` | 96 | 0.008 | 0.021 |

**Realized dispersion across generations**
(`dispersion_vs_coverage.csv`), sd against `log10(coverage)`, pooled: think
−0.005 [−0.020, +0.004], nothink +0.001 [−0.009, +0.009]. Both contain zero.

**The two sentences, and do not shorten them into one:**

1. Stated uncertainty *does* respond to information, strongly and in the right
   direction: the mechanism description widens the interval tenfold and takes
   coverage from 0.03 to 0.63. No condition reaches the nominal 0.90, and under
   `hidden` and `mismatched` the intervals are absurdly overconfident.
2. Resampling spread across generations does **not** respond to how much of the
   network was shown. That is a statement about decoding variance.

The earlier line "calibrated in rhetoric only" was an artefact of pooling
conditions and is withdrawn.

## 5. The twin contrast

`results_summary/g4/twin_arms.csv`, freeze (j). `time_agnostic_t` against
`time_respecting`: nearly the same required correction from very different
channel compositions.

| model | n instances | model gap | 95% CI | required gap |
|---|---|---|---|---|
| `qwen36-27b_think` | 32 | **+0.207** | [0.125, 0.286] | −0.021 |
| `qwen36-27b_nothink` | 32 | +0.221 | [0.170, 0.277] | −0.021 |
| `codex-gpt-5.6-sol` | 17 | +0.159 | [0.070, 0.257] | −0.053 |

The models shift by 0.21 between two arms whose correct answers differ by 0.02 (Qwen) or 0.05 (Codex, 17 instances).
The required gap is outside the model CI for every model.

**The coverage objection, answered** (`twin_coverage_confound.csv`): the twins
differ in coverage (0.138 against 0.078), so "saw less, correct more" would
produce a gap on its own. It predicts a **negative** gap, because
`time_respecting` is the lower-coverage arm. Observed: **positive in all three
models.** The rival explanation predicts the mirror image of what happened.

Supporting, one line: the within-arm correlation between the model's correction
and coverage is ≈0 (−0.041 to +0.086). Do **not** quote the arm-level
correlation (−0.24 to −0.66) as support — coverage and the required correction
are collinear at −0.629 across arms, so a model tracking the requirement
produces it too.

## 6. The wrong-direction cell — now complete

`results_summary/g4/wrong_direction.csv`, freeze (k). Correct process
description, inverted direction sentence. **Both models, three generations
each.** The two thinking generations lost to a wall clock were rerun and
finished at 64/64; the provisional label is gone.

| model | arm | `mechanism` slope | with false direction | shift toward the false claim |
|---|---|---|---|---|
| think | `time_agnostic_t` (correct: up) | 1.036 | **0.098** [−0.021, 0.226] | +0.226 [0.162, 0.286] |
| think | `event_sample` (correct: down) | 0.664 | **0.075** [−0.149, 0.318] | +0.127 [0.086, 0.168] |
| nothink | `time_agnostic_t` | 0.558 [0.337, 0.775] | 0.193 [0.034, 0.374] | +0.237 [0.207, 0.269] |
| nothink | `event_sample` | 0.635 [0.369, 0.913] | 0.340 [0.149, 0.659] | +0.072 [0.033, 0.110] |

Between-arm contrast **+0.352 [0.271, 0.429]** (think) and **+0.309
[0.250, 0.365]** (nothink), every draw positive in both. The arms carry opposite
correct directions, so opposite shifts identify deference to the claim rather
than a reaction to prompt length. **The thinking model defers more, not less.**

**The strongest single statement available from this panel:** the thinking
model is its best case-specific corrector — position slope 1.036 on
`time_agnostic_t` from the mechanism description alone. One sentence asserting
the opposite direction takes that to **0.098, with an interval containing
zero**, on both arms. The correction does not shrink, it stops existing.

Follows the evidence rather than the instruction: think 0.344 / 0.548, nothink
0.438 / 0.781 on `time_agnostic_t` / `event_sample`. Both models follow the
false instruction more often than the evidence on `time_agnostic_t`.

**Say it this way, not more strongly:** the prompt states the false direction
as fact rather than as a fallible hint, so this is not a demonstration of blind
obedience. It measures which source wins when an explicit textual claim
contradicts the structure derivable from the described process. The text wins.

## 7. Design numbers (context slides)

| number | value | source |
|---|---|---|
| panel | 32 graph instances, 12 graph groups, 5 arms | `results_summary/g3/` |
| channel split, `time_agnostic_t` | selection +0.009, censoring −0.288 | `bias_channels_by_arm.csv` |
| channel split, `recent_history_k20` | selection +0.161, censoring −0.439 (opposed) | same |
| additivity residual | max 1.1e-16 over 160 cases | `docs/BIAS_CHANNELS_2026-09-02.md` |
| mechanism-space coverage | 11 alternative samplers × 5 budgets all sit in composition band 0.22–0.42; the chosen five span 0.029–1.000 | `mechanism_space.csv` |
| bootstrap robustness | preregistered [0.734, 0.974]; wild cluster Rademacher [0.718, 0.937]; instance-level [0.704, 0.950] | `bootstrap_robustness.csv` |
| dropout | 1.1% (think), 1.5% (nothink), 0% Codex; no case removed from the primary fit except one in `node_panel` | `docs/MISSINGNESS_2026-09-02.md` |

## 8. Named limitations that belong on a slide

- **No sixth arm.** A "selection with upward correction" arm would separate
  mechanism reading from the surface heuristic that the twin and
  wrong-direction findings expose. It is not built. The mechanism-space figure
  shows the gap it would have filled.
- The pooled slope does not clear N1; only the per-arm result does.
- `qwen36-27b_nothink` pooled is consistent with an arm lookup table.
- **The effect is strongest at `rho_2` and decays across the profile**, and on
  the frozen profile loss the non-thinking model gains nothing from the
  description over the direction sentence.
- **The mechanism description is not uniformly better than naming the
  direction** — the ordering reverses on the selection arm.
- **The correction is not identified from the prompt**, and the two arms fail
  differently. On `time_agnostic_t` the reference estimator inverts an
  occupancy model with draws uniform over *active windows*, while the prompt
  says an event is drawn uniformly from the pair's *full history*; those agree
  only under equal event counts per active window, which the `(n, mask)` input
  does not carry. Measured on this panel, that assumption is nearly satisfied —
  exactly equal in 80% of pairs with ≥2 active windows, median ratio 1.00 —
  so say "identified under a homogeneity assumption this panel nearly
  satisfies". On `event_sample_then_full_history` the inclusion probability
  depends on the unshown phase-1 fraction `f` and the code approximates it away
  via `1/n`, which needs `n·f ≪ 1`; here `f` has median 0.270 and
  `max(n)·f > 1` in **29 of 32** cases, so that argument is out of its regime
  and the 94% bias removal is empirical, not guaranteed. The benchmark tests
  empirical direction sensitivity, not exact derivation.
- The wrong-direction result is one generation for the thinking model, and the
  prompt states the false direction as fact rather than as a fallible hint — so
  it measures how a model weighs an explicit textual claim against derivable
  structure, not blind obedience.
- Codex is a product screen, not evidence about a pinned model.

---

## Provisional status, as of 2026-09-03

| item | status |
|---|---|
| Qwen main run, both branches, 3 generations | **final** — all six generations settled at 976/736, verified byte-identical to the cluster |
| Wrong-direction, both models | **final** — 3 generations each, 64/64 |
| Codex Step 2 | **final** — 256 of 256 core prompts, 0 errors |
| Codex hold-A completion | **final** — 30 of 30, `mismatched` and `metadata_only` off n=7 |
