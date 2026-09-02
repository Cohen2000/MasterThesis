# Numbers cleared for slides

*2026-09-02, for the Thursday idea presentation. Every number that may appear
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
| `codex-gpt-5.6-sol` | exploratory | clean arms | 79 | **0.832** | [0.748, 0.964] | 0.82 |

**The four nulls, same scope.** These belong on the same slide as the slope;
quoting a slope without them is the thing the external review caught.

| null | what it is | clean-arm value |
|---|---|---|
| N0 | both legs move together | 0.000 |
| N1 | hidden leg pinned to the plug-in, mechanism leg a constant | 0.888 (Qwen), 0.894 (Codex) |
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

## 3. Skill score

`results_summary/g4/skill_scores.csv`. `1 − MSE(model)/MSE(best constant)`;
the best constant is the in-sample mean of the truth, so N1 scores exactly 0
and the constant-output artefact cannot enter. All n = 32, `time_agnostic_t`.

| condition | think | nothink | Codex (exploratory) |
|---|---|---|---|
| `hidden` | −1.216 | −1.149 | −1.208 |
| `direction_only` | −0.655 | −0.324 | −1.499 (n=15) |
| `mechanism` | **+0.811** | **+0.627** | +0.831 |
| `mechanism_direction` | +0.783 | +0.683 | +0.763 (n=15) |
| `metadata_only` | −1.819 | −2.242 | +0.177 (n=7) |
| `mismatched` | −1.467 | −1.466 | −1.039 (n=7) |

The headline: describing the process moves the thinking model from −1.216 to
+0.811, a swing of **2.03**, while naming the direction alone (−0.655) does
not help. Adding the direction on top of the description buys nothing
(+0.783 against +0.811).

Codex `metadata_only` (n=7) and `mismatched` (n=7) are too small for a slide
claim; quote them only as "not estimated at this n" if asked.

## 4. Properness: the two operationalizations disagree

Freeze (h) asks for both halves. They must appear together — either alone
misleads.

**Stated intervals** (`results_summary/g4/stated_intervals.csv`), nominal 90%:

| model | arm | n | empirical coverage | width vs log coverage |
|---|---|---|---|---|
| `qwen36-27b_think` | `time_agnostic_t` | 576 | **0.255** | −0.037 |
| `qwen36-27b_think` | `recent_history_k20` | 384 | **0.042** | +0.017 |
| `qwen36-27b_think` | `node_panel_full_history` | 370 | 0.732 | −0.060 |
| `qwen36-27b_nothink` | `time_agnostic_t` | 579 | 0.212 | −0.042 |
| `codex-gpt-5.6-sol` | `time_agnostic_t` | 122 | 0.738 | −0.096 |

**Realized dispersion across generations**
(`results_summary/g4/dispersion_vs_coverage.csv`), sd against `log10(coverage)`,
pooled: think −0.005 [−0.020, +0.004], nothink +0.001 [−0.009, +0.009]. Both
contain zero.

**The sentence:** stated interval width *does* widen as coverage falls, but the
answers themselves do not spread out at all. The uncertainty is **calibrated in
rhetoric only**. Nominal 90% intervals achieve 0.04 to 0.73 empirical coverage.

## 5. The twin contrast

`results_summary/g4/twin_arms.csv`, freeze (j). `time_agnostic_t` against
`time_respecting`: nearly the same required correction from very different
channel compositions.

| model | n instances | model gap | 95% CI | required gap |
|---|---|---|---|---|
| `qwen36-27b_think` | 32 | **+0.207** | [0.125, 0.286] | −0.021 |
| `qwen36-27b_nothink` | 32 | +0.221 | [0.170, 0.277] | −0.021 |
| `codex-gpt-5.6-sol` | 14 | +0.162 | [0.064, 0.265] | −0.002 |

The models shift by 0.21 between two arms whose correct answers differ by 0.02.
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

## 6. The wrong-direction cell — PROVISIONAL

`results_summary/g4/wrong_direction.csv`, freeze (k). Correct process
description, inverted direction sentence. **Non-thinking model only; the
thinking half was still running on the cluster at the time of writing.** If the
thinking half has not landed by Thursday, this slide says "non-thinking model,
thinking pending" on its face.

| arm | correct direction | `mechanism` slope | with false direction | shift toward the false claim |
|---|---|---|---|---|
| `time_agnostic_t` | upward | 0.558 [0.337, 0.775] | **0.193** [0.034, 0.374] | +0.237 [0.207, 0.269] |
| `event_sample_then_full_history` | downward | 0.635 [0.369, 0.913] | 0.239 [−0.029, 0.676] | +0.069 [0.025, 0.111] |

Between-arm contrast **+0.306 [0.242, 0.369]**, all draws positive. The two
arms carry opposite correct directions, so opposite shifts identify deference
to the claim rather than a reaction to prompt length. One false sentence
removes about two thirds of the slope the correct description bought.

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
- The wrong-direction result is half a cell.
- Codex is a product screen, not evidence about a pinned model.

---

## Provisional status, as of 2026-09-02 12:5x

| item | status |
|---|---|
| Qwen main run, both branches, 3 generations | **final** — all six generations settled at 976/736, verified byte-identical to the cluster |
| Wrong-direction, non-thinking | **final** |
| Wrong-direction, thinking | **running** — 3 SLURM jobs, watcher will sync and recompute |
| Codex Step 2 | **running** — 213 of 256 core prompts; every Codex number here may move |
