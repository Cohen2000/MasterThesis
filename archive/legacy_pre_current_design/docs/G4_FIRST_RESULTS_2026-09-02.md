# G4, first results: the twin contrast and the dispersion criterion

Both analyses were specified before any of these numbers existed — the twin
contrast in freeze (j), the dispersion regression in freeze (h) — and both are
run here for the first time. The blinding condition in (i) was satisfied first:
the channel decomposition, the mechanism-space figure and the bootstrap
robustness rows were all finished before the `mechanism` leg of any Qwen model
was read.

> **Preliminary in one respect.** Qwen thinking generation 2 stood at 963 of 976
> settled when these were computed. Thirteen prompts are outstanding, which can
> only move cases in or out of the three-generation subset. Both tables are
> recomputed on completion; the sign and magnitude of everything below would
> have to survive that, and nothing here is quoted as final until it does.

## The twin contrast: the models are reading the text, not the requirement

`time_agnostic_t` and `time_respecting` require nearly the same correction from
very different channel compositions. That is the point of the pair: if a model
responds to the *implied correction* it should shift about equally on both; if it
responds to *text surface* it should not.

Gap means the arm difference on the same instance, seed slot 0 only.
`results_summary/g4/twin_arms.csv`.

| model | instances | model gap [95%] | required gap [95%] | zero in CI | required in CI |
|---|---|---|---|---|---|
| Qwen3.6-27B thinking | 32 | **+0.207** [0.125, 0.286] | −0.021 [−0.085, 0.016] | no | **no** |
| Qwen3.6-27B non-thinking | 32 | **+0.221** [0.170, 0.277] | −0.021 [−0.085, 0.016] | no | **no** |
| Codex `gpt-5.6-sol` | 12 | **+0.161** [0.048, 0.282] | +0.012 [0.003, 0.026] | no | **no** |

The required gap is indistinguishable from zero. Every model's gap is an order of
magnitude larger, excludes zero, and excludes the required gap. For both Qwen
branches it also has the *opposite sign*: they shift more on the arm that needs
slightly less correction.

So the models are not responding only to the correction the mechanism implies.
Something about how the two mechanisms are described moves them by about 0.2
where the correct answer moves by about 0.02.

This does not overturn the main direction result — that the models move the right
way is measured against `delta_i` per case, and stands or falls on its own. It
bounds what the result can be said to show: the models pick up *that* a
correction is needed and its sign, and they do not calibrate *how much* to the
mechanism. The twin pair is the cleanest evidence for that reading in the design,
because it is the one place where the required answers are matched and the texts
are not.

Two confounds to state with it. The two arms differ in more than their mechanism
text — realized coverage differs (0.138 against 0.078) and so does the data
block. The contrast is between arms, not between texts holding everything else
fixed, and a text-only version of it would need a separate cell.

## The dispersion criterion: uncertainty does not scale with missing information

Spread of `rho_k2` across independent generations of the identical prompt,
regressed on `log10(coverage)`, cluster bootstrap over the twelve graph groups.
Properness predicts a **negative** slope. `results_summary/g4/dispersion_vs_coverage.csv`.

| model | measure | cases | slope [95%] | adjusted for mean `rho_k2` |
|---|---|---|---|---|
| Qwen3.6-27B non-thinking | sd | 709 | **+0.0008** [−0.0090, 0.0094] | +0.0012 |
| Qwen3.6-27B thinking | sd | 940 | **−0.0042** [−0.0201, 0.0046] | −0.0053 |
| Qwen3.6-27B non-thinking | IQR | 709 | +0.0010 [−0.0086, 0.0091] | +0.0014 |
| Qwen3.6-27B thinking | IQR | 940 | −0.0035 [−0.0180, 0.0046] | −0.0045 |
| Codex (2 generations, 30 cases) | sd | 30 | −0.0035 [−0.0061, −0.0002] | −0.0034 |

**Qwen's realized dispersion is flat in coverage.** Non-thinking is at zero;
thinking leans negative, in the direction properness predicts, but its interval
comfortably contains zero. Over a coverage range spanning more than a decade, the
spread of the answers barely moves.

The guards (h) built in all hold:

- **sd and IQR agree**, so this is not one outlier-sensitive estimator.
- **The adjusted slope barely differs** from the raw one, so the mean-variance
  compression of a probability near its bounds is not producing or masking it.
- **All-three and at-least-two generations agree** (thinking −0.0042 against
  −0.0057), so the coverage-dependent dropout is not driving it.
- **The dropout is small and not in the feared direction.** 33 of 973 thinking
  cells excluded, at *higher* median coverage than those included (0.065 against
  0.053); 25 of 734 non-thinking cells, at slightly lower (0.044 against 0.055).
  The threat (h) named — that low-coverage cases drop out preferentially — is not
  what happened.

The Codex row is the one negative interval that excludes zero, and it is the row
to trust least: two generations, 30 cases, four groups. (h) fixed in advance that
it belongs in the text as an order of magnitude and not in the regression, and
that is where it stays.

**Reading.** On the realized-spread operationalization, these models are not
proper in the multiple-imputation sense: they do not become measurably less
repeatable when they have less to go on. Whether their *stated* intervals widen
is the other half of the criterion and is reported beside this from the
`lo90`/`hi90` fields; the two are separate operationalizations and can disagree,
and a model whose stated intervals widen while its answers do not is calibrated
in rhetoric only.

## Reproducing

```bash
python src/build_g4_cell.py
python src/report_twin_arms.py --paired results_summary/g4/g4_paired.csv
python src/report_dispersion_coverage.py --cell results_summary/g4/g4_cell.csv
```
