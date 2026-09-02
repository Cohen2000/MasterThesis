# Does a coverage rule of thumb explain the twin gap?

*2026-09-02. One analysis, on data already collected. Reproduce with
`PYTHONPATH=src python src/report_twin_arms.py --paired
results_summary/g4/g4_paired.csv`; tables `twin_coverage_confound.csv` and
`coverage_association.csv`.*

## The objection

The twins are matched on the required correction (-0.280 against -0.301) and
contrast in the stated mechanism. But they are **not** matched on dyad
coverage: `time_agnostic_t` shows 0.138 of the dyads, `time_respecting` 0.078.
A model applying "saw less, correct more" would produce a gap between them
without reading either mechanism description, and the twin finding would say
nothing about mechanism sensitivity.

## The test, and it is one number

The heuristic makes a directional prediction. `time_respecting` is the *lower*
coverage arm, so the heuristic corrects more there. Under this module's
`A - B` convention with `A = time_agnostic_t`, that is a **negative** model gap.

| model | coverage A | coverage B | heuristic predicts | observed gap | consistent? |
|---|---|---|---|---|---|
| `qwen36-27b_think` (confirmatory) | 0.138 | 0.078 | negative | **+0.207** | no |
| `qwen36-27b_nothink` (confirmatory) | 0.138 | 0.078 | negative | **+0.221** | no |
| `codex-gpt-5.6-sol` (exploratory) | 0.101 | 0.062 | negative | **+0.162** | no |

**No.** The gap points towards the *higher* coverage arm in every model, which
is the opposite of the rival explanation's prediction. The confounder does not
generate the observed gap; it would have generated the mirror image of it.

The finding therefore stands cleaner than before, and its wording does not
change: the models shift by 0.21 between two arms whose correct answers differ
by 0.02, and coverage is not why.

## Supporting check across all five arms

| model | arm-level corr | within-arm corr |
|---|---|---|
| `qwen36-27b_think` | -0.301 | +0.010 |
| `qwen36-27b_nothink` | -0.242 | -0.041 |
| `codex-gpt-5.6-sol` | -0.655 | +0.086 |

The **within-arm** correlation is the informative one and it is essentially
zero: inside an arm, where the required correction still varies case to case,
the model's correction does not track coverage.

The arm-level correlation is negative, which looks like the heuristic -- but it
is **not diagnostic**, and it should not be quoted as if it were. Across the
five arms coverage and the required correction are collinear at -0.629: the
arms that show less also need more correction. A model tracking the requirement
perfectly would produce the same negative arm-level correlation. That
collinearity is exactly why the twin pair was constructed, and the twins break
it: they hold the requirement fixed and let coverage vary, and there the model
goes the way the heuristic does not.

## What this does not settle

It rules out coverage as *the* explanation of the twin gap. It does not
identify what the gap is: the models shift far more between the twins than
either the requirement or coverage can account for, and freeze (j) already
records that the gap is 10x the required difference. Something in the text
surface drives it, and this panel does not say which feature.
