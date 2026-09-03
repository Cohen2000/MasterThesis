# The wrong-direction cell: derivation or deference?

*2026-09-02. Analysis of a cell that had already run. The reading rule was
fixed first, as freeze (k), before any of its answers were looked at; this
document reports against that rule and adds nothing to it.*

**Complete as of 2026-09-03.** Both models, three generations each. The two
thinking generations that had been lost to a four hour wall clock were rerun
with a resumable chunk size and finished at 64/64. The one-generation reading
recorded earlier understated the effect: it is stronger and tighter with all
three.

## The manipulation

The mechanism description is the correct one. Only the sentence naming which
way the naive estimate errs is inverted. That separates deference to an
explicit claim from derivation out of a described process -- `mismatched`
cannot, because it swaps the description itself, and a model shifting under it
may simply be reacting to a process that does not fit the sample it can see.

Two arms, with opposite correct directions:

| arm | naive estimate | correct correction | the prompt says |
|---|---|---|---|
| `time_agnostic_t` | underestimates | **upward** | "an overestimate of" -> move down |
| `event_sample_then_full_history` | overestimates | **downward** | "an underestimate of" -> move up |

## The prespecified test

A model that obeys the stated word moves in **opposite** directions in the two
arms. A model reacting to prompt length, or to the mere presence of an
assertion, moves the **same** way in both. The between-arm gap in the raw shift
is therefore the quantity that separates them, and neither arm alone does.

| model | shift on the down arm | shift on the up arm | between-arm gap | 95% CI |
|---|---|---|---|---|
| `qwen36-27b_nothink` | +0.072 | -0.237 | **0.309** | [0.250, 0.365] |
| `qwen36-27b_think` | +0.127 | -0.226 | **0.352** | [0.271, 0.429] |

The thinking model defers *more*, not less.

Opposite signs, and the gap is far from zero with every bootstrap draw
positive. By the rule fixed in (k) this is the deference outcome, not the
nuisance outcome. It is not a close call.

## What it costs the correction

The like-for-like reference is the *position* slope of the `mechanism` leg on
these same cases -- the paired slope in `primary_slope.csv` is a contrast
against `hidden` and is not comparable.

Non-thinking, 3 generations:

| arm | `mechanism` slope | + false direction | shift toward the stated claim |
|---|---|---|---|
| `time_agnostic_t` | 0.558 [0.337, 0.775] | **0.193** [0.034, 0.374] | +0.237 [0.207, 0.269] |
| `event_sample_then_full_history` | 0.635 [0.369, 0.913] | **0.340** [0.149, 0.659] | +0.072 [0.033, 0.110] |

Thinking:

| arm | `mechanism` slope | + false direction | shift toward the stated claim |
|---|---|---|---|
| `time_agnostic_t` | 1.036 | **0.098** [-0.021, 0.226] | +0.226 [0.162, 0.286] |
| `event_sample_then_full_history` | 0.664 | **0.075** [-0.149, 0.318] | +0.127 [0.086, 0.168] |

This is the sharpest result in the cell. The thinking model is the panel's best
case-specific corrector -- position slope 1.036 on `time_agnostic_t` from the
mechanism description alone. Add one sentence asserting the opposite direction
and the slope is 0.098 with an interval containing zero, on both arms. The
correction does not shrink; it stops existing.

One false sentence removes roughly two thirds to three quarters of the slope
the correct description had bought, and on three of the four rows the remaining
interval no longer excludes zero.

The share of cases whose correction still goes the way the evidence requires:

| model | `time_agnostic_t` | `event_sample_then_full_history` |
|---|---|---|
| `qwen36-27b_nothink` | 0.438 | 0.781 |
| `qwen36-27b_think` | **0.344** | 0.548 |

Both models follow the false instruction more often than the evidence on
`time_agnostic_t`. The single-generation reading recorded here earlier put the
thinking figure at 0.156; with all three generations it is 0.344, so that
number was an outlier of one draw and is corrected.

## What this licenses

Per (k): the non-thinking model is deferring to the claim, so **the gain
measured in `mechanism_direction` is at least partly obedience rather than
derivation**. A correct direction sentence and a false one both move it, and
the correct one is not distinguished by the model as correct.

That is the honest form of the result. It does not say the model derives
nothing from the description -- it plainly does, or the `mechanism` slope of
0.56-0.64 would not exist -- but it does say the description is not what the
model relies on when the two disagree.

## What it does not license

- **No claim that this is blind obedience.** The prompt states the false
  direction as fact, not as a hint that might be wrong, so a model that trusts
  stated premises is behaving reasonably by one standard. What the cell
  measures is which source wins when an explicit textual claim contradicts the
  structure derivable from the described process -- and the text wins.
- **No explanation of the asymmetry.** Obedience is 3.3x stronger on
  `time_agnostic_t` (0.237) than on `event_sample_then_full_history` (0.072)
  in the non-thinking model, and 1.8x in the thinking one (0.226 against
  0.127).
  The obvious guess, that the false claim points back towards the naive anchor,
  fails: it points that way in *both* arms. Freeze (k) rules out subgroup
  slopes from this cell -- 32 instances, one seed slot -- so the asymmetry is
  recorded and left unexplained rather than fitted.
- **No claim about magnitude transfer.** The cell is powered for a direction
  and a rough size, nothing finer.

## Reproduce

```
PYTHONPATH=src python src/report_wrong_direction.py
```

Tables: `results_summary/g4/wrong_direction.csv`,
`wrong_direction_contrast.csv`, `wrong_direction_cases.csv`. The script skips
the thinking model with a printed note while its answers are absent and picks
it up automatically once they are synced.
