# The wrong-direction cell: derivation or deference?

*2026-09-02. Analysis of a cell that had already run. The reading rule was
fixed first, as freeze (k), before any of its answers were looked at; this
document reports against that rule and adds nothing to it.*

**Half the cell. The non-thinking model only.** The thinking half was still on
the cluster when this was written (`g3wd_0/1/2`, running). The thinking model
is the one carrying the headline slope, so nothing here transfers to it.

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
| `qwen36-27b_nothink` | +0.069 | -0.237 | **0.306** | [0.242, 0.369] |

Opposite signs, and the gap is far from zero with every bootstrap draw
positive. By the rule fixed in (k) this is the deference outcome, not the
nuisance outcome. It is not a close call.

## What it costs the correction

The like-for-like reference is the *position* slope of the `mechanism` leg on
these same cases -- the paired slope in `primary_slope.csv` is a contrast
against `hidden` and is not comparable.

| arm | `mechanism` slope | + false direction | shift toward the stated claim |
|---|---|---|---|
| `time_agnostic_t` | 0.558 [0.337, 0.775] | **0.193** [0.034, 0.374] | +0.237 [0.207, 0.269] |
| `event_sample_then_full_history` | 0.635 [0.369, 0.913] | **0.239** [-0.029, 0.676] | +0.069 [0.025, 0.111] |

One false sentence removes roughly two thirds of the slope the correct
description had bought, and on `event_sample_then_full_history` the remaining
interval no longer excludes zero. The share of cases whose correction still
goes the way the evidence requires falls to 0.44 on `time_agnostic_t` -- worse
than a coin -- while `event_sample_then_full_history` holds 0.74.

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

- **Nothing about the thinking model.** Its half is unread. The whole
  substantive question -- whether reasoning changes which source wins a
  conflict -- is open, and this document is not evidence about it either way.
- **No explanation of the asymmetry.** Obedience is 3.4x stronger on
  `time_agnostic_t` (0.237) than on `event_sample_then_full_history` (0.069).
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
