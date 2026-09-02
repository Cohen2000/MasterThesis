# Is the G4 dropout ignorable?

*2026-09-02. Analysis only; no new runs. Reproduce with
`PYTHONPATH=src python src/report_missingness.py`; tables under
`results_summary/g4/missingness_*.csv`.*

A dozen or so prompts per Qwen generation never terminate at any token budget.
They are dropped before any slope is fitted, and nothing so far said whether
that loss is related to the quantity being estimated. It is the standard
missing-data question and it had not been asked of this panel.

Two levels of loss are separated, because they have different causes:
`structural` (no parseable complete record -- a token-cap burn, not an answer)
and `invalid` (structurally complete, but a profile component outside `[0, 1]`
or a provider refusal). Neither is repaired.

## How much, and where

| model | attempted | structural | invalid | rate |
|---|---|---|---|---|
| `codex-gpt-5.6-sol` | 422 (rising) | 0 | 0 | 0.000 |
| `qwen36-27b_nothink` | 2,220 | 32 | 1 | 0.015 |
| `qwen36-27b_think` | 2,928 | 32 | 0 | 0.011 |

Codex loses nothing, though that row is still filling: 47 of the 256
core prompts were outstanding when this was written, so the Codex zero is
provisional and the Qwen rows are final (the main Qwen run settled at 10:39 on
2026-09-02). The Qwen loss is almost entirely structural: 64 burns and
a single invalid profile in the whole panel.

It is not spread evenly. For thinking, 29 of 32 losses fall on
`event_sample_then_full_history`; the remaining 3 fall on
`node_panel_full_history`. **`time_agnostic_t`, `time_respecting` and
`recent_history_k20` lose nothing at all in the thinking condition.** That
matters, because `time_agnostic_t` is the arm that carries the headline
separation from the prior-fallback null (slope 1.058 against N1 = 0.210): the
arm the claim rests on is complete.

## Is the loss related to `delta_i`?

Pooled, the burned records have a far smaller correct correction than the kept
ones -- a gap of -0.387 [-0.457, -0.275] for thinking. Read alone that looks
like severe informative missingness.

It is an arm artefact. `delta_i` differs by an order of magnitude across arms
and is *negative* on `event_sample_then_full_history`, which is where nearly
all the loss sits. Within arm the gap collapses:

| model | arm | n | lost | gap (lost - kept) | 95% CI |
|---|---|---|---|---|---|
| think | `event_sample_then_full_history` | 624 | 29 | -0.116 | [-0.199, +0.004] |
| think | `node_panel_full_history` | 528 | 3 | -0.052 | [-0.060, -0.042] |
| think | `recent_history_k20` | 528 | 0 | -- | -- |
| think | `time_agnostic_t` | 720 | 0 | -- | -- |
| think | `time_respecting` | 528 | 0 | -- | -- |
| nothink | `event_sample_then_full_history` | 483 | 25 | -0.022 | [-0.137, +0.051] |
| nothink | `node_panel_full_history` | 386 | 6 | -0.022 | [-0.050, +0.022] |

Two rows in `missingness_delta_gap.csv` are not reproduced above and should
not be read as estimates: `recent_history_k20` and `time_respecting` in the
non-thinking condition each lost exactly one record, so the cluster bootstrap
is resampling the presence or absence of a single observation and its interval
carries no information about a rate.

The `node_panel` row is statistically clear and substantively empty: mean
`delta_i` on that arm is 0.002, so a gap of 0.05 is noise around a correction
that does not exist. The `event_sample` thinking row is the one honest
qualification -- the interval only just includes zero (2.9% of bootstrap draws
positive), so on that arm the burns do plausibly sit on the cases with the
largest negative correction. It is not dismissed below; it is bounded.

## Why it does not reach the fitted slope

The slope is fitted on case means at `seed_slot == 0`, after the generation
axis is averaged away. A case survives if *at least one* generation of each leg
survived, so a per-record burn does not remove a case:

| model | arm | paired cases at slot 0 | of 32 |
|---|---|---|---|
| think | `event_sample_then_full_history` | 32 | complete |
| think | `time_agnostic_t` | 32 | complete |
| think | `time_respecting` | 32 | complete |
| think | `recent_history_k20` | 32 | complete |
| think | `node_panel_full_history` | 31 | one lost |

Despite 17 burned records at slot 0 on `event_sample`, the arm keeps all 32
cases. The only case the panel actually loses is one in
`node_panel_full_history` -- the arm whose within-arm slope is already
prespecified as not interpretable as a point estimate (freeze (b) addendum).

What the burns do cost is generations behind a cell mean. No Qwen cell mean in
the primary slice rests on a single draw: the minimum is 2 of 3, and only 8 of
318 thinking cells and 7 of 318 non-thinking cells are short at all. Noise in a
cell mean enters the *response*, not the predictor -- `delta_i` is computed
from the frozen truth and is exact -- so it widens the residual and the
interval without biasing the slope.

(The same diagnostic reads differently for Codex, which has two generations
only on the prompts Step 1 and Step 2 share. A Codex cell with one generation
is the design, not a loss.)

## Asymmetry between the two legs

An asymmetric loss is the one that could move `Delta_i` without moving the
predictor, so the two directions are counted separately rather than folded into
one "incomplete pair" number:

| model | arm | pairs | only `hidden` lost | only `mechanism` lost | both |
|---|---|---|---|---|---|
| think | `event_sample_then_full_history` | 168 | 9 | 6 | 2 |
| think | `node_panel_full_history` | 168 | 0 | 2 | 0 |
| nothink | `event_sample_then_full_history` | 96 | 4 | 0 | 0 |
| nothink | `node_panel_full_history` | 96 | 0 | 3 | 0 |

Where asymmetry occurs it is close to balanced (9 against 6), and it occurs at
the record level, which the generation average absorbs. No arm shows a
one-sided pattern large enough to pull the paired contrast in a direction.

## What this licenses, and what it does not

Licensed: the reported slopes are fitted on the complete case panel for every
arm except one case in `node_panel_full_history`, and the arm carrying the
headline claim has no dropout at all. The dropout is a precision cost, not a
selection effect.

Not licensed: a claim that the loss is ignorable *by mechanism*. It is not
random -- it concentrates on one arm, and within that arm the burns lean
towards the largest negative corrections. The defensible statement is narrower
and is the one to use in the text: **the loss does not remove cases from the
primary fit, and it is absent from the arms the headline rests on.** For
`event_sample_then_full_history` specifically, the cell means of the burned
cases are averaged over two generations rather than three, and that arm's
result should be read with that qualification attached.
