# Is the net correction direction on the opposed arms stable under budget?

Freeze (j) reports `recent_history_k20` and `time_respecting` as an extension
rather than part of the primary claim, because their two channels disagree and
the net direction is a quantitative trade-off. That raises the obvious follow-up:
if a moderate budget change flips the net sign, the "required direction" those
cases are scored against is not a property of the arm at all, and the extension
is testing noise.

**It does not flip.** Not on average, and not in a single case.

## Method, and the verification it turns on

The frozen walk logs were not retained, but `walk_rng_seed` is in the case table
and `src/walks.py` builds a sweep from prefixes of one log, so replaying a seed
reproduces the identical walk and every shorter budget is a prefix of it.

The replay is checked before anything is read from it: the 800-step reproduction
must match the frozen case row's plug-in and oracle values to 1e-9, and the
script exits without writing if it does not. **It matched for all 96 cases** —
32 instances x 3 arms — which is also an incidental end-to-end reproducibility
check on the frozen pipeline.

Nothing frozen was touched; the replay reads the panel event files and writes
only to `results_summary/`.

## Result

`results_summary/g3/budget_sign_stability.csv`, means over 32 cases.

| arm | budget | selection | censoring | net bias | cases with positive net |
|---|---|---|---|---|---|
| `recent_history_k20` | 200 | +0.160 | −0.467 | −0.307 | **0 / 32** |
| | 800 *(frozen)* | +0.161 | −0.439 | −0.278 | **0 / 32** |
| | 1600 | +0.155 | −0.408 | −0.253 | **0 / 32** |
| `time_respecting` | 200 | +0.206 | −0.521 | −0.315 | **0 / 32** |
| | 800 *(frozen)* | +0.197 | −0.498 | −0.301 | **0 / 32** |
| | 1600 | +0.190 | −0.476 | −0.285 | **0 / 32** |
| `time_agnostic_t` | 200 | +0.008 | −0.297 | −0.290 | 1 / 32 |
| | 800 *(frozen)* | +0.009 | −0.288 | −0.280 | 1 / 32 |
| | 1600 | +0.006 | −0.272 | −0.266 | 1 / 32 |

Across an eightfold budget range, on both opposed arms, **not one of the 32
cases has a positive net bias at any budget**. The mean shrinks monotonically —
more data, less bias, as it should — but slowly: about 0.02 per doubling on
`recent_history_k20`. Extrapolating that rate, the sign would not reach zero
within any budget this design could run.

`time_agnostic_t` has exactly one positive case, and it is the *same* case at
every budget. That is a property of one graph, not of the budget.

## Two things this also shows

**The channel composition is a mechanism property, not a budget artefact.** The
selection share moves from 0.256 to 0.275 on `recent_history_k20` and from 0.283
to 0.286 on `time_respecting` across the whole range. This is the assumption the
mechanism-space figure rests on, measured rather than assumed.

**The opposed arms' status in (j) is about inference, not instability.** They
remain an extension because a reader given only the mechanism text could not
*derive* the net sign without the magnitudes — not because the sign is fragile.
Those are different objections and only the first one applies. The distinction
belongs in the text.

## Limits

Budgets below 200 were not tried; at very small budgets the sample can be nearly
empty and the plug-in is dominated by single-event dyads. The frozen design does
not run there and neither does this check.

This varies the budget only. It does not vary the graph panel, the window count
`W`, or `history_k` on the recent-history arm, any of which could move the
balance and none of which is claimed here.

## Reproducing

```bash
python src/report_budget_sign_stability.py
```
