# Where each arm's rho_2 bias comes from

The design is argued as two error channels crossed with three correction
directions. Until now that was prose. This is the number.

## The split

    rho2_true  = mean of 1[K_true >= 2] over population dyads with K_true >= 1
    rho2_seen  = mean of 1[K_true >= 2] over the dyads the sample contains
    rho2_naive = mean of 1[K_obs  >= 2] over those same dyads

    selection = rho2_seen  - rho2_true      which dyads enter the sample
    censoring = rho2_naive - rho2_seen      how much of their history is seen
    total     = selection + censoring = rho2_naive - rho2_true = -delta_i

`rho2_seen` is `oracle__seen_label_rho_k2`, already in the frozen case table:
true labels on the sampled dyad set, computed from the full event stream.

This is the numerator/denominator split in the exact sense. Selection moves the
set the ratio is taken over and leaves the labels alone; censoring moves the
labels and leaves the set alone. Additivity is not an approximation — the
maximum residual over all 160 cases is **1.1e-16**.

### Why the order is not a choice

Two-stage decompositions are usually path-dependent, and the usual objection
applies: why sample-then-censor rather than censor-then-sample. Here the second
path does not exist. A dyad outside the sample has no observed history at all,
so "the population under censored labels" is not a defined quantity. There is
exactly one route from the truth to the plug-in, and nothing is being held fixed
by choice.

## Result

Means over 32 cases per arm, with a cluster bootstrap over the twelve graph
groups. `results_summary/g3/bias_channels_by_arm.csv`.

| arm | total bias | selection | censoring | channels |
|---|---|---|---|---|
| `event_sample_then_full_history` | +0.163 | **+0.163** [0.105, 0.219] | 0.000 | single |
| `node_panel_full_history` | −0.002 | **−0.002** [−0.016, 0.009] | 0.000 | single |
| `recent_history_k20` | −0.278 | +0.161 [0.124, 0.198] | **−0.439** [−0.515, −0.367] | opposed |
| `time_agnostic_t` | −0.280 | +0.009 [0.001, 0.019] | **−0.288** [−0.342, −0.236] | opposed |
| `time_respecting` | −0.301 | +0.197 [0.145, 0.252] | **−0.498** [−0.577, −0.422] | opposed |

Three things follow, and none of them were visible in the prose version.

**The two whole-entity arms have exactly zero censoring.** Not small — zero, to
machine precision, on every case. That is what "full history of the selected
entity" means: a dyad touching a panel node has all of its events observed, so
its label cannot be censored. Their entire bias is selection, and arm B's
+0.163 is a pure denominator effect.

**On the walks the two channels have opposite signs.** Selection is positive on
all three — a walk preferentially finds dyads that are active, and active dyads
have higher K — while censoring is negative and larger, because a walk sees only
part of even a found dyad's history. The observed downward bias is a difference
of two opposing effects, not a single mechanism. Anything reported as "the walk
underestimates" is a net figure over channels that disagree.

**`time_agnostic_t` is very nearly a pure censoring arm.** Its selection channel
is +0.009, an order of magnitude below the other two walks, so 97% of its
movement is censoring. Together with arm B at 100% selection, the panel has a
near-clean instrument at each end of the channel axis, not merely a spread.

## The share columns, and which to use

`share_of_total` (channel over total bias) is the textbook quantity and is
**not** usable here. Where the channels oppose it leaves the unit interval —
censoring is 1.58 of the total on `recent_history_k20` and selection is −0.58 —
and where the total is near zero, as on `node_panel_full_history` with −0.002,
it is numerically meaningless.

`share_of_movement`, |channel| / (|selection| + |censoring|), stays in [0, 1]
whatever the signs do and is what the figure uses. Both columns are in the CSV;
the first is kept so a reader can see why it was not used.

## What this does not establish

The decomposition is a property of the sampling designs and the ground truth. It
says nothing about whether a model recovers either channel, and it is not
evidence about the direction test. It is the design justification, computed
rather than asserted.

Coverage is not held constant across arms, so the channel sizes are not a
like-for-like comparison of mechanisms at equal information. They are the
channel composition of the arms as actually budgeted, which is what the design
argument needs.

## Reproducing

```bash
python src/report_bias_channels.py
```

Writes `bias_channels_by_case.csv` and `bias_channels_by_arm.csv` to
`results_summary/g3/`. The additivity check is an assertion in the code, not a
reported number: a non-additive decomposition raises rather than prints.
