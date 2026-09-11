# 4a: is the time_agnostic_t selection channel real? 4b: is the correction reachable?

Both analytic. No model output, no new runs.

## 4a — the +0.009 is real, and it is the expected value

The objection: a random walk is degree-biased and should oversample active
dyads, so `time_agnostic_t` at +0.009 selection against `time_respecting` at
+0.197 looks like a definition artefact.

### Reference distribution

Both sides of the channel are unweighted means over *unique dyads*.
`rho_W5_k` is the fraction of population pairs with `K >= k` among pairs with
`K >= 1` (`rho_W5_k1 = 1` in every case, confirming the denominator).
`oracle__seen_label_rho_k` is the same statistic over the dyads the sample
contains, with true labels. Neither is event-weighted, so the channel is a
composition difference over dyads and nothing else.

### It is not a k = 2 coincidence

Selection across the whole profile, mean over 32 cases per arm:

| arm | k=2 | k=3 | k=4 | k=5 |
|---|---|---|---|---|
| `event_sample_then_full_history` | +0.163 | +0.109 | +0.084 | +0.039 |
| `node_panel_full_history` | −0.002 | +0.000 | −0.006 | −0.003 |
| `recent_history_k20` | +0.161 | +0.124 | +0.086 | +0.044 |
| **`time_agnostic_t`** | **+0.009** | **+0.008** | **+0.012** | **−0.000** |
| `time_respecting` | +0.197 | +0.143 | +0.105 | +0.063 |

`time_agnostic_t` is flat at zero across all four thresholds. The other walks
show the monotone decline of a classic activity oversample.

### The activity gradient, measured directly

`oracle__hh_label_rho_k2` is the same quantity traversal-weighted. Its gap to
the unweighted version is measured **on the identical dyad set**, so it isolates
how strongly the sampler's own traversal counts track true `K`:

| arm | selection (discovery) | traversal-weighted − unweighted |
|---|---|---|
| `node_panel_full_history` | −0.002 | **+0.195** |
| `event_sample_then_full_history` | +0.163 | +0.099 |
| `time_respecting` | +0.197 | +0.027 |
| `recent_history_k20` | +0.161 | +0.016 |
| **`time_agnostic_t`** | **+0.009** | **−0.001** |

`node_panel_full_history` is the control that makes this readable: it discovers
neutrally, yet reweighting its dyads by traversal count moves `rho_2` by +0.195.
The activity-to-`K` gradient in these graphs is therefore strong. Against that,
`time_agnostic_t`'s −0.001 says its traversal counts carry no information about
`K` at all.

### Why, in one sentence for the thesis

> The degree bias of a random walk falls on the nodes it revisits, not on the
> dyads it discovers: `time_agnostic_t` steps to a uniformly random neighbour on
> the time-collapsed graph, where every event of a pair has been merged into one
> edge, so the sampler cannot see how active a pair is and discovers pairs
> essentially uniformly, while `time_respecting` steps along a uniformly chosen
> future *event* and therefore reaches a pair in proportion to its activity.

Verified in `src/walks.py`: the `time_agnostic_t` branch draws
`y = coll_adj[x][randint(len(coll_adj[x]))]` — uniform over distinct
neighbours, multiplicity invisible. The `time_respecting` branch draws
`j = randint(j0, len(times))` — uniform over later events, multiplicity is the
weight. Its edge-revisit rate is 0.16 against 0.38–0.83 for the others, the
signature of near-uniform edge coverage.

**Verdict: real finding, not an artefact.** It is what makes `time_agnostic_t`
the clean censoring instrument.

## 4b — reachable on both clean arms, not on the forward walk

The question: if the correction magnitude is not identifiable from what the
prompt contains, "correctly corrected" is not a fair target.

### `event_sample_then_full_history` — feasible, and demonstrated

Events are examined in uniformly random order and a pair enters if one of *its*
records is reached, so inclusion probability is `1 - (1-f)^n` for the realized
sampling fraction `f`. **`f` is not in the prompt** — the budget was removed
from every prompt by the leakage audit. But for small `f` the probability is
proportional to `n`, and in the Hájek ratio form the constant cancels:

    rho_k = sum_d (1/n_d) 1[K_d >= k]  /  sum_d (1/n_d)

`n_d` is in the prompt, and this arm retrieves complete histories, so the
observed mask is the true one. No external parameter is required.

| estimator | rho_2 bias | rho_2 MAE | ProfileMAE |
|---|---|---|---|
| plug-in | +0.163 | 0.164 | 0.102 |
| **Hájek inverse-activity** | **+0.010** | **0.090** | **0.046** |
| occupancy MLE | +0.317 | 0.318 | 0.253 |

94% of the bias removed, from prompt content alone.

Stated limit: exact only while `f * n` is small. As a pair's inclusion
probability approaches 1 the true weight stops being proportional to `1/n` and
the most active pairs are under-corrected. Without `f` that residual cannot be
removed.

**Measured 2026-09-03, and the limit binds harder than this section implied.**
Across the 32 primary cases the realized sampling fraction has median 0.270 and
mean 0.295, reaches 1.0 in four cases, and `max(n) * f > 1` holds in **29 of
32**. The small-`f` regime is therefore the exception on this panel, not the
rule. The heading "feasible, and demonstrated" stands for the demonstration —
94% of the bias is removed — but "no external parameter is required" was the
wrong sentence: the parameter is not required only because it is being
approximated away, and the approximation is out of its regime in most cases.
Read the Hájek row as a strong empirical reference, never as an identified
prompt-only estimator.

### `time_agnostic_t` — already solved, and its name should say so

Its mechanism text states the observation model verbatim: one timestamp per
traversal, drawn uniformly from the pair's complete history, independently and
with replacement, `n` = number of traversals. `src/corrected_estimator.py`
inverts an occupancy model in which draws land uniformly over the pair's `K`
**active windows**.

**Corrected 2026-09-03: those are not the same likelihood.** Uniform over
events equals uniform over active windows only when a pair's events are spread
equally across the windows it is active in, and the `(n, mask)` input carries no
per-window counts, so the prompt cannot verify it. The MLE is identified under
that auxiliary assumption, not from prompt content alone.

How much that costs on this panel is measurable, and the answer is: very
little. Over the 584 primary-case pairs with at least two active windows, the
ratio of the largest to the smallest per-window event count is exactly 1 in
**80%** of them, with median 1.00, mean 1.25 and 90th percentile 2.00; only
3.6% exceed 2. The assumption is nearly satisfied here, which is why the MLE
removes 80% of the bias. Say "identified under a stated homogeneity assumption
that this panel nearly satisfies" — not "identified from the prompt".

| estimator | rho_2 bias | rho_2 MAE | ProfileMAE |
|---|---|---|---|
| plug-in | −0.280 | 0.289 | 0.131 |
| **occupancy MLE** | **−0.056** | **0.086** | **0.054** |
| Hájek | −0.294 | 0.303 | 0.135 |

### `time_respecting` — not identifiable label-free

Established in the repo before this session, `src/bias_identifiability.py` and
`results/bias_identifiability/RESULTS.md`: a bias-aware MLE with one forward-bias
parameter has its profile likelihood maximised at `beta = 0`, the wrong value,
so maximum likelihood selects the no-correction answer. On one instance with
`rho_true = 0.40` the likelihood prefers `rho_hat = 0.187` over the correct
`beta ~ 2` giving 0.42. The occupancy MLE is inconsistent under forward
sampling, not merely inefficient. The bias strength must come from outside the
sample.

### Specificity: each estimator is best only on its own arm

| arm | plug-in | Hájek | occ MLE |
|---|---|---|---|
| `event_sample_then_full_history` | 0.102 | **0.046** | 0.253 |
| `node_panel_full_history` | **0.022** | 0.088 | 0.230 |
| `recent_history_k20` | 0.136 | 0.146 | **0.076** |
| `time_agnostic_t` | 0.131 | 0.135 | **0.054** |
| `time_respecting` | 0.144 | 0.149 | **0.107** |

ProfileMAE over k = 2..5. Applying inverse-activity weighting to
`node_panel_full_history`, where there is no activity-based selection, moves its
bias from −0.002 to −0.174. Neither correction is a generic improvement; each
is tied to its channel. That is the property a mechanism-aware estimator has to
have, and it is the reason these two can serve as the achievable target the
model is measured against.

## What this changes for the thesis text

"Correct correction" is a fair target in **direction** on all five arms. It is a
fair target in **magnitude** on `event_sample_then_full_history` and
`time_agnostic_t`, where a prompt-only estimator reaches it. It is **not** a
fair magnitude target on `time_respecting`, where no label-free estimator can
recover the strength, and by extension not on `recent_history_k20`, which shares
the forward-time structure. Where the model is scored on magnitude, that limit
is stated at the point of scoring, not in a footnote.

## Reproducing

```bash
python src/mechanism_aware_estimators.py
```
