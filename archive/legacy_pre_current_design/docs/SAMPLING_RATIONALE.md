# Why these five observation mechanisms

Drafted for the methods chapter. It replaces the realism argument with a
design argument, because realism is not what the arms are for and defending them
on it invites an objection the design does not need.

## The argument

The experiment asks whether a model, given a neutral description of how data
were observed, moves its estimate in the **case-specific correct direction**.
That question needs three things from the set of mechanisms, and realism is not
one of them.

1. **Two error channels, separable.** A partial observation of a temporal
   network distorts `rho_k` in two ways: *selection*, which dyads enter the
   sample at all, and *censoring*, how much of a sampled dyad's history is seen.
   `docs/BIAS_CHANNELS_2026-09-02.md` shows the split is exact — additive to
   1.1e-16 — and that the panel contains a near-clean instrument for each.
2. **Three correction directions, so direction cannot be guessed.** Upward,
   near-zero and downward all occur. A model that always corrects the same way
   scores at chance.
3. **Ground truth per case**, so "the correct direction" is a measured quantity
   rather than a judgement.

These are properties of a designed comparison. A mechanism is included because
it isolates a channel or supplies a direction, not because a practitioner would
plausibly collect data that way. Throughout, **controlled observation
mechanism**, never *realistic sampling mechanism*.

That the five span the space rather than being chosen for their results is shown
in `docs/MECHANISM_SPACE_2026-09-02.md`: eleven alternative samplers already run
in this repository all fall in one narrow band, and three of the five occupy
positions none of them reaches.

## Role of each arm — one sentence, about the design

**`time_agnostic_t`** — the **cleanest censoring instrument** in the panel and
the reason the censoring channel can be read at all: its selection component is
+0.009 across the whole `K` profile, so essentially all of its −0.280 bias is
censoring. It is **not** put forward as a realistic collection method; nobody
crawls a time-collapsed graph and draws one uniformly random timestamp per
traversal. It is constructed so that the observation model is exactly specifiable
in the prompt, and it is the arm on which a mechanism-aware estimator reaches the
truth (occupancy MLE, ProfileMAE 0.131 → 0.054).

**`time_respecting`** — the **twin** of `time_agnostic_t`: it requires nearly the
same correction, −0.301 against −0.280, from a very different channel
composition, 35% censoring with an opposing selection term against 97%
censoring. The pair is a matched contrast in the required answer and a difference
in the stated mechanism, which is what makes it possible to ask whether a model
is responding to the implied correction or to text surface. It is also the arm
where the correction is *not* identifiable label-free, so it carries the limit on
what magnitude scoring can mean.

**`event_sample_then_full_history`** — the **pure selection instrument**, and the
only mechanism in the repository whose bias points **upward** (+0.163). Its
censoring component is exactly zero by construction: complete histories are
retrieved for every pair that enters. It carries the upward direction alone,
which is why it cannot be dropped for budget.

**`node_panel_full_history`** — the **null case**: whole-entity sampling whose
bias is −0.002, indistinguishable from zero. It tests whether a model moves when
it should not, which is the failure mode a direction test cannot otherwise
detect. Its role in the argument runs through signed error and skill score, not
through a slope: `Var(delta)` collapses to 0.0026 on this arm and a within-arm
slope is not interpretable there (freeze (b)).

**`recent_history_k20`** — the **opposed-channel arm** closest to how temporal-
graph systems actually retrieve neighbours, and the one place where a practical
access pattern enters the design. Its channels disagree (+0.161 selection,
−0.439 censoring), so its net direction is a quantitative trade-off rather than
a qualitative inference, and freeze (j) reports it as an extension rather than as
part of the primary claim.

## Related work: Rocha, Masuda & Holme 2017

> **Unverified.** The characterization below is drafted from a summary of the
> paper and has not been checked against the full text; see
> `docs/CITATION_CHECKLIST.md` item 1. The *correspondence* claims are ours and
> follow from our own mechanism definitions, so they do not depend on it. The
> *direction* claims about link lifetime do, and must be verified before
> publication.

Rocha, Masuda and Holme (PRE 96, 052302, 2017) study how sampling distorts
temporal-network statistics under four strategies: **TS**, shortening the
observation window; **NS**, sampling nodes uniformly; **ES**, sampling events
uniformly; and **RS**, coarsening the resolution so that same-link events inside
an interval merge. For mean link lifetime `L_s` they report that NS and ES
estimate well, TS underestimates because of cutoffs, and RS overestimates
because rounding birth and death outward lengthens the observed span while
single-event links drop out of the mean.

**What transfers, and what does not.** Their estimand is `L_s = t_last -
t_first` over links with at least two events. Ours is `rho_k = P(K >= k)`, the
fraction of pairs active in at least `k` of five windows. These respond to
distortion differently and **the bias directions do not carry across**. RS is the
clear case: merging same-link events within an interval lowers `K` per dyad and
should push `rho_2` *down*, while Rocha finds RS pushing lifetime *up*. Anyone
reading their table as a prediction for this work will get the sign wrong.

**Correspondence.**

- `node_panel_full_history` corresponds to **NS**: nodes selected uniformly,
  their histories complete.
- `event_sample_then_full_history` is **not ES**. ES is one-stage — events are
  sampled and that is the data. Ours is two-stage: a uniform event sample is used
  only to *name* pairs, and each named pair's complete history is then retrieved.
  The distinction matters because it moves the arm from mixed-channel to pure
  selection, and because Rocha finds ES low-bias on lifetime whereas our
  two-stage variant is the largest positive `rho_2` bias in the panel.
- `time_agnostic_t`, `time_respecting` and `recent_history_k20` lie **outside**
  the Rocha taxonomy. Walk-based access is not one of their four strategies.

**Deliberately not used: TS and RS.** Both change the estimand rather than
observing it partially. TS shortens the observation window, so `rho_k` becomes a
quantity over a trimmed window and is no longer the same target. RS changes the
definition of `K` itself by merging events. Under either, "the correct
correction" would not be defined, because there would be no fixed population
quantity to correct *towards*. This is a design exclusion, not an oversight.

**What is not covered.** The panel has no resolution-coarsening mechanism, no
observation-window truncation, and — until a sixth arm is added — no
*upward*-correcting selection mechanism: every selection channel here
over-represents active dyads. Claims are therefore about censoring versus
activity-based selection, and do not extend to aggregation or truncation
distortions.
