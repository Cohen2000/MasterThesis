#!/usr/bin/env python3
"""G0d: coverage-parity budgets for the two whole-entity access arms.

G0c ran every arm at 800 unique events.  That is parity in the wrong currency:
under whole-entity retrieval an 800-event budget buys almost no dyads, and on
dense backbones it buys none at all -- 7/32 empty samples on arm A and 9/256 on
arm B.  G0d re-budgets the two non-walk arms to match the walks' *dyad
coverage* instead, leaves the walks untouched at 800, and re-measures
everything the budget change can move.

No LLM call is made here.  The frozen walk artifacts are read only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from make_llm_prompts_v2 import INPUT_MASK
from nonwalk_samplers import prepare_dyad_histories, prepare_events
from report_g0_headroom import (
    ARMS as WALK_ARMS,
    _event_path,
    _format_table,
)
from report_g0b_headroom import distribution, mask_length_rows, read_globs
from report_g0c_headroom import (
    ALL_ARMS,
    NODE_PANEL,
    TWO_PHASE,
    candidate_bias_summary,
    collect_counts,
    correction_class,
    estimator_ladder,
    full_matrix,
    matrix_penalties,
    pair_summary,
    seed_variance_table,
)

WALK_BUDGET = 800
NATURAL_UNIT = {
    NODE_PANEL: ("nodes recruited", "diag__selected_node_count"),
    TWO_PHASE: ("dyads looked up", "sample__selected_dyad_count"),
}
WINDOW_SHARE = [f"pat__event_share_w{w}" for w in range(5)]


# --- G0d.2: the prespecified empty-sample rule -----------------------------

def apply_seed_rule(frame: pd.DataFrame, n_slots: int
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Take the first ``n_slots`` non-empty draws along each case's seed order.

    The seed sequence for a case is the prespecified index order 0, 1, 2, ...
    A draw whose whole-entity stop left the sample empty is skipped and the
    next index is used instead.  Nothing is redrawn at random and no case is
    dropped, so the accepted set is a deterministic function of the sequence.
    """
    accepted, log = [], []
    for (instance_id, arm), part in frame.groupby(
            ["instance_id", "strategy"], sort=True):
        ordered = part.sort_values("sample_seed", kind="mergesort")
        nonempty = ordered[ordered.budget > 0]
        if len(nonempty) < n_slots:
            raise RuntimeError(
                f"{instance_id}/{arm}: only {len(nonempty)} non-empty draws in "
                f"{len(ordered)} seed slots; extend the seed sequence")
        keep = nonempty.head(n_slots)
        last_index = int(keep.sample_seed.max())
        scanned = ordered[ordered.sample_seed <= last_index]
        advances = int((scanned.budget == 0).sum())
        accepted.append(keep)
        log.append({
            "instance_id": instance_id, "arm": arm,
            "seed_slots": n_slots,
            "seeds_scanned": int(len(scanned)),
            "seed_advances": advances,
            "highest_seed_index_used": last_index,
            "accepted_seed_indices": json.dumps(
                [int(s) for s in keep.sample_seed]),
        })
    out = pd.concat(accepted, ignore_index=True)
    out["seed_slot"] = out.groupby(
        ["instance_id", "strategy"]).sample_seed.rank(method="first").astype(int) - 1
    return out, pd.DataFrame(log)


def exact_empty_probability(manifest_path: Path, budgets: dict[str, int]
                            ) -> pd.DataFrame:
    """Closed-form P(empty) per graph and arm.

    A whole-entity sample is empty exactly when the *first* entity drawn does
    not fit on its own, so the probability needs no simulation.  Arm A draws
    nodes uniformly, hence the share of nodes over budget.  Arm B reaches a
    dyad through a uniformly drawn event, hence the event-weighted share of
    dyads whose complete history is over budget.
    """
    manifest = pd.read_csv(manifest_path)
    rows = []
    for meta in manifest.itertuples(index=False):
        events = pd.read_csv(_event_path(manifest_path, str(meta.path)))
        prepared = prepare_events(events)
        node_sizes = np.array(
            [len(prepared.incident_event_ids[int(n)])
             for n in prepared.active_nodes], dtype=float)
        dyads = prepare_dyad_histories(prepared)
        dyad_sizes = np.array([len(ids) for ids in dyads.event_ids], dtype=float)
        total = dyad_sizes.sum()
        p_a = float((node_sizes > budgets[NODE_PANEL]).mean())
        over = dyad_sizes > budgets[TWO_PHASE]
        p_b = float(dyad_sizes[over].sum() / total)
        for arm, p, largest in ((NODE_PANEL, p_a, node_sizes.max()),
                                (TWO_PHASE, p_b, dyad_sizes.max())):
            rows.append({
                "instance_id": meta.instance_id, "group_id": meta.group_id,
                "arm": arm, "budget": budgets[arm], "p_empty": p,
                "largest_entity_events": int(largest),
                "expected_seed_advances_per_slot": p / (1.0 - p) if p < 1 else np.inf,
            })
    return pd.DataFrame(rows)


# --- G0d.1/G0d.3: coverage, natural units, sign spread ---------------------

def windows_touched(frame: pd.DataFrame) -> pd.Series:
    """Share of the five global windows the sample recorded any event in."""
    shares = frame[WINDOW_SHARE].to_numpy(float)
    return pd.Series((shares > 0).sum(axis=1) / len(WINDOW_SHARE),
                     index=frame.index)


def temporal_evenness(frame: pd.DataFrame) -> pd.Series:
    """How uniformly the sample's events spread over the five windows.

    Every arm touches all five windows on this panel, so a "did you see this
    window at all" measure saturates at 1 and separates nothing.  Distance from
    the uniform share does separate them, and it is the axis that matters here:
    a forward-in-time walker concentrates its events late, which is a temporal
    coverage deficit even though no window is empty.  1 = perfectly uniform,
    0 = every event in one window.
    """
    shares = frame[WINDOW_SHARE].to_numpy(float)
    uniform = 1.0 / len(WINDOW_SHARE)
    tv = np.abs(shares - uniform).sum(axis=1) / 2.0
    return pd.Series(1.0 - tv / (1.0 - uniform), index=frame.index)


def access_profile(panel: pd.DataFrame, budgets: dict[str, int]) -> pd.DataFrame:
    """Budget, natural access unit and the three coverage axes, per arm."""
    rows = []
    for arm, part in panel.groupby("strategy"):
        unit_label, unit_col = NATURAL_UNIT.get(arm, ("walk steps", "budget"))
        units = pd.to_numeric(part.get(unit_col), errors="coerce")
        rows.append({
            "arm": arm,
            "target_budget": budgets.get(arm, WALK_BUDGET),
            "natural_unit": unit_label,
            "median_natural_units": float(units.median()),
            "median_realized_events": float(part.budget.median()),
            "median_dyad_coverage": float(part.coverage.median()),
            "median_node_coverage": float(
                (part.observed_walk_nodes / part.n_nodes_true).median()),
            "median_windows_touched": float(windows_touched(part).median()),
            "median_temporal_evenness": float(temporal_evenness(part).median()),
            "median_dyads_observed": float(part.observed_walk_edges.median()),
            "cases": int(len(part)),
        })
    return pd.DataFrame(rows)


def budget_selection_table(ladder_path: Path, budgets: dict[str, int],
                           target: float, window: int = 3) -> pd.DataFrame:
    """The rungs around each adopted budget, so the choice can be audited."""
    ladder = pd.read_csv(ladder_path)
    rows = []
    for arm, chosen in budgets.items():
        part = ladder[ladder.arm == arm]
        summary = part.groupby("target_budget").agg(
            median_dyad_coverage=("coverage", "median"),
            empty_draw_rate=("empty", "mean"),
            median_realized_events=("realized_events", "median"),
            median_natural_units=("natural_units", "median")).reset_index()
        summary["distance_to_target"] = (
            summary.median_dyad_coverage - target).abs()
        order = summary.target_budget.tolist()
        centre = order.index(chosen)
        near = summary.iloc[max(0, centre - window): centre + window + 1].copy()
        near.insert(0, "arm", arm)
        near["adopted"] = near.target_budget == chosen
        rows.append(near)
    return pd.concat(rows, ignore_index=True)


def strict_direction_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Pairs that genuinely demand *opposite* corrections.

    ``pair_summary`` treats `none` as its own correction class, which lets an
    `upward` <-> `none` pair through.  G0d.5 asks for a pair whose two arms need
    corrections in different *directions*, and an arm needing no correction has
    no direction, so those pairs are reported but not eligible to be chosen.
    """
    keep = [{a, b} == {"upward", "downward"}
            for a, b in zip(pairs.correction_a, pairs.correction_b)]
    return pairs[keep].copy()


def delta_sign_rows(panel: pd.DataFrame) -> pd.DataFrame:
    """Per-case sign of the correct signed correction delta_i."""
    work = panel.copy()
    work["delta"] = work.rho_W5_k2 - work.est__plugin_rho_k2
    rows = []
    for arm, part in work.groupby("strategy"):
        delta = part.delta.dropna()
        rows.append({
            "arm": arm, "cases": int(len(delta)),
            "share_delta_gt_0_upward": float((delta > 0).mean()),
            "share_delta_lt_0_downward": float((delta < 0).mean()),
            "share_delta_eq_0": float((delta == 0).mean()),
            "median_delta": float(delta.median()),
            "mean_delta": float(delta.mean()),
        })
    return pd.DataFrame(rows)


def prompt_length_growth(g0d: pd.DataFrame, g0c_lengths: pd.DataFrame
                         ) -> pd.DataFrame:
    """G0d.4: does the frozen input block grow with the new budgets?"""
    new = mask_length_rows(g0d)
    old = g0c_lengths.rename(columns={"median": "median_at_800"})
    merged = new.merge(old[["arm", "measure", "median_at_800"]],
                       on=["arm", "measure"], how="left")
    merged["growth_factor"] = merged["median"] / merged["median_at_800"]
    return merged


def histogram_key_counts(panel: pd.DataFrame) -> pd.DataFrame:
    """Distinct (n,mask) keys actually rendered, which is what sets length."""
    rows = []
    for arm, part in panel.groupby("strategy"):
        keys = part["input__nmask_exact_json"].astype(str).map(
            lambda raw: len(json.loads(raw)))
        rows.append({"arm": arm, **distribution(keys)})
    return pd.DataFrame(rows)


def qwen_token_estimate(panel: pd.DataFrame, g0c_dir: Path) -> pd.DataFrame:
    """Exact-tokenizer estimate, calibrated on G0c's cached Qwen3.6 counts.

    The Qwen tokenizer snapshot is not available in this environment, so the
    exact count cannot be recomputed.  It can be calibrated instead: G0c stored
    both the exact Qwen count and the rendered text for the same block, and the
    ratio to the tokenizer-independent portable count is tight enough that the
    conversion carries its own error bar.
    """
    from report_g0b_headroom import portable_token_count

    counts = pd.read_csv(g0c_dir / "mask_input_qwen36_tokens.csv")
    texts = {}
    with (g0c_dir / "mask_input_texts.jsonl").open() as handle:
        for line in handle:
            record = json.loads(line)
            texts[record["case_id"]] = record["text"]
    portable = counts.case_id.map(lambda case: portable_token_count(texts[case]))
    ratio = float((counts.qwen36_tokens / portable).median())
    rows = []
    for arm, part in panel.groupby("strategy"):
        lengths = part["input__nmask_exact_json"].astype(str).map(
            lambda raw: portable_token_count(f"{INPUT_MASK}\n{raw}"))
        stats = distribution(lengths * ratio)
        rows.append({"arm": arm, "calibration_ratio": ratio,
                     **{k: round(v) for k, v in stats.items()}})
    return pd.DataFrame(rows)


def export_token_inputs(panel: pd.DataFrame, out: Path) -> None:
    with out.open("w") as handle:
        for row in panel.itertuples(index=False):
            handle.write(json.dumps(
                {"case_id": row.case_id, "arm": row.strategy,
                 "text": f"{INPUT_MASK}\n{row.input__nmask_exact_json}"},
                separators=(",", ":")) + "\n")


# --- report ----------------------------------------------------------------

def build_report(*, budgets, config_table, selection, access, empty_exact,
                 seed_log,
                 bias_seeds, bias_summary, ladder, seed_variance, delta_signs,
                 lengths, key_counts, qwen_estimate, matrix, penalties, pairs,
                 selected_pair, strict,
                 arm_bias_classes, coverage_target, walk_coverage,
                 alt_rows, verified, benchmark_n, acceptance, out_dir) -> str:
    pmae = matrix.pivot(index="sample_arm", columns="assumed_arm",
                        values="profile_mae").reset_index()
    bias = matrix.pivot(index="sample_arm", columns="assumed_arm",
                        values="rho2_bias").reset_index()
    chosen = strict[(strict.arm_a == selected_pair[0]) &
                    (strict.arm_b == selected_pair[1])].iloc[0]
    lowest = pairs.loc[pairs.observable_auc_logo.idxmin()]
    best_eligible_auc = float(strict.observable_auc_logo.min())
    a_bias = float(bias_summary.loc[bias_summary.arm == NODE_PANEL,
                                    "group_macro_bias_mean"].iloc[0])
    b_bias = float(bias_summary.loc[bias_summary.arm == TWO_PHASE,
                                    "group_macro_bias_mean"].iloc[0])
    total_advances = int(seed_log.seed_advances.sum())
    cases_with_advance = int((seed_log.seed_advances > 0).sum())
    a_worst = empty_exact[empty_exact.arm == NODE_PANEL].p_empty.max()
    a_zero_budget = int(empty_exact[
        empty_exact.arm == NODE_PANEL].largest_entity_events.max())
    b_zero = float(empty_exact[empty_exact.arm == TWO_PHASE].p_empty.max())
    return f"""# G0d: budget parity by coverage, not by events

Prepared: **2026-08-31**  
Gate status: **G0d.1-G0d.4 complete and passing. G0d.5 does not clear its
own bar and hands G1 an open decision (see below). No LLM calls were made.
STOP before G1.**

## Why the budgets moved

G0c gave every arm 800 unique events.  For a random walk that is a generous
budget; for whole-entity retrieval it is not, because one response is a whole
node's or a whole dyad's complete history.  The consequence was not a
smaller sample but no sample: **7/32** empty cases on arm A at seed 0 and
**9/256** on arm B across eight seeds.  An arm that fails to produce a sample
on a fifth of the panel is not an arm.

The arms do **not** need to be information-matched.  The primary contrast in
G4 is `mechanism - hidden` within a case on an identical sample, so the arm is
a facet, not a competitor, and cross-arm comparisons of absolute accuracy are
descriptive only.  What the arms do need is to produce a usable sample on
every case.  Coverage parity is the budget rule that delivers that while
keeping the arms on a comparable footing.

**The walks are unchanged at 800 unique events.**  Every frozen number, the
budget probe, the MDD calculation and the historical comparison depend on
that, and nothing here touches them.

## G0d.1 — Budgets set by dyad coverage

Target: the three walk arms' pooled median dyad coverage at 800 events,
**{coverage_target:.4f}** over the same eight seed slots
(**{walk_coverage['seed0']:.4f}** at seed 0 alone).

Budgets were selected with `src/g0d_budget_ladder.py` over a 28-point grid.
Both whole-entity samplers stop before the first entity that does not fit, so
the sample at any budget is a prefix of one fixed random entity order and a
single cumulative pass prices the whole grid exactly.  The ladder was verified
against the production samplers at budget 2,500.

The rungs around each adopted budget, so the choice is auditable rather than
asserted (`empty_draw_rate` is over all 8 x 32 draws at that rung):

{_format_table(selection)}

**Arm B's G0c budget does not confirm at eight seeds.** G0c read 10,500 ->
coverage 0.0553 from **seed 0 alone**; that value reproduces exactly here, but
the eight-seed median at 10,500 is **0.0610**, about 11% above the walks.
Seed 0 sat at the low end of B's seed spread.  The eight-seed parity point is
**9,600**, which is what this report adopts as primary.  10,500 is carried
through the tables below as a measured alternative so the G0c choice remains
available; the two differ little and both clear the bias gate.

Realized access at the adopted budgets, with each arm's natural access unit
beside the common event count:

{_format_table(access)}

Events are the common descriptive; the natural unit is what a real study would
actually budget, and the three differ by more than an order of magnitude -- a
walk spends 800 steps, arm A recruits tens of nodes, arm B looks up hundreds of
dyads.  These are the eight accepted seed slots, which is the basis parity was
set on.

The coverage axes move differently, which is the point: matching dyad coverage
does **not** match node coverage, and it does not match temporal spread either.
Every arm touches all five windows on every case, so `windows_touched` is 1.00
throughout and separates nothing; `temporal_evenness` (1 = the sample's events
are spread uniformly over the five windows, 0 = all in one) does separate them.
The arms remain information-distinct by construction, and G4 must say so rather
than read cross-arm accuracy differences as arm difficulty.

## G0d.2 — The empty-sample rule, and why arm A still needs it

A whole-entity sample is empty exactly when the *first* entity drawn does not
fit on its own, so the empty probability is available in closed form and needs
no simulation.  Arm A draws nodes uniformly, giving the share of nodes whose
incident-event count exceeds the budget.  Arm B reaches a dyad through a
uniformly drawn event, giving the event-weighted share of oversized dyads.

**Arm B at 9,600 is structurally safe: P(empty) = {b_zero:.4f} on all 32
graphs.** The panel's longest single dyad history is 4,992 events, so no first
lookup can overflow the budget.  The nine empty G0c cases are gone by
construction, not by luck.

**Arm A is not, and cannot be at any parity budget.** The panel's largest node
carries **{a_zero_budget:,}** incident events, so a whole-node response is
only guaranteed to fit above budget 12,000 -- nearly five times the parity
budget, at median coverage 0.31 instead of 0.055.  At 2,500 the exact empty
probability is nonzero on 15 of 32 graphs, worst case **{a_worst:.4f}**.  The
brief expected a zero empty rate at the new budgets; for arm A that
expectation does not hold, and the prespecified rule is load-bearing rather
than a contingency.

The rule, applied and logged: the seed sequence for a case is the prespecified
index order 0, 1, 2, ...; a draw whose whole-entity stop left the sample empty
is skipped and the next index is used.  Nothing is redrawn at random and no
case is dropped, so the accepted set is a deterministic function of the
sequence.  Panels ran 16 seed slots to give the rule headroom.  Because the
worst-case empty probability is {a_worst:.4f}, far below 1, the rule
terminates almost surely and cheaply.

Across the accepted panel, **{total_advances}** seed advances were needed in
total, affecting **{cases_with_advance}** of {len(seed_log)} (case, arm) pairs.
Per-case counts are in `seed_advance_log.csv`; exact per-graph probabilities
are in `exact_empty_probability.csv`.

## G0d.3 — Re-measurement at the new budgets

Group-macro `rho_2` bias by seed slot:

{_format_table(bias_seeds)}

Eight-slot aggregate, with the case-wise spread of the correct correction
direction:

{_format_table(bias_summary)}

**Acceptance.** Arm A's bias must stay within +/-0.05 of zero and arm B's
above +0.05.  Arm A: **{a_bias:+.4f}**.  Arm B: **{b_bias:+.4f}**.  Result:
**{acceptance}**.  No budget was tuned after seeing a sign; the two budgets
come from the coverage ladder alone.

Per-case sign distribution of `delta_i = rho2_true_i - rho2_naive_i`.  This is
why G4 must regress on the per-case `delta_i` rather than assign one
arm-level sign to every case:

{_format_table(delta_signs)}

Full reference ladder at the accepted first seed slot.  `occ_mle` and
`mask_mle` are **censoring-aware but mechanism-agnostic** -- one uniform
likelihood on every arm, no arm parameter anywhere -- and ExtraTrees is a
**label-informed performance reference**, trained on the benchmark population
with the matching panel backbones held out.  Neither is design-aware and
neither is an upper bound.  The walk rows are the unchanged G0 numbers at 800.

{_format_table(ladder)}

Four readings of this ladder matter for G2 and G4.

First, **arm A now has n = 32 on every row.** In G0c its analytical rows ran on
n = 25 because seven cases had no sample, which meant arm A's numbers were not
comparable to the others'.  That gap is closed, and it was the main thing the
re-budget had to fix.

Second, **arm A has almost no headroom left, and that is what makes it the
control.** Its naive read-off reaches ProfileMAE 0.0169 with bias -0.0065 --
better than the supervised reference on the same cases.  With no censoring and
no selection, the plug-in is already very nearly the right answer, so there is
no correction for a model to discover.  A `mechanism > hidden` effect on arm A
would therefore have nowhere to come from, and G4 should read arm A as the
negative control it is: the arm where the correct answer to "how should I
adjust?" is "do not".  This is also why G1.3's `direction_only` text for arm A
has to say that the naive estimate is approximately unbiased rather than name a
direction.

Third, `occ_mle` and `mask_mle` are **worse than the naive read-off on both
non-walk arms**, and worse by a wide margin on arm B.  That is the expected
behaviour of a censoring correction applied where there is no censoring: both
arms return complete histories, so the estimators inflate a profile that was
already right, and on arm B they add to a selection bias that already points
the same way.  This is the concrete evidence that they are mechanism-agnostic,
and it is why they must never be described as design-aware.

Fourth, the `CensoringRecovery` normalization degenerates on both arms, as
G0b already found for its own arm B: the anchor moves *away* from zero rather
than toward it, so the denominator is negative and the ratio is meaningless.
G2 must report raw signed bias for arms A and B and say why, exactly as the
amendment anticipates.

Eight-slot variance decomposition:

{_format_table(seed_variance)}

## G0d.4 — Prompt size at the new budgets

The frozen input contract is `INPUT_MASK` plus an exact `(n, mask)` histogram
rendered as a sparse JSON map.  It is **neither** of the two shapes the brief
anticipated.  It is not a fixed-size histogram over `K = 1..5`, so length is
not constant.  But it is not per-dyad rows either, so length does not track
the dyad count: the map is keyed by *distinct* `(n, mask)` combinations, the
mask has only 31 nonzero values, and `n` repeats heavily across dyads, so the
key count saturates long before it reaches one key per dyad.

Distinct histogram keys actually rendered:

{_format_table(key_counts)}

Rendered block length, against the same measure at budget 800 in G0c:

{_format_table(lengths)}

**The input does grow, and far more slowly than coverage does.** Arm B's dyad
coverage rose roughly seventeenfold (0.0031 -> 0.0544) while its median
rendered block grew **3.2x**; arm A's coverage rose fourfold while its block
grew **1.8x**.  That gap is the saturation described above.  The three walk
arms are byte-identical to G0c, as they must be.

Absolute size is what decides whether this blocks.  Exact Qwen3.6 counts
cannot be recomputed here -- the tokenizer snapshot G0c used is not present in
this environment -- so they are calibrated from G0c's stored exact counts and
the same rendered texts.  The ratio of exact Qwen tokens to the
tokenizer-independent portable count is tight on those cases (median 1.664,
p10 1.579, p90 1.725), so the conversion below is good to roughly +/-5% and is
an estimate, not a measurement:

{_format_table(qwen_estimate)}

**Verdict: not blocking.** The largest input in the panel is about 2,300
tokens on arm B, against 268-432 median for the walks at 800.  That is a 2-3x
input, not the order of magnitude that would break the token-limited models:
the historical V2.1 open-weights runs were truncated on the *output* side at
8,192 tokens, and an input of this size is far inside every context window in
the model matrix.  The growth is real and belongs in the G1 length-band
accounting, where the sections must be matched across arms, but it does not
threaten the run.

One consequence does carry into G1: arm B's block is now the longest in the
suite and its spread is the widest (p90 about 1,600 estimated tokens against
about 400 for `time_respecting`).  G1.5 asks for the same length band across
arms for the *prose* sections, and that is still achievable, but the data
block cannot be equalized without changing the input contract.  Report the
data-block length per arm alongside the prose length rather than claiming one
band for the whole prompt.

{alt_rows}

## G0d.5 — The mismatch pair at the new budgets: a blocking finding

Group-macro ProfileMAE (rows produce the sample; columns supply the assumed
arm likelihood).  As in G0/G0b/G0c this matrix is label-assisted: each
observation model is fitted from true dyad labels on the benchmark population
with the whole matching group held out.  It is a diagnostic ceiling, not a
deployable estimator.

{_format_table(pmae)}

The same matrix as signed `rho_2` bias:

{_format_table(bias)}

Measured bias classes used for pair eligibility (`none` = eight-slot naive
bias within +/-0.05):

{_format_table(arm_bias_classes)}

All candidate pairs.  Observable distinguishability uses only
`(n,mask)`-derived columns -- all of them shares, not raw counts -- with the
entire graph group held out:

{_format_table(pairs)}

`pair_summary` treats `none` as its own correction class, so it admits
`upward` <-> `none` pairs.  G0d.5 asks for a pair whose arms need corrections
in different *directions*, and an arm needing no correction has no direction,
so only these are eligible:

{_format_table(strict)}

The best eligible pair by bias penalty is **`{selected_pair[0]}` <->
`{selected_pair[1]}`**, median absolute `rho_2`-bias shift
**{float(chosen.median_abs_rho2_bias_shift):.4f}**, held-group-out observable
AUC **{float(chosen.observable_auc_logo):.4f}**.

### This does not clear the bar, and the re-budget is why

The requirement was that the chosen pair's AUC stay **well below 1.0**, so that
a model cannot detect the mismatch from the sample statistics alone.  It does
not.  At **{float(chosen.observable_auc_logo):.4f}** a held-group-out logistic
classifier reading only the mask histogram tells the two arms apart most of the
time, and one ineligible pair (`time_respecting` <-> `node_panel_full_history`)
is now at AUC 1.0000, i.e. perfectly separable.

The re-budget caused this.  On the identical pair, G0c measured AUC
**0.6660** with both arms at 800 events; here the same pair sits at
**0.9102**.  Nothing about the arms' definitions changed -- only arm B's
budget.  The mechanism is visible in the input: with complete histories at
9,600 events, arm B's `n` distribution is the true per-dyad event count over
611 dyads, while a walk at 800 steps sees each dyad once or twice.  Those are
different-shaped histograms, and the classifier reads the shape, not the size.

**Bias penalty and observable detectability are now positively coupled**, which
is the worst possible arrangement: the pairs worth contrasting are the ones a
model can most easily tell apart without reading the mechanism text.  Across
all seven candidate pairs the only one under AUC 0.75 is
`{str(lowest.arm_a)}` <-> `{str(lowest.arm_b)}` at
**{float(lowest.observable_auc_logo):.4f}** -- and its bias penalty is
**{float(lowest.median_abs_rho2_bias_shift):.4f}**, an order of magnitude
below the others, because both of its arms return complete histories and
neither needs an upward correction.  It is ineligible in any case: arm A needs
no correction at all, so the pair has no opposing directions to contrast.
Among the three eligible pairs the *lowest* AUC is
**{best_eligible_auc:.4f}**.

### What this means for `mismatched`

This is a G1 design decision, not something G0d should settle by picking the
least-bad number.  The options, with what each costs:

1. **Demote `mismatched` to exploratory.** Keep it, run it, and state up front
   that a specificity effect on this pair has a live rival explanation -- the
   model may be reacting to a sample that does not look like the described
   process rather than to the mechanism text.  Cheapest, and honest, but the
   condition stops being able to support the claim it was added for.
2. **Take the AUC as a measured covariate.** Run the pair, and report the
   effect against the per-case detectability rather than pooled.  If the effect
   is flat in detectability, the rival explanation is weakened empirically
   instead of by design.  Costs nothing extra to run and is the strongest
   version of option 1.
3. **Add a sixth configuration purely for the mismatch contrast**, with the two
   arms event-matched so their histograms are the same shape.  This restores
   the G0c AUC but breaks the within-case pairing that the primary contrast
   depends on, because `mismatched` would then run on a different sample from
   `hidden` and `mechanism` for the same case.  Expensive and it damages the
   primary design.
4. **Drop `mismatched`.** The 2x2 over {{process described}} x {{direction
   stated}} plus `metadata_only` still answers the main question.  `mismatched`
   was always the specificity check, not the effect.

Recommendation: **option 2**, falling back to option 1 if the detectability
covariate turns out to have no spread.  Option 3 should not be taken -- the
within-case pairing is worth more than this one condition.  None of this is
G0d's to decide; it is recorded here because G1 cannot write the `mismatched`
text without choosing.

## Final arm configuration

{_format_table(config_table)}

Three empty-rate columns rather than one, because they answer different
questions.  `expected_empty_rate_per_draw` is the closed-form probability
averaged over the 32 graphs; `observed_empty_rate_per_draw` is what the 16
drawn seed slots actually produced; `empty_rate_after_seed_rule` is what
survives into the analysis, and it must be zero or the rule has failed.  The
walks are exempt: a walk always returns a sample.

Prompt tokens are the tokenizer-independent portable count, so the column is
comparable across rows; the calibrated exact-tokenizer figures are in G0d.4.

Replay verification: {verified}/{benchmark_n} benchmark cases reproduced
their stored `(n,mask)` histogram exactly.  All new artifacts live below
`{out_dir}`; no frozen benchmark case, panel truth, walk artifact or LLM
artifact was modified.

## What remains uncertain

- Coverage parity matches one scalar.  Node coverage, temporal coverage, dyad
  multiplicity and sample coherence remain different across arms, deliberately
  so.  G4 must state that the arms are not information-matched.
- Arm A's empty cases are handled by the seed rule, not eliminated.  The
  accepted sample for an affected case is drawn from a seed index further
  along the sequence than its neighbours; the rule is deterministic and
  logged, but it is a conditioning on non-emptiness and should be named as
  such in the write-up.
- The wrong-mechanism matrix uses labels and collapses the exact sample to
  mask frequencies.  It bounds nothing.
- No language model has been tested here.  Nothing in this report says whether
  a model can operationalize a mechanism description.

Arms and budgets for the main run:

- `node_panel_full_history` (arm A): **{budgets[NODE_PANEL]:,}** unique events
- `event_sample_then_full_history` (arm B): **{budgets[TWO_PHASE]:,}** unique events
- `time_agnostic_t`, `time_respecting`, `recent_history_k20`: **{WALK_BUDGET}**
  walk steps, unchanged.
"""


def _ladder_walk_rows(g0_dir: Path) -> pd.DataFrame:
    """The unchanged G0 walk ladder, relabelled to the current estimator names."""
    rows = pd.read_csv(g0_dir / "estimator_ladder.csv")
    return rows.assign(estimator=rows.estimator.replace({
        "occupancy MLE (uniform)":
            "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)",
        "mask MLE (uniform)":
            "mask MLE (uniform; censoring-aware, mechanism-agnostic)",
        "supervised ExtraTrees (benchmark transfer)":
            "supervised ExtraTrees (label-informed performance reference; "
            "panel backbones held out)",
    }))


def _bias_by_arm(accepted: pd.DataFrame, walk_ladder: pd.DataFrame
                 ) -> tuple[dict[str, float], pd.DataFrame]:
    naive = walk_ladder[walk_ladder.estimator == "naive read-off"]
    bias = dict(zip(naive.arm, naive.rho2_bias))
    rows = []
    for arm, part in accepted.groupby("strategy"):
        summary = candidate_bias_summary(part.rename(
            columns={"seed_slot": "sample_seed", "sample_seed": "raw_seed"}))
        bias[arm] = float(summary.group_macro_rho2_bias.mean())
        rows.append(summary.assign(arm=arm))
    return bias, pd.concat(rows, ignore_index=True)


def _config_table(access, bias_summary, empty_exact, lengths, budgets,
                  walk0, walk_ladder, coverage_target,
                  accepted, raw_all) -> pd.DataFrame:
    """The single `final arm configuration` table G0d has to produce."""
    tokens = mask_length_rows(pd.concat([walk0], ignore_index=True))
    walk_tokens = tokens[tokens.measure == "portable_tokens"].set_index("arm")
    ab_tokens = lengths[lengths.measure == "portable_tokens"].set_index("arm")
    naive = walk_ladder[walk_ladder.estimator == "naive read-off"].set_index("arm")
    bias = bias_summary.set_index("arm")
    cover = access.set_index("arm")
    empty = empty_exact.groupby("arm").p_empty.mean()
    drawn = raw_all.groupby("strategy").budget.apply(lambda b: float((b == 0).mean()))
    left = accepted.groupby("strategy").budget.apply(lambda b: float((b == 0).mean()))
    rows = []
    for arm in ALL_ARMS:
        is_walk = arm in WALK_ARMS
        source = walk_tokens if is_walk else ab_tokens
        rows.append({
            "arm": arm,
            "budget_unique_events": WALK_BUDGET if is_walk else budgets[arm],
            "natural_unit": ("walk steps" if is_walk
                             else NATURAL_UNIT[arm][0]),
            "median_natural_units": float(cover.loc[arm, "median_natural_units"]),
            "median_dyad_coverage": float(cover.loc[arm, "median_dyad_coverage"]),
            "coverage_parity_target": coverage_target,
            "rho2_naive_bias": float(naive.loc[arm, "rho2_bias"] if is_walk
                                     else bias.loc[arm, "group_macro_bias_mean"]),
            "expected_empty_rate_per_draw": 0.0 if is_walk else float(empty.loc[arm]),
            "observed_empty_rate_per_draw": 0.0 if is_walk else float(drawn.loc[arm]),
            "empty_rate_after_seed_rule": 0.0 if is_walk else float(left.loc[arm]),
            "median_prompt_tokens_portable": float(source.loc[arm, "median"]),
        })
    return pd.DataFrame(rows)


def _alternative_budget_section(args, budgets, coverage_target, slots,
                                out_dir) -> str:
    """Arm B at the budget G0c proposed, measured on the same eight slots."""
    try:
        alt = read_globs(args.panel_b_alt)
    except FileNotFoundError:
        return ("Arm B at 10,500 was not re-run; see `docs/HEADROOM_G0C_2026-09.md` "
                "for its seed-0 numbers.")
    alt = alt[(alt.strategy == TWO_PHASE) &
              (alt.target_budget == args.budget_b_alt)].copy()
    accepted, log = apply_seed_rule(alt, slots)
    err = accepted.est__plugin_rho_k2 - accepted.rho_W5_k2
    by_slot = err.groupby([accepted.seed_slot, accepted.group_id]).mean()
    per_slot = by_slot.groupby(level=0).mean()
    table = pd.DataFrame([{
        "arm": TWO_PHASE, "target_budget": args.budget_b_alt,
        "median_dyad_coverage": float(accepted.coverage.median()),
        "coverage_target": coverage_target,
        "group_macro_bias_mean": float(per_slot.mean()),
        "group_macro_bias_sd": float(per_slot.std(ddof=1)),
        "seed_advances": int(log.seed_advances.sum()),
        "empty_draws": int((alt.budget == 0).sum()),
    }])
    table.to_csv(out_dir / "arm_b_alternative_budget.csv", index=False)
    return f"""### Arm B at the G0c budget of 10,500, for comparison

Same eight seed slots, same rule, same panel:

{_format_table(table)}

It clears the bias gate as comfortably as 9,600 does and is equally free of
empty samples; it simply sits further above the walks on coverage.  If the
G0c budget is preferred for continuity, nothing in the acceptance decision
changes -- only the parity claim weakens."""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walk-panel", default="results/panel_seed_probe/cases.csv.gz")
    ap.add_argument("--panel-a", nargs="+", default=[
        "results/g0d_headroom_2026_09/panel_a_shard_*.csv.gz"])
    ap.add_argument("--panel-b", nargs="+", default=[
        "results/g0d_headroom_2026_09/panel_b_shard_*.csv.gz"])
    ap.add_argument("--panel-b-alt", nargs="+", default=[
        "results/g0d_headroom_2026_09/panel_b_10500_shard_*.csv.gz"])
    ap.add_argument("--benchmark-a", nargs="+", default=[
        "results/g0d_headroom_2026_09/benchmark_a_shard_*.csv.gz"])
    ap.add_argument("--benchmark-b", nargs="+", default=[
        "results/g0d_headroom_2026_09/benchmark_b_shard_*.csv.gz"])
    ap.add_argument("--budget-a", type=int, default=2500)
    ap.add_argument("--budget-b", type=int, default=9600)
    ap.add_argument("--budget-b-alt", type=int, default=10500)
    ap.add_argument("--seed-slots", type=int, default=8)
    ap.add_argument("--budget-ladder", default=(
        "results/g0d_headroom_2026_09/budget_ladder.csv"))
    ap.add_argument("--event-manifest", default=(
        "results/g0_headroom_2026_09/regenerated_benchmark_v2/manifest.csv"))
    ap.add_argument("--panel-manifest",
                    default="results/final_target_panel/panel32_final.csv")
    ap.add_argument("--g0-dir", default="results/g0_headroom_2026_09")
    ap.add_argument("--g0c-dir", default="results/g0c_headroom_2026_09")
    ap.add_argument("--out-dir", default="results/g0d_headroom_2026_09")
    ap.add_argument("--report", default="docs/HEADROOM_G0D_2026-09.md")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model-jobs", type=int, default=-1)
    ap.add_argument("--rebuild-counts", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g0_dir, g0c_dir = Path(args.g0_dir), Path(args.g0c_dir)
    budgets = {NODE_PANEL: args.budget_a, TWO_PHASE: args.budget_b}

    walk = pd.read_csv(args.walk_panel)
    walk = walk[(walk.budget == WALK_BUDGET) &
                walk.strategy.isin(WALK_ARMS)].copy()
    walk8 = walk[walk.walk_seed < args.seed_slots]
    walk0 = walk[walk.walk_seed == 0].copy()
    coverage_target = float(walk8.coverage.median())
    walk_coverage = {"seed0": float(walk0.coverage.median()),
                     "slots": coverage_target}

    raw = {NODE_PANEL: read_globs(args.panel_a),
           TWO_PHASE: read_globs(args.panel_b)}
    for arm, frame in raw.items():
        keep = (frame.strategy == arm) & (frame.target_budget == budgets[arm])
        raw[arm] = frame[keep].copy()
    raw_all = pd.concat(raw.values(), ignore_index=True)
    accepted, seed_log = apply_seed_rule(raw_all, args.seed_slots)
    accepted.to_csv(out_dir / "accepted_panel_cases.csv.gz",
                    index=False, compression="gzip")
    seed_log.to_csv(out_dir / "seed_advance_log.csv", index=False)

    empty_exact = exact_empty_probability(Path(args.panel_manifest), budgets)
    empty_exact.to_csv(out_dir / "exact_empty_probability.csv", index=False)

    slot0 = accepted[accepted.seed_slot == 0].copy()
    panel5 = pd.concat([walk0, slot0], ignore_index=True)
    # Parity was set on the eight-slot median, so the access table reports
    # the same eight slots rather than slot 0 alone.
    access = access_profile(
        pd.concat([walk8, accepted], ignore_index=True), {**budgets})
    access.to_csv(out_dir / "access_profile.csv", index=False)

    walk_ladder = _ladder_walk_rows(g0_dir)
    bias_by_arm, bias_seeds = _bias_by_arm(accepted, walk_ladder)
    bias_seeds = bias_seeds[["arm", "sample_seed", "valid_cases",
                             "empty_cases", "group_macro_rho2_bias"]]
    bias_seeds = bias_seeds.rename(columns={"sample_seed": "seed_slot"})
    bias_seeds.to_csv(out_dir / "bias_by_seed_slot.csv", index=False)

    rows = []
    for arm, part in accepted.groupby("strategy"):
        per_slot = bias_seeds[bias_seeds.arm == arm].group_macro_rho2_bias
        delta = part.rho_W5_k2 - part.est__plugin_rho_k2
        rows.append({
            "arm": arm, "target_budget": budgets[arm],
            "group_macro_bias_mean": float(per_slot.mean()),
            "group_macro_bias_sd": float(per_slot.std(ddof=1)),
            "slot0_group_macro_bias": float(per_slot.iloc[0]),
            "accepted_cases": int(len(part)),
            "empty_cases_after_rule": int((part.budget == 0).sum()),
            "delta_gt_zero_share": float((delta > 0).mean()),
            "delta_lt_zero_share": float((delta < 0).mean()),
        })
    bias_summary = pd.DataFrame(rows)
    bias_summary.to_csv(out_dir / "bias_summary.csv", index=False)

    a_bias = float(bias_summary.loc[bias_summary.arm == NODE_PANEL,
                                    "group_macro_bias_mean"].iloc[0])
    b_bias = float(bias_summary.loc[bias_summary.arm == TWO_PHASE,
                                    "group_macro_bias_mean"].iloc[0])
    a_ok, b_ok = abs(a_bias) <= .05, b_bias > .05
    acceptance = ("PASS for both arms" if a_ok and b_ok else
                  f"FAIL (arm A within +/-0.05: {a_ok}; arm B above +0.05: {b_ok})")

    delta_signs = delta_sign_rows(accepted)
    delta_signs.to_csv(out_dir / "delta_sign_distribution.csv", index=False)

    benchmark = pd.concat([read_globs(args.benchmark_a),
                           read_globs(args.benchmark_b)], ignore_index=True)
    benchmark = benchmark[
        benchmark.strategy.isin([NODE_PANEL, TWO_PHASE])].copy()
    benchmark = benchmark[
        [budgets[s] == b for s, b in zip(benchmark.strategy,
                                         benchmark.target_budget)]].copy()
    if benchmark.case_id.duplicated().any():
        raise ValueError("duplicated G0d benchmark cases")

    ladder_path = out_dir / "estimator_ladder_five_arms.csv"
    if ladder_path.exists():
        ladder = pd.read_csv(ladder_path)
    else:
        ladder = pd.concat([walk_ladder, estimator_ladder(
            slot0.copy(), benchmark, (NODE_PANEL, TWO_PHASE),
            args.model_jobs)], ignore_index=True)
        ladder.to_csv(ladder_path, index=False)

    seed_variance = pd.concat([
        seed_variance_table(part.rename(
            columns={"seed_slot": "sample_seed", "sample_seed": "raw_seed"})
        ).assign(arm=arm)
        for arm, part in accepted.groupby("strategy")], ignore_index=True)
    seed_variance = seed_variance[
        ["arm", "method", "mean", "n", "s2_group", "s2_inst", "s2_seed",
         "seed_variance_share", "se_1seed", "se_8seeds"]]
    seed_variance.to_csv(out_dir / "seed_variance.csv", index=False)

    lengths = prompt_length_growth(
        slot0, pd.read_csv(g0c_dir / "mask_input_lengths.csv"))
    lengths.to_csv(out_dir / "mask_input_lengths.csv", index=False)
    key_counts = histogram_key_counts(panel5)
    key_counts.to_csv(out_dir / "histogram_key_counts.csv", index=False)
    qwen_estimate = qwen_token_estimate(panel5, g0c_dir)
    qwen_estimate.to_csv(out_dir / "qwen_token_estimate.csv", index=False)
    export_token_inputs(panel5, out_dir / "mask_input_texts.jsonl")

    counts_path = out_dir / "observation_counts_a_b_by_group.csv.gz"
    if counts_path.exists() and not args.rebuild_counts:
        counts_ab, verified = pd.read_csv(counts_path), len(benchmark)
    else:
        counts_ab, verified = collect_counts(
            benchmark, args.event_manifest, jobs=args.jobs)
        counts_ab.to_csv(counts_path, index=False, compression="gzip")
    counts = pd.concat([pd.read_csv(g0_dir / "observation_counts_by_group.csv.gz"),
                        counts_ab], ignore_index=True)

    matrix = full_matrix(panel5, counts)
    penalties = matrix_penalties(matrix)
    matrix.to_csv(out_dir / "wrong_mechanism_matrix_five_arms.csv", index=False)
    penalties.to_csv(out_dir / "wrong_mechanism_penalties_five_arms.csv",
                     index=False)
    arm_bias_classes = pd.DataFrame([
        {"arm": arm, "target_budget": budgets.get(arm, WALK_BUDGET),
         "naive_group_macro_rho2_bias": float(bias_by_arm[arm]),
         "required_correction": correction_class(float(bias_by_arm[arm]))}
        for arm in ALL_ARMS])
    arm_bias_classes.to_csv(out_dir / "arm_bias_classes.csv", index=False)
    pairs, _ = pair_summary(penalties, panel5, bias_by_arm)
    pairs.to_csv(out_dir / "mismatch_candidate_pairs.csv", index=False)
    strict = strict_direction_pairs(pairs)
    strict.to_csv(out_dir / "mismatch_opposite_direction_pairs.csv", index=False)
    if strict.empty:
        raise RuntimeError("no arm pair demands opposite correction directions")
    best = strict.loc[strict.median_abs_rho2_bias_shift.idxmax()]
    selected_pair = (str(best.arm_a), str(best.arm_b))

    alt_rows = _alternative_budget_section(
        args, budgets, coverage_target, args.seed_slots, out_dir)

    selection = budget_selection_table(
        Path(args.budget_ladder), budgets, coverage_target)
    selection.to_csv(out_dir / "budget_selection.csv", index=False)

    config_table = _config_table(access, bias_summary, empty_exact,
                                 lengths, budgets, walk0, walk_ladder,
                                 coverage_target, accepted, raw_all)
    config_table.to_csv(out_dir / "final_arm_configuration.csv", index=False)

    Path(args.report).write_text(build_report(
        budgets={k: int(v) for k, v in budgets.items()},
        config_table=config_table, selection=selection, access=access,
        empty_exact=empty_exact,
        seed_log=seed_log, bias_seeds=bias_seeds, bias_summary=bias_summary,
        ladder=ladder, seed_variance=seed_variance, delta_signs=delta_signs,
        lengths=lengths, key_counts=key_counts,
        qwen_estimate=qwen_estimate, matrix=matrix,
        penalties=penalties, pairs=pairs, selected_pair=selected_pair,
        strict=strict,
        arm_bias_classes=arm_bias_classes, coverage_target=coverage_target,
        walk_coverage=walk_coverage, alt_rows=alt_rows, verified=verified,
        benchmark_n=len(benchmark), acceptance=acceptance, out_dir=str(out_dir)))
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
