#!/usr/bin/env python3
"""G0b audit: add one opposite-direction access arm to the G0 design.

This script only augments the cached G0 results.  It never rewrites frozen V2
cases or the final panel.  Candidate cases and their regenerated benchmark
cases must already have been produced under a new results directory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import glob
import json
import math
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pandas as pd

from benchmark_features import edge_observations
from build_benchmark_data import stable_seed
from census import window_index
from make_llm_prompts_v2 import INPUT_MASK
from nonwalk_samplers import (
    activity_proportional_dyad_full_history,
    prepare_dyad_histories,
    prepare_events,
)
from report_g0_headroom import (
    ARMS as WALK_ARMS,
    KS,
    MASKS,
    PROFILE_KS,
    TRUTH,
    _event_path,
    _format_table,
    _read_cases,
    add_panel_logo_floor,
    arm_likelihood_profile,
    extra_trees_transfer,
    group_macro_k_weights,
    joint_counts_for_log,
    mask_histogram,
    metric_summary,
    observation_model,
    parse_nmask_hist,
    tv_table,
)
from report_seed_variance import decompose, profile_mae, se_macro


NODE_PANEL = "node_panel_full_history"
PPS_DYAD = "activity_proportional_dyad_full_history"
ALL_ARMS = (*WALK_ARMS, PPS_DYAD)


def read_globs(specs: list[str]) -> pd.DataFrame:
    paths = []
    for spec in specs:
        paths.extend(sorted(glob.glob(spec)))
    if not paths:
        raise FileNotFoundError(f"no files matched {specs}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def group_macro_mean(frame: pd.DataFrame, value: str,
                     seed_col: str | None = None) -> float:
    levels = ([seed_col, "group_id"] if seed_col else ["group_id"])
    grouped = frame.groupby(levels, as_index=False)[value].mean()
    return float(grouped[value].mean())


def signed_rho2_bias(frame: pd.DataFrame) -> pd.Series:
    return frame["est__plugin_rho_k2"] - frame["rho_W5_k2"]


def candidate_bias_rows(candidate: pd.DataFrame) -> pd.DataFrame:
    work = candidate.copy()
    work["rho2_error"] = signed_rho2_bias(work)
    rows = []
    for arm, part in work.groupby("strategy"):
        by_seed = (part.groupby(["sample_seed", "group_id"], as_index=False)
                   .rho2_error.mean().groupby("sample_seed").rho2_error.mean())
        rows.append({
            "arm": arm,
            "group_macro_bias_mean_8seeds": float(by_seed.mean()),
            "group_macro_bias_sd_8seeds": float(by_seed.std(ddof=1)),
            "seed0_group_macro_bias": float(by_seed.loc[0]),
            "pooled_case_bias": float(part.rho2_error.mean()),
            "positive_case_share": float((part.rho2_error > 0).mean()),
            "negative_case_share": float((part.rho2_error < 0).mean()),
            "qualifies_near_zero": bool(abs(by_seed.mean()) <= 0.05),
            "qualifies_opposite_to_walks": bool(by_seed.mean() > 0),
        })
    return pd.DataFrame(rows)


def choose_candidate(biases: pd.DataFrame) -> str | None:
    """Apply the prespecified B > A > C preference among qualifying arms."""
    qualifies = biases[
        biases["qualifies_near_zero"] | biases["qualifies_opposite_to_walks"]]
    for arm in (PPS_DYAD, NODE_PANEL, "temporal_prefix"):
        if arm in set(qualifies.arm):
            return arm
    return None


def distribution(values: pd.Series) -> dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    return {
        "median": float(x.median()), "mean": float(x.mean()),
        "p10": float(x.quantile(.1)), "p90": float(x.quantile(.9)),
        "min": float(x.min()), "max": float(x.max()),
    }


_TOKEN_RE = re.compile(
    r'"(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[A-Za-z_]+|[{}\[\]:,]')


def portable_token_count(text: str) -> int:
    """Tokenizer-independent lexical count used only for a length diagnostic."""
    return len(_TOKEN_RE.findall(text))


def mask_length_rows(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, part in panel.groupby("strategy"):
        lengths = []
        for raw in part["input__nmask_exact_json"].astype(str):
            text = f"{INPUT_MASK}\n{raw}"
            lengths.append({
                "characters": len(text),
                "utf8_bytes": len(text.encode("utf-8")),
                "portable_tokens": portable_token_count(text),
            })
        temp = pd.DataFrame(lengths)
        for measure in temp:
            stats = distribution(temp[measure])
            rows.append({"arm": arm, "measure": measure, **stats})
    return pd.DataFrame(rows)


def vectorized_true_window_counts(events: pd.DataFrame, W: int = 5,
                                  T: float = 1.0) -> dict[tuple[int, int], int]:
    """Exact K per dyad without constructing Python arrays for every edge."""
    x = events[["u", "v", "t"]].copy()
    x["window"] = window_index(x["t"].to_numpy(float), 0.0, T / W, W)
    counts = x.groupby(["u", "v"], sort=False)["window"].nunique()
    return {tuple(map(int, edge)): int(k) for edge, k in counts.items()}


def _replay_pps_instance(task):
    instance_id, group_id, event_path, records = task
    events = pd.read_csv(event_path)
    prepared = prepare_events(events)
    prepared_dyads = prepare_dyad_histories(prepared)
    idx = SimpleNamespace(T=1.0)
    true_k = vectorized_true_window_counts(prepared.events, W=5, T=1.0)
    aggregate = np.zeros((6, 32), dtype=np.int64)
    checked = 0
    for row in records:
        result = activity_proportional_dyad_full_history(
            prepared_dyads, int(row["target_budget"]), int(row["sample_rng_seed"]))
        if len(result.log) != int(row["budget"]):
            raise AssertionError(f"budget mismatch for {row['case_id']}")
        replay = Counter((int(x["n"]), int(x["mask"]))
                         for x in edge_observations(
                             result.log, len(result.log), W=5, T=idx.T))
        frozen = parse_nmask_hist(str(row["input__nmask_exact_json"]))
        if replay != frozen:
            raise AssertionError(f"PPS replay mismatch for {row['case_id']}")
        aggregate += joint_counts_for_log(
            result.log, idx, len(result.log), W=5, true_k=true_k)
        checked += 1
    output = []
    for k in KS:
        for mask in MASKS:
            output.append({"group_id": group_id, "arm": PPS_DYAD,
                           "K": k, "mask": mask,
                           "count": int(aggregate[k, mask])})
    return instance_id, checked, output


def collect_pps_counts(cases: pd.DataFrame, event_manifest: str,
                       jobs: int = 1) -> tuple[pd.DataFrame, int]:
    manifest_path = Path(event_manifest)
    manifest = pd.read_csv(manifest_path)
    lookup = {str(r.instance_id): _event_path(manifest_path, str(r.path))
              for r in manifest.itertuples(index=False)}
    tasks = []
    for instance_id, frame in cases.groupby("instance_id", sort=True):
        groups = frame.group_id.astype(str).unique()
        if len(groups) != 1:
            raise AssertionError(f"multiple groups for {instance_id}")
        tasks.append((str(instance_id), str(groups[0]),
                      str(lookup[str(instance_id)]), frame.to_dict("records")))
    totals = defaultdict(lambda: np.zeros((6, 32), dtype=np.int64))
    checked = 0
    if jobs == 1:
        iterator = map(_replay_pps_instance, tasks)
        pool = None
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        iterator = pool.map(_replay_pps_instance, tasks, chunksize=1)
    try:
        for pos, (_, n_checked, rows) in enumerate(iterator, 1):
            checked += n_checked
            for row in rows:
                totals[(row["group_id"], row["arm"])][
                    row["K"], row["mask"]] += row["count"]
            if pos % 50 == 0 or pos == len(tasks):
                print(f"[G0b replay] {pos}/{len(tasks)} instances; "
                      f"{checked}/{len(cases)} cases verified", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()
    rows = []
    for (group_id, arm), matrix in sorted(totals.items()):
        for k in KS:
            for mask in MASKS:
                rows.append({"group_id": group_id, "arm": arm, "K": k,
                             "mask": mask, "count": int(matrix[k, mask])})
    return pd.DataFrame(rows), checked


def pps_budget_for_coverage(events: pd.DataFrame, seed: int,
                            target_coverage: float) -> int:
    """A sufficient whole-dyad budget for a target dyad coverage."""
    prepared = prepare_events(events)
    sizes = (prepared.events.groupby(["u", "v"], sort=True)
             .size().to_numpy(float))
    rng = np.random.default_rng(seed)
    order = np.argsort(rng.exponential(scale=1.0 / sizes), kind="stable")
    needed = max(1, min(len(sizes), math.ceil(target_coverage * len(sizes))))
    return int(sizes[order[:needed]].sum())


def parity_budget_rows(panel_manifest: str, target_coverage: float,
                       base_seed: int = 20260831) -> pd.DataFrame:
    path = Path(panel_manifest)
    manifest = pd.read_csv(path)
    rows = []
    for row in manifest.itertuples(index=False):
        events = pd.read_csv(_event_path(path, str(row.path)))
        seed = stable_seed(base_seed, row.instance_id, PPS_DYAD, 0)
        rows.append({
            "instance_id": row.instance_id, "group_id": row.group_id,
            "target_coverage": target_coverage,
            "sufficient_event_budget": pps_budget_for_coverage(
                events, seed, target_coverage),
        })
    return pd.DataFrame(rows)


def score_arm_likelihood_cell(sample: pd.DataFrame, assumed_arm: str,
                              cross_models: dict[str, dict[str, np.ndarray]]) -> dict:
    cols = [f"g0b__arm_likelihood_rho_k{k}" for k in PROFILE_KS]
    scored = sample.copy()
    predictions = []
    for row in scored.itertuples(index=False):
        hist = mask_histogram(str(row.input__nmask_exact_json))
        predictions.append(arm_likelihood_profile(
            hist, cross_models[str(row.group_id)][assumed_arm]))
    scored[cols] = np.asarray(predictions)
    return metric_summary(scored, cols)


def full_matrix(panel: pd.DataFrame, counts: pd.DataFrame,
                old_matrix: pd.DataFrame) -> pd.DataFrame:
    cross_models = {
        group: observation_model(counts, excluded_groups=[group], arms=ALL_ARMS)
        for group in sorted(panel.group_id.astype(str).unique())
    }
    records = old_matrix.to_dict("records")
    for sample_arm in ALL_ARMS:
        sample = panel[panel.strategy == sample_arm]
        for assumed_arm in ALL_ARMS:
            if sample_arm in WALK_ARMS and assumed_arm in WALK_ARMS:
                continue
            records.append({"sample_arm": sample_arm,
                            "assumed_arm": assumed_arm,
                            **score_arm_likelihood_cell(
                                sample, assumed_arm, cross_models)})
    result = pd.DataFrame(records)
    if len(result) != len(ALL_ARMS) ** 2:
        raise AssertionError("extended wrong-mechanism matrix is incomplete")
    return result


def matrix_penalties(matrix: pd.DataFrame) -> pd.DataFrame:
    diagonal = matrix[matrix.sample_arm == matrix.assumed_arm].set_index("sample_arm")
    rows = []
    for row in matrix[matrix.sample_arm != matrix.assumed_arm].itertuples(index=False):
        base = diagonal.loc[row.sample_arm]
        shift = float(row.rho2_bias - base.rho2_bias)
        rows.append({
            "sample_arm": row.sample_arm, "assumed_arm": row.assumed_arm,
            "profile_mae_penalty": float(row.profile_mae - base.profile_mae),
            "rho2_bias_shift": shift, "abs_rho2_bias_shift": abs(shift),
        })
    return pd.DataFrame(rows)


def mismatch_pairing(penalties: pd.DataFrame) -> tuple[pd.DataFrame, tuple[str, str]]:
    rows = []
    for walk in WALK_ARMS:
        pair = penalties[
            ((penalties.sample_arm == walk) & (penalties.assumed_arm == PPS_DYAD)) |
            ((penalties.sample_arm == PPS_DYAD) & (penalties.assumed_arm == walk))]
        if len(pair) != 2:
            raise AssertionError(f"missing bidirectional mismatch cells for {walk}")
        rows.append({
            "walk_arm": walk,
            "other_arm": PPS_DYAD,
            "median_profile_mae_penalty": float(pair.profile_mae_penalty.median()),
            "median_abs_rho2_bias_shift": float(pair.abs_rho2_bias_shift.median()),
            "min_abs_rho2_bias_shift": float(pair.abs_rho2_bias_shift.min()),
            "max_abs_rho2_bias_shift": float(pair.abs_rho2_bias_shift.max()),
        })
    summary = pd.DataFrame(rows)
    best = summary.loc[summary.median_abs_rho2_bias_shift.idxmax()]
    return summary, (str(best.walk_arm), PPS_DYAD)


def seed_variance_rows(panel_b: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "naive read-off": [f"est__plugin_rho_k{k}" for k in PROFILE_KS],
        "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__occ_mle_rho_k{k}" for k in PROFILE_KS],
        "mask MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__mask_mle_rho_k{k}" for k in PROFILE_KS],
    }
    rows = []
    for name, cols in methods.items():
        parts = decompose(panel_b, profile_mae(panel_b, cols))
        total = parts["s2_group"] + parts["s2_inst"] + parts["s2_seed"]
        rows.append({
            "method": name, **parts,
            "seed_variance_share": (parts["s2_seed"] / total if total else 0.0),
            "se_1seed": se_macro(parts, n_seed=1),
            "se_8seeds": se_macro(parts, n_seed=8),
        })
    return pd.DataFrame(rows)


def _fmt_distribution(label: str, stats: dict[str, float]) -> dict:
    return {"scope": label, **stats}


def build_report(*, candidate_bias: pd.DataFrame, spread: pd.DataFrame,
                 candidate_viability: pd.DataFrame,
                 budget_parity: pd.DataFrame, lengths: pd.DataFrame,
                 qwen_lengths: pd.DataFrame,
                 seed_variance: pd.DataFrame, tv: pd.DataFrame,
                 tv_observed: pd.DataFrame, matrix: pd.DataFrame,
                 penalties: pd.DataFrame, pair_summary: pd.DataFrame,
                 selected_pair: tuple[str, str], ladder: pd.DataFrame,
                 censoring_anchor: pd.DataFrame, verified: int,
                 benchmark_n: int, node_panel_checks: dict,
                 old_pair_score: float, out_dir: str) -> str:
    tv_wide = tv.pivot(index=["arm_a", "arm_b"], columns="K", values="tv").reset_index()
    tv_obs_wide = tv_observed.pivot(
        index=["arm_a", "arm_b"], columns="K", values="tv").reset_index()
    pmae = matrix.pivot(index="sample_arm", columns="assumed_arm",
                        values="profile_mae").reset_index()
    bias = matrix.pivot(index="sample_arm", columns="assumed_arm",
                        values="rho2_bias").reset_index()
    selected_score = float(pair_summary.loc[
        pair_summary.walk_arm == selected_pair[0],
        "median_abs_rho2_bias_shift"].iloc[0])
    replacement = selected_score > old_pair_score
    final_pair = selected_pair if replacement else (
        "time_respecting", "recent_history_k20")
    candidate_c_note = (
        "Candidate C was not run: it was optional, both priority candidates "
        "already qualified, and the rule permits adding at most one arm.")
    return f"""# G0b headroom audit: widening the signed-bias axis

Prepared: **2026-08-31**  
Gate status: **G0 + G0b complete; blocked pending joint confirmation. No G1 work has begun.**

## Scope and implementation checks

No LLM was called and no frozen artifact was modified. G0b added new case
files only below `{out_dir}`. Candidate A samples a random priority ordering of
all event-active nodes and reveals a selected node's complete incident event
record. On this panel the event-active universe equals the manifest node
universe for all 32 graphs ({node_panel_checks['matching_node_universes']}/32).
It stops before the first whole-node response that would exceed the budget;
there are no partial node responses, but the adaptive stopping time means the
fixed-p identity `1-(1-p)^2` is only a conceptual reference, not the exact
inclusion probability of this implementation. Candidate B uses an exact
exponential-race PPS ordering by full dyad event count. Whole histories that do
not fit the remaining budget are skipped, never truncated.

{candidate_c_note}

## G0b.1–2 — Candidate bias and mechanical selection

Signed naive read-off bias on `rho_2` (primary group-macro; candidate means
average the eight prespecified sample seeds):

{_format_table(candidate_bias)}

Bias spread for each candidate set:

{_format_table(spread)}

Both A (within 0.05 of zero) and B (opposite aggregate sign) qualify. The
prespecified priority `B > A > C` therefore selects **`{PPS_DYAD}`**. Its
aggregate correction on this panel is downward, whereas all three walk arms
require an upward correction.

## G0b.3 — Viability of the selected arm

Coverage and budget diagnostics:

{_format_table(candidate_viability)}

At seed 0 the selected arm's coverage is substantially below the walk panel's
historical median 0.056. The following is a sufficient whole-dyad event budget
for each graph to include the first 5.6% of dyads in the same PPS order; it is
reported only as a parity diagnostic and does **not** change budget 800:

{_format_table(budget_parity)}

Static mask-input length (`INPUT_MASK` plus exact histogram, before any G1
mechanism text):

{_format_table(lengths)}

`portable_tokens` is an explicitly tokenizer-independent lexical count, not a
Qwen/Codex token claim. Exact Qwen3.6-27B tokenizer counts for the same static
mask inputs (cached snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
`add_special_tokens=False`) are:

{_format_table(qwen_lengths)}

These are token counts for the fixed input block only. Exact **full-prompt**
counts cannot exist until G1 fixes the mechanism texts.

Eight-seed variance decomposition for the selected arm, using the same
implementation as `src/report_seed_variance.py`:

{_format_table(seed_variance)}

## G0b.4 — Widened observation-model audit

The new benchmark cases were replayed from regenerated event streams and
**{verified:,}/{benchmark_n:,}** exact `(n,mask)` histograms matched. Existing
walk counts and 3×3 cells were reused rather than regenerated.

Full `P(m | K, arm)` total-variation distances (mask 0 included):

{_format_table(tv_wide)}

Conditional on observation (`m != 0`):

{_format_table(tv_obs_wide)}

Label-assisted arm-likelihood ceiling, group-macro ProfileMAE:

{_format_table(pmae)}

The same matrix as signed `rho_2` bias:

{_format_table(bias)}

All off-diagonal changes relative to the matching diagonal:

{_format_table(penalties)}

This remains a **label-assisted diagnostic ceiling**, not a deployable
estimator and not an information-theoretic bound.

## Estimator ladder and naming correction

The uniform occupancy and mask MLEs are **censoring-aware,
mechanism-agnostic**. They receive no arm parameter and are not design-aware.

{_format_table(ladder)}

Signed-bias denominator of the renamed `CensoringRecovery` scale:

{_format_table(censoring_anchor)}

`CensoringRecovery = 1` anchors to the existing uniform mask MLE: it means
matching that estimator's recovery of uniform occupancy censoring on the arm,
not recovering the arm's observation design. A near-zero denominator or an
anchor that moves farther from zero is left unnormalized, and raw signed bias
must be used.

## G0b.5 — Mismatch pairing

Only pairs with opposite required correction directions are eligible:

{_format_table(pair_summary)}

The best new pair is **`{selected_pair[0]}` ↔ `{selected_pair[1]}`**, with a
median bidirectional absolute `rho_2`-bias shift of **{selected_score:.4f}**.
The old walk-only pair scored **{old_pair_score:.4f}** on the same bias-axis
summary. Therefore the new pair **{'replaces' if replacement else 'does not replace'}**
the old pair. Final mismatch pair: **`{final_pair[0]}` ↔ `{final_pair[1]}`**.

## Decision and limitations

**Arms for the main run: [`time_agnostic_t`, `time_respecting`,
`recent_history_k20`, `{PPS_DYAD}`].** Conditions retained: `hidden`,
`mechanism`, `mechanism_direction`, `metadata_only`, and `mismatched` only for
the final pair above.

The new arm cleanly widens the aggregate bias direction, but it is deliberately
synthetic and oracle-like: selection weights require full dyad event counts and
selected dyads return full histories. It is a specificity stimulus/control,
not a claimed real-world limited-access mechanism. Its much lower dyad
coverage at equal event budget is a genuine arm difference.

**Crucial qualification for the revised G4 wording:** positive recurrence bias
is not a label-free theorem of activity-PPS. In the with-replacement idealization
its sign is the sign of `Cov(event_count, I[K>=2])`; a K=1 dyad can have many
events, so the covariance can be negative. The observed direction is an
empirical panel-level fact, and individual cases can have the opposite sign
(see `negative_case_share` above). The planned direction contrast is therefore
an aggregate panel specificity test, not a casewise label-free ground truth.
G0b does not show that an LLM can use the mechanism description, and no prompt
wording has yet been approved.

## Stop

**STOP at G0b. Await explicit confirmation of G0 and G0b together before any
G1 prompt-contract work.**
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walk-panel", default="results/panel_seed_probe/cases.csv.gz")
    ap.add_argument("--candidate-a", nargs="+", default=[
        "results/g0b_headroom_2026_09/panel_candidates_shard_*.csv.gz"])
    ap.add_argument("--candidate-b", nargs="+", default=[
        "results/g0b_headroom_2026_09/panel_candidate_b_shard_*.csv.gz"])
    ap.add_argument("--benchmark-b", nargs="+", default=[
        "results/g0b_headroom_2026_09/benchmark_winner_shard_*.csv.gz"])
    ap.add_argument("--event-manifest", default=(
        "results/g0_headroom_2026_09/regenerated_benchmark_v2/manifest.csv"))
    ap.add_argument("--panel-manifest", default=(
        "results/final_target_panel/panel32_final.csv"))
    ap.add_argument("--g0-dir", default="results/g0_headroom_2026_09")
    ap.add_argument("--out-dir", default="results/g0b_headroom_2026_09")
    ap.add_argument("--qwen-token-counts", default=(
        "results/g0b_headroom_2026_09/mask_input_qwen36_tokens.csv"))
    ap.add_argument("--report", default="docs/HEADROOM_G0B_2026-09.md")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model-jobs", type=int, default=-1)
    ap.add_argument("--rebuild-counts", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g0_dir = Path(args.g0_dir)

    a = read_globs(args.candidate_a)
    a = a[(a.strategy == NODE_PANEL) & (a.target_budget == args.budget)].copy()
    b = read_globs(args.candidate_b)
    b = b[(b.strategy == PPS_DYAD) & (b.target_budget == args.budget)].copy()
    if len(a) != 32 * 8 or len(b) != 32 * 8:
        raise ValueError(f"expected 256 cases per priority candidate; got {len(a)}, {len(b)}")
    candidate = pd.concat([a, b], ignore_index=True)
    candidate_bias = candidate_bias_rows(candidate)
    winner = choose_candidate(candidate_bias)
    if winner != PPS_DYAD:
        raise RuntimeError(f"prespecified rule did not select expected winner: {winner}")
    candidate_bias.to_csv(out_dir / "candidate_bias.csv", index=False)

    old_ladder = pd.read_csv(g0_dir / "estimator_ladder.csv")
    old_ladder["estimator"] = old_ladder["estimator"].replace({
        "occupancy MLE (uniform)":
            "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)",
        "mask MLE (uniform)":
            "mask MLE (uniform; censoring-aware, mechanism-agnostic)",
    })
    old_naive = old_ladder[old_ladder.estimator == "naive read-off"]
    base_bias = dict(zip(old_naive.arm, old_naive.rho2_bias))
    a_bias = float(candidate_bias.loc[candidate_bias.arm == NODE_PANEL,
                                      "group_macro_bias_mean_8seeds"].iloc[0])
    b_bias = float(candidate_bias.loc[candidate_bias.arm == PPS_DYAD,
                                      "group_macro_bias_mean_8seeds"].iloc[0])
    sets = [
        ("three walks", list(base_bias.values())),
        ("three walks + candidate A", [*base_bias.values(), a_bias]),
        ("three walks + candidate B", [*base_bias.values(), b_bias]),
        ("three walks + A + B (diagnostic only)",
         [*base_bias.values(), a_bias, b_bias]),
    ]
    spread = pd.DataFrame([
        {"arm_set": name, "minimum_bias": min(values),
         "maximum_bias": max(values), "signed_bias_spread": max(values)-min(values)}
        for name, values in sets])
    spread.to_csv(out_dir / "candidate_bias_spread.csv", index=False)

    b0 = b[b.sample_seed == 0].copy()
    a0 = a[a.sample_seed == 0].copy()
    walk = pd.read_csv(args.walk_panel)
    walk = walk[(walk.budget == args.budget) & (walk.walk_seed == 0) &
                walk.strategy.isin(WALK_ARMS)].copy()
    if len(b0) != 32 or len(walk) != 96:
        raise ValueError("seed-0 panel is incomplete")
    viability = pd.DataFrame([
        _fmt_distribution("winner coverage, seed 0", distribution(b0.coverage)),
        _fmt_distribution("winner coverage, all 8 seeds", distribution(b.coverage)),
        _fmt_distribution("winner realized event budget", distribution(b.budget)),
        _fmt_distribution("winner event-budget slack", distribution(b.sample__budget_slack)),
        _fmt_distribution("candidate A coverage, seed 0", distribution(a0.coverage)),
        _fmt_distribution("candidate A event-budget slack", distribution(a.sample__budget_slack)),
        _fmt_distribution("walk coverage, seed 0", distribution(walk.coverage)),
    ])
    viability.to_csv(out_dir / "candidate_viability.csv", index=False)

    parity_path = out_dir / "pps_budget_for_walk_median_coverage.csv"
    if parity_path.exists():
        parity = pd.read_csv(parity_path)
    else:
        parity = parity_budget_rows(args.panel_manifest, target_coverage=.056)
        parity.to_csv(parity_path, index=False)
    parity_summary = pd.DataFrame([
        _fmt_distribution("sufficient budget for 0.056 dyad coverage",
                          distribution(parity.sufficient_event_budget))])

    panel4 = pd.concat([walk, b0], ignore_index=True)
    lengths = mask_length_rows(panel4)
    lengths.to_csv(out_dir / "mask_input_lengths.csv", index=False)
    qwen_raw = pd.read_csv(args.qwen_token_counts)
    if (qwen_raw.case_id.duplicated().any() or len(qwen_raw) != len(panel4) or
            set(qwen_raw.case_id) != set(panel4.case_id)):
        raise ValueError("Qwen static mask token counts do not match the 128 panel cases")
    qwen_rows = []
    for arm, part in qwen_raw.groupby("arm"):
        qwen_rows.append({"arm": arm, **distribution(part.qwen36_tokens)})
    qwen_lengths = pd.DataFrame(qwen_rows)
    seed_variance = seed_variance_rows(b)
    seed_variance.to_csv(out_dir / "winner_seed_variance.csv", index=False)

    benchmark_b = read_globs(args.benchmark_b)
    benchmark_b = benchmark_b[
        (benchmark_b.target_budget == args.budget) &
        (benchmark_b.strategy == PPS_DYAD)].copy()
    if benchmark_b.case_id.duplicated().any():
        raise ValueError("duplicate PPS benchmark case IDs")
    expected_benchmark = 745 * 4
    if len(benchmark_b) != expected_benchmark:
        raise ValueError(f"expected {expected_benchmark} PPS benchmark cases, got {len(benchmark_b)}")
    counts_path = out_dir / "observation_counts_b_by_group.csv.gz"
    if counts_path.exists() and not args.rebuild_counts:
        counts_b = pd.read_csv(counts_path)
        verified = len(benchmark_b)
    else:
        counts_b, verified = collect_pps_counts(
            benchmark_b, args.event_manifest, jobs=args.jobs)
        counts_b.to_csv(counts_path, index=False, compression="gzip")
    counts_walk = pd.read_csv(g0_dir / "observation_counts_by_group.csv.gz")
    counts = pd.concat([counts_walk, counts_b], ignore_index=True)

    models = observation_model(counts, arms=ALL_ARMS)
    weights = group_macro_k_weights(counts, arm=WALK_ARMS[0])
    tv = tv_table(models, weights, arms=ALL_ARMS)
    tv_observed = tv_table(models, weights, conditional_observed=True,
                           arms=ALL_ARMS)
    old_tv = pd.read_csv(g0_dir / "tv_full.csv")
    old_check = tv[tv.arm_a.isin(WALK_ARMS) & tv.arm_b.isin(WALK_ARMS)]
    merged = old_check.merge(old_tv, on=["arm_a", "arm_b", "K"],
                             suffixes=("_new", "_old"))
    if len(merged) != len(old_tv) or not np.allclose(merged.tv_new, merged.tv_old):
        raise AssertionError("cached walk TV values changed during G0b")
    tv.to_csv(out_dir / "tv_full_widened.csv", index=False)
    tv_observed.to_csv(out_dir / "tv_conditional_observed_widened.csv", index=False)

    old_matrix = pd.read_csv(g0_dir / "wrong_mechanism_matrix.csv")
    panel4 = pd.concat([walk, b0], ignore_index=True)
    matrix = full_matrix(panel4, counts, old_matrix)
    penalties = matrix_penalties(matrix)
    matrix.to_csv(out_dir / "wrong_mechanism_matrix_widened.csv", index=False)
    penalties.to_csv(out_dir / "wrong_mechanism_penalties_widened.csv", index=False)
    pair_summary, selected_pair = mismatch_pairing(penalties)
    pair_summary.to_csv(out_dir / "mismatch_opposite_direction_pairs.csv", index=False)

    floor_cols = add_panel_logo_floor(b0)
    et_cols = extra_trees_transfer(benchmark_b, b0, jobs=args.model_jobs,
                                   arms=[PPS_DYAD])
    methods = {
        "mean floor (panel LOGO)": floor_cols,
        "naive read-off": [f"est__plugin_rho_k{k}" for k in PROFILE_KS],
        "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__occ_mle_rho_k{k}" for k in PROFILE_KS],
        "mask MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__mask_mle_rho_k{k}" for k in PROFILE_KS],
        "supervised ExtraTrees (benchmark transfer)": et_cols,
    }
    b_ladder = pd.DataFrame([
        {"arm": PPS_DYAD, "estimator": name,
         **metric_summary(b0, cols)} for name, cols in methods.items()])
    ladder = pd.concat([old_ladder, b_ladder], ignore_index=True)
    ladder.to_csv(out_dir / "estimator_ladder_widened.csv", index=False)

    anchors = []
    for arm, part in ladder.groupby("arm"):
        naive = part[part.estimator == "naive read-off"].iloc[0]
        mask_rows = part[part.estimator.str.startswith("mask MLE")]
        mask = mask_rows.iloc[0]
        denominator = float(naive.rho2_bias - mask.rho2_bias)
        moves_toward_zero = (
            abs(float(mask.rho2_bias)) < abs(float(naive.rho2_bias)) and
            (float(mask.rho2_bias) - float(naive.rho2_bias)) *
            float(naive.rho2_bias) < 0)
        anchors.append({
            "arm": arm, "naive_rho2_bias": naive.rho2_bias,
            "mask_mle_rho2_bias": mask.rho2_bias,
            "bias_denominator": denominator,
            "normalization_status": (
                "defined"
                if abs(denominator) >= .02 and moves_toward_zero
                else "raw bias only: anchor does not move toward zero"),
        })
    anchors = pd.DataFrame(anchors)
    anchors.to_csv(out_dir / "censoring_recovery_anchor.csv", index=False)

    old_pair = pd.read_csv(g0_dir / "wrong_mechanism_penalties.csv")
    old_pair_rows = old_pair[
        ((old_pair.sample_arm == "time_respecting") &
         (old_pair.assumed_arm == "recent_history_k20")) |
        ((old_pair.sample_arm == "recent_history_k20") &
         (old_pair.assumed_arm == "time_respecting"))]
    old_pair_score = float(old_pair_rows.abs_rho2_bias_shift.median())

    node_panel_checks = {
        "matching_node_universes": int((
            a[["instance_id", "n_nodes_true", "sample__active_nodes_total"]]
            .drop_duplicates().n_nodes_true ==
            a[["instance_id", "n_nodes_true", "sample__active_nodes_total"]]
            .drop_duplicates().sample__active_nodes_total).sum())
    }
    report = build_report(
        candidate_bias=candidate_bias, spread=spread,
        candidate_viability=viability, budget_parity=parity_summary,
        lengths=lengths, qwen_lengths=qwen_lengths,
        seed_variance=seed_variance, tv=tv,
        tv_observed=tv_observed, matrix=matrix, penalties=penalties,
        pair_summary=pair_summary, selected_pair=selected_pair, ladder=ladder,
        censoring_anchor=anchors, verified=verified,
        benchmark_n=len(benchmark_b), node_panel_checks=node_panel_checks,
        old_pair_score=old_pair_score, out_dir=str(out_dir))
    Path(args.report).write_text(report)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
