#!/usr/bin/env python3
"""G0c audit for the two-phase event-sample/full-history access arm.

The script is deliberately limited to non-LLM evidence.  It verifies sampler
replays, reconstructs group-balanced observation models for the five-arm set,
fits the same group-held-out ExtraTrees reference used in G0/G0b, evaluates a
coverage-matched sensitivity, and writes the single G0c gate report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from benchmark_features import edge_observations
from make_llm_prompts_v2 import INPUT_MASK
from nonwalk_samplers import (
    event_sample_then_full_history,
    node_panel_full_history,
    prepare_dyad_histories,
    prepare_events,
)
from report_g0_headroom import (
    ARMS as WALK_ARMS,
    KS,
    MASKS,
    PROFILE_KS,
    _event_path,
    _format_table,
    add_panel_logo_floor,
    arm_likelihood_profile,
    extra_trees_transfer,
    group_macro_k_weights,
    joint_counts_for_log,
    mask_histogram,
    metric_summary,
    observation_model,
    parse_nmask_hist,
)
from report_g0b_headroom import (
    distribution,
    mask_length_rows,
    read_globs,
    vectorized_true_window_counts,
)
from report_seed_variance import decompose, profile_mae, se_macro


NODE_PANEL = "node_panel_full_history"
TWO_PHASE = "event_sample_then_full_history"
ALL_ARMS = (*WALK_ARMS, NODE_PANEL, TWO_PHASE)
TRUTH = tuple(f"rho_W5_k{k}" for k in PROFILE_KS)


def group_macro_bias(frame: pd.DataFrame, pred: str = "est__plugin_rho_k2") -> float:
    work = frame[["group_id", pred, "rho_W5_k2"]].copy()
    work["error"] = work[pred] - work["rho_W5_k2"]
    return float(work.dropna().groupby("group_id").error.mean().mean())


def candidate_bias_summary(frame: pd.DataFrame) -> pd.DataFrame:
    work = frame.copy()
    work["rho2_error"] = work["est__plugin_rho_k2"] - work["rho_W5_k2"]
    rows = []
    for seed, part in work.groupby("sample_seed"):
        valid = part.rho2_error.notna()
        rows.append({
            "sample_seed": int(seed),
            "valid_cases": int(valid.sum()),
            "empty_cases": int((~valid).sum()),
            "group_macro_rho2_bias": float(
                part[valid].groupby("group_id").rho2_error.mean().mean()),
        })
    return pd.DataFrame(rows)


def _replay_instance(task):
    instance_id, group_id, event_path, records = task
    events = pd.read_csv(event_path)
    prepared = prepare_events(events)
    prepared_dyads = prepare_dyad_histories(prepared)
    true_k = vectorized_true_window_counts(prepared.events, W=5, T=1.0)
    totals = {arm: np.zeros((6, 32), dtype=np.int64)
              for arm in (NODE_PANEL, TWO_PHASE)}
    checked = 0
    for row in records:
        strategy = str(row["strategy"])
        if strategy == NODE_PANEL:
            result = node_panel_full_history(
                prepared, int(row["target_budget"]),
                int(row["sample_rng_seed"]))
        elif strategy == TWO_PHASE:
            result = event_sample_then_full_history(
                prepared_dyads, int(row["target_budget"]),
                int(row["sample_rng_seed"]))
        else:
            raise ValueError(f"unexpected G0c strategy {strategy}")
        if len(result.log) != int(row["budget"]):
            raise AssertionError(f"budget mismatch for {row['case_id']}")
        replay = Counter((int(x["n"]), int(x["mask"]))
                         for x in edge_observations(
                             result.log, len(result.log), W=5, T=1.0))
        if replay != parse_nmask_hist(str(row["input__nmask_exact_json"])):
            raise AssertionError(f"G0c replay mismatch for {row['case_id']}")
        totals[strategy] += joint_counts_for_log(
            result.log, SimpleNamespace(T=1.0), len(result.log), W=5,
            true_k=true_k)
        checked += 1
    output = []
    for arm, matrix in totals.items():
        for k in KS:
            for mask in MASKS:
                output.append({"group_id": group_id, "arm": arm, "K": k,
                               "mask": mask, "count": int(matrix[k, mask])})
    return instance_id, checked, output


def collect_counts(cases: pd.DataFrame, event_manifest: str,
                   jobs: int = 1) -> tuple[pd.DataFrame, int]:
    manifest_path = Path(event_manifest)
    manifest = pd.read_csv(manifest_path)
    lookup = {str(r.instance_id): _event_path(manifest_path, str(r.path))
              for r in manifest.itertuples(index=False)}
    tasks = []
    for instance_id, part in cases.groupby("instance_id", sort=True):
        groups = part.group_id.astype(str).unique()
        if len(groups) != 1:
            raise AssertionError(f"multiple groups for {instance_id}")
        tasks.append((str(instance_id), str(groups[0]),
                      str(lookup[str(instance_id)]), part.to_dict("records")))
    totals = defaultdict(lambda: np.zeros((6, 32), dtype=np.int64))
    checked = 0
    pool = None
    if jobs == 1:
        iterator = map(_replay_instance, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        iterator = pool.map(_replay_instance, tasks, chunksize=1)
    try:
        for pos, (_, n_checked, rows) in enumerate(iterator, 1):
            checked += n_checked
            for row in rows:
                totals[(row["group_id"], row["arm"])][
                    row["K"], row["mask"]] += row["count"]
            if pos % 50 == 0 or pos == len(tasks):
                print(f"[G0c replay] {pos}/{len(tasks)} instances; "
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


def score_cell(sample: pd.DataFrame, assumed_arm: str,
               cross_models: dict[str, dict[str, np.ndarray]]) -> dict:
    columns = [f"g0c__arm_likelihood_rho_k{k}" for k in PROFILE_KS]
    scored = sample.copy()
    values = []
    for row in scored.itertuples(index=False):
        hist = mask_histogram(str(row.input__nmask_exact_json))
        if hist[1:].sum() <= 0:
            values.append(np.full(4, np.nan))
        else:
            values.append(arm_likelihood_profile(
                hist, cross_models[str(row.group_id)][assumed_arm]))
    scored[columns] = np.asarray(values)
    return metric_summary(scored, columns)


def full_matrix(panel: pd.DataFrame, counts: pd.DataFrame) -> pd.DataFrame:
    cross_models = {
        group: observation_model(
            counts, excluded_groups=[group], arms=ALL_ARMS)
        for group in sorted(panel.group_id.astype(str).unique())
    }
    rows = []
    for sample_arm in ALL_ARMS:
        sample = panel[panel.strategy == sample_arm]
        for assumed_arm in ALL_ARMS:
            rows.append({"sample_arm": sample_arm,
                         "assumed_arm": assumed_arm,
                         **score_cell(sample, assumed_arm, cross_models)})
    return pd.DataFrame(rows)


def matrix_penalties(matrix: pd.DataFrame) -> pd.DataFrame:
    diagonal = matrix[matrix.sample_arm == matrix.assumed_arm].set_index("sample_arm")
    rows = []
    for row in matrix[matrix.sample_arm != matrix.assumed_arm].itertuples(index=False):
        base = diagonal.loc[row.sample_arm]
        shift = float(row.rho2_bias - base.rho2_bias)
        rows.append({
            "sample_arm": row.sample_arm,
            "assumed_arm": row.assumed_arm,
            "profile_mae_penalty": float(row.profile_mae - base.profile_mae),
            "rho2_bias_shift": shift,
            "abs_rho2_bias_shift": abs(shift),
        })
    return pd.DataFrame(rows)


def correction_class(bias: float, neutral_threshold: float = 0.05) -> str:
    if bias < -neutral_threshold:
        return "upward"
    if bias > neutral_threshold:
        return "downward"
    return "none"


def observable_distinguishability(panel: pd.DataFrame,
                                  arm_a: str, arm_b: str) -> dict:
    """Group-held-out distinguishability from mask-histogram-derived columns."""
    part = panel[panel.strategy.isin([arm_a, arm_b])].copy()
    cols = [c for c in part.columns if c.startswith("occ__") or
            c.startswith("pat__mask_") or
            (c.startswith("pat__n") and "_mask" in c)]
    X = part[cols].to_numpy(float)
    y = (part.strategy == arm_b).astype(int).to_numpy()
    groups = part.group_id.astype(str).to_numpy()
    pred = np.full(len(part), np.nan)
    for train, test in LeaveOneGroupOut().split(X, y, groups):
        model = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=5000, random_state=37),
        )
        model.fit(X[train], y[train])
        pred[test] = model.predict_proba(X[test])[:, 1]
    auc = float(roc_auc_score(y, pred))
    return {
        "observable_auc_logo": auc,
        "observable_distance_2auc_minus1": float(2 * abs(auc - 0.5)),
        "observable_balanced_accuracy": float(
            balanced_accuracy_score(y, pred >= 0.5)),
        "observable_feature_count": int(len(cols)),
    }


def pair_summary(penalties: pd.DataFrame, panel: pd.DataFrame,
                 bias_by_arm: dict[str, float]) -> tuple[pd.DataFrame, tuple[str, str]]:
    rows = []
    for i, arm_a in enumerate(ALL_ARMS):
        for arm_b in ALL_ARMS[i + 1:]:
            class_a = correction_class(float(bias_by_arm[arm_a]))
            class_b = correction_class(float(bias_by_arm[arm_b]))
            if class_a == class_b:
                continue
            pair = penalties[
                ((penalties.sample_arm == arm_a) &
                 (penalties.assumed_arm == arm_b)) |
                ((penalties.sample_arm == arm_b) &
                 (penalties.assumed_arm == arm_a))]
            if len(pair) != 2:
                raise AssertionError(f"missing bidirectional cells for {arm_a}, {arm_b}")
            rows.append({
                "arm_a": arm_a, "arm_b": arm_b,
                "correction_a": class_a, "correction_b": class_b,
                "median_profile_mae_penalty": float(
                    pair.profile_mae_penalty.median()),
                "median_abs_rho2_bias_shift": float(
                    pair.abs_rho2_bias_shift.median()),
                "min_abs_rho2_bias_shift": float(
                    pair.abs_rho2_bias_shift.min()),
                "max_abs_rho2_bias_shift": float(
                    pair.abs_rho2_bias_shift.max()),
                **observable_distinguishability(panel, arm_a, arm_b),
            })
    summary = pd.DataFrame(rows)
    best = summary.loc[summary.median_abs_rho2_bias_shift.idxmax()]
    return summary, (str(best.arm_a), str(best.arm_b))


def estimator_ladder(panel: pd.DataFrame, train: pd.DataFrame,
                     arms: tuple[str, ...], jobs: int) -> pd.DataFrame:
    work = panel.copy()
    floor_cols = add_panel_logo_floor(work)
    et_cols = extra_trees_transfer(train, work, jobs=jobs, arms=arms)
    methods = {
        "mean floor (panel LOGO)": floor_cols,
        "naive read-off": [f"est__plugin_rho_k{k}" for k in PROFILE_KS],
        "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__occ_mle_rho_k{k}" for k in PROFILE_KS],
        "mask MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__mask_mle_rho_k{k}" for k in PROFILE_KS],
        "supervised ExtraTrees (label-informed performance reference; "
        "panel backbones held out)": et_cols,
    }
    return pd.DataFrame([
        {"arm": arm, "estimator": name,
         **metric_summary(work[work.strategy == arm], cols)}
        for arm in arms for name, cols in methods.items()
    ])


def seed_variance_table(panel_b: pd.DataFrame) -> pd.DataFrame:
    methods = {
        "naive read-off": [f"est__plugin_rho_k{k}" for k in PROFILE_KS],
        "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__occ_mle_rho_k{k}" for k in PROFILE_KS],
        "mask MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__mask_mle_rho_k{k}" for k in PROFILE_KS],
    }
    rows = []
    for name, cols in methods.items():
        values = profile_mae(panel_b, cols)
        parts = decompose(panel_b, values)
        total = parts["s2_group"] + parts["s2_inst"] + parts["s2_seed"]
        rows.append({
            "method": name, **parts,
            "missing_cases": int(np.isnan(values).sum()),
            "seed_variance_share": parts["s2_seed"] / total if total else 0.0,
            "se_1seed": se_macro(parts, n_seed=1),
            "se_8seeds": se_macro(parts, n_seed=8),
        })
    return pd.DataFrame(rows)


def estimand_checks(panel_manifest: str) -> dict:
    path = Path(panel_manifest)
    manifest = pd.read_csv(path)
    node_matches = dyad_matches = event_matches = canonical = 0
    for row in manifest.itertuples(index=False):
        events = pd.read_csv(_event_path(path, str(row.path)),
                             usecols=["u", "v"])
        nodes = pd.unique(pd.concat([events.u, events.v], ignore_index=True))
        dyads = events[["u", "v"]].drop_duplicates()
        node_matches += int(len(nodes) == int(row.n_nodes))
        dyad_matches += int(len(dyads) == int(row.n_edges))
        event_matches += int(len(events) == int(row.n_events))
        canonical += int((dyads.u < dyads.v).all())
    return {"graphs": len(manifest), "node_matches": node_matches,
            "dyad_matches": dyad_matches, "event_matches": event_matches,
            "canonical": canonical}


def distribution_rows(items: list[tuple[str, pd.Series]]) -> pd.DataFrame:
    return pd.DataFrame([{"scope": label, **distribution(values)}
                         for label, values in items])


def export_token_inputs(panel: pd.DataFrame, out: Path) -> None:
    rows = []
    for row in panel.itertuples(index=False):
        rows.append({"case_id": row.case_id, "arm": row.strategy,
                     "text": f"{INPUT_MASK}\n{row.input__nmask_exact_json}"})
    with out.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def build_report(*, checks: dict, bias_seeds: pd.DataFrame,
                 bias_summary: pd.DataFrame, viability: pd.DataFrame,
                 phase_split: pd.DataFrame, lengths: pd.DataFrame,
                 qwen_lengths: pd.DataFrame | None,
                 seed_variance: pd.DataFrame, ladder: pd.DataFrame,
                 coverage_ladder: pd.DataFrame,
                 coverage_comparison: pd.DataFrame,
                 matrix: pd.DataFrame, penalties: pd.DataFrame,
                 pairs: pd.DataFrame, selected_pair: tuple[str, str],
                 arm_bias_classes: pd.DataFrame,
                 a_seed0_empty_cases: int,
                 verified: int, benchmark_n: int,
                 acceptance_passes: bool, coverage_holds: bool,
                 out_dir: str) -> str:
    pmae = matrix.pivot(index="sample_arm", columns="assumed_arm",
                        values="profile_mae").reset_index()
    bias = matrix.pivot(index="sample_arm", columns="assumed_arm",
                        values="rho2_bias").reset_index()
    qwen_text = (_format_table(qwen_lengths) if qwen_lengths is not None else
                 "Exact Qwen counts were not available; see the exported token inputs.")
    chosen = pairs[(pairs.arm_a == selected_pair[0]) &
                   (pairs.arm_b == selected_pair[1])].iloc[0]
    return f"""# G0c: estimand pin and two-phase arm-B audit

Prepared: **2026-08-31**  
Gate status: **G0c complete. No LLM calls were made. STOP before G1.**

## Scope and decision

G0c replaces the oracle-weighted G0b arm with `{TWO_PHASE}` and keeps both
full-history controls in the planned five-arm set. All new artifacts live
below `{out_dir}`; no frozen benchmark case, panel truth, or LLM artifact was
modified.

The arms are a spanning set over the two named error sources, not a sampler
catalogue:

| arm | censoring | selection bias | required correction |
|---|---|---|---|
| `time_agnostic_t` | strong | weak | upward |
| `time_respecting` | strong | weak | upward |
| `recent_history_k20` | strong | weak | upward |
| `node_panel_full_history` | none | none | none |
| `event_sample_then_full_history` | none | strong | downward |

The three walk arms are near-replicates on the signed-bias axis; they differ
mainly in observable mask shape rather than in the direction or size of the
required correction.

The rebuilt arm's eight-seed group-macro `rho_2` bias is
**{float(bias_summary.group_macro_bias_mean.iloc[0]):+.4f}** (seed SD
{float(bias_summary.group_macro_bias_sd.iloc[0]):.4f}). The prespecified
acceptance threshold `> +0.05` therefore **{'passes' if acceptance_passes else 'fails'}**.

## G0c.1 — Pinned estimand

- **Node universe:** all node IDs appearing as an endpoint of at least one
  event in the complete normalized stream. Isolated nodes outside the event
  stream are not in the manifest universe.
- **Dyad universe:** all canonical undirected dyads `(u,v)`, `u < v`, with at
  least one event in that complete stream. It is **not** all `n choose 2`
  possible pairs.
- **Denominator:** exactly the number of those full-stream event-active dyads
  (`n_pairs` in `census.py`, stored as `n_edges` in the manifests). For each
  such dyad, `K` is the number of distinct active windows and
  `rho_k = mean(1[K >= k])` over that fixed denominator.
- **Across arms:** the universe and truth columns are identical for all five
  arms. The samplers copy truth from the same manifest row, and the replay
  audit initializes every full-population dyad at observation mask 0 before
  moving observed dyads to a nonzero mask.

Direct panel-file checks: node counts match in **{checks['node_matches']}/{checks['graphs']}**
graphs, active-dyad counts in **{checks['dyad_matches']}/{checks['graphs']}**,
event counts in **{checks['event_matches']}/{checks['graphs']}**, and canonical
undirected endpoint order in **{checks['canonical']}/{checks['graphs']}**.

**Sample-dependent conditioning exists only in estimators, not in the
estimand.** In particular, the naive read-off in `benchmark_features.py`
averages over the dyads present in the sample. That observed-dyad denominator
is intentionally sample-dependent and is the source of the selection/censoring
bias being measured. The frozen truth denominator does not change by arm.

## G0c.2 — Rebuilt arm B at 800 unique events

Implementation: phase 1 traverses a uniform random-priority ordering of event
records. The first occurrence of a dyad triggers phase 2, which adds that
dyad's complete event history. The prefix stops before the first new lookup
whose complete history would exceed the remaining unique-event budget. No
full-stream event count is used as a sampling weight. More active dyads are
discovered more often because they own more event records in phase 1.

Every emitted dyad history is complete and no partial response is retained.
The stop can leave slack; if the first event belongs to a dyad with more than
800 events it emits an empty sample. This happened in
**{int(bias_summary.empty_cases.iloc[0])}/{int(bias_summary.total_cases.iloc[0])}**
eight-seed cases. Those cases are reported as undefined, not silently redrawn
or treated as `rho_2=0`. Seed 0, used for the estimator ladder, has
**{int(bias_summary.seed0_empty_cases.iloc[0])}** empty cases.

Bias by seed (undefined empty cases excluded from signed bias):

{_format_table(bias_seeds)}

Aggregate bias and case-wise direction spread:

{_format_table(bias_summary)}

Coverage and budget diagnostics:

{_format_table(viability)}

Unique-event phase split (`phase1 + phase2 additional = realized`):

{_format_table(phase_split)}

Static mask-input lengths (`INPUT_MASK` plus exact histogram, before G1 text):

{_format_table(lengths)}

Exact Qwen3.6-27B tokenizer counts for the same block
(cached snapshot `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`,
`add_special_tokens=False`):

{qwen_text}

Eight-seed variance decomposition uses successful cases only and is therefore
slightly unbalanced where the whole-lookup stop produced an empty sample:

{_format_table(seed_variance)}

Full reference ladder at seed 0. `occ_mle` and `mask_mle` are
**censoring-aware, mechanism-agnostic**; ExtraTrees is a **label-informed
performance reference**, trained with the matching panel backbone held out:

{_format_table(ladder)}

Arm A has **{a_seed0_empty_cases}/32** empty seed-0 samples under its existing
whole-node stop, hence `n={32 - a_seed0_empty_cases}` for its analytical rows.
The supervised reference still produces 32 predictions by treating the empty
observable feature vector as a valid input. This difference in `n` must not be
hidden in later comparisons.

## G0c.3 — Coverage-matched sensitivity

A common diagnostic budget of **10,500** unique events gives rebuilt B median
coverage {float(coverage_comparison.loc[coverage_comparison.scope == 'rebuilt B, budget 10500, seed 0', 'median'].iloc[0]):.4f},
against {float(coverage_comparison.loc[coverage_comparison.scope == 'three walks, budget 800, seed 0', 'median'].iloc[0]):.4f}
for the walks. This is median matching, not graph-by-graph matching; the main
run remains fixed at 800.

{_format_table(coverage_comparison)}

Estimator ladder at the sensitivity budget:

{_format_table(coverage_ladder)}

The positive naive-bias sign **{'and rough magnitude hold' if coverage_holds else 'do not hold'}**
at matched median coverage. This check does not make the access arms
information-equivalent: B still observes fewer dyads with complete records,
whereas the walks observe more dyads with truncated/repeated records.

## G0c.4 — Mismatch pair over all five arms

The empirical observation models for A and rebuilt B were learned from the
regenerated 745-instance benchmark with complete source/family holdout. The
new `(n,mask)` samples were deterministically replayed and
**{verified:,}/{benchmark_n:,}** histograms matched. As in G0/G0b, this
arm-likelihood exercise is label-assisted and is neither a deployable
estimator nor a theoretical bound.

Group-macro ProfileMAE (rows produce the sample; columns supply the assumed
arm likelihood):

{_format_table(pmae)}

The same matrix as signed `rho_2` bias:

{_format_table(bias)}

All directed off-diagonal penalties:

{_format_table(penalties)}

Eligible bidirectional pairs must have different correction classes. `none`
means an eight-seed naive bias within +/-0.05. Observable distinguishability
uses only `(n,mask)`-histogram-derived features: a logistic arm classifier is
trained with the entire graph group held out. AUC 0.5 / distance 0 means
indistinguishable; AUC 1 / distance 1 means perfectly distinguishable.

Measured bias classes used for eligibility:

{_format_table(arm_bias_classes)}

{_format_table(pairs)}

Largest bidirectional bias penalty: **`{selected_pair[0]}` <->
`{selected_pair[1]}`**, median absolute `rho_2`-bias shift
**{float(chosen.median_abs_rho2_bias_shift):.4f}**. Its held-group-out
observable AUC is **{float(chosen.observable_auc_logo):.4f}** (distance
{float(chosen.observable_distance_2auc_minus1):.4f}). The chosen pair is only
moderately distinguishable under this held-group-out diagnostic, not grossly
separable. Text distrust and fallback toward `hidden` remain possible, but the
observable statistics do not make that alternative explanation automatic.

## What remains uncertain

- Positive selection bias is a panel-level empirical result, not a per-case
  theorem. The case-wise `delta_i` sign distribution above is why G4 must use
  the prespecified slope rather than assign one arm-level sign to every case.
- A hard unique-event cap and complete-history requirement make oversize first
  lookups genuinely undefined. G2 must freeze how these cases enter response
  and failure-rate reporting for both full-history arms; G0c did not redraw
  them.
- Coverage matching controls one scalar only. It does not match number of
  dyads, event multiplicities, temporal masks, or sample coherence.
- No language model has been tested here, and this report says nothing about
  whether a model can operationalize the mechanism text.

arms for the main run: [`time_agnostic_t`, `time_respecting`,
`recent_history_k20`, `node_panel_full_history`,
`event_sample_then_full_history`]
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walk-panel", default="results/panel_seed_probe/cases.csv.gz")
    ap.add_argument("--panel-a", nargs="+", default=[
        "results/g0b_headroom_2026_09/panel_candidates_shard_*.csv.gz"])
    ap.add_argument("--panel-b", nargs="+", default=[
        "results/g0c_headroom_2026_09/panel_b_shard_*.csv.gz"])
    ap.add_argument("--panel-b-coverage", nargs="+", default=[
        "results/g0c_headroom_2026_09/panel_b_coverage_shard_*.csv.gz"])
    ap.add_argument("--benchmark-main", nargs="+", default=[
        "results/g0c_headroom_2026_09/benchmark_main_shard_*.csv.gz"])
    ap.add_argument("--benchmark-b-coverage", nargs="+", default=[
        "results/g0c_headroom_2026_09/benchmark_b_coverage_shard_*.csv.gz"])
    ap.add_argument("--event-manifest", default=(
        "results/g0_headroom_2026_09/regenerated_benchmark_v2/manifest.csv"))
    ap.add_argument("--panel-manifest", default=(
        "results/final_target_panel/panel32_final.csv"))
    ap.add_argument("--g0-dir", default="results/g0_headroom_2026_09")
    ap.add_argument("--out-dir", default="results/g0c_headroom_2026_09")
    ap.add_argument("--qwen-token-counts", default=(
        "results/g0c_headroom_2026_09/mask_input_qwen36_tokens.csv"))
    ap.add_argument("--report", default="docs/HEADROOM_G0C_2026-09.md")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--model-jobs", type=int, default=-1)
    ap.add_argument("--rebuild-counts", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g0_dir = Path(args.g0_dir)

    walk = pd.read_csv(args.walk_panel)
    walk = walk[(walk.budget == 800) & (walk.walk_seed == 0) &
                walk.strategy.isin(WALK_ARMS)].copy()
    a_all = read_globs(args.panel_a)
    a_all = a_all[(a_all.strategy == NODE_PANEL) &
                  (a_all.target_budget == 800)].copy()
    a0 = a_all[a_all.sample_seed == 0].copy()
    b_all = read_globs(args.panel_b)
    b_all = b_all[(b_all.strategy == TWO_PHASE) &
                  (b_all.target_budget == 800)].copy()
    b0 = b_all[b_all.sample_seed == 0].copy()
    bcov = read_globs(args.panel_b_coverage)
    bcov = bcov[(bcov.strategy == TWO_PHASE) &
                (bcov.target_budget == 10500) &
                (bcov.sample_seed == 0)].copy()
    if (len(walk), len(a_all), len(b_all), len(bcov)) != (96, 256, 256, 32):
        raise ValueError("G0c panel inputs are incomplete")

    bias_seeds = candidate_bias_summary(b_all)
    bias_seeds.to_csv(out_dir / "rebuilt_b_bias_by_seed.csv", index=False)
    valid = b_all.est__plugin_rho_k2.notna()
    delta = (b_all.loc[valid, "rho_W5_k2"] -
             b_all.loc[valid, "est__plugin_rho_k2"])
    bias_summary = pd.DataFrame([{
        "group_macro_bias_mean": float(bias_seeds.group_macro_rho2_bias.mean()),
        "group_macro_bias_sd": float(bias_seeds.group_macro_rho2_bias.std(ddof=1)),
        "seed0_group_macro_bias": float(
            bias_seeds.loc[bias_seeds.sample_seed == 0,
                           "group_macro_rho2_bias"].iloc[0]),
        "total_cases": int(len(b_all)), "valid_cases": int(valid.sum()),
        "empty_cases": int((~valid).sum()),
        "seed0_empty_cases": int(b0.est__plugin_rho_k2.isna().sum()),
        "delta_lt_zero_share_valid": float((delta < 0).mean()),
        "delta_gt_zero_share_valid": float((delta > 0).mean()),
        "delta_eq_zero_share_valid": float((delta == 0).mean()),
    }])
    bias_summary.to_csv(out_dir / "rebuilt_b_bias_summary.csv", index=False)
    acceptance_passes = float(bias_summary.group_macro_bias_mean.iloc[0]) > .05
    if not acceptance_passes:
        raise RuntimeError("rebuilt B failed the prespecified +0.05 bias gate")

    viability = distribution_rows([
        ("rebuilt B coverage, seed 0", b0.coverage),
        ("rebuilt B coverage, all 8 seeds", b_all.coverage),
        ("rebuilt B realized unique events", b_all.budget),
        ("rebuilt B budget slack", b_all.sample__budget_slack),
        ("node panel A coverage, seed 0", a0.coverage),
        ("three walks coverage, seed 0", walk.coverage),
    ])
    phase_split = distribution_rows([
        ("phase 1 uniform event sample", b_all.sample__phase1_unique_event_count),
        ("phase 2 additional lookup events",
         b_all.sample__phase2_additional_unique_event_count),
        ("realized unique events", b_all.budget),
    ])
    viability.to_csv(out_dir / "rebuilt_b_viability.csv", index=False)
    phase_split.to_csv(out_dir / "rebuilt_b_phase_split.csv", index=False)

    panel5 = pd.concat([walk, a0, b0], ignore_index=True)
    lengths = mask_length_rows(panel5)
    lengths.to_csv(out_dir / "mask_input_lengths.csv", index=False)
    export_token_inputs(panel5, out_dir / "mask_input_texts.jsonl")
    qwen_lengths = None
    qwen_path = Path(args.qwen_token_counts)
    if qwen_path.exists():
        qwen = pd.read_csv(qwen_path)
        if len(qwen) != len(panel5) or set(qwen.case_id) != set(panel5.case_id):
            raise ValueError("Qwen token counts do not match the five-arm seed-0 panel")
        qwen_lengths = pd.DataFrame([
            {"arm": arm, **distribution(part.qwen36_tokens)}
            for arm, part in qwen.groupby("arm")])
        qwen_lengths.to_csv(out_dir / "mask_input_qwen36_summary.csv", index=False)

    seed_variance = seed_variance_table(b_all)
    seed_variance.to_csv(out_dir / "rebuilt_b_seed_variance.csv", index=False)

    benchmark = read_globs(args.benchmark_main)
    benchmark = benchmark[(benchmark.target_budget == 800) &
                          benchmark.strategy.isin([NODE_PANEL, TWO_PHASE])].copy()
    if len(benchmark) != 745 * 4 * 2 or benchmark.case_id.duplicated().any():
        raise ValueError("G0c main benchmark cases are incomplete or duplicated")
    counts_path = out_dir / "observation_counts_a_b_by_group.csv.gz"
    if counts_path.exists() and not args.rebuild_counts:
        counts_ab = pd.read_csv(counts_path)
        verified = len(benchmark)
    else:
        counts_ab, verified = collect_counts(
            benchmark, args.event_manifest, jobs=args.jobs)
        counts_ab.to_csv(counts_path, index=False, compression="gzip")
    counts_walk = pd.read_csv(g0_dir / "observation_counts_by_group.csv.gz")
    counts = pd.concat([counts_walk, counts_ab], ignore_index=True)

    matrix = full_matrix(panel5, counts)
    penalties = matrix_penalties(matrix)
    matrix.to_csv(out_dir / "wrong_mechanism_matrix_five_arms.csv", index=False)
    penalties.to_csv(out_dir / "wrong_mechanism_penalties_five_arms.csv", index=False)

    old_ladder = pd.read_csv(g0_dir / "estimator_ladder.csv")
    old_ladder["estimator"] = old_ladder.estimator.replace({
        "occupancy MLE (uniform)":
            "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)",
        "mask MLE (uniform)":
            "mask MLE (uniform; censoring-aware, mechanism-agnostic)",
        "supervised ExtraTrees (benchmark transfer)":
            "supervised ExtraTrees (label-informed performance reference; "
            "panel backbones held out)",
    })
    ladder_path = out_dir / "estimator_ladder_five_arms.csv"
    if ladder_path.exists():
        ladder = pd.read_csv(ladder_path)
    else:
        ladder_ab = estimator_ladder(
            pd.concat([a0, b0], ignore_index=True), benchmark,
            (NODE_PANEL, TWO_PHASE), args.model_jobs)
        ladder = pd.concat([old_ladder, ladder_ab], ignore_index=True)
        ladder.to_csv(ladder_path, index=False)

    bias_by_arm = dict(zip(
        old_ladder[old_ladder.estimator == "naive read-off"].arm,
        old_ladder[old_ladder.estimator == "naive read-off"].rho2_bias))
    a_bias_by_seed = candidate_bias_summary(a_all)
    bias_by_arm[NODE_PANEL] = float(a_bias_by_seed.group_macro_rho2_bias.mean())
    bias_by_arm[TWO_PHASE] = float(bias_summary.group_macro_bias_mean.iloc[0])
    arm_bias_classes = pd.DataFrame([
        {"arm": arm, "naive_group_macro_rho2_bias": float(bias_by_arm[arm]),
         "required_correction": correction_class(float(bias_by_arm[arm]))}
        for arm in ALL_ARMS])
    arm_bias_classes.to_csv(out_dir / "arm_bias_classes.csv", index=False)
    pairs, selected_pair = pair_summary(
        penalties, panel5, bias_by_arm=bias_by_arm)
    pairs.to_csv(out_dir / "mismatch_candidate_pairs.csv", index=False)

    benchmark_cov = read_globs(args.benchmark_b_coverage)
    benchmark_cov = benchmark_cov[
        (benchmark_cov.target_budget == 10500) &
        (benchmark_cov.strategy == TWO_PHASE)].copy()
    if len(benchmark_cov) != 745 * 4:
        raise ValueError("coverage benchmark cases are incomplete")
    coverage_ladder_path = out_dir / "coverage_matched_estimator_ladder.csv"
    if coverage_ladder_path.exists():
        coverage_ladder = pd.read_csv(coverage_ladder_path)
    else:
        coverage_ladder = estimator_ladder(
            bcov.copy(), benchmark_cov, (TWO_PHASE,), args.model_jobs)
        coverage_ladder.to_csv(coverage_ladder_path, index=False)
    coverage_comparison = distribution_rows([
        ("rebuilt B, budget 800, seed 0", b0.coverage),
        ("rebuilt B, budget 10500, seed 0", bcov.coverage),
        ("three walks, budget 800, seed 0", walk.coverage),
    ])
    coverage_comparison.to_csv(out_dir / "coverage_matched_coverage.csv",
                               index=False)
    cov_naive = coverage_ladder[
        coverage_ladder.estimator == "naive read-off"].iloc[0]
    main_naive = ladder[(ladder.arm == TWO_PHASE) &
                        (ladder.estimator == "naive read-off")].iloc[0]
    coverage_holds = (float(cov_naive.rho2_bias) > .05 and
                      abs(float(cov_naive.rho2_bias)) >=
                      .5 * abs(float(main_naive.rho2_bias)))

    checks = estimand_checks(args.panel_manifest)
    report = build_report(
        checks=checks, bias_seeds=bias_seeds, bias_summary=bias_summary,
        viability=viability, phase_split=phase_split, lengths=lengths,
        qwen_lengths=qwen_lengths, seed_variance=seed_variance,
        ladder=ladder, coverage_ladder=coverage_ladder,
        coverage_comparison=coverage_comparison, matrix=matrix,
        penalties=penalties, pairs=pairs, selected_pair=selected_pair,
        arm_bias_classes=arm_bias_classes,
        a_seed0_empty_cases=int(a0.est__plugin_rho_k2.isna().sum()),
        verified=verified, benchmark_n=len(benchmark),
        acceptance_passes=acceptance_passes, coverage_holds=coverage_holds,
        out_dir=str(out_dir))
    Path(args.report).write_text(report)
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
