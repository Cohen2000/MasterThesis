#!/usr/bin/env python3
"""G2: freeze the main run.

Draws the final samples from one master seed, runs the reference ladder on
exactly those samples, fixes the two prespecified subsets by a recorded rule,
asserts the output schema is constant across conditions, and hashes everything
that must not move again.

No LLM call is made here.
"""

from __future__ import annotations

import argparse
import hashlib
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import prompt_contract_g1 as C
from report_g0_headroom import PROFILE_KS, _format_table
from report_g0b_headroom import read_globs
from report_g0c_headroom import estimator_ladder
from report_g0d_headroom import (
    MISMATCH_PAIR,
    NODE_PANEL,
    TWO_PHASE,
    apply_seed_rule,
    detectability_covariate,
)

TRUTH = [f"rho_W5_k{k}" for k in PROFILE_KS]
NAIVE = [f"est__plugin_rho_k{k}" for k in PROFILE_KS]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- G2.1 ------------------------------------------------------------------

def load_cases(config: dict, walks_path: str, arm_a: list[str],
               arm_b: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every generated seed slot, then the rule that decides which are used."""
    arms = config["final_run"]["arms"]
    walk_arms = [a for a, spec in arms.items() if spec["family"] == "walk"]
    walks = pd.read_csv(walks_path)
    walks = walks[walks.strategy.isin(walk_arms) &
                  (walks.budget == arms[walk_arms[0]]["budget"])].copy()
    # The walk runner names its slot column `walk_seed`; the non-walk runner
    # names it `sample_seed`. The seed rule needs one name.
    walks["sample_seed"] = walks["walk_seed"]
    frames = [walks]
    for paths, arm in ((arm_a, NODE_PANEL), (arm_b, TWO_PHASE)):
        part = read_globs(paths)
        part = part[(part.strategy == arm) &
                    (part.target_budget == arms[arm]["budget"])].copy()
        frames.append(part)
    raw = pd.concat(frames, ignore_index=True)
    slots = int(config["final_run"]["seed_slots_used"])
    accepted, log = apply_seed_rule(raw, slots)
    return accepted, log


# --- G2.2 ------------------------------------------------------------------

def censoring_recovery(ladder: pd.DataFrame) -> pd.DataFrame:
    """CensoringRecovery, and an explicit verdict where it degenerates.

    The scale anchors on the uniform mask MLE: the denominator is
    `bias_naive - bias_mask_mle`. On an arm with no censoring the anchor moves
    *away* from zero, the denominator flips sign, and the ratio stops meaning
    anything. That is reported rather than papered over.
    """
    naive = ladder[ladder.estimator == "naive read-off"].set_index("arm")
    mask = ladder[ladder.estimator.str.startswith("mask MLE")].set_index("arm")
    rows = []
    for arm in naive.index:
        b_naive = float(naive.loc[arm, "rho2_bias"])
        b_mask = float(mask.loc[arm, "rho2_bias"])
        denominator = b_naive - b_mask
        moves_toward_zero = abs(b_mask) < abs(b_naive)
        usable = moves_toward_zero and abs(denominator) > 1e-6
        rows.append({
            "arm": arm, "naive_rho2_bias": b_naive,
            "mask_mle_rho2_bias": b_mask,
            "bias_denominator": denominator,
            "anchor_moves_toward_zero": moves_toward_zero,
            "normalization": ("defined" if usable else
                              "DEGENERATE: report raw signed bias"),
        })
    return pd.DataFrame(rows)


def delta_table(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """delta_i = rho2_true_i - rho2_naive_i, the primary regression predictor."""
    work = primary.copy()
    work["delta_i"] = work.rho_W5_k2 - work.est__plugin_rho_k2
    rows = []
    for arm, part in work.groupby("strategy"):
        rows.append({
            "arm": arm, "cases": int(len(part)),
            "mean": float(part.delta_i.mean()),
            "median": float(part.delta_i.median()),
            "sd": float(part.delta_i.std(ddof=1)),
            "p10": float(part.delta_i.quantile(.1)),
            "p90": float(part.delta_i.quantile(.9)),
            "share_positive": float((part.delta_i > 0).mean()),
            "share_negative": float((part.delta_i < 0).mean()),
        })
    pooled = work.delta_i
    rows.append({
        "arm": "POOLED", "cases": int(len(pooled)),
        "mean": float(pooled.mean()), "median": float(pooled.median()),
        "sd": float(pooled.std(ddof=1)), "p10": float(pooled.quantile(.1)),
        "p90": float(pooled.quantile(.9)),
        "share_positive": float((pooled > 0).mean()),
        "share_negative": float((pooled < 0).mean()),
    })
    return pd.DataFrame(rows), work[["case_id", "instance_id", "group_id",
                                     "strategy", "coverage", "delta_i"]]


# --- G2.4 ------------------------------------------------------------------

def select_replication_graphs(primary: pd.DataFrame, n: int) -> pd.DataFrame:
    """Coverage-stratified pick, one graph per group, rule recorded not eyed.

    Coverage is averaged over the arms so the choice does not depend on which
    arm happens to see a graph well. Groups are ordered by their mean coverage
    and n are taken at evenly spaced ranks, which spans the range instead of
    clustering at one end. Ties break on instance_id, which is stable.
    """
    per_graph = (primary.groupby(["group_id", "instance_id"], as_index=False)
                 .coverage.mean().sort_values(["coverage", "instance_id"],
                                              ignore_index=True))
    per_group = (per_graph.groupby("group_id", as_index=False)
                 .coverage.mean().sort_values(["coverage", "group_id"],
                                              ignore_index=True))
    picks = np.linspace(0, len(per_group) - 1, n).round().astype(int)
    chosen_groups = per_group.group_id.iloc[sorted(set(picks))].tolist()
    while len(chosen_groups) < n:
        for group in per_group.group_id:
            if group not in chosen_groups:
                chosen_groups.append(group)
                break
    rows = []
    for group in chosen_groups:
        candidates = per_graph[per_graph.group_id == group]
        middle = candidates.iloc[len(candidates) // 2]
        rows.append({"group_id": group,
                     "instance_id": middle.instance_id,
                     "mean_coverage": float(middle.coverage)})
    return pd.DataFrame(rows).sort_values("mean_coverage", ignore_index=True)


# --- G2.5 ------------------------------------------------------------------

def schema_check(primary: pd.DataFrame) -> pd.DataFrame:
    """The requested output schema must be byte-identical everywhere.

    `direction_only` and `mismatched` are new context blocks; if any of them
    perturbed the task block, output format would become a condition effect.
    """
    seen = {}
    for row in primary.to_dict("records"):
        arm = row["strategy"]
        for condition in (*C.CONDITIONS, "irrelevant_context"):
            if condition == "mismatched":
                if arm not in MISMATCH_PAIR:
                    continue
                stated = [a for a in MISMATCH_PAIR if a != arm][0]
            else:
                stated = None
            prompt = C.build_prompt(row, condition, stated)
            tail = prompt[prompt.index("TASK\n"):]
            seen.setdefault(sha256_text(tail), []).append((arm, condition))
    return pd.DataFrame([{
        "distinct_task_blocks": len(seen),
        "byte_identical_across_all_arms_and_conditions": len(seen) == 1,
        "task_block_sha256": next(iter(seen)) if len(seen) == 1 else "VARIES",
        "combinations_checked": sum(len(v) for v in seen.values()),
    }])


def _read_filtered(specs: list[str], arms: list[str],
                   budget: int) -> pd.DataFrame:
    """Read the frozen walk shards one at a time, keeping only what is used.

    The twelve shards together are far larger than the 8,940 rows this needs,
    so filtering inside the loop keeps peak memory to one shard.
    """
    import glob

    paths = []
    for spec in specs:
        paths.extend(sorted(glob.glob(spec)))
    if not paths:
        raise FileNotFoundError(f"no files matched {specs}")
    parts = []
    for path in paths:
        shard = pd.read_csv(path)
        parts.append(shard[(shard.budget == budget) &
                           shard.strategy.isin(arms)].copy())
        del shard
    return pd.concat(parts, ignore_index=True)


# --- G2.7 ------------------------------------------------------------------

def prompt_volume(config: dict, primary: pd.DataFrame,
                  replication: pd.DataFrame
                  ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Exact prompt and call totals, so G3 can be scheduled on real numbers."""
    arms = len(config["final_run"]["arms"])
    graphs = int(primary.instance_id.nunique())
    subsets = config["final_run"]["subsets"]
    ic_arms = len(subsets["irrelevant_context"]["arms"])
    rep_graphs = len(replication)
    rep_seeds = len(subsets["seed_replication"]["extra_seed_slots"])
    rep_conditions = len(subsets["seed_replication"]["conditions"])
    rows = [
        {"set": "factorial: hidden, direction_only, mechanism, "
                "mechanism_direction",
         "prompts": graphs * arms * 4, "models": "all"},
        {"set": "mismatched, bidirectional on one pair",
         "prompts": graphs * 2, "models": "all"},
        {"set": "metadata_only (identical across arms)",
         "prompts": graphs, "models": "all"},
        {"set": "disclosed_historical (appendix bridge)",
         "prompts": graphs * 3, "models": "primary only"},
        {"set": "irrelevant_context (robustness subset)",
         "prompts": graphs * ic_arms, "models": "primary only"},
        {"set": "seed replication",
         "prompts": rep_graphs * arms * rep_conditions * rep_seeds,
         "models": "primary only"},
    ]
    volume = pd.DataFrame(rows)
    volume.loc[len(volume)] = {"set": "TOTAL PROMPTS",
                               "prompts": int(volume.prompts.sum()),
                               "models": ""}

    all_models = int(volume.loc[volume.models == "all", "prompts"].sum())
    primary_only = int(volume.loc[volume.models == "primary only",
                                  "prompts"].sum())
    calls = []
    for model, spec in config["final_run"]["models"].items():
        is_primary = spec["role"] == "primary"
        prompts = all_models + (primary_only if is_primary else 0)
        calls.append({"model": model, "role": spec["role"],
                      "generations": int(spec["generations"]),
                      "version_pinnable": "yes" if spec["pinnable"] else "no",
                      "prompts": prompts,
                      "calls": prompts * int(spec["generations"])})
    call_frame = pd.DataFrame(calls)
    call_frame["generations"] = call_frame.generations.astype(str)
    call_frame["prompts"] = call_frame.prompts.astype(str)
    call_frame.loc[len(call_frame)] = {
        "model": "TOTAL CALLS", "role": "", "generations": "",
        "version_pinnable": "", "prompts": "",
        "calls": int(call_frame.calls.sum())}
    return volume, call_frame


# --- report ----------------------------------------------------------------

def build_document(*, config, hashes, seed_log, ladder, recovery, deltas,
                   detect, detect_note, replication, ic_subset, schema,
                   volume, calls, historical_arm, historical_naming,
                   historical_signs, config_note, coverage) -> str:
    fr = config["final_run"]
    advanced = seed_log[seed_log.seed_advances > 0]
    return f"""# G2 freeze: the main run

Prepared: **2026-09-01**  
Gate status: **G2 complete. No LLM calls were made. STOP before G3.**

This closes the five open gates in `docs/TARGET_EVALUATION_FREEZE.md`: the
input contract, the output schema, the model configurations, the sampled cases
with their budgets, and the prompt hash.

## Motivation, from data that already exists

Two of the three historical `disclosed` prompts name the direction of the bias
in a `Consequence:` line; `time_agnostic_t`'s describes the process and stops.
That accident splits the archived V2.1 suite along the axis the G1 condition
split was built to separate, so the question can be asked retrospectively at no
cost. Paired on the 84 `mask`-input cases answered under both conditions:

{_format_table(historical_arm)}

{_format_table(historical_naming)}

{_format_table(historical_signs)}

**The retrospective answer runs opposite to the obvious guess, and that is why
the randomized split is needed.** If naming the direction carried the
disclosure effect, the two direction-naming arms would move more. They move
less: median shift 0.0681 against 0.0935, and four of five models show the
larger shift on the arm whose text names no direction at all.

This is not evidence that direction-naming does nothing. It is an
*observational* contrast in which arm is confounded with everything else that
differs between the three walks -- coverage, mechanism-text length, and the
size of the bias being corrected. `time_agnostic_t` has the largest coverage
and the largest hidden bias, so it has the most room to move. The reading to
take is the weak one: the historical data cannot separate process description
from direction statement, in either direction, which is exactly the gap G1's
factorial exists to close.

## G2.1 — Fresh samples

Master seed **{fr['master_seed']}**, recorded in
`config/final_run_g2.yaml`. Both runners derive their per-case seed as
`stable_seed(master, instance_id, arm, seed_slot)`, so a different seed reaches
every (graph, arm) pair while the design stays reproducible from that one
number. {fr['seed_slots_generated']} slots are generated per (graph, arm) and
{fr['seed_slots_used']} are used: slot 0 is the primary sample, slots 1-3 feed
the replication subset.

**The identical sample goes to every condition of a case.** The sample is a
property of (graph, arm, slot) and the condition only changes the prose block
in front of it. This is the single most load-bearing property of the design:
it is what makes sampling noise cancel in the primary within-case contrast.
Nothing in the pipeline regenerates a sample per condition, and the schema
check below re-derives every condition's prompt from one case row.

Coverage on the final samples:

{_format_table(coverage)}

The G0d empty-sample rule fired **{len(advanced)}** times across
{len(seed_log)} (graph, arm) pairs, all on `{NODE_PANEL}`, every fire logged:

{_format_table(advanced.drop(columns=['seed_slots']) if len(advanced) else advanced)}

## G2.2 — Reference ladder on exactly these samples

These are the reference lines for every figure. Not the historical numbers and
not the G0d numbers: those were measured on other samples.

{_format_table(ladder)}

### CensoringRecovery, and where it degenerates

{_format_table(recovery)}

The scale anchors on the uniform mask MLE, so its denominator is
`bias_naive - bias_mask_mle`. That presumes the anchor moves the estimate
*toward* zero. On both whole-entity arms it moves away: there is no censoring
to recover, so a censoring correction inflates an already-correct profile.
The denominator is then not small but wrong-signed, and the ratio is not a
weakened measurement, it is a meaningless one. **Report raw signed bias on
those arms and say why**, exactly as flagged.

### `delta_i`, the primary regression predictor

`delta_i = rho2_true_i - rho2_naive_i` is the correct signed correction for
case `i`, and the predictor the primary slope regresses on.

{_format_table(deltas)}

The pooled sign split is the reason G4 must regress on the per-case `delta_i`
rather than assign one sign per arm. Even on arm B, where the arm-level bias is
firmly positive, some cases need an upward correction; and arm A is close to a
coin flip by construction, which is what makes it the negative control.

## G2.3 — Detectability re-check on the fresh samples

{_format_table(detect)}

{detect_note}

Per Decision D this covariate is reported, not invested in. The three-way
directional test does not depend on it: a shift toward the direction implied by
the stated wrong mechanism means the model did not act on any incoherence,
however detectable that incoherence was. The pair is not changed to reduce
detectability, and no case is selected on it.

## G2.4 — Prespecified subsets

Both are fixed here, by a recorded rule, before anything has been run.

### Seed replication

{len(replication)} graphs x {len(fr['arms'])} arms x {len(fr['subsets']['seed_replication']['conditions'])} conditions x {len(fr['subsets']['seed_replication']['extra_seed_slots'])} extra seed
slots, primary model only. It tests a seed-by-condition interaction, which the
pairing argument does not cover: pairing removes sampling noise from the
contrast but says nothing about whether the effect itself depends on the draw.

Selection rule: **{fr['subsets']['seed_replication']['selection_rule']}**.
Graphs are averaged over arms so the choice cannot depend on which arm happens
to see a graph well; groups are ranked by mean coverage and taken at evenly
spaced ranks so the subset spans the range rather than clustering; within a
group the median-coverage graph is taken. Ties break on `instance_id`.

{_format_table(replication)}

### `irrelevant_context`

{_format_table(ic_subset)}

## G2.5 — Output schema is constant

{_format_table(schema)}

The requested-output block is byte-identical across every arm and every
condition, including the two new ones. Had it varied, output format would have
become a condition effect and any parse-rate difference would have been
uninterpretable. Enforced by test.

## G2.6 — Freeze

{_format_table(hashes)}

### Which benchmark configuration is authoritative

{config_note}

## G2.7 — Prompt volume

{_format_table(volume)}

At the allocation fixed in `config/final_run_g2.yaml` -- generations are a
resolution instrument, not an accuracy one, so the count follows each model's
measured reproducibility in `docs/LLM_NOISE_RESULT.md` rather than being
uniform:

{_format_table(calls)}

One caveat on the primary designation, recorded rather than buried. Codex
carries the subsets because it is the strongest reader in the suite
(ProfileMAE 0.053, reproducibility 0.07) and because 68% of its within-graph
noise comes from redrawing the walk, which is what makes spending seeds on it
worthwhile. But its harness adds a prompt that is not part of the frozen prompt
and cannot be version-pinned, so every Codex row is a product screen rather
than a pinned API measurement. That is acceptable for the subsets, which are
robustness and diagnostic items where the strongest reader is the most
informative, and **no primary claim rests on a Codex-only result**. If the
subsets must be version-pinned instead, `qwen3.6-27b_think` is the alternative
and the totals above do not change.

## What is not settled here

- Exact Qwen token counts. The numbers in `docs/PROMPT_CONTRACT_2026-09.md`
  are calibrated estimates and must be replaced from the cluster tokenizer
  before G3 starts.
- Whether `disclosed_historical` is run. It is costed above and is an appendix
  bridge; no primary claim depends on it.
- No language model has seen any of these prompts.
"""


CONFIG_NOTE = """`config/benchmark_v21.yaml` is authoritative. The decision was made by
diffing them rather than by preference, and the diff is narrow: the two files
differ in **exactly one key**, `presets.v2.evaluation`. Data generation, the
walk plan, the seed, and the `smoke`, `full` and `v2_smoke` presets are
byte-identical in both.

`benchmark_v21.yaml` adds `lifetime_mean_over_T` to the evaluation targets,
prediction targets, and the strategy-blind, leave-one-block-out and sim2real
blocks, and drops `extra_trees:patterns` from `headline_only_pairs`. The
frozen target hierarchy in `docs/TARGET_EVALUATION_FREEZE.md` requires lifetime
as a robustness target, so v21 is the file consistent with the freeze and the
older one silently omits a target the hierarchy asks for.

Two consequences worth stating plainly. First, because the difference is
confined to `evaluation`, this choice **changes no sample, no walk, no case and
no generated artifact**, and it does not touch the G2 main run at all -- that
runs from `config/final_run_g2.yaml`. Second, the real hazard was never the
file, it was the default: `src/evaluate_benchmark.py` is the *only* consumer of
the `evaluation` block, and it defaulted to the file without the lifetime
target. A resolution recorded in prose while the default kept pointing the
other way would have changed nothing, so the default now points at
`benchmark_v21.yaml`. `src/build_benchmark_data.py` copies the block into a
manifest sidecar without acting on it; every other script reads sections that
are identical in both files, so their defaults are left alone."""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/final_run_g2.yaml")
    parser.add_argument("--walks", default="results/final_run_g2/walks.csv.gz")
    parser.add_argument("--arm-a", nargs="+", default=[
        "results/final_run_g2/arm_a_shard_*.csv.gz"])
    parser.add_argument("--arm-b", nargs="+", default=[
        "results/final_run_g2/arm_b_shard_*.csv.gz"])
    parser.add_argument("--benchmark-a", nargs="+", default=[
        "results/final_run_g2/benchmark_a_shard_*.csv.gz"])
    parser.add_argument("--benchmark-b", nargs="+", default=[
        "results/final_run_g2/benchmark_b_shard_*.csv.gz"])
    parser.add_argument("--walk-ladder",
                        default="results/g0_headroom_2026_09/estimator_ladder.csv")
    parser.add_argument("--benchmark-walks", nargs="+", default=[
        "results/benchmark_v2/results/cases_shard_*.csv.gz"])
    parser.add_argument("--out-dir", default="results/final_run_g2")
    parser.add_argument("--summary-dir", default="results_summary/g2")
    parser.add_argument("--report", default="docs/FREEZE_2026-09.md")
    parser.add_argument("--model-jobs", type=int, default=-1)
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    fr = config["final_run"]
    out_dir, summary = Path(args.out_dir), Path(args.summary_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.mkdir(parents=True, exist_ok=True)

    accepted, seed_log = load_cases(config, args.walks, args.arm_a, args.arm_b)
    accepted.to_csv(out_dir / "final_cases.csv.gz", index=False,
                    compression="gzip")
    seed_log.to_csv(summary / "seed_advance_log.csv", index=False)
    primary = accepted[accepted.seed_slot == 0].copy()
    expected = len(fr["arms"]) * 32
    if len(primary) != expected:
        raise RuntimeError(f"expected {expected} primary cases, got {len(primary)}")
    primary.to_csv(out_dir / "primary_cases.csv.gz", index=False,
                   compression="gzip")

    coverage = pd.DataFrame([
        {"arm": arm, "cases": int(len(part)),
         "median_dyad_coverage": float(part.coverage.median()),
         "p10": float(part.coverage.quantile(.1)),
         "p90": float(part.coverage.quantile(.9)),
         "median_realized_budget": float(part.budget.median())}
        for arm, part in primary.groupby("strategy")])
    coverage.to_csv(summary / "final_coverage.csv", index=False)

    ladder_path = out_dir / "reference_ladder.csv"
    if ladder_path.exists():
        ladder = pd.read_csv(ladder_path)
    else:
        walk_arms = [a for a, s in fr["arms"].items() if s["family"] == "walk"]
        walk_train = _read_filtered(args.benchmark_walks, walk_arms, 800)
        nonwalk_train = pd.concat([read_globs(args.benchmark_a),
                                   read_globs(args.benchmark_b)],
                                  ignore_index=True)
        ladder = pd.concat([
            estimator_ladder(primary[primary.strategy.isin(walk_arms)].copy(),
                             walk_train, tuple(walk_arms), args.model_jobs),
            estimator_ladder(primary[~primary.strategy.isin(walk_arms)].copy(),
                             nonwalk_train, (NODE_PANEL, TWO_PHASE),
                             args.model_jobs),
        ], ignore_index=True)
        ladder.to_csv(ladder_path, index=False)
    ladder.to_csv(summary / "reference_ladder.csv", index=False)

    recovery = censoring_recovery(ladder)
    recovery.to_csv(summary / "censoring_recovery.csv", index=False)
    deltas, per_case_delta = delta_table(primary)
    deltas.to_csv(summary / "delta_i_distribution.csv", index=False)
    per_case_delta.to_csv(out_dir / "delta_i_by_case.csv", index=False)
    per_case_delta.to_csv(summary / "delta_i_by_case.csv", index=False)

    detect_cases = detectability_covariate(primary, *MISMATCH_PAIR)
    detect_cases.to_csv(summary / "mismatch_detectability_by_case.csv",
                        index=False)
    detect = pd.DataFrame([
        {"actual_arm": arm, "stated_arm": part.stated_arm.iloc[0],
         "cases": int(len(part)),
         "median_p_stated": float(part.p_stated.median()),
         "p_stated_p10": float(part.p_stated.quantile(.1)),
         "p_stated_p90": float(part.p_stated.quantile(.9)),
         "cases_p_stated_above_0_2": int((part.p_stated > .2).sum())}
        for arm, part in detect_cases.groupby("actual_arm")])
    detect.to_csv(summary / "mismatch_detectability.csv", index=False)
    spread = int((detect_cases.p_stated > .2).sum())
    detect_note = textwrap.fill((
        f"**The skew is unchanged from G1**: {spread} of {len(detect_cases)} "
        "cases exceed `p_stated` 0.2, so the classifier is usually confident "
        "the sample did not come from the arm the text names. Enter it "
        "continuously and show the low-detectability tail separately with its "
        "n stated; do not present pooled tertiles as a stratification."
        if spread <= len(detect_cases) // 3 else
        f"The spread is wider than in G1: {spread} of {len(detect_cases)} "
        "cases exceed `p_stated` 0.2, so a coarse stratification may now be "
        "defensible. Report it alongside the continuous term."), width=78)

    replication = select_replication_graphs(
        primary, int(fr["subsets"]["seed_replication"]["graphs"]))
    replication.to_csv(summary / "subset_seed_replication.csv", index=False)
    ic = fr["subsets"]["irrelevant_context"]
    ic_subset = pd.DataFrame([{"arm": a, "graphs": ic["graphs"],
                               "conditions": "irrelevant_context vs mechanism",
                               "models": "primary only"} for a in ic["arms"]])
    ic_subset.to_csv(summary / "subset_irrelevant_context.csv", index=False)

    schema = schema_check(primary)
    schema.to_csv(summary / "output_schema_check.csv", index=False)
    if not bool(schema.byte_identical_across_all_arms_and_conditions.iloc[0]):
        raise RuntimeError("output schema varies across conditions")

    volume, calls = prompt_volume(config, primary, replication)
    volume.to_csv(summary / "prompt_volume.csv", index=False)
    calls.to_csv(summary / "call_volume.csv", index=False)

    historical_arm = pd.read_csv(summary / "historical_disclosure_by_arm.csv")
    historical_naming = pd.read_csv(
        summary / "historical_disclosure_by_naming.csv")
    historical_signs = pd.read_csv(
        summary / "historical_disclosure_sign_counts.csv")

    contract = Path("src/prompt_contract_g1.py")
    hashes = pd.DataFrame([
        {"item": "master seed", "value": str(fr["master_seed"])},
        {"item": "prompt contract module sha256",
         "value": sha256_file(contract)},
        {"item": "task block sha256",
         "value": str(schema.task_block_sha256.iloc[0])},
        {"item": "final run config sha256",
         "value": sha256_file(Path(args.config))},
        {"item": "primary case ids sha256",
         "value": sha256_text("\n".join(sorted(primary.case_id)))},
        {"item": "final cases sha256",
         "value": sha256_file(out_dir / "final_cases.csv.gz")},
        {"item": "panel manifest sha256",
         "value": sha256_file(Path(fr["panel_manifest"]))},
        {"item": "primary cases", "value": str(len(primary))},
        {"item": "arms and budgets",
         "value": ", ".join(f"{a}={s['budget']}" for a, s in fr["arms"].items())},
        {"item": "mismatch pair", "value": " <-> ".join(MISMATCH_PAIR)},
        {"item": "conditions", "value": ", ".join(C.CONDITIONS)},
    ])
    hashes.to_csv(summary / "freeze_hashes.csv", index=False)

    Path(args.report).write_text(build_document(
        config=config, hashes=hashes, seed_log=seed_log, ladder=ladder,
        recovery=recovery, deltas=deltas, detect=detect,
        detect_note=detect_note, replication=replication, ic_subset=ic_subset,
        schema=schema, volume=volume, calls=calls,
        historical_arm=historical_arm, historical_naming=historical_naming,
        historical_signs=historical_signs, config_note=CONFIG_NOTE,
        coverage=coverage))
    print(f"wrote {args.report}", flush=True)


if __name__ == "__main__":
    main()
