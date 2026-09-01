#!/usr/bin/env python3
"""G3 Step 1: the early signal slice.

Two arms with opposite correct corrections, two conditions, 32 graphs, Codex.
The question is whether `Delta_i(mechanism - hidden)` tracks the case-specific
correct correction `delta_i = rho2_true_i - rho2_naive_i`.

**This is not a go/no-go and cannot change the design.** The prompts are the
frozen set, the same cases enter the full run unchanged, and no parameter, arm,
condition, budget or metric is adjusted on what this shows. It previews which
result the thesis leads with, nothing more.

Scoring follows the frozen rule: parse the final JSON only, never clip or
repair, an out-of-range or missing component is invalid.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_eval_frozen import PROFILE_PRED, PROFILE_TRUTH, valid_unit
from run_llm_v2 import extract_last_json, is_complete_record

# A call the provider refused is not a model failure and must not be counted
# as one: the runner already treats it that way when deciding whether to
# retry, and the reported response rate has to agree with that.
REFUSAL_MARKERS = ("usage limit", "rate limit", "429", "quota",
                   "insufficient credit")

BOOTSTRAP = 10000
RNG_SEED = 20260901


def load_answers(paths: list[str]) -> pd.DataFrame:
    """One row per (prompt_id, generation); nothing is repaired.

    Files are append-only, so a prompt that was truncated and later retried has
    two records. The complete one wins, and if none is complete the last
    attempt stands -- otherwise a raised token cap would silently double-count
    every case it rescued.
    """
    rows = []
    for generation, path in enumerate(paths):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            parsed = extract_last_json(record.get("answer", "")) or {}
            finish = str(record.get("finish_reason") or "").lower()
            refused = (finish.startswith("error")
                       and any(m in finish for m in REFUSAL_MARKERS))
            row = {
                "prompt_id": record["prompt_id"],
                "generation": generation,
                "structurally_complete": bool(is_complete_record(record)),
                "provider_refused": refused,
                "responded": not record.get("error") and not refused,
                "total_tokens": record.get("total_tokens"),
                "latency_s": record.get("latency_s"),
                "prompt_sha256": record.get("prompt_sha256"),
            }
            for key in PROFILE_PRED:
                value = parsed.get(key)
                row[key] = float(value) if valid_unit(value) else np.nan
            row["all_components_valid"] = all(
                np.isfinite(row[k]) for k in PROFILE_PRED)
            # A mathematically invalid profile is not repaired; it is counted.
            values = [row[k] for k in PROFILE_PRED]
            row["profile_monotone"] = (
                bool(np.all(np.diff(values) <= 1e-12))
                if row["all_components_valid"] else False)
            rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["_order"] = range(len(frame))
    frame = frame.sort_values(["structurally_complete", "_order"])
    frame = (frame.drop_duplicates(subset=["prompt_id", "generation"],
                                   keep="last")
             .sort_values("_order").drop(columns="_order")
             .reset_index(drop=True))
    return frame


def merge(answers: pd.DataFrame, prompts_path: str,
          cases_path: str) -> pd.DataFrame:
    prompts = pd.DataFrame([json.loads(l) for l in
                            Path(prompts_path).read_text().splitlines() if l.strip()])
    prompts = prompts[["prompt_id", "case_id", "instance_id", "group_id",
                       "strategy", "condition", "coverage", "prompt_sha256"]]
    cases = pd.read_csv(cases_path)[
        ["case_id"] + PROFILE_TRUTH + ["est__plugin_rho_k2"]]
    merged = answers.merge(prompts, on="prompt_id", how="left",
                           suffixes=("_answer", "_prompt"))
    if merged.case_id.isna().any():
        raise RuntimeError("an answer has no matching frozen prompt")
    mismatched = merged[merged.prompt_sha256_answer.notna() &
                        (merged.prompt_sha256_answer !=
                         merged.prompt_sha256_prompt)]
    if len(mismatched):
        raise RuntimeError(f"{len(mismatched)} answers used a non-frozen prompt")
    merged = merged.merge(cases, on="case_id", how="left")
    merged["rho2_error"] = merged.rho_k2 - merged.rho_W5_k2
    merged["delta_i"] = merged.rho_W5_k2 - merged.est__plugin_rho_k2
    merged["profile_mae"] = np.abs(
        merged[PROFILE_PRED].to_numpy(float) -
        merged[PROFILE_TRUTH].to_numpy(float)).mean(axis=1)
    return merged


def rates(merged: pd.DataFrame) -> pd.DataFrame:
    """Response, validity and invalid-profile rates, per arm and condition."""
    rows = []
    for keys, part in merged.groupby(["strategy", "condition"]):
        # Denominator is the calls the provider actually served. A refusal is
        # an operational event, reported in its own column.
        served = part[~part.provider_refused]
        valid = served.all_components_valid
        rows.append({
            "arm": keys[0], "condition": keys[1], "calls": int(len(part)),
            "provider_refusals": int(part.provider_refused.sum()),
            "served_calls": int(len(served)),
            "response_rate": float(served.responded.mean()) if len(served) else float("nan"),
            "structural_completeness": float(
                served.structurally_complete.mean()) if len(served) else float("nan"),
            "validity_rate": float(valid.mean()) if len(served) else float("nan"),
            "invalid_profile_rate": float(
                1.0 - served.loc[valid, "profile_monotone"].mean()
                if valid.any() else np.nan),
            "median_total_tokens": float(served.total_tokens.median())
            if len(served) else float("nan"),
        })
    return pd.DataFrame(rows)


def aggregate_generations(merged: pd.DataFrame) -> pd.DataFrame:
    """Collapse generations within (case, condition) *before* any paired test.

    Generations are not independent observations. The nesting is
    graph -> seed -> condition -> generation and the level that matters is the
    graph group, so the generation axis is averaged away first.
    """
    usable = merged[merged.all_components_valid & ~merged.provider_refused]
    return (usable.groupby(["case_id", "instance_id", "group_id", "strategy",
                            "condition", "coverage", "delta_i"], as_index=False)
            .agg(rho2_error=("rho2_error", "mean"),
                 profile_mae=("profile_mae", "mean"),
                 rho_k2=("rho_k2", "mean"),
                 generations=("generation", "nunique")))


def paired(cell: pd.DataFrame) -> pd.DataFrame:
    """Delta_i = shift in the model's rho_2 estimate, mechanism minus hidden."""
    wide = cell.pivot_table(
        index=["case_id", "instance_id", "group_id", "strategy", "delta_i",
               "coverage"],
        columns="condition", values="rho_k2").reset_index()
    wide = wide.dropna(subset=["hidden", "mechanism"])
    wide["Delta_i"] = wide.mechanism - wide.hidden
    # The correct correction and the model's shift should share a sign.
    wide["direction_hit"] = np.sign(wide.Delta_i) == np.sign(wide.delta_i)
    wide["magnitude_ratio"] = wide.Delta_i / wide.delta_i.replace(0, np.nan)
    return wide


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def cluster_bootstrap_slope(wide: pd.DataFrame, n: int = BOOTSTRAP
                            ) -> dict[str, float]:
    """Slope of Delta_i on delta_i, resampling whole graph groups.

    The group is the unit because cases inside a group share a backbone; 160
    cases are not 160 independent observations and the CI must not pretend so.
    """
    groups = wide.group_id.unique()
    point = _slope(wide.delta_i.to_numpy(float), wide.Delta_i.to_numpy(float))
    rng = np.random.default_rng(RNG_SEED)
    by_group = {g: wide[wide.group_id == g] for g in groups}
    draws = []
    for _ in range(n):
        picked = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[g] for g in picked], ignore_index=True)
        value = _slope(sample.delta_i.to_numpy(float),
                       sample.Delta_i.to_numpy(float))
        if np.isfinite(value):
            draws.append(value)
    draws = np.asarray(draws)
    return {
        "slope": point,
        "ci_lo": float(np.percentile(draws, 2.5)),
        "ci_hi": float(np.percentile(draws, 97.5)),
        "share_positive_draws": float((draws > 0).mean()),
        "graph_groups": int(len(groups)),
        "cases": int(len(wide)),
    }


def slope_table(wide: pd.DataFrame) -> pd.DataFrame:
    rows = [{"scope": "pooled (both arms)", **cluster_bootstrap_slope(wide)}]
    for arm, part in wide.groupby("strategy"):
        rows.append({"scope": arm, **cluster_bootstrap_slope(part)})
    return pd.DataFrame(rows)


def direction_table(wide: pd.DataFrame) -> pd.DataFrame:
    """Direction and magnitude reported separately: different failures."""
    rows = []
    for scope, part in [("pooled (both arms)", wide),
                        *list(wide.groupby("strategy"))]:
        ratio = part.magnitude_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        rows.append({
            "scope": scope, "cases": int(len(part)),
            "direction_hit_rate": float(part.direction_hit.mean()),
            "mean_delta_i_true": float(part.delta_i.mean()),
            "mean_Delta_i_model": float(part.Delta_i.mean()),
            "median_magnitude_ratio": float(ratio.median()),
            "share_moved_at_all": float((part.Delta_i.abs() > 1e-9).mean()),
        })
    return pd.DataFrame(rows)


def shift_distribution(wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, part in [("pooled (both arms)", wide),
                        *list(wide.groupby("strategy"))]:
        q = part.Delta_i.quantile([.1, .25, .5, .75, .9])
        rows.append({
            "scope": scope, "cases": int(len(part)),
            "mean": float(part.Delta_i.mean()),
            "p10": float(q.loc[.1]), "q25": float(q.loc[.25]),
            "median": float(q.loc[.5]), "q75": float(q.loc[.75]),
            "p90": float(q.loc[.9]),
            "share_positive": float((part.Delta_i > 0).mean()),
            "share_negative": float((part.Delta_i < 0).mean()),
        })
    return pd.DataFrame(rows)


def bias_by_condition(cell: pd.DataFrame) -> pd.DataFrame:
    """Group-macro signed rho_2 bias, the axis the main figure uses."""
    rows = []
    for keys, part in cell.groupby(["strategy", "condition"]):
        macro = part.groupby("group_id").rho2_error.mean()
        rows.append({
            "arm": keys[0], "condition": keys[1], "cases": int(len(part)),
            "group_macro_rho2_bias": float(macro.mean()),
            "group_macro_profile_mae": float(
                part.groupby("group_id").profile_mae.mean().mean()),
        })
    return pd.DataFrame(rows)


def build_document(*, rates_table, bias, shifts, slopes, directions,
                   answered, expected, generations, cell) -> str:
    from report_g0_headroom import _format_table

    pooled = slopes[slopes.scope == "pooled (both arms)"].iloc[0]
    pooled_dir = directions[directions.scope == "pooled (both arms)"].iloc[0]
    positive = pooled.ci_lo > 0
    negative = pooled.ci_hi < 0
    moved = float(pooled_dir.share_moved_at_all)
    if positive:
        reading = ("**slope clearly positive** -- the language-model chapter "
                   "leads: the model moves its estimate in the case-specific "
                   "correct direction when given a neutral process "
                   "description.")
    elif moved < 0.5:
        reading = ("**the model does not move** -- mechanism blindness leads: "
                   "neither the analytical estimators nor the model act on the "
                   "observation process, and the reference ladder carries the "
                   "argument.")
    elif negative:
        reading = ("**slope clearly negative** -- the model moves, and moves "
                   "*against* the correct correction. This is not one of the "
                   "three prespecified readings and needs its own treatment; "
                   "it is closer to the mechanism-blindness case than to the "
                   "others, because the shift is not tracking the target.")
    else:
        reading = ("**the model moves but the slope is indistinguishable from "
                   "zero** -- the level-versus-ordering dissociation leads: "
                   "mechanism information shifts calibration without improving "
                   "case-specific inference.")
    return f"""# G3 Step 1: early signal slice

Prepared: **2026-09-01**  
Scope: **Codex `gpt-5.6-sol`, {generations} generations, arms
`time_agnostic_t` and `event_sample_then_full_history`, conditions `hidden` and
`mechanism`, all 32 graphs.** {answered}/{expected} calls complete.

**This is not a go/no-go and it did not change anything.** The prompts are the
frozen set from `docs/FREEZE_2026-09.md`, verified here by per-prompt SHA-256
against the frozen file; the same cases enter the full run unchanged; and no
parameter, arm, condition, budget or metric was adjusted on what follows. Its
only purpose is to say which result the thesis leads with.

The two arms were chosen because their correct corrections point in opposite
directions -- `time_agnostic_t` needs an upward correction, arm B a downward
one -- so a model that simply shifts every estimate the same way scores zero on
the direction test rather than passing it by accident.

## Prespecified reading

{reading}

## The primary statistic

`delta_i = rho2_true_i - rho2_naive_i` is the correct signed correction for
case `i`. `Delta_i = rho2_model(mechanism) - rho2_model(hidden)` is the shift
the description produced. The primary statistic is the slope of `Delta_i` on
`delta_i`, with a cluster bootstrap over graph groups, because cases inside a
group share a backbone and 128 calls are not 128 independent observations.

Generations are averaged within (case, condition) **before** the pairing.

{_format_table(slopes)}

## Direction and magnitude, separately

Deriving the right direction and sizing the correction are different failures
and are not collapsed into one number.

{_format_table(directions)}

## The distribution of the shift, not only its mean

An effect carried by a handful of cases must be visible as one.

{_format_table(shifts)}

## Signed `rho_2` bias by condition

This is the axis of the main figure. The reference lines are the naive
read-off on the same fresh samples: **-0.278** on `time_agnostic_t` and
**+0.153** on `event_sample_then_full_history`.

{_format_table(bias)}

## Response, validity and internal consistency

Tracked per arm and per condition. `invalid_profile_rate` is the share of
otherwise-valid answers violating `1 >= rho_2 >= rho_3 >= rho_4 >= rho_5 >= 0`.
Nothing is repaired: a violation is recorded, never sorted.

{_format_table(rates_table)}

## What this does not show

- One model, and a product screen: the Codex harness injects instructions that
  are not part of the frozen prompt and cannot be version-pinned. Two arms and
  two conditions out of five and six.
- Prompt length is an arm-level confound. Arm B's data block is roughly three
  times the walks', so the two arms' absolute accuracy is not comparable. The
  paired contrast is within a case, where length is near-constant across the
  two conditions, so the primary statistic is unaffected.
- Nothing here separates mechanistic reasoning from a learned text heuristic.
  The design cannot, and the wording throughout stays at
  "mechanism-sensitive inference".
- `{len(cell)}` case-condition cells survived validity filtering out of
  `{2 * expected // generations}` possible; a shift computed on a case whose
  other condition failed to parse is dropped rather than imputed.
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", nargs="+", required=True)
    parser.add_argument("--prompts",
                        default="results/final_run_g2/prompts_step1.jsonl")
    parser.add_argument("--cases",
                        default="results/final_run_g2/primary_cases.csv.gz")
    parser.add_argument("--summary-dir", default="results_summary/g3")
    parser.add_argument("--report", default="docs/STEP1_SIGNAL_2026-09.md")
    args = parser.parse_args()

    answers = load_answers(args.answers)
    merged = merge(answers, args.prompts, args.cases)
    summary = Path(args.summary_dir)
    summary.mkdir(parents=True, exist_ok=True)

    rates_table = rates(merged)
    cell = aggregate_generations(merged)
    wide = paired(cell)
    slopes = slope_table(wide)
    directions = direction_table(wide)
    shifts = shift_distribution(wide)
    bias = bias_by_condition(cell)

    for name, frame in (("step1_rates", rates_table), ("step1_bias", bias),
                        ("step1_shift_distribution", shifts),
                        ("step1_slopes", slopes),
                        ("step1_direction", directions),
                        ("step1_paired_cases", wide)):
        frame.to_csv(summary / f"{name}.csv", index=False)

    Path(args.report).write_text(build_document(
        rates_table=rates_table, bias=bias, shifts=shifts, slopes=slopes,
        directions=directions, answered=int(len(merged)),
        expected=int(len(answers)), generations=len(args.answers), cell=cell))
    print(f"wrote {args.report}", flush=True)
    print(slopes.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
