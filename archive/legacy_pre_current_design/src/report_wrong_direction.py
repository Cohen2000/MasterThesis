"""Freeze (k): the wrong-direction cell.

The mechanism description is correct; only the sentence naming which way the
naive estimate errs is inverted. The question is whether the model derives its
correction from the described process or defers to the stated claim.

The two arms carry opposite correct directions -- `time_agnostic_t` needs an
upward correction, `event_sample_then_full_history` a downward one -- so a
model that obeys the false sentence moves in opposite directions in the two
arms, while a nuisance response to prompt length or to the mere presence of an
assertion moves the same way in both. That is the identification, and it is why
the pooled mean shift is not the headline: pure deference averages it away.

Three quantities, all fixed in the freeze before any answer was read:
position slope against `delta_i`, the paired deference shift against the
`mechanism` leg, and the conflict resolution rate.

Read-only.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/final_run_g2"
SEED = 20260901
DRAWS = 4000
CELL = "mechanism_wrong_direction"
REFERENCE = "mechanism"
# Correct direction of the required correction, from the measured naive bias.
# Inverting the stated word therefore points the model the other way.
CORRECT_UP = "time_agnostic_t"
CORRECT_DOWN = "event_sample_then_full_history"


def load_cell(prompts_path: str, cases: str) -> pd.DataFrame:
    """The wrong-direction answers, merged onto their own prompt file."""
    from report_step1_signal import load_answers, PROFILE_TRUTH

    q = str(BASE / "answers/qwen")
    frames = []
    for model, tag in (("qwen36-27b_think", "think"),
                       ("qwen36-27b_nothink", "nothink")):
        groups = [[f"{q}/answers_vllm_wrongdir_qwen36-27b_{tag}_g{g}.jsonl"]
                  for g in (0, 1, 2)]
        parts = []
        for generation, patterns in enumerate(groups):
            paths = [h for p in patterns for h in sorted(glob.glob(p))]
            if not paths:
                continue
            part = load_answers(paths)
            part["generation"] = generation
            parts.append(part)
        if not parts:
            print(f"{model}: no wrong-direction answers yet -- skipped")
            continue
        frame = pd.concat(parts, ignore_index=True)
        frame["model"] = model
        frames.append(frame)
    if not frames:
        raise SystemExit("no wrong-direction answers on disk")
    answers = pd.concat(frames, ignore_index=True)

    prompts = pd.DataFrame([json.loads(l) for l in
                            Path(prompts_path).read_text().splitlines()
                            if l.strip()])
    prompts = prompts[["prompt_id", "case_id", "instance_id", "group_id",
                       "strategy", "condition", "coverage", "correct_direction",
                       "prompt_sha256"]]
    merged = answers.merge(prompts, on="prompt_id", how="inner",
                           suffixes=("_answer", "_prompt"))
    bad = merged[merged.prompt_sha256_answer.notna() &
                 (merged.prompt_sha256_answer != merged.prompt_sha256_prompt)]
    if len(bad):
        raise RuntimeError(f"{len(bad)} answers used a non-frozen prompt")

    case_table = pd.read_csv(cases)[
        ["case_id"] + PROFILE_TRUTH + ["est__plugin_rho_k2"]]
    merged = merged.merge(case_table, on="case_id", how="left")
    merged["delta_i"] = merged.rho_W5_k2 - merged.est__plugin_rho_k2
    merged = merged[merged.all_components_valid & ~merged.provider_refused]
    return merged


def load_reference(cases: str, prompts: str) -> pd.DataFrame:
    """The `mechanism` leg on the same cases, from the main G4 cell."""
    cell = pd.read_csv(REPO / "results_summary/g4/g4_cell.csv")
    part = cell[(cell.condition == REFERENCE)
                & cell.model.str.startswith("qwen")
                & (cell.seed_slot == 0)
                & cell.strategy.isin([CORRECT_UP, CORRECT_DOWN])]
    return part


def cluster_bootstrap(values: pd.DataFrame, column: str,
                      rng: np.random.Generator) -> tuple[float, float, float, float]:
    groups = sorted(values.group_id.unique())
    by_group = {g: values[values.group_id == g][column].to_numpy(float)
                for g in groups}
    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(groups, size=len(groups), replace=True)
        sample = np.concatenate([by_group[g] for g in pick])
        draws.append(sample.mean())
    draws = np.asarray(draws)
    return (float(values[column].mean()), float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float((draws > 0).mean()))


def cluster_bootstrap_slope(frame: pd.DataFrame, x: str, y: str,
                            rng: np.random.Generator):
    groups = sorted(frame.group_id.unique())
    by_group = {g: frame[frame.group_id == g] for g in groups}

    def fit(part):
        if part[x].std(ddof=0) == 0 or len(part) < 3:
            return np.nan
        return float(np.polyfit(part[x], part[y], 1)[0])

    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(groups, size=len(groups), replace=True)
        value = fit(pd.concat([by_group[g] for g in pick], ignore_index=True))
        if np.isfinite(value):
            draws.append(value)
    point = fit(frame)
    if not draws:
        return point, np.nan, np.nan
    draws = np.asarray(draws)
    return (point, float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts",
                    default=str(BASE / "prompts_wrongdir_qwen36-27b_nothink_g0.jsonl"),
                    help="the wrong-direction prompt file; all six are the "
                         "same 64 prompts, only the generation seed differs")
    ap.add_argument("--cases", default=str(BASE / "final_cases.csv.gz"))
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    cell = load_cell(args.prompts, args.cases)
    reference = load_reference(args.cases, args.prompts)
    rng = np.random.default_rng(SEED)

    # Generations are averaged away before any contrast, as everywhere else.
    keys = ["model", "case_id", "instance_id", "group_id", "strategy",
            "coverage", "delta_i"]
    wrong = (cell.groupby(keys, as_index=False)
             .agg(rho2_wrong=("rho_k2", "mean"),
                  generations=("generation", "nunique"),
                  naive=("est__plugin_rho_k2", "first")))
    ref = (reference.groupby(["model", "case_id"], as_index=False)
           .agg(rho2_mechanism=("rho_k2", "mean")))
    joined = wrong.merge(ref, on=["model", "case_id"], how="left")

    # Positive = moved the way the false sentence said. The stated word is the
    # opposite of the correct one, so the sign flips with the arm.
    stated_sign = np.where(joined.strategy == CORRECT_UP, -1.0, +1.0)
    joined["shift_raw"] = joined.rho2_mechanism.notna() * (
        joined.rho2_wrong - joined.rho2_mechanism)
    joined.loc[joined.rho2_mechanism.isna(), "shift_raw"] = np.nan
    joined["shift_toward_stated"] = joined.shift_raw * stated_sign
    joined["Delta_i"] = joined.rho2_wrong - joined.naive
    # Did the model's own correction go the way the evidence requires?
    joined["follows_evidence"] = np.sign(joined.Delta_i) == np.sign(joined.delta_i)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    joined.to_csv(out / "wrong_direction_cases.csv", index=False)

    rows = []
    for (model, arm), part in joined.groupby(["model", "strategy"]):
        slope, lo, hi = cluster_bootstrap_slope(part, "delta_i", "Delta_i", rng)
        paired = part.dropna(subset=["shift_toward_stated"])
        if len(paired):
            shift, s_lo, s_hi, s_pos = cluster_bootstrap(
                paired, "shift_toward_stated", rng)
            raw, r_lo, r_hi, _ = cluster_bootstrap(paired, "shift_raw", rng)
        else:
            shift = s_lo = s_hi = s_pos = raw = r_lo = r_hi = np.nan
        rows.append({
            "model": model, "arm": arm, "n": len(part),
            "n_paired": len(paired),
            "correct_direction": "upward" if arm == CORRECT_UP else "downward",
            "position_slope": slope, "slope_lo": lo, "slope_hi": hi,
            "shift_raw": raw, "raw_lo": r_lo, "raw_hi": r_hi,
            "shift_toward_stated": shift, "shift_lo": s_lo, "shift_hi": s_hi,
            "share_positive_draws": s_pos,
            "follows_evidence_rate": float(part.follows_evidence.mean()),
        })
    table = pd.DataFrame(rows)

    # The reference has to be the *position* slope of the `mechanism` leg on
    # these same cases. The slope in primary_slope.csv is the paired contrast
    # against `hidden` and is not comparable to the position slope above.
    ref_position = (reference.groupby(
        ["model", "case_id", "instance_id", "group_id", "strategy", "delta_i",
         "est__plugin_rho_k2"], as_index=False)
        .agg(rho2_mechanism=("rho_k2", "mean")))
    ref_position["Delta_i"] = (ref_position.rho2_mechanism
                               - ref_position.est__plugin_rho_k2)
    ref_rows = []
    for (model, arm), part in ref_position.groupby(["model", "strategy"]):
        slope, lo, hi = cluster_bootstrap_slope(part, "delta_i", "Delta_i", rng)
        ref_rows.append({"model": model, "arm": arm,
                         "n_mechanism": len(part),
                         "mechanism_position_slope": slope,
                         "mechanism_slope_lo": lo,
                         "mechanism_slope_hi": hi})
    table = table.merge(pd.DataFrame(ref_rows), on=["model", "arm"], how="left")
    table.to_csv(out / "wrong_direction.csv", index=False)

    # The between-arm difference in the raw shift: the sign test that separates
    # deference from a nuisance response. Deference makes it large, a length or
    # assertion artefact makes it approximately zero.
    contrast = []
    for model, part in joined.dropna(subset=["shift_raw"]).groupby("model"):
        up = part[part.strategy == CORRECT_UP]
        down = part[part.strategy == CORRECT_DOWN]
        if up.empty or down.empty:
            continue
        groups = sorted(part.group_id.unique())
        draws = []
        for _ in range(DRAWS):
            pick = rng.choice(groups, size=len(groups), replace=True)
            sample = pd.concat([part[part.group_id == g] for g in pick],
                               ignore_index=True)
            a = sample[sample.strategy == CORRECT_DOWN].shift_raw
            b = sample[sample.strategy == CORRECT_UP].shift_raw
            if a.empty or b.empty:
                continue
            draws.append(a.mean() - b.mean())
        draws = np.asarray(draws)
        contrast.append({
            "model": model,
            "n_up": len(up), "n_down": len(down),
            "mean_shift_down_arm": float(down.shift_raw.mean()),
            "mean_shift_up_arm": float(up.shift_raw.mean()),
            "between_arm_gap": float(down.shift_raw.mean() - up.shift_raw.mean()),
            "ci_lo": float(np.percentile(draws, 2.5)) if len(draws) else np.nan,
            "ci_hi": float(np.percentile(draws, 97.5)) if len(draws) else np.nan,
            "share_positive_draws": float((draws > 0).mean()) if len(draws) else np.nan,
        })
    pd.DataFrame(contrast).to_csv(out / "wrong_direction_contrast.csv",
                                  index=False)

    pd.set_option("display.width", 220)
    print("== per arm\n", table.round(4).to_string(index=False))
    print("\n== between-arm contrast (deference makes this large)\n",
          pd.DataFrame(contrast).round(4).to_string(index=False))
    print(f"\n{len(joined)} cases -> {out / 'wrong_direction_cases.csv'}")


if __name__ == "__main__":
    main()
