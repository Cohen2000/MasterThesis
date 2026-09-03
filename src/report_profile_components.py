"""Freeze (d): does the direction effect hold along the profile, or only at k=2?

The frozen target is the joint profile (rho_2 ... rho_5) with ProfileMAE as its
loss; rho_2 is the headline component, not the target. (d) asks for the same
analysis on rho_3 at minimum, on data already collected, "because a direction
effect that appears only on the headline component is a weaker claim than one
that holds along the profile".

It had not been run: `build_g4_cell.py` validates all four components and then
exports only `rho_k2`. This module answers it directly from the answer files.

Two quantities:

* the position slope of each component k, so the decay across k is visible;
* the frozen group-macro ProfileMAE per condition, which is the loss the target
  freeze actually specifies -- averaged within graph group first, so the twelve
  groups weigh equally rather than the instance count deciding.

Read-only.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/final_run_g2"
BOOTSTRAP = 4000
SEED = 20260901
KS = (2, 3, 4, 5)
CLEAN = ("event_sample_then_full_history", "time_agnostic_t",
         "node_panel_full_history")


def _fit(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def _boot(frame, x, y, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = sorted(frame.group_id.unique())
    by_group = {g: frame[frame.group_id == g] for g in groups}
    draws = []
    for _ in range(BOOTSTRAP):
        pick = rng.choice(groups, size=len(groups), replace=True)
        value = _fit(*(pd.concat([by_group[g] for g in pick],
                                 ignore_index=True)[[x, y]].to_numpy(float).T))
        if np.isfinite(value):
            draws.append(value)
    if not draws:
        return np.nan, np.nan
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load(prompts: str, cases: str) -> pd.DataFrame:
    from build_g4_cell import MODEL_GROUPS, load_raw
    frames = []
    for name, groups in MODEL_GROUPS.items():
        try:
            frames.append(load_raw(name, groups, prompts, cases))
        except SystemExit as exc:
            print(f"{name}: skipped -- {exc}")
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame.all_components_valid & ~frame.provider_refused]
    frame["seed_slot"] = frame.case_id.str.extract(r"\|(?:ss|ws)(\d+)\|").astype(int)
    return frame[frame.seed_slot == 0]


def component_slopes(frame: pd.DataFrame, cases: str) -> pd.DataFrame:
    """Paired slope per component: Delta_k on delta_k, exactly as for k=2."""
    truth = pd.read_csv(cases)[
        ["case_id"] + [f"rho_W5_k{k}" for k in KS]
        + [f"est__plugin_rho_k{k}" for k in KS]]
    rows = []
    for k in KS:
        pred, true, naive = f"rho_k{k}", f"rho_W5_k{k}", f"est__plugin_rho_k{k}"
        part = frame[["model", "case_id", "group_id", "strategy", "condition",
                      "generation", pred]].copy()
        averaged = (part.groupby(["model", "case_id", "group_id", "strategy",
                                  "condition"], as_index=False)[pred].mean())
        wide = averaged.pivot_table(
            index=["model", "case_id", "group_id", "strategy"],
            columns="condition", values=pred).reset_index()
        if not {"hidden", "mechanism"} <= set(wide.columns):
            continue
        wide = wide.dropna(subset=["hidden", "mechanism"])
        wide = wide.merge(truth[["case_id", true, naive]], on="case_id",
                          how="left")
        wide["delta_k"] = wide[true] - wide[naive]
        wide["Delta_k"] = wide.mechanism - wide.hidden
        for model, sub in wide.groupby("model"):
            for scope, part2 in [("clean_arms", sub[sub.strategy.isin(CLEAN)]),
                                 ("time_agnostic_t",
                                  sub[sub.strategy == "time_agnostic_t"])]:
                if len(part2) < 4:
                    continue
                slope = _fit(part2.delta_k, part2.Delta_k)
                lo, hi = _boot(part2, "delta_k", "Delta_k")
                rows.append({"model": model, "scope": scope, "k": k,
                             "n": len(part2), "slope": slope,
                             "ci_lo": lo, "ci_hi": hi,
                             "mean_required": float(part2.delta_k.mean())})
    return pd.DataFrame(rows)


def profile_mae(frame: pd.DataFrame, cases: str) -> pd.DataFrame:
    """The frozen loss: mean |pred - truth| over the four components.

    Group-macro: averaged within graph group first, so the twelve groups weigh
    equally. That is the aggregation the target freeze specifies, and it is not
    the same as a flat mean over cases.
    """
    truth = pd.read_csv(cases)[["case_id"] + [f"rho_W5_k{k}" for k in KS]]
    part = frame.merge(truth, on="case_id", how="left", suffixes=("", "_t"))
    pred = part[[f"rho_k{k}" for k in KS]].to_numpy(float)
    real = part[[f"rho_W5_k{k}" for k in KS]].to_numpy(float)
    part["profile_mae"] = np.abs(pred - real).mean(axis=1)
    per_case = (part.groupby(["model", "strategy", "condition", "group_id",
                              "case_id"], as_index=False)
                .profile_mae.mean())
    per_group = (per_case.groupby(["model", "strategy", "condition",
                                   "group_id"], as_index=False)
                 .profile_mae.mean())
    out = (per_group.groupby(["model", "strategy", "condition"])
           .agg(groups=("profile_mae", "size"),
                profile_mae=("profile_mae", "mean")).reset_index())
    return out.sort_values(["model", "strategy", "condition"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(BASE / "prompts.jsonl"))
    ap.add_argument("--cases", default=str(BASE / "final_cases.csv.gz"))
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    frame = load(args.prompts, args.cases)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    slopes = component_slopes(frame, args.cases)
    slopes.to_csv(out / "profile_component_slopes.csv", index=False)
    mae = profile_mae(frame, args.cases)
    mae.to_csv(out / "profile_mae_by_condition.csv", index=False)

    pd.set_option("display.width", 200)
    print("== paired slope per component\n",
          slopes.round(4).to_string(index=False))
    print("\n== frozen group-macro ProfileMAE, clean arms\n",
          mae[mae.strategy.isin(CLEAN)].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
