#!/usr/bin/env python3
"""Retrospective preview of `mechanism` vs `mechanism_direction`, for free.

Two of the three historical `disclosed` prompts name the direction of the bias
in a `Consequence:` line -- `time_respecting` ("later windows are easier to
record") and `recent_history_k20` ("biased toward recent activity").
`time_agnostic_t`'s does not: it describes the process and stops.

That accident splits the archived V2.1 suite along exactly the axis the G1
condition split was built to separate.  If the `disclosed - hidden` effect is
weaker on `time_agnostic_t` than on the two direction-naming arms, the
historical data already hints that naming the direction is doing part of the
work that "disclosure" was credited with.

This reads archived answers only.  No LLM call, no new prompt.  It is a
retrospective slice of an observational contrast, not an experiment: arm is
confounded with everything else that differs between the three walks, and the
comparison is three arms, not three randomized levels.  It motivates the G1
design; it cannot substitute for it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# The historical `disclosed` text names the direction for these two arms only.
DIRECTION_NAMING = ("time_respecting", "recent_history_k20")
NON_NAMING = ("time_agnostic_t",)


def load_pairs(path: Path, input_kind: str = "mask") -> pd.DataFrame:
    """Cases answered under both conditions, on the one shared input format.

    `hidden` was only ever run on the `mask` input, so restricting to it is
    what makes the two conditions paired rather than merely comparable.
    """
    frame = pd.read_csv(path)
    frame = frame[frame.input_kind == input_kind]
    frame = frame[frame.condition.isin(["hidden", "disclosed"])].copy()
    frame["rho2_error"] = frame.rho_k2 - frame.rho_W5_k2
    wide = frame.pivot_table(
        index=["model", "strategy", "group_id", "case_id"],
        columns="condition", values="rho2_error").reset_index()
    wide = wide.dropna(subset=["hidden", "disclosed"])
    wide["bias_shift"] = wide.disclosed - wide.hidden
    return wide


def _group_macro(frame: pd.DataFrame, column: str) -> float:
    return float(frame.groupby("group_id")[column].mean().mean())


def by_arm(pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, arm), part in pairs.groupby(["model", "strategy"]):
        rows.append({
            "model": model, "arm": arm,
            "direction_named": arm in DIRECTION_NAMING,
            "cases": int(len(part)),
            "hidden_bias": _group_macro(part, "hidden"),
            "disclosed_bias": _group_macro(part, "disclosed"),
            "bias_shift": _group_macro(part, "bias_shift"),
        })
    return pd.DataFrame(rows).sort_values(["model", "arm"], ignore_index=True)


def by_naming(arm_rows: pd.DataFrame) -> pd.DataFrame:
    """Direction-naming arms against the one that only describes the process."""
    rows = []
    for model, part in arm_rows.groupby("model"):
        named = part[part.direction_named]
        plain = part[~part.direction_named]
        rows.append({
            "model": model,
            "shift_direction_named": float(named.bias_shift.mean()),
            "shift_process_only": float(plain.bias_shift.mean()),
            "difference": float(named.bias_shift.mean() - plain.bias_shift.mean()),
            "named_arms": len(named), "process_only_arms": len(plain),
        })
    frame = pd.DataFrame(rows)
    # One model (mistral-small-4_high) overcorrects on `time_agnostic_t` from
    # -0.35 to +0.17, which alone moves the mean. The median across models is
    # reported beside it so the reading does not rest on that single cell.
    for label, value in (("ALL MODELS (mean)", "mean"),
                         ("ALL MODELS (median)", "median")):
        frame.loc[len(frame)] = {
            "model": label,
            "shift_direction_named": getattr(
                frame.shift_direction_named.iloc[:len(rows)], value)(),
            "shift_process_only": getattr(
                frame.shift_process_only.iloc[:len(rows)], value)(),
            "difference": getattr(
                frame.difference.iloc[:len(rows)], value)(),
            "named_arms": np.nan, "process_only_arms": np.nan,
        }
    return frame


def sign_counts(naming: pd.DataFrame, n_models: int) -> pd.DataFrame:
    per_model = naming.iloc[:n_models]
    return pd.DataFrame([{
        "models": n_models,
        "models_with_larger_shift_on_process_only_arm": int(
            (per_model.difference < 0).sum()),
        "models_with_larger_shift_on_direction_naming_arms": int(
            (per_model.difference > 0).sum()),
    }])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", default=(
        "results/llm_v21/eval/parsed_answers.csv.gz"))
    parser.add_argument("--out-dir", default="results_summary/g2")
    args = parser.parse_args()

    pairs = load_pairs(Path(args.answers))
    arm_rows = by_arm(pairs)
    naming = by_naming(arm_rows)
    signs = sign_counts(naming, arm_rows.model.nunique())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arm_rows.to_csv(out / "historical_disclosure_by_arm.csv", index=False)
    naming.to_csv(out / "historical_disclosure_by_naming.csv", index=False)
    signs.to_csv(out / "historical_disclosure_sign_counts.csv", index=False)
    print(arm_rows.to_markdown(index=False, floatfmt=".4f"))
    print()
    print(naming.to_markdown(index=False, floatfmt=".4f"))
    print()
    print(signs.to_markdown(index=False))


if __name__ == "__main__":
    main()
