#!/usr/bin/env python3
"""Evaluate the bounded non-walk Qwen screen with the frozen scoring rule."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_eval_frozen import is_complete_record, score


def load_answer_groups(patterns):
    """Return one answer map per (model, thinking), preferring completeness."""
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(pattern))
    latest, complete = {}, {}
    for path in sorted(set(paths)):
        if "smoke" in Path(path).name:
            continue
        with open(path) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_id = record.get("prompt_id")
                if not prompt_id:
                    continue
                mode = (record.get("model", "unknown"),
                        record.get("thinking", "unknown"))
                key = mode + (prompt_id,)
                latest[key] = record
                if is_complete_record(record):
                    complete[key] = record
    grouped = {}
    for key, record in latest.items():
        model, thinking, prompt_id = key
        grouped.setdefault((model, thinking), {})[prompt_id] = \
            complete.get(key, record)
    return grouped


def load_prompts(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_truth(path):
    frame = pd.read_csv(path, low_memory=False)
    return frame, {str(row["case_id"]): row for row in frame.to_dict("records")}


def llm_metrics(answer_groups, prompts, truth):
    rows = []
    strategies = sorted({p["strategy"] for p in prompts})
    conditions = sorted({p["condition"] for p in prompts})
    for (model, thinking), answers in sorted(answer_groups.items()):
        for condition in conditions:
            for strategy in strategies + ["ALL"]:
                subset = [p for p in prompts
                          if p["condition"] == condition and
                          (strategy == "ALL" or p["strategy"] == strategy)]
                mapping = {p["case_id"]: p["prompt_id"] for p in subset}
                case_ids = sorted(mapping)
                result = score(answers, mapping, case_ids, truth)
                if result is None:
                    result = {
                        "n": 0, "missing": len(case_ids),
                        "response_rate": np.nan, "validity": np.nan,
                        "profile_mae_penalized": np.nan,
                        "profile_mae_complete": np.nan,
                        "rho2_mae": np.nan, "rho2_bias": np.nan,
                        "rho2_spearman": np.nan, "c_mae": np.nan,
                        "c_validity": np.nan, "cover90": np.nan,
                        "violation_rate": np.nan,
                    }
                result.pop("per_case", None)
                rows.append({
                    "source": "llm", "model": model,
                    "thinking": thinking, "condition": condition,
                    "strategy": strategy, "n_expected": len(case_ids),
                    **result,
                })
    return rows


def baseline_metrics(prediction_path, selected_cases, prompts):
    if not prediction_path:
        return []
    predictions = pd.read_csv(prediction_path, low_memory=False)
    selected_ids = set(selected_cases.case_id.astype(str))
    predictions = predictions[
        predictions.case_id.astype(str).isin(selected_ids) &
        predictions.target.isin(
            ["rho_W5_k2", "rho_W5_k3", "rho_W5_k4", "rho_W5_k5"])]
    strategies = sorted({p["strategy"] for p in prompts})
    condition_model = {
        "sample": "ET_prompt_parity",
        "metadata_only_no_sample": "ET_metadata_only",
    }
    rows = []
    for condition in sorted({p["condition"] for p in prompts}):
        model = condition_model.get(condition)
        if model is None:
            continue
        for strategy in strategies + ["ALL"]:
            expected = {p["case_id"] for p in prompts
                        if p["condition"] == condition and
                        (strategy == "ALL" or p["strategy"] == strategy)}
            group = predictions[(predictions.model == model) &
                                predictions.case_id.astype(str).isin(expected)]
            group = group.copy()
            group["ae"] = (group.y_true - group.prediction).abs()
            case_loss = group.groupby("case_id").ae.mean()
            n = len(case_loss)
            mae = float(case_loss.mean()) if n else np.nan
            rows.append({
                "source": "baseline", "model": model, "thinking": "n/a",
                "condition": condition, "strategy": strategy,
                "n_expected": len(expected), "n": n,
                "missing": len(expected) - n, "response_rate": 1.0,
                "validity": 1.0, "profile_mae_penalized": mae,
                "profile_mae_complete": mae, "rho2_mae": np.nan,
                "rho2_bias": np.nan, "rho2_spearman": np.nan,
                "c_mae": np.nan, "c_validity": np.nan,
                "cover90": np.nan, "violation_rate": 0.0,
            })
    return rows


def fmt(value, digits=3):
    try:
        return f"{float(value):.{digits}f}" if np.isfinite(float(value)) else "-"
    except (TypeError, ValueError):
        return "-"


def write_summary(metrics, out_path, title=None):
    """Write the markdown report.

    The title is derived from the models actually present rather than fixed.
    The same evaluator serves the bounded Qwen screen and the multi-model
    expansion, and a report headed "Qwen screen" while listing five models is
    the kind of label that gets a table quoted for the wrong experiment.
    """
    models = sorted({str(m) for m in metrics.model.unique()
                     if str(m) != "nan" and not str(m).startswith("ET_")})
    if title is None:
        title = ("# Non-walk access evaluation"
                 if len(models) > 1 else "# Non-walk Qwen screen")
    lines = [
        title, "",
        f"Models in this report: {len(models)} "
        f"({', '.join(models)}).",
        "",
        "Failure-penalized profile MAE is the primary metric; lower is better.",
        "Missing jobs are reported separately and are not silently scored as answers.",
        "",
        "| source | model/mode | condition | strategy | answered/expected | valid | penalized MAE | complete MAE |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    detail = metrics[metrics.strategy != "ALL"].sort_values(
        ["condition", "strategy", "source", "thinking"])
    for _, row in detail.iterrows():
        label = f"{row.model} ({row.thinking})"
        lines.append(
            f"| {row.source} | {label} | {row.condition} | {row.strategy} | "
            f"{int(row.n)}/{int(row.n_expected)} | {fmt(row.validity, 2)} | "
            f"{fmt(row.profile_mae_penalized)} | "
            f"{fmt(row.profile_mae_complete)} |")

    sample = metrics[(metrics.source == "llm") &
                     (metrics.strategy != "ALL")].copy()
    if not sample.empty:
        pivot = sample.pivot_table(
            index=["model", "thinking", "strategy"], columns="condition",
            values="profile_mae_penalized", aggfunc="first")
        if {"sample", "metadata_only_no_sample"}.issubset(pivot.columns):
            pivot["sample_gain"] = (pivot["metadata_only_no_sample"] -
                                    pivot["sample"])
            lines += [
                "", "## Sample contribution", "",
                "Positive values mean the sampled observation beat metadata-only.",
                "", "| model/mode | strategy | metadata MAE - sample MAE |",
                "|---|---|---:|",
            ]
            for (model, thinking, strategy), row in pivot.sort_index().iterrows():
                lines.append(
                    f"| {model} ({thinking}) | {strategy} | "
                    f"{fmt(row.sample_gain)} |")
    out_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--answers", action="append", required=True,
                        help="answer JSONL path or quoted glob; repeatable")
    parser.add_argument("--baseline-predictions", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    prompts = load_prompts(args.prompts)
    selected_cases, truth = load_truth(args.cases)
    rows = llm_metrics(load_answer_groups(args.answers), prompts, truth)
    rows += baseline_metrics(args.baseline_predictions, selected_cases, prompts)
    metrics = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    write_summary(metrics, out / "SUMMARY.md")
    overall = metrics[metrics.strategy == "ALL"]
    if not overall.empty:
        print(overall[[
            "source", "model", "thinking", "condition", "n", "n_expected",
            "validity", "profile_mae_penalized", "profile_mae_complete",
        ]].round(4).to_string(index=False))
    print(f"wrote {out / 'metrics.csv'} and {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
