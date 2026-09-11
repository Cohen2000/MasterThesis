#!/usr/bin/env python3
"""Validate and summarize the Panel32 non-walk sampling screen."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_STRATEGIES = {
    "uniform_event_reservoir", "time_prefix_events",
    "time_random_window_events", "node_panel_full_history",
    "ego_recent_k1", "ego_recent_k5", "ego_recent_k20",
    "ego_recent_kall",
}


def validate(cases):
    errors = []
    if cases.case_id.duplicated().any():
        errors.append("duplicate case_id")
    missing = EXPECTED_STRATEGIES - set(cases.strategy)
    if missing:
        errors.append(f"missing strategies: {sorted(missing)}")
    if not cases.filter(regex=r"^(occ__|pat__|crawl__|wcnt__|est__)").columns.is_unique:
        errors.append("duplicate model feature columns")
    forbidden = [c for c in cases if c.startswith(
        ("occ__", "pat__", "crawl__", "wcnt__", "est__")) and
        ("true" in c or "oracle" in c or "rho_target" in c)]
    if forbidden:
        errors.append(f"truth leaked into feature prefixes: {forbidden}")
    exact_budget_designs = {
        "uniform_event_reservoir", "time_prefix_events",
        "time_random_window_events",
    }
    ordinary = cases[cases.strategy.isin(exact_budget_designs)]
    mismatch = ordinary[ordinary.budget != ordinary.target_budget]
    if len(mismatch):
        errors.append(f"{len(mismatch)} fixed-size cases missed exact event budget")
    panel100 = cases[(cases.strategy == "node_panel_full_history") &
                     (cases.target_budget == 100)]
    if len(panel100):
        errors.append("node panel unexpectedly contains budget 100")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="results/nonwalk_screen/panel32_cases.csv.gz")
    parser.add_argument("--out-dir", default="results/nonwalk_screen/diagnostics")
    args = parser.parse_args()
    cases = pd.read_csv(args.cases)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    errors = validate(cases)

    stationarity_cols = [
        "instance_id", "group_id", "graph_category",
        "diag__event_rate_cv", "diag__event_rate_linear_slope_per_bin_mean",
        "diag__late_early_event_rate_ratio",
        "diag__nodes_seen_first_half_share", "diag__dyads_seen_first_half_share",
    ]
    stationarity_cols = [c for c in stationarity_cols if c in cases]
    stationarity = cases[stationarity_cols].drop_duplicates("instance_id")
    stationarity.to_csv(out / "stationarity_by_graph.csv", index=False)

    b800 = cases[cases.target_budget == 800]
    degree = (b800.groupby("strategy", as_index=False)
              .agg(n=("case_id", "size"),
                   degree_ratio_median=("diag__selected_degree_mean_ratio", "median"),
                   degree_ratio_q10=("diag__selected_degree_mean_ratio",
                                     lambda x: x.quantile(.1)),
                   degree_ratio_q90=("diag__selected_degree_mean_ratio",
                                     lambda x: x.quantile(.9)),
                   degree_ks_median=("diag__selected_degree_ks", "median")))
    degree.to_csv(out / "degree_selection_b800.csv", index=False)

    panel = cases[cases.strategy == "node_panel_full_history"].copy()
    panel["realized_to_target"] = panel.budget / panel.target_budget
    panel_summary = (panel.groupby("target_budget", as_index=False)
                     .agg(n=("case_id", "size"), empty=("budget", lambda x: int((x == 0).sum())),
                          realized_median=("budget", "median"),
                          ratio_median=("realized_to_target", "median"),
                          ratio_q10=("realized_to_target", lambda x: x.quantile(.1)),
                          ratio_q90=("realized_to_target", lambda x: x.quantile(.9))))
    panel_summary.to_csv(out / "node_panel_budget.csv", index=False)

    ego = cases[cases.strategy.str.startswith("ego_recent_k")].copy()
    ego_summary = (ego.groupby(["strategy", "target_budget"], as_index=False)
                   .agg(n=("case_id", "size"),
                        queries_median=("sample__query_budget_realized", "median"),
                        partial_share=("sample__partial_query_count",
                                       lambda x: float((x > 0).mean())),
                        coverage_median=("coverage", "median")))
    ego_summary.to_csv(out / "ego_budget.csv", index=False)
    ego["budget_fill_ratio"] = ego.budget / ego.target_budget
    ego_saturation = (ego.groupby(["strategy", "target_budget"], as_index=False)
                      .agg(n=("case_id", "size"),
                           saturated_share=("budget_fill_ratio",
                                            lambda x: float((x < 1).mean())),
                           fill_ratio_median=("budget_fill_ratio", "median"),
                           fill_ratio_min=("budget_fill_ratio", "min")))
    ego_saturation.to_csv(out / "ego_saturation.csv", index=False)

    windows = cases[cases.strategy.isin(
        ["time_prefix_events", "time_random_window_events"])]
    window_summary = (windows.groupby(["strategy", "target_budget"], as_index=False)
                      .agg(n=("case_id", "size"),
                           width_median=("sample__observed_time_width", "median"),
                           width_q10=("sample__observed_time_width", lambda x: x.quantile(.1)),
                           width_q90=("sample__observed_time_width", lambda x: x.quantile(.9))))
    window_summary.to_csv(out / "time_window_width.csv", index=False)

    counts = (cases.groupby(["strategy", "target_budget"], as_index=False)
              .agg(n_cases=("case_id", "size"), n_graphs=("instance_id", "nunique"),
                   n_seeds=("sample_seed", "nunique")))
    counts.to_csv(out / "case_counts.csv", index=False)

    lines = ["# Non-walk sampling diagnostics", "",
             f"Cases: {len(cases)}; graphs: {cases.instance_id.nunique()}; "
             f"errors: {len(errors)}.", ""]
    if errors:
        lines += ["## Validation errors", ""] + [f"- {e}" for e in errors] + [""]
    lines += ["## Degree selection at requested budget 800", "",
              degree.to_markdown(index=False, floatfmt=".3f"), "",
              "## Node-panel realized budgets", "",
              panel_summary.to_markdown(index=False, floatfmt=".3f"), ""]
    (out / "SUMMARY.md").write_text("\n".join(lines))
    print(f"wrote {out}; validation errors={len(errors)}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
