#!/usr/bin/env python3
"""Create compact v2 diagnostic tables after the main evaluation."""

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd


def read_cases(pattern):
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(pattern)
    header = pd.read_csv(paths[0], nrows=0).columns.tolist()
    wanted = [c for c in header if c in {
        "case_id", "instance_id", "group_id", "strategy", "data_block",
        "domain", "budget", "coverage", "walk_seed", "rho_W5_k2",
        "est__plugin_rho_k2", "est__conditional_rho_k2",
        "est__occ_mle_rho_k2", "est__mask_mle_rho_k2",
        "diag__edgebank_frequency_auc", "diag__edgebank_recency_auc",
    }]
    return pd.concat([pd.read_csv(p, usecols=wanted) for p in paths], ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/benchmark_v2/manifest.csv")
    ap.add_argument("--cases", default="results/benchmark_v2/cases_shard_*.csv.gz")
    ap.add_argument("--metrics", default="results/benchmark_v2/metrics.csv")
    ap.add_argument("--out-dir", default="results/benchmark_v2")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    mf = pd.read_csv(args.manifest)
    targets = ["rho_W5_k2", "rho_W5_k3", "rho_W5_k4", "rho_W5_k5",
               "C_one_step", "mean_span_frac", "rho_event_weighted"]
    dist = (mf.groupby("data_block")[targets]
              .agg(["count", "mean", "std", "min", "median", "max"]))
    dist.columns = [f"{a}__{b}" for a, b in dist.columns]
    dist.reset_index().to_csv(out / "target_distribution_by_block.csv", index=False)

    cases = read_cases(args.cases)
    diag_cols = [c for c in cases if c.startswith("diag__")]
    if diag_cols:
        diag = (cases.groupby(["strategy", "data_block", "budget"])[diag_cols]
                     .agg(["count", "mean", "median", "std"]))
        diag.columns = [f"{a}__{b}" for a, b in diag.columns]
        diag.reset_index().to_csv(out / "edgebank_auc_summary.csv", index=False)

    est_cols = [c for c in cases if c.startswith("est__")]
    if est_cols and "walk_seed" in cases:
        seed = (cases.groupby(["instance_id", "strategy", "budget"])[est_cols + ["coverage"]]
                     .std(ddof=0).reset_index())
        summary = (seed.groupby(["strategy", "budget"])[est_cols + ["coverage"]]
                       .agg(["mean", "median", "max"]))
        summary.columns = [f"{a}__seed_sd_{b}" for a, b in summary.columns]
        summary.reset_index().to_csv(out / "seed_noise_floor.csv", index=False)

    metrics = pd.read_csv(args.metrics)
    headline = metrics[(metrics["target"] == "rho_W5_k2") &
                       (metrics["slice_type"] == "overall")].copy()
    headline.sort_values(["protocol", "strategy", "mae_group_macro"]).to_csv(
        out / "headline_all_protocols.csv", index=False)

    lines = ["# V2 diagnostic overview", "",
             f"- Instances: {len(mf):,}",
             f"- Independent groups: {mf.group_id.nunique():,}",
             f"- Cases: {len(cases):,}", "",
             "## Data blocks", "",
             mf.groupby("data_block").size().to_markdown(), "",
             "## Best headline rows per protocol/access", ""]
    for (protocol, strategy), g in headline.groupby(["protocol", "strategy"]):
        r = g.nsmallest(1, "mae_group_macro").iloc[0]
        lines.append(f"- **{protocol} / {strategy}:** {r['model']} "
                     f"[{r['input']}] — group-macro MAE {r['mae_group_macro']:.4f}, "
                     f"worst-group {r['mae_worst_group']:.4f}")
    (out / "V2_DIAGNOSTICS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote v2 diagnostics to {out}")


if __name__ == "__main__":
    main()
