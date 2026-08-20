#!/usr/bin/env python3
"""Truth-only W robustness for an explicit panel of complete event streams."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from census import normalize, one_step_pair_persistence, profile, spans, window_index


def read_events(path):
    d = pd.read_csv(path, usecols=["u", "v", "t"])
    d["u"] = d["u"].astype(str)
    d["v"] = d["v"].astype(str)
    d["t"] = pd.to_numeric(d["t"], errors="coerce")
    d = d.dropna(subset=["t"])
    return normalize(d)


def targets(events, W):
    pair = events.pair.to_numpy()
    t = events.t.to_numpy(float)
    lo, hi = float(t.min()), float(t.max())
    wi = window_index(t, lo, (hi - lo) / W, W)
    k = spans(pair, wi)
    out = {f"rho_k{j}": profile(k, W)[j] for j in range(1, W + 1)}
    out["mean_occupancy"] = float(k.mean() / W)
    out["C_one_step"] = one_step_pair_persistence(pair, wi, W)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", required=True,
                    help="CSV with instance_id, graph_category, path")
    ap.add_argument("--data-root", default="results/benchmark_v2/data")
    ap.add_argument("--windows", default="4,5,8")
    ap.add_argument("--out-dir", default="results/target_diagnostics/w_robustness")
    args = ap.parse_args()
    panel = pd.read_csv(args.panel)
    required = {"instance_id", "graph_category", "path"}
    if not required.issubset(panel):
        raise SystemExit(f"panel lacks {sorted(required - set(panel.columns))}")
    if len(panel) != 32:
        raise SystemExit(f"panel must contain exactly 32 rows, found {len(panel)}")

    root = Path(args.data_root)
    Ws = [int(x) for x in args.windows.split(",")]
    rows = []
    missing = []
    for r in panel.itertuples(index=False):
        path = root / r.path
        if not path.exists():
            missing.append(str(path))
            continue
        ev = read_events(path)
        for W in Ws:
            rows.append({"instance_id": r.instance_id,
                         "graph_category": r.graph_category, "W": W,
                         **targets(ev, W)})
    if missing:
        sample = "\n".join(missing[:8])
        raise SystemExit(
            f"{len(missing)} panel event files are missing under {root}.\n"
            f"First missing paths:\n{sample}\n"
            "Point --data-root at the benchmark data directory or copy only "
            "the 32 paths listed in the panel.")

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(rows)
    long.to_csv(out / "targets_by_W.csv", index=False)
    base = long[long.W == 5].set_index("instance_id")
    summary = []
    for category in ["all"] + sorted(long.graph_category.unique()):
        ids = (base.index if category == "all"
               else base[base.graph_category == category].index)
        for W in Ws:
            if W == 5:
                continue
            cur = long[(long.W == W) & long.instance_id.isin(ids)].set_index("instance_id")
            joined = base.loc[ids].join(cur, lsuffix="_W5", rsuffix=f"_W{W}")
            # Compare every profile component defined under both windowings.
            # rho_5 has no counterpart for W=4 and is therefore not fabricated.
            common_profile = [f"rho_k{k}" for k in range(2, min(5, W) + 1)]
            for target in common_profile + ["mean_occupancy", "C_one_step"]:
                a, b = joined[f"{target}_W5"], joined[f"{target}_W{W}"]
                summary.append({
                    "graph_category": category, "comparison": f"W5_vs_W{W}",
                    "target": target, "n": len(joined),
                    "spearman": a.corr(b, method="spearman"),
                    "mae_change": (a - b).abs().mean(),
                    "max_abs_change": (a - b).abs().max(),
                })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
