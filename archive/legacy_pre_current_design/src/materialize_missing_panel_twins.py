#!/usr/bin/env python3
"""Generate only missing low/high controlled variants for the final panel."""

import argparse
import json
from pathlib import Path

import pandas as pd

from benchmark_generators import family_from_events, normalize_event_stream
from build_benchmark_data import _instance_row, stable_seed
from generator import make_instance


def signature(events):
    x = normalize_event_stream(events)
    counts = x.groupby(["u", "v"]).size().sort_index()
    nodes = set(x.u) | set(x.v)
    return nodes, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/benchmark_v2/data/manifest.csv")
    ap.add_argument("--data-root", default="results/benchmark_v2/data")
    ap.add_argument("--out-manifest",
                    default="results/benchmark_v2/data/panel_supplement.csv")
    ap.add_argument("--seed", type=int, default=20260711)
    args = ap.parse_args()
    root = Path(args.data_root)
    m = pd.read_csv(args.manifest)
    empirical = m[m.data_block.eq("real_empirical")]
    existing = set(m.instance_id)
    rows = []

    for r in empirical.itertuples(index=False):
        original = pd.read_csv(root / r.path)
        original = normalize_event_stream(original)
        fam = family_from_events(original, name=f"real::{r.source}", W=5)
        fam.timestamps = "bursty"
        for target, role in ((0.15, "low"), (0.55, "high")):
            iid = (f"controlled__{r.source}__bursty__contiguous__"
                   f"rho{target:.2f}__r0")
            if iid in existing:
                continue
            seed = stable_seed(args.seed, "controlled", r.source, "bursty",
                               "contiguous", target, 0)
            inst = make_instance(fam, target, seed=seed,
                                 span_layout="contiguous")
            nodes_a, counts_a = signature(original)
            nodes_b, counts_b = signature(inst.events)
            if nodes_a != nodes_b or not counts_a.equals(counts_b):
                raise RuntimeError(f"invariants failed for {iid}")
            metadata = {
                "source": r.source, "domain": r.domain,
                "substrate": "empirical", "group_id": f"real::{r.source}",
                "family": f"real::{r.source}", "rep": 0,
                "rho_target": target, "alpha": float("nan"),
                "chi": float("nan"), "data_block": "real_controlled",
                "generator": "steered_P[w,t]",
                "generator_params_json": json.dumps({
                    "seed": seed, "deviations": inst.deviations,
                    "timestamp_mode": "bursty", "span_layout": "contiguous",
                    "panel_role": role}),
            }
            rows.append(_instance_row(root, iid, inst.events, metadata, W=5))

    out = Path(args.out_manifest)
    if out.exists():
        old = pd.read_csv(out)
        rows = old.to_dict("records") + rows
    result = pd.DataFrame(rows).drop_duplicates("instance_id", keep="last")
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    print(f"wrote {len(result)} supplemental rows to {out}")
    if not len(result):
        print("No variants were missing.")


if __name__ == "__main__":
    main()
