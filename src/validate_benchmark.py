#!/usr/bin/env python3
"""Fail-fast validation for a generated manifest or case table."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_manifest(path):
    p = Path(path); d = pd.read_csv(p)
    assert not d.empty, "manifest is empty"
    assert d["instance_id"].is_unique, "duplicate instance_id"
    assert d["group_id"].notna().all(), "missing group_id"
    missing = []
    for rel in d["path"]:
        q = Path(str(rel)); q = q if q.is_absolute() else p.parent / q
        if not q.exists(): missing.append(str(q))
    assert not missing, f"missing event files (first five): {missing[:5]}"
    targets = ([f"rho_W5_k{k}" for k in range(1, 6)] +
               ["C_one_step", "mean_span_frac", "rho_event_weighted"])
    assert np.isfinite(d[targets].to_numpy(float)).all(), "non-finite target"
    assert ((d[targets] >= 0) & (d[targets] <= 1)).all().all(), "target outside [0,1]"
    print(f"manifest OK: {len(d)} instances, {d.group_id.nunique()} groups, "
          f"blocks={d.data_block.value_counts().to_dict()}")


def validate_cases(patterns):
    paths = []
    for pat in patterns:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    d = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    assert not d.empty, "case table is empty"
    assert d["case_id"].is_unique, "duplicate case_id"
    feature_cols = [c for c in d if c.startswith(("occ__", "pat__", "crawl__"))]
    assert feature_cols, "no model feature columns"
    forbidden_exact = {"coverage", "total_edges", "n_nodes_true", "n_events_true",
                       "rho_target", "alpha", "chi"}
    assert not (set(feature_cols) & forbidden_exact), "metadata leaked into model features"
    for col in ("input__nw_exact_json", "input__nmask_exact_json"):
        for value in d[col].head(50):
            assert isinstance(json.loads(value), dict), f"invalid JSON in {col}"
    target_cols = ([f"rho_W5_k{k}" for k in range(2, 6)] +
                   ["C_one_step", "mean_span_frac", "rho_event_weighted"])
    assert np.isfinite(d[target_cols].to_numpy(float)).all(), "non-finite case target"
    main = d[d.strategy != "time_agnostic"]
    assert (main.groupby("strategy")["group_id"].nunique() >= 2).all(), (
        "a main strategy has fewer than two independent groups")
    print(f"cases OK: {len(d)} rows, {len(feature_cols)} legal features, "
          f"strategies={d.strategy.value_counts().to_dict()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest")
    ap.add_argument("--cases", nargs="*")
    args = ap.parse_args()
    if not args.manifest and not args.cases:
        ap.error("provide --manifest and/or --cases")
    if args.manifest: validate_manifest(args.manifest)
    if args.cases: validate_cases(args.cases)


if __name__ == "__main__":
    main()
