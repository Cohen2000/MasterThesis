#!/usr/bin/env python3
"""Test whether adjacent-window retention C adds information beyond rho profile."""

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors


PROFILE = [f"rho_W5_k{k}" for k in range(2, 6)]


def cv_predictions(X, y, groups, model, folds=5):
    cv = GroupKFold(n_splits=min(folds, pd.Series(groups).nunique()))
    return cross_val_predict(model, X, y, groups=groups, cv=cv, n_jobs=1)


def profile_analysis(manifest, out, jobs):
    d = pd.read_csv(manifest).dropna(subset=PROFILE + ["C_one_step", "group_id"])
    X, y, groups = d[PROFILE], d.C_one_step, d.group_id
    models = {
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=500, min_samples_leaf=2, random_state=20260731,
            n_jobs=jobs, max_features=1.0),
    }
    rows = []
    for name, model in models.items():
        pred = cv_predictions(X, y, groups, model)
        rows.append({"analysis": "C_from_true_profile", "model": name,
                     "n": len(y), "n_groups": pd.Series(groups).nunique(),
                     "r2_oof": r2_score(y, pred),
                     "mae_oof": mean_absolute_error(y, pred),
                     "residual_variance": float(np.var(y - pred)),
                     "target_variance": float(np.var(y))})
        if name == "extra_trees":
            d = d.assign(C_profile_oof=pred, C_profile_residual=y - pred)

    z = StandardScaler().fit_transform(d[PROFILE])
    nn = NearestNeighbors(n_neighbors=2).fit(z)
    dist, idx = nn.kneighbors(z)
    pairs = []
    seen = set()
    for i, j, ds in zip(range(len(d)), idx[:, 1], dist[:, 1]):
        key = tuple(sorted((i, int(j))))
        if key in seen:
            continue
        seen.add(key)
        a, b = d.iloc[i], d.iloc[int(j)]
        pairs.append({
            "instance_a": a.instance_id, "instance_b": b.instance_id,
            "block_a": a.data_block, "block_b": b.data_block,
            "profile_distance_z": float(ds),
            "profile_max_abs_delta": float(
                np.max(np.abs(a[PROFILE].to_numpy(float) -
                              b[PROFILE].to_numpy(float)))),
            "C_a": a.C_one_step, "C_b": b.C_one_step,
            "C_abs_delta": abs(a.C_one_step - b.C_one_step),
            **{f"{k}_a": a[k] for k in PROFILE},
            **{f"{k}_b": b[k] for k in PROFILE},
        })
    pairs = pd.DataFrame(pairs).sort_values(
        ["profile_max_abs_delta", "C_abs_delta"], ascending=[True, False])
    # Most persuasive examples: near profiles first, then largest C gap.
    candidates = pairs[pairs.profile_max_abs_delta <= 0.03].sort_values(
        "C_abs_delta", ascending=False)
    if len(candidates) < 20:
        candidates = pairs.sort_values(
            ["profile_max_abs_delta", "C_abs_delta"], ascending=[True, False])
    candidates.head(100).to_csv(out / "matched_profile_different_C.csv", index=False)
    return rows


def input_analysis(spec, out, jobs, walk_seed):
    paths = sorted(glob.glob(spec))
    if not paths:
        raise FileNotFoundError(spec)
    use = lambda c: (c in {"case_id", "group_id", "strategy", "walk_seed",
                            "C_one_step"} or c.startswith("occ__") or
                     c.startswith("pat__"))
    d = pd.concat([pd.read_csv(p, usecols=use) for p in paths], ignore_index=True)
    d = d[d.walk_seed == walk_seed].dropna(subset=["C_one_step", "group_id"])
    rows = []
    for strategy, g in d.groupby("strategy"):
        positionless = [c for c in g if c.startswith("occ__")]
        strict_mask = [
            c for c in g
            if c.startswith("pat__mask_")
            or re.match(r"^pat__n.+_mask(?:_|$)", c)
        ]
        for label, cols in {
            "positionless_nw": positionless,
            # Excludes lifetime, first/last-window and IET summaries: this is
            # strictly the added value of retaining window positions.
            "window_mask": positionless + strict_mask,
        }.items():
            cols = [c for c in cols if g[c].notna().any()]
            model = make_pipeline(
                SimpleImputer(strategy="median"),
                ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2,
                                    random_state=20260731, n_jobs=jobs,
                                    max_features=0.7))
            pred = cv_predictions(g[cols], g.C_one_step, g.group_id, model)
            rows.append({
                "analysis": "C_from_sample_input", "model": "extra_trees",
                "input": label, "strategy": strategy, "walk_seed": walk_seed,
                "n": len(g), "n_groups": g.group_id.nunique(),
                "r2_oof": r2_score(g.C_one_step, pred),
                "mae_oof": mean_absolute_error(g.C_one_step, pred),
                "residual_variance": float(np.var(g.C_one_step - pred)),
                "target_variance": float(np.var(g.C_one_step)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/benchmark_v2/data/manifest.csv")
    ap.add_argument("--cases", default="results/benchmark_v2/results/cases_shard_*.csv.gz")
    ap.add_argument("--out-dir", default="results/target_diagnostics/C_complementarity")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--walk-seed", type=int, default=0)
    ap.add_argument("--skip-input", action="store_true",
                    help="only analyze truth-profile complementarity")
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    rows = profile_analysis(args.manifest, out, args.jobs)
    if not args.skip_input:
        rows += input_analysis(args.cases, out, args.jobs, args.walk_seed)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    print(metrics.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
