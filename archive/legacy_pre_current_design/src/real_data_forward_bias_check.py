#!/usr/bin/env python3
"""Real-data forward-bias check: does the (n,w) occupancy-MLE collapse under a
forward-time crawl on REAL datasets too, not only on the synthetic generator?

Two-stage design, both stages reuse census.py's own parser (no reliance on any
config/datasets.yaml -- the format spec for each dataset is inline below):

  Stage A (--check-formats, runs first, always): parse each raw file, run it
  through census.py's own normalize() + census_row() (exactly the function
  that produced results/census/census_results_full.csv), and compare the
  freshly recomputed rho_headline against the value already stored in that
  CSV. This is a SELF-CHECK, not a courtesy: if the column layout / delimiter
  guessed below is wrong for a file, the parse will silently produce garbage
  pairs, and the only way to catch that without re-deriving the exact format
  is to check it reproduces a number you already trust. A dataset that fails
  this check is skipped entirely from Stage B -- never trust its walk numbers.

  Stage B (the actual check): for every dataset that PASSED Stage A, build a
  node-indexed event stream (real u,v ids, not the pair-collapsed form
  normalize() uses -- walks.py needs actual endpoints to traverse), run
  time_agnostic_t (clean control) and time_respecting (forward-biased) walks
  at several budgets/seeds, read off the occupancy MLE (corrected_estimator,
  beta=0) from each walk, and also report the MLE at the two beta values
  already calibrated on synthetic data (0.5 for time_agnostic_t, 2.0 for
  time_respecting) as a transfer check. All compared against the TRUE
  rho_headline from census_results_full.csv (not recomputed -- the trusted
  number this repo already reports).

Picked datasets (rationale: high format confidence -- plain 3-column SNAP or
SocioPatterns files, no bipartite/rating columns to get wrong -- spanning a
useful rho/size/domain range): CollegeMsg, email-Eu-core-temporal,
Hospital ward, Hypertext 2009 conference, High school 2013. Swap/add datasets
via the FORMATS dict below; re-run --check-formats after any change.

Run:
  python real_data_forward_bias_check.py --peek CollegeMsg   # eyeball one file
  python real_data_forward_bias_check.py                     # full check

Requires: census.py, walks.py, corrected_estimator.py, bias_identifiability.py
alongside, and results/census/census_results_full.csv already built.
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import census
import walks
from corrected_estimator import rho_mle
from bias_identifiability import _dense, obs_nw, rho_at_beta

W = 5
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
CANON_BETA = {"time_agnostic_t": 0.5, "time_respecting": 2.0}
BUDGETS = [400, 1600, 3200]
SEEDS = range(5)

# path is relative to --data-dir (default: ../data/raw)
FORMATS = {
    "CollegeMsg": {
        "file": "CollegeMsg.txt.gz", "delimiter": "whitespace",
        "columns": {"u": 0, "v": 1, "t": 2},
        "label": "CollegeMsg (SNAP)",
    },
    "email-Eu-core-temporal": {
        "file": "email-Eu-core-temporal.txt.gz", "delimiter": "whitespace",
        "columns": {"u": 0, "v": 1, "t": 2},
        "label": "email-Eu-core-temporal (SNAP)",
    },
    "HighSchool2013": {
        "file": "HighSchool2013_proximity_net.csv.gz", "delimiter": "whitespace",
        "columns": {"t": 0, "u": 1, "v": 2},
        "label": "High school 2013 (SocioPatterns)",
    },
    "hospital": {
        "file": "hospital_lyon_contacts.dat.gz", "delimiter": "whitespace",
        "columns": {"t": 0, "u": 1, "v": 2},
        "label": "Hospital ward (SocioPatterns)",
    },
    "ht2009": {
        "file": "ht2009_contact_list.dat.gz", "delimiter": "whitespace",
        "columns": {"t": 0, "u": 1, "v": 2},
        "label": "Hypertext 2009 conference (SocioPatterns)",
    },
}


def _count_nonempty_lines(path):
    n = 0
    with census._open(path) as fh:
        for line in fh:
            if line.strip():
                n += 1
    return n


def robust_parse(path, fmt):
    """Parse with the given format; if suspiciously few rows come out (wrong
    delimiter is the classic failure), retry with the other delimiter."""
    df = census.parse_events(path, fmt)
    raw_n = _count_nonempty_lines(path)
    if raw_n > 0 and len(df) < 0.5 * raw_n:
        alt = dict(fmt)
        alt["delimiter"] = "," if fmt.get("delimiter", "whitespace") != "," else "whitespace"
        df2 = census.parse_events(path, alt)
        if len(df2) > len(df):
            print(f"    [format] {path.name}: '{fmt.get('delimiter')}' gave "
                  f"{len(df)}/{raw_n} rows, switched to '{alt['delimiter']}' "
                  f"-> {len(df2)} rows")
            return df2
    return df


def to_walk_events(raw: pd.DataFrame) -> pd.DataFrame:
    """raw has columns u,v (raw ids, any type) and t (float). Returns a
    node-indexed, self-loop-free, time-sorted, [0,1]-rescaled event stream
    with dense integer node ids, ready for walks.build_index."""
    raw = raw[raw["u"].astype(str) != raw["v"].astype(str)].copy()
    all_ids = pd.concat([raw["u"], raw["v"]], ignore_index=True)
    codes, _ = pd.factorize(all_ids)
    m = len(raw)
    raw = raw.assign(u=codes[:m], v=codes[m:])
    raw = raw.sort_values("t", kind="mergesort").reset_index(drop=True)
    t = raw["t"].to_numpy(np.float64)
    tmin, tmax = float(t.min()), float(t.max())
    raw["t"] = (t - tmin) / (tmax - tmin) if tmax > tmin else np.zeros_like(t)
    return raw[["u", "v", "t"]].astype({"u": "int64", "v": "int64"})


def stage_a_check_formats(data_dir, truth):
    print("=== Stage A: format self-check (recomputed rho_headline vs "
          "census_results_full.csv) ===")
    passed = {}
    for key, fmt in FORMATS.items():
        path = data_dir / fmt["file"]
        if not path.exists():
            print(f"  [MISSING] {key}: {path} not found -- skipping")
            continue
        raw = robust_parse(path, fmt)
        if len(raw) < 100:
            print(f"  [FAIL] {key}: only {len(raw)} rows parsed, format spec "
                  f"is almost certainly wrong -- fix FORMATS['{key}'] and re-run")
            continue
        norm = census.normalize(raw.astype({"u": str, "v": str}))
        row = census.census_row(norm, label=fmt["label"])
        mine = row["rho_headline"]
        label = fmt["label"]
        if label not in truth.index:
            print(f"  [WARN] {key}: label '{label}' not found in "
                  f"census_results_full.csv -- cannot cross-check, treating as FAIL")
            continue
        true_val = float(truth.loc[label, "rho_headline"])
        diff = abs(mine - true_val)
        ok = diff < 0.01
        tag = "PASS" if ok else "FAIL"
        print(f"  [{tag}] {key:16s} n_pairs={row['n_pairs']:>8d}  "
              f"recomputed rho={mine:.4f}  census.csv rho={true_val:.4f}  diff={diff:.4f}")
        if ok:
            passed[key] = {"fmt": fmt, "true_rho": true_val,
                           "n_pairs_census": row["n_pairs"]}
    print(f"\n{len(passed)}/{len(FORMATS)} datasets passed the format self-check.\n")
    return passed


def stage_b_walk_check(data_dir, passed):
    print("=== Stage B: clean vs forward walk MLE on real data ===")
    DT = {b: _dense(b) for b in BETAS}
    rows = []
    for key, info in passed.items():
        fmt = info["fmt"]
        path = data_dir / fmt["file"]
        raw = robust_parse(path, fmt)
        events = to_walk_events(raw)
        idx = walks.build_index(events)
        tot = len(idx.edge_times)
        print(f"  {key}: {idx.n_nodes} nodes, {tot} collapsed edges, "
              f"{len(events)} events, true_rho={info['true_rho']:.3f}")
        for strat in ["time_agnostic_t", "time_respecting"]:
            for seed in SEEDS:
                log = walks.run_walk(idx, strat, max_budget=max(BUDGETS), seed=seed)
                for b in BUDGETS:
                    na, wa = obs_nw(log, b)
                    if len(na) == 0:
                        continue
                    r_uniform, _ = rho_mle(na, wa)
                    cb = CANON_BETA[strat]
                    r_transfer = rho_at_beta(na, wa, DT[cb])
                    rows.append({
                        "dataset": key, "label": fmt["label"],
                        "true_rho": info["true_rho"], "strategy": strat,
                        "budget": b, "seed": seed, "coverage": len(na) / tot,
                        "n_obs_edges": len(na),
                        "rho_mle_uniform": r_uniform,
                        "rho_mle_transfer_beta": r_transfer,
                        "transfer_beta_used": cb,
                    })
    return pd.DataFrame(rows)


def summarize(df):
    print("\n=== Summary (mean +/- std over seeds, per dataset x walk x budget) ===")
    g = df.groupby(["dataset", "strategy", "budget"])
    summ = g.agg(
        true_rho=("true_rho", "first"),
        coverage=("coverage", "mean"),
        n_seeds=("seed", "count"),
        rho_mle_mean=("rho_mle_uniform", "mean"),
        rho_mle_std=("rho_mle_uniform", "std"),
        rho_transfer_mean=("rho_mle_transfer_beta", "mean"),
    ).reset_index()
    summ["mae_uniform"] = (summ["rho_mle_mean"] - summ["true_rho"]).abs()
    summ["mae_transfer"] = (summ["rho_transfer_mean"] - summ["true_rho"]).abs()
    print(f"{'dataset':16s}{'walk':16s}{'budget':>7}{'cov':>7}{'true':>7}"
          f"{'MLE':>8}{'MAE_u':>7}{'transf':>8}{'MAE_t':>7}")
    print("-" * 90)
    for _, r in summ.sort_values(["dataset", "budget", "strategy"]).iterrows():
        print(f"{r.dataset:16s}{r.strategy:16s}{r.budget:7.0f}{r.coverage:7.3f}"
              f"{r.true_rho:7.3f}{r.rho_mle_mean:8.3f}{r.mae_uniform:7.3f}"
              f"{r.rho_transfer_mean:8.3f}{r.mae_transfer:7.3f}")
    return summ


def peek(data_dir, key):
    fmt = FORMATS[key]
    path = data_dir / fmt["file"]
    print(f"{fmt['label']}  ({path})")
    raw = robust_parse(path, fmt)
    print(f"parsed {len(raw)} rows; first 5:")
    print(raw.head().to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../data/raw")
    ap.add_argument("--truth-csv", default="../results/census/census_results_full.csv")
    ap.add_argument("--out", default="../results/census/real_data_bias_check.csv")
    ap.add_argument("--out-summary", default="../results/census/real_data_bias_check_summary.csv")
    ap.add_argument("--peek", help="print parsed rows of one dataset key and exit")
    args = ap.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir)

    if args.peek:
        peek(data_dir, args.peek)
        return

    truth = pd.read_csv(args.truth_csv).set_index("dataset")
    t0 = time.time()
    passed = stage_a_check_formats(data_dir, truth)
    if not passed:
        print("No dataset passed the format self-check -- fix FORMATS and re-run "
              "with --peek <key> first. Stopping before Stage B.")
        return
    df = stage_b_walk_check(data_dir, passed)
    summ = summarize(df)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    summ.to_csv(args.out_summary, index=False)
    print(f"\nwrote {args.out} ({len(df)} rows)")
    print(f"wrote {args.out_summary}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
