#!/usr/bin/env python3
"""Run all walk strategies over the grid instances and write summaries.

Changes vs the original:
  - carries n (node count) and mean_span_frac (occupancy ground truth) through
    META, so the evaluation can use coverage and a second target without
    re-reading the manifest.
  - logs COVERAGE per summary row: coverage = unique_edges_seen / total_edges.
    total_edges is the number of distinct collapsed edges of the instance
    (len(idx.edge_times)). Coverage is THE x-axis for the plug-in-vs-regressor
    analysis: it is what a fixed walk budget actually sees of a graph, and it
    drops as graphs grow, which is the partial-access regime we want to probe.

For every instance x strategy x walk seed, one walk log is produced up to the
largest budget; sweep points are prefixes of that log (checkpoint trick).

Usage:
  python run_pilot_walks.py --manifest data_grid/manifest.csv --out summaries.csv
"""

import argparse
import zlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from walks import build_index, run_walk, summaries_at_checkpoints
from census import window_index as _window_index
from corrected_estimator import rho_mle as _rho_mle

STRATEGIES = ["time_agnostic", "time_agnostic_t", "time_respecting", "recency_biased"]
BUDGETS = [50, 100, 200, 400, 800, 1600, 3200]
# carry size (n) and occupancy (mean_span_frac) alongside the rho label
META = ["substrate", "n", "family", "rho_target", "rep", "hub_bias",
        "rho_headline", "mean_span_frac"]


def load_events(path, T: float = 1.0) -> pd.DataFrame:
    """Load an event CSV (columns u, v, t) and rescale timestamps to [0, T].

    Synthetic grid data is already in [0, 1] with T=1, so this is a no-op there.
    Real datasets carry absolute timestamps which the window math would otherwise
    clip; the affine map t -> (t - t_min)/(t_max - t_min) * T preserves event
    ORDER and all RELATIVE gaps exactly.
    """
    ev = pd.read_csv(path)
    t = ev["t"].to_numpy(np.float64)
    if len(t) == 0:
        return ev
    tmin, tmax = float(t.min()), float(t.max())
    ev["t"] = (t - tmin) / (tmax - tmin) * T if tmax > tmin else np.zeros_like(t)
    return ev


def mle_readouts(log, budget, T=1.0, W=5, iters=200):
    """Occupancy-MLE readouts from the (n, w) of the first `budget` log rows.

    Returns (rho_mle, occ_mle):
      rho_mle = 1 - pi_1            (window persistence, bias-naive MLE)
      occ_mle = (sum_k k*pi_k) / W  (mean occupancy, SAME fit)
    NaN if the prefix has no usable timed observations (e.g. time_agnostic).
    """
    L = log.iloc[:budget]
    s = L[L["kind"] == 1]
    if len(s) == 0:
        return float("nan"), float("nan")
    t = s["t"].to_numpy(np.float64)
    ok = np.isfinite(t)
    if not ok.any():
        return float("nan"), float("nan")
    win = _window_index(t[ok], 0.0, T / W, W)
    u = s["u"].to_numpy()[ok]
    v = s["v"].to_numpy()[ok]
    nw = {}
    for a, b, w in zip(u, v, win):
        d = nw.setdefault((int(a), int(b)), [0, set()])
        d[0] += 1
        d[1].add(int(w))
    if not nw:
        return float("nan"), float("nan")
    n_arr = np.fromiter((d[0] for d in nw.values()), dtype=np.int64, count=len(nw))
    w_arr = np.fromiter((len(d[1]) for d in nw.values()), dtype=np.int64, count=len(nw))
    _, pi = _rho_mle(n_arr, w_arr, W=W, iters=iters)
    if pi is None:
        return float("nan"), float("nan")
    occ = float((np.arange(1, W + 1) * pi).sum() / W)
    return float(1.0 - pi[0]), occ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data_grid/manifest.csv")
    ap.add_argument("--out", default="summaries.csv")
    ap.add_argument("--budgets", default=",".join(map(str, BUDGETS)))
    ap.add_argument("--strategies", default=",".join(STRATEGIES))
    ap.add_argument("--walk-seeds", type=int, default=2)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--mle-iters", type=int, default=200,
                    help="EM iterations for the occupancy-MLE readout per checkpoint")
    ap.add_argument("--limit", type=int, default=0, help="only first N instances (testing)")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    strategies = args.strategies.split(",")
    mf = pd.read_csv(args.manifest)
    if args.limit:
        mf = mf.head(args.limit)

    rows = []
    t0 = time.time()
    for i, r in mf.iterrows():
        events = load_events(r["path"], T=args.T)
        idx = build_index(events, T=args.T, W=args.W)
        total_edges = len(idx.edge_times)            # coverage denominator
        for strat in strategies:
            for ws in range(args.walk_seeds):
                seed = zlib.crc32(
                    f"{r['family']}|{r['rho_target']}|{r['rep']}|"
                    f"{r['hub_bias']}|{strat}|{ws}".encode()) % 1_000_000
                log = run_walk(idx, strat, max_budget=max(budgets), seed=seed)
                for feats in summaries_at_checkpoints(log, idx, budgets):
                    feats["total_edges"] = total_edges
                    feats["coverage"] = (
                        feats.get("unique_edges", 0) / total_edges
                        if total_edges else float("nan"))
                    b_ck = int(feats.get("budget", max(budgets)))
                    rho_mle_v, occ_mle_v = mle_readouts(
                        log, b_ck, T=args.T, W=args.W, iters=args.mle_iters)
                    feats["rho_mle"] = rho_mle_v
                    feats["occ_mle"] = occ_mle_v
                    rows.append({**{k: r[k] for k in META},
                                 "strategy": strat, "walk_seed": ws, **feats})
        if (i + 1) % 20 == 0:
            print(f"[{i+1}/{len(mf)}] {time.time()-t0:.0f}s elapsed", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out}  ({len(out)} summary rows, "
          f"{time.time()-t0:.0f}s total)")

    # sanity: mean coverage by size at the top budget (should drop as n grows)
    top = out[out["budget"] == max(budgets)]
    if len(top):
        piv = top.pivot_table(index="n", columns="strategy",
                              values="coverage", aggfunc="mean")
        print("\nmean coverage at top budget (drops as n grows -> partial access):")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    main()
