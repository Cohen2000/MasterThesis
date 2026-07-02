#!/usr/bin/env python3
"""Phase 1 analysis: turn the cluster outputs into the per-walk coverage band.

Reads the outputs of run_pilot_walks.py / pilot_eval.py (which must be the PATCHED
versions that emit rho_mle/occ_mle and the 'mle' estimator tier) plus the
manifest, and produces:

  - a per-walk x coverage-band table (floor / plugin / MLE / oracle + MLE-fraction)
    for both estimands, showing the uniform-MLE failure under forward bias while
    the oracle recovers the signal;
  - the ranking-vs-calibration dissociation under the forward walk
    (Spearman(rho_mle, rho_true) stays high while MAE collapses);
  - the occupancy / C redundancy check on the full grid;
  - band_per_walk_rho.png (the per-walk figure the pooled plot hides).

Run:
  python analyze_coverage_band.py \
      --summaries ../results/coverage/summaries.csv \
      --grid-results ../results/coverage/grid_results.csv \
      --manifest ../data/synthetic/<run>/manifest.csv \
      --fig ../results/coverage/band_per_walk_rho.png

summaries.csv may be gzip'd (.gz); pandas reads it either way.
"""
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

WALKS = [("time_agnostic", "negative control (no timestamps)"),
         ("time_agnostic_t", "unbiased observation (clean reference)"),
         ("recency_biased", "recency-weighted (intermediate bias)"),
         ("time_respecting", "forward-time (the realistic biased crawl)")]


def _mae(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[m] - b[m]).mean()) if m.sum() else np.nan


def band_table(g):
    g = g.copy()
    g["cb"] = g["coverage"].map(lambda c: "lo(<.01)" if c < 0.01 else ("hi(>.1)" if c > 0.1 else "mid"))
    for tgt in ["rho_headline", "mean_span_frac"]:
        print(f"\n================ TARGET: {tgt} ================")
        print(f"{'walk':17s}{'band':10s}{'floor':>7}{'plugin':>8}{'MLE':>7}{'oracle':>8}{'MLE_frac':>9}")
        print("-" * 66)
        for w, _ in WALKS:
            for cb in ["lo(<.01)", "mid", "hi(>.1)"]:
                d = g[(g.target == tgt) & (g.strategy == w) & (g.cb == cb)]
                if d.empty:
                    continue
                p = {e: d[d.estimator == e]["mae"].mean() for e in d.estimator.unique()}
                fl, pl = p.get("mean_floor", np.nan), p.get("plugin", np.nan)
                ml, orc = p.get("mle", np.nan), p.get("xgb_full", np.nan)
                frac = (fl - ml) / (fl - orc) if np.isfinite(ml) and (fl - orc) > 1e-9 else np.nan
                plS = f"{pl:8.3f}" if np.isfinite(pl) else f"{'--':>8}"
                mlS = f"{ml:7.3f}" if np.isfinite(ml) else f"{'--':>7}"
                frS = f"{frac:9.2f}" if np.isfinite(frac) else f"{'--':>9}"
                print(f"{w:17s}{cb:10s}{fl:7.3f}{plS}{mlS}{orc:8.3f}{frS}")
            print()


def ranking_dissociation(s):
    from scipy.stats import spearmanr
    print("\n================ ranking vs calibration (per walk) ================")
    s = s.copy()
    s["cb"] = s["coverage"].map(lambda c: "lo(<.01)" if c < 0.01 else ("hi(>.1)" if c > 0.1 else "mid"))
    print(f"{'walk':17s}{'band':10s}{'Spearman(mle,true)':>20}{'MLE_MAE':>9}")
    for w in ["time_agnostic_t", "recency_biased", "time_respecting"]:
        for cb in ["lo(<.01)", "mid", "hi(>.1)"]:
            d = s[(s.strategy == w) & (s.cb == cb)].dropna(subset=["rho_mle"])
            if len(d) < 10:
                continue
            sp = spearmanr(d.rho_mle, d.rho_headline)[0]
            print(f"{w:17s}{cb:10s}{sp:20.3f}{_mae(d.rho_mle, d.rho_headline):9.3f}")
        print()


def redundancy(m):
    from scipy.stats import pearsonr, spearmanr
    print("================ occupancy / C redundancy on the full grid ================")
    for a, b in [("rho_headline", "mean_span_frac"), ("rho_headline", "C_one_step"),
                 ("mean_span_frac", "C_one_step")]:
        if a in m and b in m:
            d = m[[a, b]].dropna()
            print(f"  {a:15s} vs {b:15s}: Pearson {pearsonr(d[a], d[b])[0]:+.3f}  "
                  f"Spearman {spearmanr(d[a], d[b])[0]:+.3f}")


def figure(g, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    EST = [("mean_floor", "floor", "#888888", "--"), ("plugin", "plug-in", "#ff7f0e", "-"),
           ("mle", "occupancy MLE", "#d62728", "-"), ("xgb_full", "oracle", "#2ca02c", "-")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True, sharey=True)
    for ax, (w, title) in zip(axes.flat, WALKS):
        sub = g[(g.target == "rho_headline") & (g.strategy == w)]
        for est, lab, col, ls in EST:
            e = sub[sub.estimator == est]
            if e.empty:
                continue
            q = pd.qcut(e["coverage"], q=min(8, e["coverage"].nunique()), duplicates="drop")
            tr = e.groupby(q, observed=True).agg(c=("coverage", "mean"), mae=("mae", "mean"))
            ax.plot(tr["c"], tr["mae"], marker="o", color=col, ls=ls, lw=2, label=lab, ms=4)
        ax.set_xscale("log"); ax.set_title(w + "\n" + title, fontsize=10)
        ax.grid(alpha=0.25); ax.set_ylim(0, 0.30)
    for ax in axes[1]:
        ax.set_xlabel("walk coverage (unique edges / total, log)")
    for ax in axes[:, 0]:
        ax.set_ylabel("MAE vs true rho")
    axes.flat[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("Window persistence recovery per walk: floor / plug-in / MLE / oracle vs coverage")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"\nwrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summaries", default="../results/coverage/summaries.csv")
    ap.add_argument("--grid-results", default="../results/coverage/grid_results.csv")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--fig", default="../results/coverage/band_per_walk_rho.png")
    args = ap.parse_args()

    g = pd.read_csv(args.grid_results)
    band_table(g)
    figure(g, args.fig)
    try:
        s = pd.read_csv(args.summaries)
        ranking_dissociation(s)
    except Exception as e:
        print(f"[skip ranking dissociation: {e}]")
    if args.manifest:
        try:
            redundancy(pd.read_csv(args.manifest))
        except Exception as e:
            print(f"[skip redundancy: {e}]")


if __name__ == "__main__":
    main()
