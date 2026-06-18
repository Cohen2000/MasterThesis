#!/usr/bin/env python3
"""Persistence census over public temporal graph datasets.

Implements the measures from Definition Note v3:
  - Horizon [t_min, t_max], W equal-length half-open windows, last window closed.
  - a_e = number of distinct active windows per observed node pair (indicator).
  - rho_{k,W} = share of observed pairs with a_e >= k.
  - Full persistence profile (k = 1..W) for W in {4, 5, 8}.
  - Both sensitivity slices: fixed k = 2 and fixed alpha = 0.4 (k = ceil(0.4 W)).
  - Half-window shifted grid (boundary check), equal-event windows (census
    sensitivity), mean normalized span (threshold-free scalar).
  - One-step pair persistence C (adjacent windows; pair-level variant of the
    Nicosia et al. idea; their original coefficient is node-averaged).
  - Lifetime L (first-to-last event per pair), absolute and normalized by T.
  - Burstiness B = (sigma - mu) / (sigma + mu) of inter-event times,
    pooled and median per pair (pairs with >= 3 events).
  - Censoring share: pairs first observed in the final window (W = 5).
  - Context: window length / median inter-event time.

Usage:
  python census.py --peek snap_collegemsg          # verify parsing
  python census.py --only snap_collegemsg          # one dataset
  python census.py                                  # all downloaded datasets
  python census.py --markdown --plots --classify   # full report
"""

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HEADLINE_W = 5
HEADLINE_K = 2
W_GRID = (4, 5, 8)
ALPHA = 0.4
WINDOW_EPS = 1e-9       # float-boundary guard for window_index (see its docstring)


# ----------------------------------------------------------------------------
# Parsing and normalization
# ----------------------------------------------------------------------------

def _open(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return open(path, "r")


def parse_events(path: Path, fmt: dict) -> pd.DataFrame:
    """Read raw event list into DataFrame[u, v, t] according to format spec."""
    delim = fmt.get("delimiter", "whitespace")
    sep = r"\s+" if delim == "whitespace" else delim
    comment_chars = tuple(fmt.get("comment", "#%"))
    cols = fmt["columns"]
    skip = fmt.get("skiprows", 0)
    bipartite = fmt.get("bipartite", False)

    rows_u, rows_v, rows_t = [], [], []
    iu, iv, it = cols["u"], cols["v"], cols["t"]
    max_idx = max(iu, iv, it)
    with _open(path) as fh:
        for line_no, line in enumerate(fh):
            if line_no < skip:
                continue
            s = line.strip()
            if not s or s.startswith(comment_chars):
                continue
            parts = s.split() if delim == "whitespace" else s.split(delim)
            if len(parts) <= max_idx:
                continue
            if bipartite:
                rows_u.append("u:" + parts[iu])
                rows_v.append("i:" + parts[iv])
            else:
                rows_u.append(parts[iu])
                rows_v.append(parts[iv])
            rows_t.append(parts[it])
    df = pd.DataFrame({"u": rows_u, "v": rows_v, "t": rows_t})
    df["t"] = pd.to_numeric(df["t"], errors="coerce")
    df = df.dropna(subset=["t"])
    df["t"] = df["t"].astype(np.float64)
    return df


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Undirected canonical pairs, no self-loops, integer pair ids, sorted by t."""
    df = df[df["u"] != df["v"]].copy()
    a = np.minimum(df["u"].astype(str), df["v"].astype(str))
    b = np.maximum(df["u"].astype(str), df["v"].astype(str))
    df["pair"] = pd.factorize(a + "\x1f" + b)[0]
    df = df.sort_values(["pair", "t"], kind="mergesort").reset_index(drop=True)
    return df[["pair", "t"]]


# ----------------------------------------------------------------------------
# Core metric helpers
# ----------------------------------------------------------------------------

def window_index(t: np.ndarray, t_min: float, win: float, n_windows: int,
                 offset: float = 0.0) -> np.ndarray:
    """Half-open equal windows; last window closed (clip catches t == t_max).

    The + EPS guards the half-open boundary against floating-point error: an
    event exactly on a boundary k*win can compute (k*win)/win = k - 1e-16 in
    IEEE arithmetic, so a bare floor would drop it into window k-1. EPS is far
    below any real timestamp spacing (data is normalized to [0, T]), and is the
    single windowing convention shared by census, generator and walks so their
    window assignments never diverge on a boundary-exact timestamp.
    """
    idx = np.floor((t - t_min + offset) / win + WINDOW_EPS).astype(np.int64)
    return np.clip(idx, 0, n_windows - 1)


def spans(pair: np.ndarray, widx: np.ndarray) -> np.ndarray:
    """a_e per pair: number of distinct active windows (indicator semantics).

    Name collision to avoid: "span" here is a *count* of distinct active
    windows (possibly non-contiguous), NOT the contiguous time interval
    [t_first, t_last] that Galimberti et al. (2018, span-cores) call a span.
    """
    key = pd.DataFrame({"pair": pair, "w": widx}).drop_duplicates()
    return key.groupby("pair").size().sort_index().to_numpy()


def profile(a: np.ndarray, n_windows: int) -> dict:
    """rho_k for k = 1..W (k = 1 is identically 1.0 by construction).

    rho_k adapts the "support set" notion of Lahiri & Berger-Wolf (a subgraph
    is persistent if present in more than a threshold number of windows; see
    Holme & Saramaki 2012) to node pairs, normalized to a [0,1] fraction. This
    is distinct from lifetime (t_last - t_first) and from binary future-window
    survival (Navarro et al. 2017), neither of which counts active windows.
    """
    return {k: float(np.mean(a >= k)) for k in range(1, n_windows + 1)}


def one_step_pair_persistence(pair: np.ndarray, widx: np.ndarray,
                              n_windows: int) -> float:
    """C: P(pair active in window w+1 | active in window w), w = 0..W-2.

    Pair-level variant of Nicosia et al.'s adjacent-window persistence (their
    temporal correlation coefficient is node-averaged; Bauza Mingueza et al.
    2023 use a network-level adjacent-snapshot Jaccard -- both coarser).
    """
    act = pd.DataFrame({"pair": pair, "w": widx}).drop_duplicates()
    cur = act[act["w"] <= n_windows - 2]
    nxt = act.copy()
    nxt["w"] = nxt["w"] - 1
    hits = cur.merge(nxt, on=["pair", "w"], how="inner")
    if len(cur) == 0:
        return float("nan")
    return float(len(hits) / len(cur))


def equal_event_windows(pair: np.ndarray, t: np.ndarray, n_windows: int) -> np.ndarray:
    """Window index by event-count quantiles (events already sorted per pair,
    so sort globally by t first)."""
    order = np.argsort(t, kind="mergesort")
    widx = np.empty(len(t), dtype=np.int64)
    widx[order] = (np.arange(len(t)) * n_windows) // len(t)
    return spans(pair, widx)


def burstiness(gaps: np.ndarray) -> float:
    """Goh-Barabasi B = (sigma - mu) / (sigma + mu); -1 regular, 0 Poisson-like,
    -> 1 extremely bursty. NaN if fewer than 2 gaps.

    Caveat: B is biased low on short sequences (finite-size effect; Kim & Jo
    2016), and B -> 1 is only reachable as the number of events grows. Per-pair
    B (few events) is therefore a diagnostic, not an estimation target; the
    pooled B over all gaps is the more stable summary.
    """
    if len(gaps) < 2:
        return float("nan")
    mu, sigma = float(np.mean(gaps)), float(np.std(gaps))
    if mu + sigma == 0:
        return float("nan")
    return (sigma - mu) / (sigma + mu)


# ----------------------------------------------------------------------------
# Per-dataset census
# ----------------------------------------------------------------------------

def census_row(df: pd.DataFrame, label: str = "") -> dict:
    pair = df["pair"].to_numpy()
    t = df["t"].to_numpy()
    t_min, t_max = float(t.min()), float(t.max())
    T = t_max - t_min
    if T <= 0:
        raise ValueError("Degenerate horizon (all events share one timestamp).")

    g = df.groupby("pair")["t"]
    first = g.min().to_numpy()
    last = g.max().to_numpy()
    sizes = g.size().to_numpy()
    n_pairs = len(sizes)
    lifetimes = last - first

    # inter-event gaps within pairs
    dt = np.diff(t)
    same = pair[1:] == pair[:-1]
    gaps = dt[same]
    median_iet = float(np.median(gaps)) if len(gaps) else float("nan")

    # per-pair burstiness (pairs with >= 3 events -> >= 2 gaps), vectorized
    b_pair_median = float("nan")
    if len(gaps):
        gap_pair = pair[1:][same]
        gdf = pd.DataFrame({"pair": gap_pair, "g": gaps, "g2": gaps ** 2})
        agg = gdf.groupby("pair").agg(s=("g", "sum"), s2=("g2", "sum"), n=("g", "size"))
        agg = agg[agg["n"] >= 2]
        if len(agg):
            mu = agg["s"] / agg["n"]
            var = (agg["s2"] / agg["n"] - mu ** 2).clip(lower=0)
            sig = np.sqrt(var)
            denom = sig + mu
            b = ((sig - mu) / denom).where(denom > 0)
            b_pair_median = float(np.nanmedian(b.to_numpy(dtype=float)))

    row = {
        "dataset": label,
        "n_nodes": None,  # filled by the CLI from the raw id columns
        "n_pairs": int(n_pairs),
        "n_events": int(len(t)),
        "horizon_T": T,
        "median_iet": median_iet,
        "events_per_pair_median": float(np.median(sizes)),
        "share_single_event_pairs": float(np.mean(sizes == 1)),
        "burstiness_pooled": burstiness(gaps),
        "burstiness_pair_median": b_pair_median,
        "lifetime_mean": float(np.mean(lifetimes)),
        "lifetime_median": float(np.median(lifetimes)),
        "lifetime_mean_over_T": float(np.mean(lifetimes) / T),
    }

    # window-based metrics over the W grid
    for W in W_GRID:
        win = T / W
        widx = window_index(t, t_min, win, W)
        a = spans(pair, widx)
        prof = profile(a, W)
        for k, val in prof.items():
            row[f"rho_W{W}_k{k}"] = val
        k_alpha = int(np.ceil(ALPHA * W))
        row[f"rho_W{W}_fixedk2"] = prof[2]
        row[f"rho_W{W}_fixedalpha(k={k_alpha})"] = prof[k_alpha]
        if W == HEADLINE_W:
            row["rho_headline"] = prof[HEADLINE_K]
            row["mean_span_frac"] = float(np.mean(a) / W)
            # event-weighted rho: share of all events on persistent pairs.
            # Bridges pair-level rho to event-level recurrence (EdgeBank/TGB).
            # Cross-check vs TGX reoccurrence (Shirzadkhani et al. 2024) is by
            # correlation of dataset ordering, not identity (theirs is split-
            # based, ours window-based).
            row["rho_event_weighted"] = float(sizes[a >= HEADLINE_K].sum() / sizes.sum())
            row["C_one_step"] = one_step_pair_persistence(pair, widx, W)
            # shifted grid: W+1 bins, first/last half-width
            widx_s = window_index(t, t_min, win, W + 1, offset=win / 2)
            a_s = spans(pair, widx_s)
            row["rho_shifted"] = float(np.mean(a_s >= HEADLINE_K))
            # equal-event windows
            a_ee = equal_event_windows(pair, t, W)
            row["rho_equal_event"] = float(np.mean(a_ee >= HEADLINE_K))
            # censoring: pairs first observed in the final window
            first_w = window_index(first, t_min, win, W)
            row["censoring_share_lastwin"] = float(np.mean(first_w == W - 1))
            row["winlen_over_median_iet"] = (
                float(win / median_iet) if median_iet and median_iet > 0 else float("nan")
            )
    return row


def count_nodes(df_raw: pd.DataFrame) -> int:
    return int(pd.unique(pd.concat([df_raw["u"], df_raw["v"]]).astype(str)).size)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def load_registry(path: Path) -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)["datasets"]


def classify(df_out: pd.DataFrame) -> pd.DataFrame:
    """Terciles over headline rho across datasets -> easy / medium / hard.

    'easy'/'hard' refer to link-prediction-style difficulty: high rho = easy
    recurrence regime, low rho = hard novelty regime.
    """
    q1, q2 = df_out["rho_headline"].quantile([1 / 3, 2 / 3])
    def lab(x):
        return "hard (low rho)" if x <= q1 else ("easy (high rho)" if x > q2 else "medium")
    df_out["class_terciles"] = df_out["rho_headline"].apply(lab)
    df_out.attrs["tercile_thresholds"] = (float(q1), float(q2))
    return df_out


def make_plots(df_out: pd.DataFrame, out_dir: Path, profile_df: pd.DataFrame = None):
    """Census figures, all written to out_dir.

    Always: rho_distribution, rho_vs_C. With >= 3 datasets also rho_profile
    (full k = 1..W curve per difficulty tercile, so the headline k = 2 is shown
    as one slice) and rho_drivers (single-event share with the rho <= 1 - share
    ceiling, plus how strongly rho correlates with single-event share vs
    burstiness). With profile_df (long table dataset,W,k,rho over the W sweep)
    also rho_vs_W (the window-count sweep at k=2) and rho_rank_stability (a
    Spearman heatmap of the dataset ranking under rho(k,W) vs the headline
    rho(2,5), over a W x threshold grid).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # --- original two ---
    d = df_out.sort_values("rho_headline")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(d["dataset"], d["rho_headline"])
    ax.set_xlabel("rho (W=5, k=2)")
    ax.set_title("Edge persistence across datasets")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_distribution.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_out["rho_headline"], df_out["C_one_step"])
    for _, r in df_out.iterrows():
        ax.annotate(r["dataset"], (r["rho_headline"], r["C_one_step"]), fontsize=7)
    ax.set_xlabel("rho (W=5, k=2)")
    ax.set_ylabel("C (one-step pair persistence)")
    ax.set_title("Recurrence vs. short-range stability")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_vs_C.png", dpi=150)
    plt.close(fig)

    if len(df_out) < 3:
        print("[plots] < 3 datasets: skipping profile / drivers / sweep", file=sys.stderr)
        return

    col = {"easy (high rho)": "#1d9e75", "medium": "#ba7517", "hard (low rho)": "#a32d2d"}
    lab = {"easy (high rho)": "easy (high \u03c1)", "medium": "medium",
           "hard (low rho)": "hard (low \u03c1)"}
    order = ["easy (high rho)", "medium", "hard (low rho)"]
    if "class_terciles" not in df_out.columns:   # recompute if --classify was off
        df_out = classify(df_out.copy())

    # --- persistence profile rho(k) per tercile (W = 5) ---
    ks = [1, 2, 3, 4, 5]
    cols = [f"rho_W5_k{k}" for k in ks]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for g in order:
        dd = df_out[df_out["class_terciles"] == g]
        if dd.empty:
            continue
        v = dd[cols].to_numpy(dtype=float)
        ax.fill_between(ks, v.min(0), v.max(0), color=col[g], alpha=0.15)
        ax.plot(ks, v.mean(0), color=col[g], lw=2, marker="o",
                label=f"{lab[g]}  (n={len(dd)})")
    ax.axvline(2, color="0.4", ls="--", lw=1)
    ax.text(2.05, 0.97, "headline (k=2)", color="0.4", fontsize=9, va="top")
    ax.set_xticks(ks)
    ax.set_ylim(-0.02, 1.03)
    ax.set_xlabel("k  (pair active in at least k of 5 windows)")
    ax.set_ylabel("\u03c1  (fraction of observed pairs)")
    ax.set_title("Persistence profile by tercile (W=5)\n"
                 "the dashed line (k=2) is one slice of the full curve")
    ax.legend(title="tercile (line = mean, band = min..max)", fontsize=9, title_fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_profile.png", dpi=150)
    plt.close(fig)

    # --- drivers: both candidates shown over datasets (honest contrast).
    # left carries the rho <= 1 - share ceiling (a real bound); burstiness has none.
    rs = float(df_out["rho_headline"].corr(df_out["share_single_event_pairs"], method="spearman"))
    rb = float(df_out["rho_headline"].corr(df_out["burstiness_pooled"], method="spearman"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.4))
    for g in order:
        dd = df_out[df_out["class_terciles"] == g]
        axL.scatter(dd["share_single_event_pairs"], dd["rho_headline"],
                    c=col[g], s=55, label=lab[g], zorder=3)
        axR.scatter(dd["burstiness_pooled"], dd["rho_headline"], c=col[g], s=55, zorder=3)
    xx = np.linspace(0, 1, 100)
    axL.plot(xx, 1 - xx, color="0.4", ls="--", lw=1.2, label="ceiling  \u03c1 = 1 - share")

    def mark(ax, xcol, names):
        for name in names:
            if (df_out["dataset"] == name).any():
                r = df_out[df_out["dataset"] == name].iloc[0]
                ax.annotate(name.split(" (")[0], (r[xcol], r["rho_headline"]),
                            xytext=(5, 5), textcoords="offset points", fontsize=8)
    mark(axL, "share_single_event_pairs", ["ia-digg-reply (NetRepo)", "ia-radoslaw-email (NetRepo)"])
    mark(axR, "burstiness_pooled", ["ia-digg-reply (NetRepo)", "Hospital ward (SocioPatterns)"])

    axL.set_xlim(0, 1.02)
    axL.set_ylim(-0.02, 1.0)
    axR.set_xlim(0.2, 0.8)
    axR.set_ylim(-0.02, 1.0)
    axL.set_xlabel("share of single-event pairs")
    axR.set_xlabel("burstiness (pooled)")
    axL.set_ylabel("\u03c1  (W=5, k=2)")
    axR.set_ylabel("\u03c1  (W=5, k=2)")
    axL.set_title(f"\u03c1 is driven by one-shot pairs  (Spearman {rs:+.2f})")
    axR.set_title(f"\u03c1 is NOT driven by burstiness  (Spearman {rb:+.2f})")
    axL.legend(fontsize=8, loc="upper right")
    axL.grid(alpha=0.25)
    axR.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_drivers.png", dpi=150)
    plt.close(fig)

    # --- window-count sweep + rank-stability heatmap (need profile_df from main) ---
    if profile_df is None or profile_df.empty:
        print("[plots] no profile data: skipping rho_vs_W and rho_rank_stability",
              file=sys.stderr)
        return
    g2t = dict(zip(df_out["dataset"], df_out["class_terciles"]))

    # rho vs W at the headline k=2 (one line per dataset)
    sweep = profile_df[profile_df["k"] == HEADLINE_K]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for name, sub in sweep.groupby("dataset"):
        sub = sub.sort_values("W")
        g = g2t.get(name, "medium")
        ax.plot(sub["W"], sub["rho"], color=col.get(g, "#888780"),
                lw=1.3, alpha=0.75, marker="o", markersize=3)
    handles = [plt.Line2D([0], [0], color=col[g], lw=2, marker="o", markersize=4) for g in order]
    ax.legend(handles, [lab[g] for g in order], title="tercile", fontsize=9, title_fontsize=9)
    ax.set_xlabel("W  (number of windows)")
    ax.set_ylabel("\u03c1  (k=2)")
    ax.set_title("Edge persistence vs number of windows W\n"
                 "\u03c1 rises as windows get finer (toward the 1 - single-pair ceiling)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "rho_vs_W.png", dpi=150)
    plt.close(fig)

    # --- rank-stability heatmap over the (W, threshold) grid ---
    # Each cell: Spearman of the dataset ranking under rho(k, W) vs the headline
    # rho(2, 5), across datasets. Columns are the threshold as a fraction of W
    # (alpha = k/W); the actual count is k = max(2, round(alpha*W)), capped at W,
    # so k=1 (trivially 1.0, zero variance) is never used. This answers whether
    # rho(2,5) is an isolated choice or representative of a family of thresholds.
    # Caveat: the strict corner (high alpha) is noisy because rho there is near 0
    # for most datasets, so its low variance makes the Spearman unreliable.
    by_wk = {(int(W), int(k)): sub.set_index("dataset")["rho"]
             for (W, k), sub in profile_df.groupby(["W", "k"])}
    if (HEADLINE_W, HEADLINE_K) not in by_wk:
        print("[plots] headline (W,k) not in sweep: skipping rho_rank_stability",
              file=sys.stderr)
        return
    headline = by_wk[(HEADLINE_W, HEADLINE_K)]
    Ws = sorted({int(w) for w in profile_df["W"].unique() if w >= 4})
    alphas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
    M = np.full((len(Ws), len(alphas)), np.nan)
    for i, W in enumerate(Ws):
        for j, al in enumerate(alphas):
            k = min(max(HEADLINE_K, int(round(al * W))), W)
            ser = by_wk.get((W, k))
            if ser is not None:
                M[i, j] = headline.corr(ser, method="spearman")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.9")
    fig, ax = plt.subplots(figsize=(7.8, 5.2))
    im = ax.imshow(M, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(alphas)))
    ax.set_xticklabels([f"{a:.1f}" for a in alphas])
    ax.set_yticks(range(len(Ws)))
    ax.set_yticklabels(Ws)
    ax.set_xlabel("threshold as fraction of W  (\u03b1 = k/W,  k = max(2, round(\u03b1\u00b7W)))")
    ax.set_ylabel("W  (number of windows)")
    ax.set_title("Rank stability of \u03c1 across thresholds\n"
                 "cell = Spearman of the dataset ranking vs the headline \u03c1(2,5)")
    for i in range(len(Ws)):
        for j in range(len(alphas)):
            v = M[i, j]
            if np.isnan(v):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=8, color="0.45")
            else:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="white" if v < 0.55 else "black")
    if HEADLINE_W in Ws and 0.4 in alphas:   # outline the headline cell rho(2,5)
        ax.add_patch(plt.Rectangle((alphas.index(0.4) - 0.5, Ws.index(HEADLINE_W) - 0.5),
                                   1, 1, fill=False, edgecolor="red", lw=2))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Spearman vs \u03c1(2,5)")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_rank_stability.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="config/datasets.yaml")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="results/census/census_results_full.csv")
    ap.add_argument("--only", help="single dataset key")
    ap.add_argument("--peek", help="print first parsed rows of a dataset and exit")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--w-sweep", default="2,3,4,5,6,8,10,12,16,20",
                    help="window counts for the rho-vs-W plot (used with --plots)")
    args = ap.parse_args()

    reg = load_registry(Path(args.registry))
    data_dir = Path(args.data_dir)

    if args.peek:
        spec = reg[args.peek]
        path = data_dir / spec["file"]
        df = parse_events(path, spec["format"]).head(5)
        print(f"{spec['label']}  ({path})")
        print(df.to_string(index=False))
        return

    keys = [args.only] if args.only else list(reg.keys())
    rows = []
    profile_rows = []
    w_sweep = [int(x) for x in args.w_sweep.split(",")] if args.plots else []
    for key in keys:
        spec = reg[key]
        path = data_dir / spec["file"]
        if not path.exists():
            print(f"[skip] {key}: {path} not found", file=sys.stderr)
            continue
        print(f"[run ] {key} ...", file=sys.stderr)
        raw = parse_events(path, spec["format"])
        df = normalize(raw)
        row = census_row(df, label=spec["label"])
        row["n_nodes"] = count_nodes(raw)
        row["domain"] = spec.get("domain", "")
        rows.append(row)
        if args.plots:   # full rho(k) profile at each sweep W, same windowing as census_row
            pr, tt = df["pair"].to_numpy(), df["t"].to_numpy()
            tmin = float(tt.min())
            span_T = float(tt.max()) - tmin
            for W in w_sweep:
                a = spans(pr, window_index(tt, tmin, span_T / W, W))
                for k, r in profile(a, W).items():
                    profile_rows.append({"dataset": spec["label"], "W": W,
                                         "k": k, "rho": r})

    if not rows:
        print("No datasets found. Download files into --data-dir first.", file=sys.stderr)
        return

    out = pd.DataFrame(rows)
    lead = ["dataset", "domain", "n_nodes", "n_pairs", "n_events", "horizon_T",
            "rho_headline", "rho_event_weighted", "mean_span_frac", "C_one_step",
            "rho_shifted", "rho_equal_event", "lifetime_mean_over_T",
            "burstiness_pooled", "censoring_share_lastwin", "winlen_over_median_iet"]
    out = out[[c for c in lead if c in out.columns]
              + [c for c in out.columns if c not in lead]]
    if args.classify and len(out) >= 3:
        out = classify(out)
        print(f"tercile thresholds: {out.attrs['tercile_thresholds']}", file=sys.stderr)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(out)} datasets)", file=sys.stderr)
    if args.markdown:
        md = out_path.with_suffix(".md")
        md.write_text(out.round(3).to_markdown(index=False))
        print(f"wrote {md}", file=sys.stderr)
    if args.plots:
        profile_df = pd.DataFrame(profile_rows) if profile_rows else None
        make_plots(out, out_path.parent, profile_df)
        print("wrote census plots: rho_distribution, rho_vs_C, rho_profile, "
              "rho_drivers, rho_vs_W, rho_rank_stability", file=sys.stderr)


if __name__ == "__main__":
    main()
