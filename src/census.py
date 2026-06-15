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
    """Half-open equal windows; last window closed (clip catches t == t_max)."""
    idx = np.floor((t - t_min + offset) / win).astype(np.int64)
    return np.clip(idx, 0, n_windows - 1)


def spans(pair: np.ndarray, widx: np.ndarray) -> np.ndarray:
    """a_e per pair: number of distinct active windows (indicator semantics)."""
    key = pd.DataFrame({"pair": pair, "w": widx}).drop_duplicates()
    return key.groupby("pair").size().sort_index().to_numpy()


def profile(a: np.ndarray, n_windows: int) -> dict:
    """rho_k for k = 1..W (k = 1 is identically 1.0 by construction)."""
    return {k: float(np.mean(a >= k)) for k in range(1, n_windows + 1)}


def one_step_pair_persistence(pair: np.ndarray, widx: np.ndarray,
                              n_windows: int) -> float:
    """C: P(pair active in window w+1 | active in window w), w = 0..W-2.

    Pair-level variant of Nicosia et al.'s adjacent-window persistence
    (their temporal correlation coefficient is node-averaged overlap).
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
    -> 1 extremely bursty. NaN if fewer than 2 gaps."""
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
            # event-weighted rho: share of all events that lie on persistent
            # pairs. Bridges pair-level rho to event-level recurrence numbers
            # (EdgeBank/TGB style); high values with low rho_headline mean few
            # recurring pairs carry most of the traffic.
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


def make_plots(df_out: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = df_out.sort_values("rho_headline")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(d["dataset"], d["rho_headline"])
    ax.set_xlabel("rho (W=5, k=2)")
    ax.set_title("Edge persistence across datasets")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_distribution.png", dpi=150)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(df_out["rho_headline"], df_out["C_one_step"])
    for _, r in df_out.iterrows():
        ax.annotate(r["dataset"], (r["rho_headline"], r["C_one_step"]), fontsize=7)
    ax.set_xlabel("rho (W=5, k=2)")
    ax.set_ylabel("C (one-step pair persistence)")
    ax.set_title("Recurrence vs. short-range stability")
    fig.tight_layout()
    fig.savefig(out_dir / "rho_vs_C.png", dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default="datasets.yaml")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="census_results.csv")
    ap.add_argument("--only", help="single dataset key")
    ap.add_argument("--peek", help="print first parsed rows of a dataset and exit")
    ap.add_argument("--markdown", action="store_true")
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--classify", action="store_true")
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
    out.to_csv(args.out, index=False)
    print(f"wrote {args.out} ({len(out)} datasets)", file=sys.stderr)
    if args.markdown:
        md = Path(args.out).with_suffix(".md")
        md.write_text(out.round(3).to_markdown(index=False))
        print(f"wrote {md}", file=sys.stderr)
    if args.plots:
        make_plots(out, Path(args.out).parent)
        print("wrote rho_distribution.png, rho_vs_C.png", file=sys.stderr)


if __name__ == "__main__":
    main()
