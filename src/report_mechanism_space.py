"""Where the five chosen arms sit in the space of observation mechanisms.

The design is defended on channel coverage rather than realism, which invites
the obvious question: were these five picked because they gave the wanted
answer? This places them against every other sampler in the repository that has
already been run against ground truth, on the two axes that define the design.

    x  channel composition -- |selection| / (|selection| + |censoring|)
       0 = the bias is entirely censoring, 1 = entirely selection
    y  signed rho_2 bias of the naive plug-in

Both are analytic. Nothing here reads a model.

The composition axis alone hides one thing, so it is marked separately: a
sampler whose two channels have *opposite* signs sits at the same x as one
whose channels agree, but its net bias is a difference rather than a sum. Those
are drawn as open markers.

Sources, all pre-existing:
  results/final_run_g2/final_cases.csv.gz        the five chosen arms
  results/nonwalk_screen/panel32_cases.csv.gz    reservoir, prefix, window,
                                                 ego-recent, node panel
  results/nonwalk_crawl_screen/crawl_cases_*.gz  bfs crawl, forest fire
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TRUE, SEEN, NAIVE = "rho_W5_k2", "oracle__seen_label_rho_k2", "est__plugin_rho_k2"
COLS = ["case_id", "instance_id", "group_id", "strategy", "coverage",
        TRUE, SEEN, NAIVE]
CHOSEN = ("event_sample_then_full_history", "node_panel_full_history",
          "recent_history_k20", "time_agnostic_t", "time_respecting")


def _load(path: str, budget=None) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=lambda c: c in COLS + ["target_budget",
                                                            "seed_slot"])
    if budget is not None and "target_budget" in frame:
        frame = frame[frame.target_budget == budget]
    if "seed_slot" in frame:
        frame = frame[frame.seed_slot == 0]
    return frame


def collect(budget=None) -> pd.DataFrame:
    """Chosen arms from the frozen run, alternatives from the screens.

    The screens also contain `node_panel_full_history`, at their own budgets.
    Pooling the two sources would silently average the frozen arm with a
    differently budgeted namesake, so the chosen arms are taken only from the
    frozen run and the screen rows for those names are dropped.
    """
    chosen = _load(str(REPO / "results/final_run_g2/final_cases.csv.gz"))
    chosen["target_budget"] = np.nan
    chosen["source"] = "frozen"

    others = []
    panel = REPO / "results/nonwalk_screen/panel32_cases.csv.gz"
    if panel.exists():
        others.append(_load(str(panel), budget))
    crawl = sorted(glob.glob(str(REPO / "results/nonwalk_crawl_screen/crawl_cases_shard_*.csv.gz")))
    if crawl:
        others.append(pd.concat([_load(f, budget) for f in crawl]))
    other = pd.concat(others, ignore_index=True)
    other = other[~other.strategy.isin(CHOSEN)]
    other["source"] = "screen"

    frame = pd.concat([chosen, other], ignore_index=True)
    return frame.dropna(subset=[TRUE, SEEN, NAIVE])


def channels(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["selection"] = frame[SEEN] - frame[TRUE]
    frame["censoring"] = frame[NAIVE] - frame[SEEN]
    frame["total_bias"] = frame[NAIVE] - frame[TRUE]
    keys = ["strategy"] + (["target_budget"] if "target_budget" in frame else [])
    rows = []
    for key, part in sorted(frame.groupby(keys, dropna=False),
                            key=lambda kv: str(kv[0])):
        arm = key[0] if isinstance(key, tuple) else key
        sel = float(part.selection.mean())
        cens = float(part.censoring.mean())
        movement = abs(sel) + abs(cens)
        rows.append({
            "strategy": arm,
            "target_budget": (key[1] if isinstance(key, tuple) and len(key) > 1
                              else np.nan),
            "n": len(part),
            "chosen": arm in CHOSEN,
            "selection": sel,
            "censoring": cens,
            "total_bias": float(part.total_bias.mean()),
            "coverage": float(part.coverage.mean()),
            "selection_share": abs(sel) / movement if movement else np.nan,
            "opposed": bool(sel * cens < 0) and min(abs(sel), abs(cens)) > 1e-6,
        })
    return pd.DataFrame(rows).sort_values("selection_share")


def plot(table: pd.DataFrame, path: Path) -> None:
    """Chosen arms as large markers, every alternative as its budget trail.

    The trail matters. The alternatives are read at five budgets, so plotting
    one point each would invite the objection that a different budget would
    move them; the whole trajectory is drawn instead, and it does not leave the
    band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(10, 6.4))
    ax.axhline(0, color="0.6", lw=1, ls="--")

    other = table[~table.chosen]
    for name, part in other.groupby("strategy"):
        part = part.sort_values("target_budget")
        ax.plot(part.selection_share, part.total_bias, "-", color="#9aa7c7",
                lw=1.0, alpha=0.85, zorder=1)
        ax.scatter(part.selection_share, part.total_bias, s=16,
                   color="#9aa7c7", edgecolor="none", zorder=1)

    # Eleven names inside one narrow band is unreadable and says nothing the
    # band itself does not. Only the two samplers that define its edges are
    # labelled; the rest are the band.
    if len(other):
        lo, hi = other.selection_share.min(), other.selection_share.max()
        ax.axvspan(lo, hi, color="#9aa7c7", alpha=0.18, zorder=0)
        edges = {other.loc[other.selection_share.idxmin(), "strategy"],
                 other.loc[other.selection_share.idxmax(), "strategy"]}
        for name in edges:
            row = other[other.strategy == name].iloc[-1]
            ax.annotate(name, (row.selection_share, row.total_bias),
                        textcoords="offset points", xytext=(6, -10),
                        fontsize=7, color="0.4")

    for _, r in table[table.chosen].iterrows():
        ax.scatter(r.selection_share, r.total_bias, s=230, marker="o",
                   facecolor="#a32d2d" if r.opposed else "#1d3f8f",
                   edgecolor="0.15", linewidth=1.4, zorder=4)
        # Labels on the right edge would run off the canvas.
        right = r.selection_share > 0.7
        ax.annotate(r.strategy, (r.selection_share, r.total_bias),
                    textcoords="offset points",
                    xytext=(-12, 12) if right else (10, 9),
                    ha="right" if right else "left",
                    fontsize=9, fontweight="bold", color="0.1", zorder=5)

    ax.set_xlabel("channel composition   |selection| / (|selection| + |censoring|)\n"
                  "0 = bias is entirely censoring                    1 = entirely selection")
    ax.set_ylabel("signed rho_2 bias of the naive plug-in")
    ax.set_title("Observation-mechanism space\n"
                 "the five chosen arms against every other sampler in the "
                 "repository, at five budgets each")
    ax.set_xlim(-0.10, 1.10)
    ax.grid(alpha=0.22)
    if len(other):
        ax.annotate("all eleven alternatives,\nevery budget, sit in this band",
                    xy=(other.selection_share.mean(), 0.62),
                    xycoords=("data", "axes fraction"),
                    xytext=(0.13, 0.80), textcoords=("data", "axes fraction"),
                    ha="center", fontsize=8.5, color="#41527d",
                    arrowprops=dict(arrowstyle="->", color="#41527d", lw=1))
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", markersize=12, markerfacecolor="#1d3f8f",
               markeredgecolor="0.15", label="chosen arm, one channel only"),
        Line2D([], [], marker="o", ls="", markersize=12, markerfacecolor="#a32d2d",
               markeredgecolor="0.15", label="chosen arm, channels opposed"),
        Line2D([], [], marker="o", ls="-", markersize=5, color="#9aa7c7",
               label="alternative sampler, trail over budgets 100-3200"),
    ], loc="lower right", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--budget", type=int, default=None,
                    help="restrict the alternatives to one budget; the default "
                         "keeps all five and draws each sampler's trail")
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g3"))
    ap.add_argument("--figure", default=str(REPO / "docs/figures/mechanism_space.png"))
    args = ap.parse_args()

    table = channels(collect(args.budget))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "mechanism_space.csv", index=False)
    print(table.round(4).to_string(index=False))

    figure = Path(args.figure)
    figure.parent.mkdir(parents=True, exist_ok=True)
    plot(table, figure)


if __name__ == "__main__":
    main()
