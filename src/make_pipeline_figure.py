#!/usr/bin/env python3
"""Draw the project pipeline figure.

The figure is generated rather than drawn by hand so that it cannot drift away
from the code the way a one-off image does. Every quantity it shows is either
computed here from the toy example (the rho values in panel 2) or taken from a
named artefact in the repository, and the docstrings below record which.

Run from the repository root:

    python src/make_pipeline_figure.py

Writes figures/pipeline.png and figures/pipeline.pdf.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Ellipse, FancyArrowPatch

FIG_W, FIG_H = 17.0, 6.8
# One y-unit is this many x-units tall on screen. Vertical extents of anything
# that has to look round or square must be multiplied by it.
ASPECT = FIG_W / FIG_H

W = 5  # windows; matches HEADLINE_W in src/census.py and the frozen prompt

INK = "#12233f"
MUTED = "#6b7785"
LINE = "#c6ccd4"
ACCENT = "#d1495b"
COOL = "#2a6f97"
PANEL_EDGE = "#b9c2cc"
PANEL_FILL = "#ffffff"

# Toy example for panel 2. Same four nodes and same set of pairs in both rows;
# only the temporal placement differs. This is the benchmark's central control:
# topology is held fixed so that a difference in rho cannot be attributed to a
# difference in structure.
NODES = {"A": (0.5, 1.0), "B": (0.0, 0.45), "C": (1.0, 0.45), "D": (0.5, 0.0)}
PAIRS = [("A", "B"), ("A", "C"), ("B", "C"), ("B", "D"), ("C", "D")]

LOW = [  # active pairs per window: contacts move on, almost nothing recurs
    [("A", "B"), ("B", "C")],
    [("A", "C"), ("B", "C")],
    [("B", "D")],
    [("C", "D")],
    [("B", "C")],
]
HIGH = [  # the same pairs recur across windows
    [("A", "B"), ("B", "C"), ("B", "D")],
    [("A", "B"), ("B", "C"), ("C", "D")],
    [("A", "B"), ("B", "C"), ("A", "C")],
    [("A", "B"), ("B", "D"), ("C", "D")],
    [("A", "B"), ("B", "C")],
]


def rho(windows, k):
    """Share of pairs active in at least k windows.

    Mirrors `profile()` in src/census.py: the denominator is every pair with at
    least one event in the full stream, not the pairs some sample happened to
    reveal. Keeping the figure's numbers on the same definition is the point of
    computing them here instead of typing them in.
    """
    counts = {}
    for active in windows:
        for pair in active:
            counts[frozenset(pair)] = counts.get(frozenset(pair), 0) + 1
    if not counts:
        return 0.0
    return sum(1 for c in counts.values() if c >= k) / len(counts)


def panel(ax, x0, x1, y0, y1, number, title, subtitle=None):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle="round,pad=0,rounding_size=1.2",
        linewidth=1.0, edgecolor=PANEL_EDGE, facecolor=PANEL_FILL, zorder=1))
    ax.add_patch(Ellipse((x0 + 2.2, y1 - 2.6), 2.5, 2.5 * ASPECT,
                         facecolor=INK, edgecolor="none", zorder=3))
    ax.text(x0 + 2.2, y1 - 2.6, str(number), color="white", fontsize=8,
            ha="center", va="center", fontweight="bold", zorder=4)
    # The title sits below the badge rather than beside it: in the narrow
    # panels a centred title runs straight through the number.
    ax.text((x0 + x1) / 2, y1 - 7.2, title, color=INK, fontsize=9.5,
            ha="center", va="center", fontweight="bold", zorder=4)
    if subtitle:
        ax.text((x0 + x1) / 2, y1 - 11.2, subtitle, color=MUTED, fontsize=7.0,
                ha="center", va="center", zorder=4, linespacing=1.4)


def mini_graph(ax, cx, cy, scale, active):
    """One window snapshot: fixed topology, highlighted active pairs."""
    act = {frozenset(p) for p in active}
    pos = {n: (cx + (x - 0.5) * scale, cy + (y - 0.5) * scale * ASPECT)
           for n, (x, y) in NODES.items()}
    for u, v in PAIRS:
        on = frozenset((u, v)) in act
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]],
                color=ACCENT if on else LINE, lw=1.6 if on else 0.8,
                solid_capstyle="round", zorder=2)
    for n, (x, y) in pos.items():
        ax.add_patch(Ellipse((x, y), scale * 0.30, scale * 0.30 * ASPECT,
                             facecolor="white", edgecolor=MUTED, lw=0.7,
                             zorder=3))
        ax.text(x, y, n, fontsize=5.0, color=INK, ha="center", va="center",
                zorder=4)


def arrow(ax, x, y):
    ax.add_patch(FancyArrowPatch((x, y), (x + 1.6, y), arrowstyle="-|>",
                                 mutation_scale=9, color=MUTED, lw=1.0,
                                 zorder=5))


def build():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---------------------------------------------------------------- panel 1
    panel(ax, 1, 13.5, 6, 94, 1, "Temporal network",
          "continuous-time\nevent stream")
    ax.text(7.25, 72, r"$[(u,v,t_1),\ (u,w,t_2),\ \ldots]$", fontsize=7.4,
            color=INK, ha="center", va="center")
    ax.plot([3.2, 11.3], [63, 63], color=INK, lw=1.0)
    for i in range(6):
        x = 3.2 + i * 1.62
        ax.plot([x, x], [61.6, 64.4], color=INK, lw=1.0)
    ax.text(7.25, 58.5, "time", fontsize=7.0, color=MUTED, ha="center")
    ax.text(7.25, 42, "The full pair list is\nnot the observation.\n"
                      "Access is partial\nand can be biased.",
            fontsize=7.0, color=MUTED, ha="center", va="center",
            linespacing=1.6)
    arrow(ax, 14.0, 50)

    # ---------------------------------------------------------------- panel 2
    panel(ax, 16.5, 47.5, 6, 94, 2, "Controlled benchmark",
          "same pairs, different temporal placement "
          r"$\rightarrow$ different $\rho$")
    xs = [20.2 + i * 6.2 for i in range(W)]
    for row, (windows, label, ylabel, ygraph) in enumerate([
            (LOW, "low persistence", 78.0, 66.0),
            (HIGH, "high persistence", 47.0, 35.0)]):
        ax.text(32.0, ylabel,
                f"{label}  " + r"($\rho_2 = $" + f"{rho(windows, 2):.2f})",
                fontsize=8.0, color=ACCENT, ha="center", fontweight="bold")
        for i, (x, active) in enumerate(zip(xs, windows)):
            ax.text(x, ygraph + 8.5, f"$W_{i + 1}$", fontsize=6.6,
                    color=MUTED, ha="center")
            mini_graph(ax, x, ygraph, 4.0, active)
        if row == 0:
            ax.plot([19.5, 44.5], [53.0, 53.0], color=LINE, lw=0.8,
                    ls=(0, (4, 3)))
    ax.text(32.0, 20.0,
            r"$K$ = number of the $W = 5$ windows in which a pair is active"
            "\n"
            r"$\rho_k = $ share of pairs of the FULL network with $K \geq k$,"
            r"  $k = 2 \ldots 5$",
            fontsize=7.4, color=INK, ha="center", va="center", linespacing=1.9)
    ax.text(32.0, 10.5, "the profile, not one number: the benchmark scores all\n"
                        "four levels, with $k = 2$ as the headline slice",
            fontsize=6.8, color=MUTED, ha="center", va="center",
            linespacing=1.5)
    arrow(ax, 48.0, 50)

    # ---------------------------------------------------------------- panel 3
    panel(ax, 50.5, 65.5, 6, 94, 3, "Partial access",
          "7 mechanisms in the\nbenchmark, 2 families")
    ax.text(52.2, 74.0, "walk-based", fontsize=7.6, color=COOL,
            fontweight="bold", ha="left")
    # Names and roles follow the module docstring of src/walks.py. The two
    # time-agnostic rows are deliberately separate: only the first one sees no
    # timestamps at all, which is what makes it the leakage control. Collapsing
    # them into one row would hide the arm the design leans on.
    for i, (name, note, dashed) in enumerate([
            ("time-agnostic", "no timestamps; leakage control", True),
            ("time-agnostic + times", "static hops, observed times", False),
            ("time-respecting", "CTDNE-style temporal order", False),
            ("recency-biased", "favours recent events", False),
            ("multi-start", "several entry points", False)]):
        y = 69.0 - i * 5.6
        ax.plot([52.4, 54.2], [y + 0.6, y + 0.6],
                color=MUTED if dashed else COOL, lw=1.2,
                ls=(0, (2, 1.6)) if dashed else "-", solid_capstyle="round")
        ax.add_patch(Ellipse((54.4, y + 0.6), 0.85, 0.85 * ASPECT,
                             facecolor=MUTED if dashed else COOL,
                             edgecolor="none"))
        ax.text(55.6, y + 0.9, name, fontsize=6.9, color=INK, va="center")
        ax.text(55.6, y - 1.5, note, fontsize=6.0, color=MUTED, va="center")
    ax.plot([52.2, 63.8], [40.0, 40.0], color=LINE, lw=0.8, ls=(0, (4, 3)))
    ax.text(52.2, 36.0, "non-walk", fontsize=7.6, color=ACCENT,
            fontweight="bold", ha="left")
    for i, (name, note) in enumerate([
            ("recent history, k = 5", "complete history, 5 nodes"),
            ("recent history, k = 20", "complete history, 20 nodes")]):
        y = 30.5 - i * 6.0
        ax.add_patch(Ellipse((53.3, y + 0.6), 1.7, 1.7 * ASPECT,
                             facecolor="none", edgecolor=ACCENT, lw=1.0))
        ax.text(55.6, y + 0.9, name, fontsize=6.9, color=INK, va="center")
        ax.text(55.6, y - 1.5, note, fontsize=6.0, color=MUTED, va="center")
    ax.text(58.0, 13.0, "the frozen LLM suite uses\nthree of these: two walks\n"
                        "and one non-walk panel",
            fontsize=6.4, color=MUTED, ha="center", va="center",
            linespacing=1.6)
    arrow(ax, 66.0, 50)

    # ---------------------------------------------------------------- panel 4
    panel(ax, 68.0, 82.0, 6, 94, 4, "Compact summary",
          "what the estimator\nactually receives")
    ax.text(75.0, 74.5, r"$(n,\ \mathrm{window\ mask})$ histogram",
            fontsize=7.2, color=COOL, ha="center", fontweight="bold")
    ax.text(75.0, 70.0, "per observed pair: how often it was\n"
                        "recorded, and in which windows",
            fontsize=6.2, color=MUTED, ha="center", va="center",
            linespacing=1.5)
    rows = [("n, mask", "pairs"), ("1, 01", "6"), ("1, 02", "19"),
            ("1, 04", "23"), ("2, 01", "2"), ("3, 01", "1")]
    for i, (a, b) in enumerate(rows):
        y = 63.0 - i * 3.5
        head_row = i == 0
        if head_row:
            ax.add_patch(FancyBboxPatch(
                (69.6, y - 1.6), 10.8, 3.2,
                boxstyle="round,pad=0,rounding_size=0.4",
                facecolor=INK, edgecolor="none", zorder=2))
        ax.text(71.0, y, a, fontsize=6.3, color="white" if head_row else INK,
                va="center", zorder=3, family="monospace")
        ax.text(79.2, y, b, fontsize=6.3, color="white" if head_row else INK,
                va="center", ha="right", zorder=3, family="monospace")
        if not head_row:
            ax.plot([69.6, 80.4], [y - 1.8, y - 1.8], color=LINE, lw=0.5)
    ax.text(75.0, 34.0, "no hand-built feature vector:\n"
                        "the summary is the raw counts,\n"
                        "so the estimator has to do the\n"
                        "bias correction itself",
            fontsize=6.4, color=INK, ha="center", va="center",
            linespacing=1.6)
    ax.text(75.0, 18.0, "variants ablate the input:\n"
                        "crawl order, timestamps,\n"
                        "recency, node panel",
            fontsize=6.4, color=MUTED, ha="center", va="center",
            linespacing=1.6)
    arrow(ax, 82.5, 50)

    # ---------------------------------------------------------------- panel 5
    panel(ax, 85.0, 99.0, 6, 94, 5, "Estimation and scoring",
          "three families on the\nsame cases")
    for i, (name, note, col) in enumerate([
            ("analytical estimators", "explicit bias corrections", COOL),
            ("supervised baselines", "scikit-learn ExtraTrees\nand RandomForest",
             COOL),
            ("language models", "zero-shot and\ndisclosed-examples", ACCENT)]):
        y = 72.0 - i * 11.5
        ax.add_patch(FancyBboxPatch(
            (86.4, y - 4.6), 11.2, 7.0,
            boxstyle="round,pad=0,rounding_size=0.6",
            facecolor="white", edgecolor=col, lw=1.0, zorder=2))
        ax.text(92.0, y + 0.6, name, fontsize=6.8, color=INK, ha="center",
                va="center", zorder=3, fontweight="bold")
        ax.text(92.0, y - 2.4, note, fontsize=5.8, color=MUTED, ha="center",
                va="center", zorder=3, linespacing=1.4)
    ax.plot([86.6, 97.4], [42.0, 42.0], color=LINE, lw=0.8, ls=(0, (4, 3)))
    ax.text(92.0, 37.0, "leakage-aware paired evaluation",
            fontsize=6.8, color=INK, ha="center", fontweight="bold")
    ax.text(92.0, 30.0, r"ProfileMAE $= \frac{1}{4}\sum_{k=2}^{5}"
                        r"|\hat{\rho}_k - \rho_k|$",
            fontsize=7.2, color=INK, ha="center", va="center")
    ax.text(92.0, 22.0, "invalid or missing output is\n"
                        "penalized, never repaired",
            fontsize=6.3, color=ACCENT, ha="center", va="center",
            linespacing=1.5)
    ax.text(92.0, 12.5, "secondary targets:\n"
                        "mean occupancy, one-step\n"
                        "persistence $C$, link lifetime",
            fontsize=6.2, color=MUTED, ha="center", va="center",
            linespacing=1.5)

    fig.tight_layout(pad=0.4)
    out = Path("figures")
    out.mkdir(exist_ok=True)
    fig.savefig(out / "pipeline.png", dpi=200, facecolor="white")
    fig.savefig(out / "pipeline.pdf", facecolor="white")
    plt.close(fig)
    print(f"rho_2 low  = {rho(LOW, 2):.2f}")
    print(f"rho_2 high = {rho(HIGH, 2):.2f}")
    print("wrote figures/pipeline.png and figures/pipeline.pdf")


if __name__ == "__main__":
    build()
