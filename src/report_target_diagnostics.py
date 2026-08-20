#!/usr/bin/env python3
"""Create shareable tables, figures and a Markdown target-diagnostics report."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


LABELS = {
    "rho_k2": r"$\rho_2$", "rho_k3": r"$\rho_3$",
    "rho_k4": r"$\rho_4$", "rho_k5": r"$\rho_5$",
    "mean_occupancy": "Mean occupancy", "C_one_step": r"$C_{one-step}$",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="results/final_target_panel/panel32_final.csv")
    ap.add_argument("--w-summary",
                    default="results/target_diagnostics/w_robustness_final32/summary.csv")
    ap.add_argument("--c-metrics",
                    default="results/target_diagnostics/C_complementarity_final32/metrics.csv")
    ap.add_argument("--c-pairs",
                    default="results/target_diagnostics/C_complementarity_final32/matched_profile_different_C.csv")
    ap.add_argument("--out-dir", default="results/target_diagnostics/shareable")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, w = pd.read_csv(args.panel), pd.read_csv(args.w_summary)
    cm, cp = pd.read_csv(args.c_metrics), pd.read_csv(args.c_pairs)

    # Compact manuscript/meeting table.
    compact = w[w.graph_category.ne("all")].copy()
    compact["target_label"] = compact.target.map(LABELS)
    compact.to_csv(out / "w_stability_by_category.csv", index=False)

    categories = ["empirical", "literature_synthetic", "controlled_variant"]
    targets = ["rho_k2", "mean_occupancy", "C_one_step"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    for col, comparison in enumerate(["W5_vs_W4", "W5_vs_W8"]):
        x = range(len(targets))
        width = 0.24
        for i, category in enumerate(categories):
            d = w[(w.comparison == comparison) &
                  (w.graph_category == category)].set_index("target")
            vals = [d.loc[t, "spearman"] for t in targets]
            axes[0, col].bar([v + (i - 1) * width for v in x], vals, width,
                             label=category.replace("_", " "))
            changes = [d.loc[t, "mae_change"] for t in targets]
            axes[1, col].bar([v + (i - 1) * width for v in x], changes,
                             width, label=category.replace("_", " "))
        for row in range(2):
            axes[row, col].set_xticks(list(x), [LABELS[t] for t in targets])
            axes[row, col].grid(axis="y", alpha=.25)
        axes[0, col].set_ylim(0, 1.05)
        axes[0, col].set_title(comparison.replace("_", " "))
        axes[0, col].set_ylabel("Spearman rank correlation")
        axes[1, col].set_ylabel("Mean absolute change")
    axes[0, 1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "w_stability_rank_correlation.png", dpi=220)
    plt.close(fig)

    truth_models = cm[cm.analysis == "C_from_true_profile"].set_index("model")
    ridge = truth_models.loc["ridge"]
    trees = truth_models.loc["extra_trees"]
    examples = cp.sort_values("C_abs_delta", ascending=False).head(5)
    examples.to_csv(out / "near_profile_different_C_examples.csv", index=False)
    floor_rows = []
    for k in range(2, 6):
        values = panel[f"rho_W5_k{k}"]
        floor_rows.append({
            "target": f"rho_{k}", "share_below_0.05": (values < .05).mean(),
            "share_zero": (values == 0).mean(), "median": values.median(),
        })
    floors = pd.DataFrame(floor_rows)
    floors.to_csv(out / "profile_floor_effects.csv", index=False)
    twins = panel[panel.graph_category == "controlled_variant"][
        ["matched_backbone", "panel_role", "rho_target", "rho_W5_k2"]
    ].sort_values(["matched_backbone", "panel_role"])
    twins.to_csv(out / "controlled_twin_achieved_rho2.csv", index=False)

    lines = [
        "# Target diagnostics for the frozen 32-graph panel", "",
        "## Panel", "",
        "The panel contains 8 empirical graphs, 8 literature-based mechanistic "
        "graphs (4 DAR and 4 activity/memory), and 16 matched timing variants "
        "(low/high for each empirical backbone).", "",
        "All controlled variants preserve the node set, time-collapsed topology, "
        "and per-edge event counts; only event timing changes.", "",
        "## Window-count robustness", "",
        "Spearman rank correlations against the W=5 definition:", "",
        w[(w.graph_category != "all") &
          w.target.isin(targets)][
            ["graph_category", "comparison", "target", "spearman",
             "mae_change", "max_abs_change"]
        ].round(4).to_markdown(index=False), "",
        "Interpretation: ranking is generally stable, especially for empirical "
        "graphs and mean occupancy. C_one_step changes more in absolute scale "
        "because adjacency itself changes when W changes.", "",
        "This tests sensitivity to temporal resolution, not sensitivity to "
        "randomly shifted window boundaries.", "",
        "## Profile floor effects", "",
        floors.round(4).to_markdown(index=False), "",
        "ProfileMAE is therefore accompanied by rho2 MAE, bias, validity and "
        "skill against the prespecified mean-prediction floor.", "",
        "## Controlled low/high variants", "",
        twins.round(4).to_markdown(index=False), "",
        "The labels low/high describe a within-backbone contrast. rho_target "
        "is the requested construction target; rho_W5_k2 is the achieved "
        "value. In particular, not every high variant reaches 0.55. No seed "
        "was selected post hoc to improve this contrast.", "",
        "## Is C_one_step complementary to the rho profile?", "",
        f"On the frozen 32-graph panel, group-held-out Ridge predicts "
        f"C_one_step from the true (rho2,...,rho5) profile with "
        f"R²={ridge.r2_oof:.4f} and MAE={ridge.mae_oof:.4f}; ExtraTrees "
        f"obtains R²={trees.r2_oof:.4f} and MAE={trees.mae_oof:.4f}. "
        "Thus C is largely redundant on average, but not identical to the "
        "profile.", "",
        "Near-profile examples with large C differences:", "",
        examples[["instance_a", "instance_b", "profile_max_abs_delta",
                  "C_a", "C_b", "C_abs_delta"]].round(4).to_markdown(index=False),
        "", "## Recommended target hierarchy", "",
        "1. Primary target: the full dyadic active-window survival profile "
        "(rho2, rho3, rho4, rho5), scored by ProfileMAE.",
        "2. Headline component: rho2 (multi-window dyad share).",
        "3. Secondary target: pooled adjacent-window dyad retention C_one_step.",
        "4. Mean occupancy: report as a profile-derived descriptive quantity.",
        "5. Lifetime: robustness/appendix target.", "",
        "The figure `w_stability_rank_correlation.png` and accompanying CSV "
        "files are intended for meetings, manuscript drafting, and review by "
        "other models.", "",
    ]
    (out / "TARGET_DIAGNOSTICS_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8")
    print(f"wrote shareable report bundle to {out}")


if __name__ == "__main__":
    main()
