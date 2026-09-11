#!/usr/bin/env python3
"""Select the prespecified 32-instance main-study panel."""

import argparse
from pathlib import Path

import pandas as pd


# A small balanced parameter design, not a post-hoc performance selection:
# two sizes crossed with low/high temporal memory, holding f0 and chi fixed.
DAR_IDS = [
    f"dar__dcsbm__n{n}__f0__a{alpha}__c0.15"
    for n in (500, 1500) for alpha in (0.1, 0.9)
]
ACTIVITY_IDS = [
    f"activity__n{n}__f0__{memory}"
    for n in (500, 1500) for memory in ("memoryless", "beta1")
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="results/benchmark_v2/data/manifest.csv")
    ap.add_argument("--supplement",
                    default="results/benchmark_v2/data/panel_supplement.csv")
    ap.add_argument("--out", default="results/target_diagnostics/panel32_final.csv")
    args = ap.parse_args()

    m = pd.read_csv(args.manifest)
    supplement = Path(args.supplement)
    if supplement.exists():
        m = pd.concat([m, pd.read_csv(supplement)], ignore_index=True)
        m = m.drop_duplicates("instance_id", keep="last")

    empirical = m[m.data_block.eq("real_empirical")].copy()
    if len(empirical) != 8:
        raise SystemExit(f"expected exactly 8 empirical instances, found {len(empirical)}")
    empirical["graph_category"] = "empirical"
    empirical["panel_role"] = "empirical"
    empirical["matched_backbone"] = empirical["source"]

    synthetic_ids = DAR_IDS + ACTIVITY_IDS
    synthetic = m[m.instance_id.isin(synthetic_ids)].copy()
    missing_synthetic = sorted(set(synthetic_ids) - set(synthetic.instance_id))
    if missing_synthetic:
        raise SystemExit("missing prespecified synthetic instances:\n" +
                         "\n".join(missing_synthetic))
    synthetic["graph_category"] = "literature_synthetic"
    synthetic["panel_role"] = synthetic.instance_id.map(
        lambda x: "DAR" if x.startswith("dar__") else "activity_memory")
    synthetic["matched_backbone"] = ""

    twin_ids = []
    for source in sorted(empirical.source):
        for target, role in ((0.15, "controlled_low"),
                             (0.55, "controlled_high")):
            twin_ids.append((
                f"controlled__{source}__bursty__contiguous__rho{target:.2f}__r0",
                source, role))
    controlled = m[m.instance_id.isin([x[0] for x in twin_ids])].copy()
    missing_twins = sorted(set(x[0] for x in twin_ids) -
                           set(controlled.instance_id))
    if missing_twins:
        raise SystemExit(
            "missing controlled variants required by the 8-backbone design:\n" +
            "\n".join(missing_twins) +
            "\nRun materialize_missing_panel_twins.py first.")
    role = {iid: r for iid, _, r in twin_ids}
    backbone = {iid: s for iid, s, _ in twin_ids}
    controlled["graph_category"] = "controlled_variant"
    controlled["panel_role"] = controlled.instance_id.map(role)
    controlled["matched_backbone"] = controlled.instance_id.map(backbone)

    panel = pd.concat([empirical, synthetic, controlled], ignore_index=True)
    counts = panel.graph_category.value_counts().to_dict()
    expected = {"empirical": 8, "literature_synthetic": 8,
                "controlled_variant": 16}
    if counts != expected or len(panel) != 32:
        raise SystemExit(f"panel composition {counts}, expected {expected}")
    pair_counts = controlled.groupby("matched_backbone").panel_role.nunique()
    if len(pair_counts) != 8 or not pair_counts.eq(2).all():
        raise SystemExit("each empirical backbone must have one low and one high twin")

    keep = ["instance_id", "graph_category", "panel_role",
            "matched_backbone", "data_block", "group_id", "family", "source",
            "generator", "path", "n_nodes", "n_edges", "n_events",
            "rho_target", "rho_W5_k2", "rho_W5_k3", "rho_W5_k4",
            "rho_W5_k5", "C_one_step", "mean_span_frac",
            "generator_params_json"]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    panel[keep].to_csv(out, index=False)
    print(panel.groupby(["graph_category", "panel_role"]).size().to_string())
    print(f"\nwrote frozen-design panel: {out}")


if __name__ == "__main__":
    main()
