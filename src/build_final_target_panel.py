#!/usr/bin/env python3
"""Materialize only the prespecified 32 temporal graphs for the main study."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

import census
from benchmark_generators import (
    activity_memory_event_stream,
    dar_event_stream,
    family_from_events,
    normalize_event_stream,
)
from build_benchmark_data import _instance_row, stable_seed
from generator import make_dcsbm_graph, make_instance


# The eight sources the frozen 32-graph panel was built from. Five further
# datasets have become available since; they are not added here, because the
# panel is a frozen design decision and each source contributes three instances
# (one empirical plus two controlled twins). Pass --real-keys to build a wider
# panel into a scratch directory and see what it would buy.
#
# Every per-source seed goes through stable_seed(seed, "controlled", key, ...),
# so it depends on the key and not on the position in this list: a wider panel
# contains the frozen one as a subset instead of reshuffling it.
REAL_KEYS = [
    "sp_hospital", "sp_primaryschool", "sp_highschool2013",
    "sp_hypertext2009", "snap_collegemsg", "snap_email_eu",
    "snap_mathoverflow", "snap_bitcoin_otc",
]


def edge_signature(events):
    x = normalize_event_stream(events)
    nodes = frozenset(x.u) | frozenset(x.v)
    counts = x.groupby(["u", "v"]).size().sort_index()
    return nodes, counts


def assert_timing_twin(original, twin, label):
    na, ca = edge_signature(original)
    nb, cb = edge_signature(twin)
    if na != nb:
        raise RuntimeError(f"{label}: node set changed")
    if not ca.equals(cb):
        raise RuntimeError(f"{label}: topology or per-edge event counts changed")


def real_and_twins(out, registry_path, raw_dir, seed, W, real_keys=None):
    registry = census.load_registry(Path(registry_path))
    rows = []
    for key in (real_keys or REAL_KEYS):
        spec = registry[key]
        path = Path(raw_dir) / spec["file"]
        if not path.exists():
            raise FileNotFoundError(f"{key}: expected raw file {path}")
        original = normalize_event_stream(census.parse_events(path, spec["format"]))
        common = {
            "source": key, "domain": spec.get("domain", "unknown"),
            "substrate": "empirical", "group_id": f"real::{key}",
            "family": f"real::{key}", "rep": 0,
            "alpha": float("nan"), "chi": float("nan"),
        }
        row = _instance_row(
            out, f"real__{key}", original,
            {**common, "rho_target": float("nan"),
             "data_block": "real_empirical", "generator": "empirical",
             "generator_params_json": "{}"}, W=W)
        row.update(graph_category="empirical", panel_role="empirical",
                   matched_backbone=key)
        rows.append(row)

        family = family_from_events(original, name=f"real::{key}", W=W)
        family.timestamps = "bursty"
        for target, role in ((0.15, "controlled_low"),
                             (0.55, "controlled_high")):
            twin_seed = stable_seed(
                seed, "controlled", key, "bursty", "contiguous", target, 0)
            instance = make_instance(
                family, target, seed=twin_seed, span_layout="contiguous")
            iid = (f"controlled__{key}__bursty__contiguous__"
                   f"rho{target:.2f}__r0")
            assert_timing_twin(original, instance.events, iid)
            row = _instance_row(
                out, iid, instance.events,
                {**common, "rho_target": target,
                 "data_block": "real_controlled",
                 "generator": "steered_P[w,t]",
                 "generator_params_json": json.dumps({
                     "seed": twin_seed, "deviations": instance.deviations,
                     "timestamp_mode": "bursty",
                     "span_layout": "contiguous"})}, W=W)
            row.update(graph_category="controlled_variant", panel_role=role,
                       matched_backbone=key)
            rows.append(row)
    return rows


def mechanistic(out, plan, seed, W):
    rows = []
    dcfg = plan["dar"]
    for n in (500, 1500):
        fam_i, substrate, chi = 0, "dcsbm", 0.15
        gid = f"dar::{substrate}::n{n}::f{fam_i}"
        graph_seed = stable_seed(seed, "dar_graph", substrate, n, fam_i)
        graph = make_dcsbm_graph(
            n, graph_seed, average_degree=float(dcfg["average_degree"]))
        for alpha in (0.1, 0.9):
            event_seed = stable_seed(seed, gid, "homogeneous", alpha, chi)
            events = dar_event_stream(
                graph, alpha=alpha, chi=chi, seed=event_seed, W=W,
                event_rate=float(dcfg["event_rate"]))
            iid = f"dar__{substrate}__n{n}__f0__a{alpha}__c{chi}"
            row = _instance_row(
                out, iid, events,
                {"data_block": "mechanistic_dar", "source": "synthetic",
                 "domain": "mechanistic", "substrate": substrate,
                 "generator": "DAR(1)", "group_id": gid, "family": gid,
                 "rep": 0, "rho_target": float("nan"), "alpha": alpha,
                 "chi": chi, "generator_params_json": json.dumps({
                     "alpha": alpha, "chi": chi,
                     "event_rate": float(dcfg["event_rate"]),
                     "candidate_edges": graph.number_of_edges(),
                     "seed": event_seed})}, W=W)
            row.update(graph_category="literature_synthetic", panel_role="DAR",
                       matched_backbone="")
            rows.append(row)

    acfg = plan["activity_memory"]
    for n in (500, 1500):
        gid = f"activity::n{n}::f0"
        for mode, beta in (("memoryless", None), ("1.0", 1.0)):
            event_seed = stable_seed(seed, gid, mode)
            events = activity_memory_event_stream(
                n, seed=event_seed, W=W,
                slots_per_window=int(acfg["slots_per_window"]),
                mean_activity=float(acfg["mean_activity"]),
                memory_beta=beta)
            tag = "memoryless" if beta is None else "beta1"
            row = _instance_row(
                out, f"activity__n{n}__f0__{tag}", events,
                {"data_block": "mechanistic_activity", "source": "synthetic",
                 "domain": "mechanistic", "substrate": "activity_driven",
                 "generator": "activity_memory", "group_id": gid,
                 "family": gid, "rep": 0, "rho_target": float("nan"),
                 "alpha": float("nan"), "chi": float("nan"),
                 "generator_params_json": json.dumps({
                     "memory_beta": beta,
                     "mean_activity": float(acfg["mean_activity"]),
                     "slots_per_window": int(acfg["slots_per_window"]),
                     "seed": event_seed})}, W=W)
            row.update(graph_category="literature_synthetic",
                       panel_role="activity_memory", matched_backbone="")
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/benchmark_v21.yaml")
    ap.add_argument("--registry", default="config/datasets.yaml")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out-dir", default="results/final_target_panel")
    ap.add_argument("--real-keys", default=None,
                    help="comma-separated registry keys, or 'all'. Default is "
                         "the eight sources of the frozen panel. Anything else "
                         "builds a different panel and belongs in a scratch "
                         "--out-dir, never on top of the frozen one.")
    args = ap.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    plan = config["presets"]["v2"]
    seed, W = int(plan["seed"]), int(plan["W"])
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.real_keys == "all":
        real_keys = sorted(census.load_registry(Path(args.registry)))
    elif args.real_keys:
        real_keys = [k.strip() for k in args.real_keys.split(",") if k.strip()]
    else:
        real_keys = list(REAL_KEYS)
    if real_keys != list(REAL_KEYS):
        print(f"NOTE: building a {len(real_keys)}-source panel, not the frozen "
              f"eight. Expect {3 * len(real_keys)} real instances.")
    rows = real_and_twins(out, args.registry, args.raw_dir, seed, W,
                          real_keys=real_keys)
    rows += mechanistic(out, plan, seed, W)
    panel = pd.DataFrame(rows)
    order = {"empirical": 0, "literature_synthetic": 1,
             "controlled_variant": 2}
    panel["_order"] = panel.graph_category.map(order)
    panel = panel.sort_values(
        ["_order", "matched_backbone", "panel_role", "instance_id"]
    ).drop(columns="_order").reset_index(drop=True)
    expected = {"empirical": 8, "literature_synthetic": 8,
                "controlled_variant": 16}
    if len(panel) != 32 or panel.graph_category.value_counts().to_dict() != expected:
        raise RuntimeError("final panel composition invariant failed")

    manifest = out / "panel32_final.csv"
    panel.to_csv(manifest, index=False)
    hashes = []
    for rel in panel.path:
        path = out / rel
        hashes.append((hashlib.sha256(path.read_bytes()).hexdigest(), rel))
    with open(out / "SHA256SUMS", "w") as fh:
        for digest, rel in hashes:
            fh.write(f"{digest}  {rel}\n")
    with open(out / "DESIGN.json", "w") as fh:
        json.dump({
            "W": W, "seed": seed, "n_instances": 32,
            "composition": expected, "controlled_targets": [0.15, 0.55],
            "controlled_timestamp_mode": "bursty",
            "controlled_span_layout": "contiguous",
            "dar_design": {"n": [500, 1500], "alpha": [0.1, 0.9],
                           "chi": 0.15, "family": 0},
            "activity_design": {"n": [500, 1500],
                                "memory": ["memoryless", "beta1"],
                                "family": 0},
        }, fh, indent=2)
    print(panel.groupby(["graph_category", "panel_role"]).size().to_string())
    print(f"\nwrote {manifest}")
    print(f"wrote {out / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
