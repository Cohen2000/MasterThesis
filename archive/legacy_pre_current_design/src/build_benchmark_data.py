#!/usr/bin/env python3
"""Build v1 or v2 estimator-screen event streams.

The output is a portable manifest plus compressed ``u,v,t`` files.  The legacy
``smoke``/``full`` presets remain available; v2 lives in independent
``v2_smoke``/``v2`` folders and never overwrites v1 unless explicitly asked.
"""

import argparse
import json
from pathlib import Path
import zlib

import networkx as nx
import numpy as np
import pandas as pd
import yaml

import census
from benchmark_generators import (
    activity_memory_event_stream,
    dar_event_stream,
    edge_rewire_surrogate,
    family_from_events,
    lifetime_resample,
    normalize_event_stream,
    renewal_event_stream,
    temporal_chunks,
    timestamp_shuffle,
    truth_for_events,
    within_window_timestamp_shuffle,
)
from census import window_index
from generator import make_dcsbm_graph, make_family, make_instance, make_substrate


def stable_seed(base, *parts):
    payload = "|".join(map(str, (base,) + parts)).encode()
    return int(zlib.crc32(payload) & 0xFFFFFFFF)


def load_plan(config_path, preset):
    with open(config_path) as fh:
        cfg = yaml.safe_load(fh)
    if preset not in cfg["presets"]:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(cfg['presets'])}")
    return cfg, cfg["presets"][preset]


def _instance_row(out, instance_id, events, metadata, W=5):
    safe = instance_id.replace("/", "_").replace("|", "__")
    block = metadata["data_block"]
    path = out / "events" / block / f"{safe}.csv.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    events = events[["u", "v", "t"]].sort_values("t", kind="mergesort")
    events.to_csv(path, index=False, compression="gzip")
    truth = truth_for_events(events, label=instance_id, W=W)
    nodes = pd.unique(pd.concat([events["u"], events["v"]], ignore_index=True))
    return {
        "instance_id": instance_id,
        "path": str(path.relative_to(out)),
        "n_nodes": int(len(nodes)),
        "n_edges": int(truth["n_pairs"]),
        "n_events": int(len(events)),
        **metadata,
        **truth,
    }


def _edge_window_counts(events, W):
    x = events[["u", "v", "t"]].copy()
    x["w"] = window_index(x["t"].to_numpy(float), 0.0, 1.0 / W, W)
    return x.groupby(["u", "v", "w"]).size().sort_index()


def _assert_shuffle_invariants(original, shuffled, mode, W):
    assert len(original) == len(shuffled)
    if mode in {"global", "within_window"}:
        assert np.array_equal(np.sort(original["t"]), np.sort(shuffled["t"]))
    if mode in {"global", "within_window", "lifetime_resample"}:
        a = original.groupby(["u", "v"]).size().sort_index()
        b = shuffled.groupby(["u", "v"]).size().sort_index()
        assert a.equals(b)
    if mode == "within_window":
        assert _edge_window_counts(original, W).equals(_edge_window_counts(shuffled, W))
    if mode == "lifetime_resample":
        a = original.groupby(["u", "v"])["t"].agg(["min", "max"]).sort_index()
        b = shuffled.groupby(["u", "v"])["t"].agg(["min", "max"]).sort_index()
        assert np.allclose(a.to_numpy(float), b.to_numpy(float))
    if mode == "edge_rewire":
        ga = nx.Graph(); ga.add_edges_from(original[["u", "v"]].drop_duplicates().itertuples(index=False, name=None))
        gb = nx.Graph(); gb.add_edges_from(shuffled[["u", "v"]].drop_duplicates().itertuples(index=False, name=None))
        assert sorted(dict(ga.degree()).values()) == sorted(dict(gb.degree()).values())
        sa = sorted(tuple(np.round(np.sort(g["t"].to_numpy(float)), 12))
                    for _, g in original.groupby(["u", "v"]))
        sb = sorted(tuple(np.round(np.sort(g["t"].to_numpy(float)), 12))
                    for _, g in shuffled.groupby(["u", "v"]))
        assert sa == sb


def _shuffle(events, mode, seed, W):
    if mode == "global":
        return timestamp_shuffle(events, seed)
    if mode == "within_window":
        return within_window_timestamp_shuffle(events, seed, W=W)
    if mode == "lifetime_resample":
        return lifetime_resample(events, seed)
    if mode == "edge_rewire":
        return edge_rewire_surrogate(events, seed)
    raise ValueError(f"unknown shuffle mode: {mode}")


def _real_instances(plan, out, registry_path, raw_dir, base_seed, W):
    rcfg = plan["real"]
    if not rcfg.get("enabled", False):
        return [], []
    registry = census.load_registry(Path(registry_path))
    variant_keys = set(rcfg.get("variant_datasets", []))
    rows, missing = [], []
    for key in rcfg.get("datasets", []):
        if key not in registry:
            raise KeyError(f"real dataset {key!r} missing from {registry_path}")
        spec = registry[key]
        path = Path(raw_dir) / spec["file"]
        if not path.exists():
            missing.append((key, spec["file"], spec.get("page_url", "")))
            continue
        print(f"[real] parsing {key}: {path}", flush=True)
        raw = census.parse_events(path, spec["format"])
        events = normalize_event_stream(raw)
        common = {
            "source": key, "domain": spec.get("domain", "unknown"),
            "substrate": "empirical", "group_id": f"real::{key}",
            "family": f"real::{key}", "rep": 0,
            "rho_target": np.nan, "alpha": np.nan, "chi": np.nan,
        }
        rows.append(_instance_row(
            out, f"real__{key}", events,
            {**common, "data_block": "real_empirical", "generator": "empirical",
             "generator_params_json": "{}"}, W=W))

        # Disjoint temporal panels.  group_id deliberately remains the source.
        n_chunks = int(rcfg.get("chunks", {}).get(key, 0))
        for ci, chunk in enumerate(temporal_chunks(
                events, n_chunks, min_events=int(rcfg.get("min_chunk_events", 500)))):
            rows.append(_instance_row(
                out, f"real_chunk__{key}__c{ci}", chunk,
                {**common, "rep": ci, "data_block": "real_chunk",
                 "generator": "empirical_chunk",
                 "generator_params_json": json.dumps({"chunk": ci, "n_chunks": n_chunks})}, W=W))

        if key not in variant_keys:
            continue

        # v1 had one implicit global P[w,t] mode.  Keep that default unchanged.
        modes = rcfg.get("shuffle_modes", ["global"])
        for mode in modes:
            for rep in range(int(rcfg.get("shuffle_reps", 0))):
                seed = stable_seed(base_seed, "shuffle", mode, key, rep)
                # Preserve v1 seeds exactly when no explicit mode list exists.
                if "shuffle_modes" not in rcfg:
                    seed = stable_seed(base_seed, "shuffle", key, rep)
                shuffled = _shuffle(events, str(mode), seed, W)
                _assert_shuffle_invariants(events, shuffled, str(mode), W)
                old_v1_shuffle = "shuffle_modes" not in rcfg
                shuffle_iid = (f"shuffle__{key}__r{rep}" if old_v1_shuffle
                               else f"shuffle_{mode}__{key}__r{rep}")
                shuffle_block = "real_shuffle" if old_v1_shuffle else f"real_shuffle_{mode}"
                shuffle_generator = "P[w,t]" if old_v1_shuffle else str(mode)
                shuffle_params = ({"seed": seed} if old_v1_shuffle
                                  else {"seed": seed, "mode": mode})
                rows.append(_instance_row(
                    out, shuffle_iid, shuffled,
                    {**common, "rep": rep, "data_block": shuffle_block,
                     "generator": shuffle_generator,
                     "generator_params_json": json.dumps(shuffle_params)}, W=W))

        timestamp_modes = rcfg.get("controlled_timestamp_modes", ["empirical"])
        span_layouts = rcfg.get("controlled_span_layouts", ["legacy"])
        for tmode in timestamp_modes:
            fam = family_from_events(events, name=f"real::{key}", W=W)
            fam.timestamps = str(tmode)
            for layout in span_layouts:
                for target in rcfg.get("controlled_targets", []):
                    for rep in range(int(rcfg.get("controlled_reps", 0))):
                        seed = stable_seed(base_seed, "controlled", key, tmode, layout, target, rep)
                        if "controlled_timestamp_modes" not in rcfg and "controlled_span_layouts" not in rcfg:
                            seed = stable_seed(base_seed, "controlled", key, target, rep)
                        inst = make_instance(fam, float(target), seed=seed,
                                             span_layout=str(layout))
                        old_v1_controlled = ("controlled_timestamp_modes" not in rcfg
                                             and "controlled_span_layouts" not in rcfg)
                        iid = (f"controlled__{key}__rho{float(target):.2f}__r{rep}"
                               if old_v1_controlled else
                               f"controlled__{key}__{tmode}__{layout}__rho{float(target):.2f}__r{rep}")
                        params = ({"seed": seed, "deviations": inst.deviations}
                                  if old_v1_controlled else
                                  {"seed": seed, "deviations": inst.deviations,
                                   "timestamp_mode": tmode, "span_layout": layout})
                        rows.append(_instance_row(
                            out, iid, inst.events,
                            {**common, "rep": rep, "rho_target": float(target),
                             "data_block": "real_controlled",
                             "generator": "steered_P[w,t]",
                             "generator_params_json": json.dumps(params)}, W=W))
    return rows, missing


def _graph_for_substrate(substrate, n, seed, cfg):
    if substrate == "dcsbm":
        return make_dcsbm_graph(int(n), seed,
                                 average_degree=float(cfg.get("average_degree", 24)))
    return make_substrate(substrate, int(n), seed)


def _dar_instances(plan, out, base_seed, W):
    cfg = plan["dar"]
    if not cfg.get("enabled", False):
        return []
    rows = []
    for substrate in cfg["substrates"]:
        for n in cfg["sizes"]:
            for fam_i in range(int(cfg["families"])):
                gseed = stable_seed(base_seed, "dar_graph", substrate, n, fam_i)
                g = _graph_for_substrate(substrate, n, gseed, cfg)
                gid = f"dar::{substrate}::n{n}::f{fam_i}"
                for alpha in cfg["alphas"]:
                    for chi in cfg["chis"]:
                        seed = stable_seed(base_seed, gid, "homogeneous", alpha, chi)
                        if "heterogeneous" not in cfg and "community_correlated" not in cfg:
                            seed = stable_seed(base_seed, gid, alpha, chi)
                        events = dar_event_stream(
                            g, alpha=float(alpha), chi=float(chi), seed=seed, W=W,
                            event_rate=float(cfg.get("event_rate", 1.0)))
                        iid = f"dar__{substrate}__n{n}__f{fam_i}__a{alpha}__c{chi}"
                        params = {"alpha": float(alpha), "chi": float(chi),
                                  "event_rate": float(cfg.get("event_rate", 1.0)),
                                  "candidate_edges": g.number_of_edges(), "seed": seed}
                        if "heterogeneous" in cfg or "community_correlated" in cfg:
                            params = {"mode": "homogeneous", **params}
                        rows.append(_instance_row(
                            out, iid, events,
                            {"data_block": "mechanistic_dar", "source": "synthetic",
                             "domain": "mechanistic", "substrate": substrate,
                             "generator": "DAR(1)", "group_id": gid, "family": gid,
                             "rep": 0, "rho_target": np.nan,
                             "alpha": float(alpha), "chi": float(chi),
                             "generator_params_json": json.dumps(params)}, W=W))

                hcfg = cfg.get("heterogeneous", {})
                if hcfg.get("enabled", False):
                    for mean in hcfg.get("means", []):
                        for conc in hcfg.get("concentrations", []):
                            for chi in hcfg.get("chis", []):
                                seed = stable_seed(base_seed, gid, "heterogeneous", mean, conc, chi)
                                events = dar_event_stream(
                                    g, alpha=float(mean), chi=float(chi), seed=seed, W=W,
                                    event_rate=float(cfg.get("event_rate", 1.0)),
                                    alpha_concentration=float(conc))
                                iid = (f"darhet__{substrate}__n{n}__f{fam_i}"
                                       f"__am{mean}__ac{conc}__c{chi}")
                                rows.append(_instance_row(
                                    out, iid, events,
                                    {"data_block": "mechanistic_dar_heterogeneous",
                                     "source": "synthetic", "domain": "mechanistic",
                                     "substrate": substrate, "generator": "DAR(1)-heterogeneous",
                                     "group_id": gid, "family": gid, "rep": 0,
                                     "rho_target": np.nan, "alpha": float(mean), "chi": float(chi),
                                     "generator_params_json": json.dumps(
                                         {"alpha_mean": mean, "alpha_concentration": conc,
                                          "chi": chi, "seed": seed})}, W=W))

                ccfg = cfg.get("community_correlated", {})
                if ccfg.get("enabled", False):
                    for pair in ccfg.get("pairs", []):
                        aw, ab = map(float, pair)
                        for chi in ccfg.get("chis", []):
                            seed = stable_seed(base_seed, gid, "correlated", aw, ab, chi)
                            events = dar_event_stream(
                                g, alpha=(aw + ab) / 2.0, chi=float(chi), seed=seed, W=W,
                                event_rate=float(cfg.get("event_rate", 1.0)),
                                alpha_within=aw, alpha_between=ab)
                            iid = (f"darcorr__{substrate}__n{n}__f{fam_i}"
                                   f"__aw{aw}__ab{ab}__c{chi}")
                            rows.append(_instance_row(
                                out, iid, events,
                                {"data_block": "mechanistic_dar_correlated",
                                 "source": "synthetic", "domain": "mechanistic",
                                 "substrate": substrate, "generator": "DAR(1)-community",
                                 "group_id": gid, "family": gid, "rep": 0,
                                 "rho_target": np.nan, "alpha": (aw + ab) / 2.0,
                                 "chi": float(chi),
                                 "generator_params_json": json.dumps(
                                     {"alpha_within": aw, "alpha_between": ab,
                                      "chi": chi, "seed": seed})}, W=W))
    return rows


def _activity_instances(plan, out, base_seed, W):
    cfg = plan["activity_memory"]
    if not cfg.get("enabled", False):
        return []
    rows = []
    for n in cfg["sizes"]:
        for fam_i in range(int(cfg["families"])):
            gid = f"activity::n{n}::f{fam_i}"
            for mode in cfg["modes"]:
                beta = None if str(mode) == "memoryless" else float(mode)
                seed = stable_seed(base_seed, gid, mode)
                events = activity_memory_event_stream(
                    int(n), seed=seed, W=W,
                    slots_per_window=int(cfg.get("slots_per_window", 40)),
                    mean_activity=float(cfg.get("mean_activity", 0.04)),
                    memory_beta=beta)
                tag = "memoryless" if beta is None else f"beta{beta:g}"
                rows.append(_instance_row(
                    out, f"activity__n{n}__f{fam_i}__{tag}", events,
                    {"data_block": "mechanistic_activity", "source": "synthetic",
                     "domain": "mechanistic", "substrate": "activity_driven",
                     "generator": "activity_memory", "group_id": gid,
                     "family": gid, "rep": 0, "rho_target": np.nan,
                     "alpha": np.nan, "chi": np.nan,
                     "generator_params_json": json.dumps(
                         ({"memory_beta": beta, "seed": seed} if "renewal" not in plan else
                          {"memory_beta": beta, "mean_activity": cfg.get("mean_activity"),
                           "seed": seed}))}, W=W))
    return rows


def _renewal_instances(plan, out, base_seed, W):
    cfg = plan.get("renewal", {})
    if not cfg.get("enabled", False):
        return []
    rows = []
    for substrate in cfg.get("substrates", ["dcsbm"]):
        for n in cfg["sizes"]:
            for fam_i in range(int(cfg["families"])):
                gseed = stable_seed(base_seed, "renewal_graph", substrate, n, fam_i)
                g = _graph_for_substrate(substrate, n, gseed, cfg)
                gid = f"renewal::{substrate}::n{n}::f{fam_i}"
                for life in cfg["lifetime_means"]:
                    for shape in cfg["iet_shapes"]:
                        seed = stable_seed(base_seed, gid, life, shape)
                        events = renewal_event_stream(
                            g, lifetime_mean_windows=float(life), iet_shape=float(shape),
                            seed=seed, W=W,
                            events_per_active_window=float(cfg.get("events_per_active_window", 1.0)))
                        iid = (f"renewal__{substrate}__n{n}__f{fam_i}"
                               f"__life{life}__shape{shape}")
                        rows.append(_instance_row(
                            out, iid, events,
                            {"data_block": "mechanistic_renewal", "source": "synthetic",
                             "domain": "mechanistic", "substrate": substrate,
                             "generator": "edge_renewal", "group_id": gid,
                             "family": gid, "rep": 0, "rho_target": np.nan,
                             "alpha": np.nan, "chi": np.nan,
                             "generator_params_json": json.dumps(
                                 {"lifetime_mean_windows": life, "iet_shape": shape,
                                  "seed": seed})}, W=W))
    return rows


def _twin_instances(plan, out, base_seed, W):
    cfg = plan["synthetic_twins"]
    if not cfg.get("enabled", False):
        return []
    substrates = list(cfg["substrates"])
    legacy = plan.get("legacy_controls", {})
    if legacy.get("enabled", False):
        substrates.extend(legacy.get("substrates", []))
    layouts = cfg.get("span_layouts", ["legacy"])
    rows = []
    for mode in cfg.get("timestamp_modes", ["uniform"]):
        for layout in layouts:
            for substrate in substrates:
                block = "legacy_control" if substrate in {"er", "ba"} else "synthetic_controlled"
                for n in cfg["sizes"]:
                    for fam_i in range(int(cfg["families"])):
                        seed = stable_seed(base_seed, "twin_family", mode, substrate, n, fam_i)
                        old_v1 = "span_layouts" not in cfg
                        gid = (f"twin::{mode}::{substrate}::n{n}::f{fam_i}" if old_v1
                               else f"twin::{substrate}::n{n}::f{fam_i}")
                        fam = make_family(gid, substrate, int(n), seed, timestamps=str(mode))
                        for target in cfg["targets"]:
                            for rep in range(int(cfg.get("reps", 1))):
                                iseed = stable_seed(base_seed, gid, layout, target, rep)
                                if old_v1:
                                    iseed = stable_seed(base_seed, gid, target, rep)
                                inst = make_instance(fam, float(target), seed=iseed,
                                                     span_layout=str(layout))
                                iid = ((f"twin__{mode}__{substrate}__n{n}__f{fam_i}"
                                        f"__rho{float(target):.2f}__r{rep}") if old_v1 else
                                       (f"twin__{mode}__{layout}__{substrate}__n{n}__f{fam_i}"
                                        f"__rho{float(target):.2f}__r{rep}"))
                                rows.append(_instance_row(
                                    out, iid, inst.events,
                                    {"data_block": block, "source": "synthetic",
                                     "domain": "controlled", "substrate": substrate,
                                     "generator": "steered_P[w,t]", "group_id": gid,
                                     "family": gid, "rep": rep,
                                     "rho_target": float(target), "alpha": np.nan,
                                     "chi": np.nan,
                                     "generator_params_json": json.dumps(
                                         ({"timestamp_mode": mode, "seed": iseed,
                                           "deviations": inst.deviations} if old_v1 else
                                          {"timestamp_mode": mode, "span_layout": layout,
                                           "seed": iseed, "deviations": inst.deviations}))}, W=W))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/benchmark.yaml")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--out", help="default: data/benchmark_<preset>")
    ap.add_argument("--registry", default="config/datasets.yaml")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--allow-missing-real", action="store_true",
                    help="permit fewer real datasets than the configured minimum")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    full_cfg, plan = load_plan(args.config, args.preset)
    out = Path(args.out or f"data/benchmark_{args.preset}")
    if (out / "manifest.csv").exists() and not args.overwrite:
        raise FileExistsError(f"{out}/manifest.csv exists; use --overwrite or a new --out")
    out.mkdir(parents=True, exist_ok=True)
    W, base_seed = int(plan["W"]), int(plan["seed"])

    rows, missing = _real_instances(plan, out, args.registry, args.raw_dir, base_seed, W)
    present_real = len({r["source"] for r in rows if r["data_block"] == "real_empirical"})
    min_real = int(plan["real"].get("min_datasets", 0))
    if present_real < min_real and not args.allow_missing_real:
        details = "\n".join(f"  {k}: {fn} ({url})" for k, fn, url in missing)
        raise RuntimeError(
            f"only {present_real} real datasets found; preset requires {min_real}.\n"
            f"Place files in {args.raw_dir}, or inspect with --allow-missing-real:\n{details}")
    rows += _dar_instances(plan, out, base_seed, W)
    rows += _activity_instances(plan, out, base_seed, W)
    rows += _renewal_instances(plan, out, base_seed, W)
    rows += _twin_instances(plan, out, base_seed, W)
    manifest = pd.DataFrame(rows)
    if manifest.empty:
        raise RuntimeError("benchmark plan produced no instances")
    manifest.to_csv(out / "manifest.csv", index=False)
    with open(out / "effective_plan.yaml", "w") as fh:
        yaml.safe_dump({"preset": args.preset, "plan": plan,
                        "evaluation": full_cfg.get("evaluation", {})}, fh,
                       sort_keys=False)
    with open(out / "missing_real_datasets.tsv", "w") as fh:
        fh.write("key\tfilename\tpage_url\n")
        for item in missing:
            fh.write("\t".join(item) + "\n")
    print(f"\nwrote {out / 'manifest.csv'}: {len(manifest)} instances, "
          f"{manifest.group_id.nunique()} independent groups")
    print(manifest.groupby("data_block").size().to_string())


if __name__ == "__main__":
    main()
