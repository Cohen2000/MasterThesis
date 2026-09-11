#!/usr/bin/env python3
"""Select the LLM benchmark cases from the frozen v2 estimator screen.

Selection design (supervisor-aligned):
* three access models: clean historical reference, biased causal forward walk,
  realistic recent-history retrieval;
* fixed block shares with the real block first (real 8, real_controlled 7,
  DAR 6, synthetic twins 4, renewal 3 per strategy), every present real source
  guaranteed at least once per strategy;
* spread over coverage bands and the true rho spectrum inside each block;
* per case: exact input JSONs (already frozen in the case rows), full truth
  profile, and the out-of-fold baseline predictions from the v2 evaluation so
  the later leaderboard compares LLMs and classical methods on identical cases;
* flags for the input-ablation subset and the tool-use subset;
* three worked examples per strategy (low/mid/high rho) drawn from groups that
  are DISJOINT from every selected evaluation group (no leakage).

No instance is regenerated; truth and inputs come from the frozen case table.
"""

import argparse
import glob
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

META = ["case_id", "instance_id", "group_id", "data_block", "source",
        "strategy", "walk_seed", "budget", "coverage"]
TRUTH = ["rho_W5_k2", "rho_W5_k3", "rho_W5_k4", "rho_W5_k5",
         "C_one_step", "mean_span_frac", "rho_event_weighted",
         "lifetime_mean_over_T"]
JSONS = ["input__nw_exact_json", "input__nmask_exact_json",
         "input__recent_events_json"]
CRAWL = ["observed_walk_nodes", "observed_walk_edges", "observed_timed_edges",
         "crawl__restart_fraction", "crawl__edge_revisit_rate",
         "crawl__discovery_010", "crawl__discovery_050", "crawl__discovery_100",
         "crawl__dt_q25", "crawl__dt_q50", "crawl__dt_q75", "crawl__dt_q90",
         "crawl__observed_time_span", "crawl__first_node_collision_frac",
         "crawl__node_hits_q25", "crawl__node_hits_q50",
         "crawl__node_hits_q75", "crawl__node_hits_q90",
         "crawl__edge_hits_q25", "crawl__edge_hits_q50",
         "crawl__edge_hits_q75", "crawl__edge_hits_q90",
         "crawl__observed_degree_mean", "crawl__observed_degree_max"]
PAT_TEMPORAL = (["pat__adjacent_observed_C", "pat__noncontiguous_edge_share",
                 "pat__mean_mask_width",
                 "pat__lifetime_mean", "pat__lifetime_q25", "pat__lifetime_q50",
                 "pat__lifetime_q75", "pat__lifetime_q90",
                 "pat__iet_mean", "pat__iet_q25", "pat__iet_q50",
                 "pat__iet_q75", "pat__iet_q90"]
                + [f"pat__first_w{w}" for w in range(5)]
                + [f"pat__last_w{w}" for w in range(5)]
                + [f"pat__event_share_w{w}" for w in range(5)])
EST_READOUTS = ["est__plugin_mean_occupancy", "est__occ_mle_mean_occupancy",
                "est__mask_mle_mean_occupancy", "est__plugin_C_one_step",
                "est__mask_mle_C_one_step"]

BLOCK_GROUP = {
    "real_empirical": "real", "real_chunk": "real",
    "real_controlled": "real_controlled",
    "mechanistic_dar": "dar", "mechanistic_dar_heterogeneous": "dar",
    "mechanistic_dar_correlated": "dar",
    "synthetic_controlled": "synthetic_controlled",
    "mechanistic_renewal": "renewal",
}
BLOCK_QUOTA = [("real", 8), ("real_controlled", 7), ("dar", 6),
               ("synthetic_controlled", 4), ("renewal", 3)]
BANDS = pd.IntervalIndex.from_breaks([-1.0, 0.01, 0.05, 0.2, 1.01])
BAND_LABELS = ["very_low(<.01)", "low(.01-.05)", "mid(.05-.20)", "high(>.20)"]

BASELINES = {
    "pred__floor": ("mean_floor", None),
    "pred__occ_mle": ("occ_mle", None),
    "pred__mask_mle": ("mask_mle", None),
    "pred__beta_block": ("beta_mle_block_lofo", None),
    "pred__et_combined": ("extra_trees", "combined"),
    "pred__et_stacked": ("extra_trees", "combined_plus_estimators"),
}


def crc(*parts):
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0xFFFFFFFF


def coverage_band(cov):
    idx = BANDS.get_indexer(pd.Series(cov, dtype=float))
    return [BAND_LABELS[i] if i >= 0 else BAND_LABELS[-1] for i in idx]


def load_cases(spec):
    paths = sorted(glob.glob(spec))
    if not paths:
        raise FileNotFoundError(spec)
    cols = META + TRUTH + JSONS + CRAWL + PAT_TEMPORAL + EST_READOUTS
    frames = [pd.read_csv(p, usecols=lambda c: c in cols) for p in paths]
    d = pd.concat(frames, ignore_index=True)
    missing = [c for c in cols if c not in d.columns]
    if missing:
        raise ValueError(f"case table lacks columns: {missing}")
    return d


BUDGET_LEVELS = [100, 400, 1600, 3200]


def _greedy_pick(cand, picked_y, taken_ids):
    """One row: farthest from already-picked y values, unused instances first."""
    pool = cand[~cand["instance_id"].isin(taken_ids)]
    if pool.empty:
        pool = cand
    y = pool["rho_W5_k2"].to_numpy(float)
    if not picked_y:
        score = -np.abs(y - np.median(y))          # start near the median
    else:
        score = np.min(np.abs(y[:, None] - np.array(picked_y)[None, :]), axis=1)
    pool = pool.assign(_s=score).sort_values(["_s", "case_id"],
                                             ascending=[False, True])
    return pool.iloc[0].drop(labels="_s")


def select_block(cand, quota, rng, taken_ids, real_sources=None):
    """Fill one block quota.

    real  -> round-robin over sources (every source before any repeats),
             budget balanced across the picks;
    other -> round-robin over budget levels;
    inside a cell, greedy farthest-point spread over true rho_k2, preferring
    instances not selected elsewhere.  Deterministic given the frozen table.
    """
    picked, picked_y = [], []
    used = set()
    if real_sources:
        order = sorted(cand["source"].unique(),
                       key=lambda s: (s not in real_sources, s))
        axis = [order[i % len(order)] for i in range(quota)]
        col = "source"
    else:
        levels = [b for b in BUDGET_LEVELS if (cand.budget == b).any()]
        axis = [levels[i % len(levels)] for i in range(quota)]
        col = "budget"
    budget_count = {b: 0 for b in BUDGET_LEVELS}
    for want in axis:
        sub = cand[(cand[col] == want) & (~cand.case_id.isin(used))]
        if sub.empty:
            sub = cand[~cand.case_id.isin(used)]
        if sub.empty:
            break
        if col == "source":                        # balance budgets for real
            least = min((budget_count.get(int(b), 0), int(b))
                        for b in sub.budget.unique())[1]
            sub = sub[sub.budget == least]
        row = _greedy_pick(sub, picked_y, taken_ids)
        picked.append(row)
        picked_y.append(float(row["rho_W5_k2"]))
        used.add(row["case_id"])
        budget_count[int(row["budget"])] = budget_count.get(int(row["budget"]), 0) + 1
    return pd.DataFrame(picked)


def flag_subset(sel, per_strategy, colname, seed):
    sel[colname] = 0
    for strat, g in sel.groupby("strategy"):
        rng = np.random.default_rng(crc(seed, colname, strat))
        chosen = []
        for _, sub in g.groupby("block_group"):
            share = max(1, round(per_strategy * len(sub) / len(g)))
            sub = sub.sort_values(["coverage_band", "rho_W5_k2", "case_id"])
            idx = np.linspace(0, len(sub) - 1,
                              min(share, len(sub))).round().astype(int)
            chosen.extend(sub.iloc[sorted(set(idx))].case_id.tolist())
        rng.shuffle(chosen)
        sel.loc[sel.case_id.isin(chosen[:per_strategy]), colname] = 1
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases",
                    default="results/benchmark_v2/results/cases_shard_*.csv.gz")
    ap.add_argument("--predictions",
                    default="results/benchmark_v2/results/predictions.csv.gz")
    ap.add_argument("--out-dir", default="results/llm_v2")
    ap.add_argument("--strategies", default=
                    "time_agnostic_t,time_respecting,recent_history_k20")
    ap.add_argument("--n-per-strategy", type=int, default=28)
    ap.add_argument("--ablation-per-strategy", type=int, default=12)
    ap.add_argument("--tool-per-strategy", type=int, default=8)
    ap.add_argument("--walk-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=20260712)
    args = ap.parse_args()
    if args.n_per_strategy != sum(q for _, q in BLOCK_QUOTA):
        raise SystemExit("n-per-strategy must equal the block quota sum "
                         f"({sum(q for _, q in BLOCK_QUOTA)}) or adjust BLOCK_QUOTA")

    strategies = [s.strip() for s in args.strategies.split(",")]
    d = load_cases(args.cases)
    d = d[(d.strategy.isin(strategies)) & (d.walk_seed == args.walk_seed) &
          (d.budget.isin(BUDGET_LEVELS))].copy()
    d["block_group"] = d.data_block.map(BLOCK_GROUP)
    d = d.dropna(subset=["block_group"])
    d["coverage_band"] = coverage_band(d.coverage)
    real_sources = sorted(d.loc[d.block_group == "real", "source"].unique())

    picks, taken_ids = [], set()
    for strat in strategies:
        for block, quota in BLOCK_QUOTA:
            cand = d[(d.strategy == strat) & (d.block_group == block)]
            if cand.empty:
                print(f"[warn] no candidates for {strat}/{block}")
                continue
            rng = np.random.default_rng(crc(args.seed, strat, block))
            got = select_block(cand, quota, rng, taken_ids,
                               real_sources if block == "real" else None)
            taken_ids.update(got.instance_id.tolist())
            picks.append(got)
    sel = pd.concat(picks, ignore_index=True).drop_duplicates("case_id")

    # guarantee: every real source at least once per strategy
    for strat in strategies:
        have = set(sel[(sel.strategy == strat) &
                       (sel.block_group == "real")]["source"])
        for src in real_sources:
            if src in have:
                continue
            cand = d[(d.strategy == strat) & (d.source == src)]
            if len(cand):
                sel = pd.concat([sel, cand.sort_values("case_id").head(1)])
                print(f"[fix] added missing real source {src} for {strat}")
    sel = sel.drop_duplicates("case_id").reset_index(drop=True)

    sel = flag_subset(sel, args.ablation_per_strategy, "ablation_subset", args.seed)
    sel = flag_subset(sel, args.tool_per_strategy, "tool_use_subset", args.seed + 1)

    # baseline join (k2, out-of-fold group_kfold)
    P = pd.read_csv(args.predictions,
                    usecols=["case_id", "target", "model", "input",
                             "protocol", "prediction"])
    P = P[(P.target == "rho_W5_k2") & (P.protocol == "group_kfold")]
    for col, (model, inp) in BASELINES.items():
        q = P[P.model == model]
        if inp is not None:
            q = q[q.input == inp]
        sel = sel.merge(q[["case_id", "prediction"]].rename(
            columns={"prediction": col}), on="case_id", how="left")
        miss = sel[col].isna().mean()
        if miss > 0.01:
            raise RuntimeError(f"baseline {col}: {miss:.1%} cases without "
                               "out-of-fold prediction")

    # per-case baselines for the additional targets, straight from the frozen
    # readout columns (no rerun of walks or evaluation needed)
    sel["pred_occ__plugin"] = sel["est__plugin_mean_occupancy"]
    sel["pred_occ__occ_mle"] = sel["est__occ_mle_mean_occupancy"]
    sel["pred_occ__mask_mle"] = sel["est__mask_mle_mean_occupancy"]
    sel["pred_C__plugin"] = sel["est__plugin_C_one_step"]
    sel["pred_C__mask_mle"] = sel["est__mask_mle_C_one_step"]

    # lifetime plug-ins. pat__lifetime_mean is CONDITIONAL on >= 2 recorded
    # events per pair; the census truth lifetime_mean_over_T averages over ALL
    # active pairs (single-event pairs contribute 0). The zero-inclusive
    # plug-in matches the truth population:
    #   zero_mean = conditional_mean * share(observed pairs with n >= 2).
    def share_n_ge2(nw_json):
        h = json.loads(nw_json)
        tot = sum(h.values())
        n1 = sum(v for k, v in h.items() if k.split(",")[0] == "1")
        return (tot - n1) / tot if tot else float("nan")

    sel["obs_share_n_ge2"] = sel["input__nw_exact_json"].map(share_n_ge2)
    sel["pred_lt__plugin_cond"] = sel["pat__lifetime_mean"]
    zero = sel["pat__lifetime_mean"] * sel["obs_share_n_ge2"]
    zero = zero.where(~(sel["pat__lifetime_mean"].isna()
                        & (sel["obs_share_n_ge2"] == 0)), 0.0)
    sel["pred_lt__plugin_zero"] = zero

    # example pool: groups disjoint from every selected group, small budgets
    sel_groups = set(sel.group_id)
    ex_rows = []
    for strat in strategies:
        pool = d[(d.strategy == strat) & (~d.group_id.isin(sel_groups)) &
                 (d.budget.isin([100, 400]))]
        if pool.empty:
            raise RuntimeError(f"no leakage-free example pool for {strat}")
        y = sel[sel.strategy == strat]["rho_W5_k2"]
        for tag, target in [("low", y.quantile(0.10)),
                            ("mid", y.quantile(0.50)),
                            ("high", y.quantile(0.90))]:
            row = pool.iloc[(pool.rho_W5_k2 - target).abs().argsort()].iloc[0]
            r = row.copy(); r["example_tag"] = tag
            ex_rows.append(r)
            pool = pool[pool.case_id != row.case_id]
    ex = pd.DataFrame(ex_rows).reset_index(drop=True)
    assert not (set(ex.group_id) & sel_groups), "example/eval group overlap"

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / "llm_cases.csv", index=False)
    ex.to_csv(out / "llm_examples.csv", index=False)

    lines = [f"selected {len(sel)} cases "
             f"({args.n_per_strategy} target/strategy), seed={args.seed}",
             f"walk_seed={args.walk_seed}, strategies={strategies}", ""]
    for name, tab in [("strategy x block", ["strategy", "block_group"]),
                      ("strategy x band", ["strategy", "coverage_band"]),
                      ("strategy x budget", ["strategy", "budget"])]:
        lines += [name, sel.groupby(tab).size().unstack(fill_value=0)
                  .to_string(), ""]
    yq = sel.groupby("strategy")["rho_W5_k2"].describe()[
        ["min", "25%", "50%", "75%", "max"]].round(3)
    lines += ["rho_k2 spectrum of selected cases", yq.to_string(), ""]
    lines += ["restricted-set baseline MAE (k2, selected cases only)"]
    for col in BASELINES:
        ae = (sel[col] - sel.rho_W5_k2).abs()
        g = sel.assign(ae=ae).groupby("strategy")["ae"].mean().round(4)
        lines.append(f"{col:18s} " + "  ".join(
            f"{k}={v}" for k, v in g.items()))
    lines += ["", "restricted-set baseline MAE (mean occupancy, selected cases)"]
    for col in ["pred_occ__plugin", "pred_occ__occ_mle", "pred_occ__mask_mle"]:
        g = sel.assign(ae=(sel[col] - sel.mean_span_frac).abs()) \
               .groupby("strategy")["ae"].mean().round(4)
        lines.append(f"{col:22s} " + "  ".join(f"{k}={v}" for k, v in g.items()))
    fl = sel.assign(ae=(sel.mean_span_frac.mean() - sel.mean_span_frac).abs()) \
            .groupby("strategy")["ae"].mean().round(4)
    lines.append(f"{'occ floor (set mean)':22s} " + "  ".join(
        f"{k}={v}" for k, v in fl.items()))
    lines += ["", "restricted-set baseline MAE (lifetime/T, selected cases)"]
    for col in ["pred_lt__plugin_zero", "pred_lt__plugin_cond"]:
        g = sel.assign(ae=(sel[col] - sel.lifetime_mean_over_T).abs()) \
               .groupby("strategy")["ae"].mean().round(4)
        lines.append(f"{col:22s} " + "  ".join(
            f"{k}={v}" for k, v in g.items()))
    fl = sel.assign(ae=(sel.lifetime_mean_over_T.mean()
                        - sel.lifetime_mean_over_T).abs()) \
            .groupby("strategy")["ae"].mean().round(4)
    lines.append(f"{'lifetime floor':22s} " + "  ".join(
        f"{k}={v}" for k, v in fl.items()))
    lines += ["", f"ablation_subset: {int(sel.ablation_subset.sum())}, "
              f"tool_use_subset: {int(sel.tool_use_subset.sum())}",
              f"examples: {len(ex)} (groups disjoint from eval groups)"]
    (out / "selection_report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {out/'llm_cases.csv'}, {out/'llm_examples.csv'}, "
          f"{out/'selection_report.txt'}")


if __name__ == "__main__":
    main()
