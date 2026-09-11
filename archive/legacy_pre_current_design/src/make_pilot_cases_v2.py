#!/usr/bin/env python3
"""Build the LLM pilot: select ~60 cases from the frozen reference band,
REGENERATE their (n,w) walk data deterministically, VERIFY each case against
the frozen CSV, and write the prompts for three conditions.

Why regenerate+verify instead of trusting seeds: the reference band dropped
the raw (n,w) tables, and time_respecting walks were shown NOT to reproduce
across environments (different numpy/libm builds diverge the forward
trajectory). On the SAME machine + venv that produced reference_band_cases.csv
the regeneration is bit-exact; this script proves that per case (n_obs_edges,
coverage, rho_true, uniform-MLE all re-derived and compared) and ABORTS if any
selected case fails. Run it on the machine that built the band, nowhere else.

Conditions written to prompts.jsonl (one record per case x condition):
  hidden           data + task, no information about the crawl mechanism
  disclosed        + faithful description of the forward-in-time walk
                     (mechanism only, no conclusions about its effect)
  disclosed_calib  + one labeled calibration example from a RESERVED family
                     (er8000f1, excluded from the 60 cases), same procedure.
                     This operationalises "externally supplied bias strength".

Outputs (results/phase3/):
  pilot_cases.csv          selection + per-case reference predictions (labels
                           stay LOCAL; this file is never sent to the cluster)
  prompts.jsonl            case_id, condition, prompt text, minimal metadata
  pilot_selection_report.txt

Usage:
  python make_pilot_cases.py \
      --cases ../results/phase3/reference_band_cases.csv \
      --out-dir ../results/phase3
"""
import argparse
import json
import os
import time
import warnings
import zlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import generator
import walks
from bias_identifiability import _dense, obs_nw, rho_at_beta

# ---- must mirror phase3_reference_band.py exactly (seed reconstruction) ----
SIZES = [1500, 8000, 20000]
SUBSTRATES = ["ba", "er"]
FAMILIES = 2
TARGETS_PER_FAMILY = 6
REPS = 1
BUDGETS = [400, 1600, 3200]
WALK_STRATS = ["time_agnostic_t", "time_respecting"]

RESERVED_FAMILY = "er8000f1"        # calibration examples come from here
N_PER_CELL = 10                     # cases per (strategy x band)
SELECT_SEED = 20260702
CALIB_BUDGET = 1600
CALIB_TARGET_RHO = 0.30             # pick reserved-family instance closest to this

N_BINS = [(1, "n = 1"), (2, "n = 2"), (3, "n = 3"), (4, "n = 4"), (5, "n = 5"),
          (6, "n = 6-10"), (11, "n >= 11")]


def hist_matrix(na, wa):
    """7x5 binned (n,w) count matrix, same binning as the oracle features."""
    M = np.zeros((7, 5), dtype=int)
    for n, w in zip(na, wa):
        bi = 0 if n == 1 else 1 if n == 2 else 2 if n == 3 else 3 if n == 4 \
            else 4 if n == 5 else 5 if n <= 10 else 6
        M[bi, w - 1] += 1
    return M


def hist_text(na, wa):
    M = hist_matrix(na, wa)
    lines = ["observations |  w=1   w=2   w=3   w=4   w=5"]
    for (lo, lab), row in zip(N_BINS, M):
        lines.append(f"{lab:12s} |" + "".join(f"{c:6d}" for c in row))
    lines.append(f"distinct edges observed: {len(na)}")
    return "\n".join(lines)


SETUP = """You are estimating a property of a temporal graph from partial crawl data.

Setup: a temporal graph was observed over a time horizon normalized to [0, 1], divided into W = 5 equal windows. A crawler with a limited budget explored the graph and recorded timestamped edge events. Only a subset of all edges was observed.

For every DISTINCT edge the crawler observed, two numbers were recorded: n = how many times that edge was observed, and w = in how many distinct windows (1 to 5) it was observed. The table below aggregates all observed edges into bins of n (rows) and exact w (columns); each cell is a count of edges."""

MECHANISMS = {
    "time_respecting": """The crawling procedure is known: it is a forward-in-time temporal walk. It starts at a uniformly random node at a uniformly random time tau in [0, 1]. From its current node at time tau it selects uniformly at random one of that node's edge events with timestamp t > tau, records that event, moves to the other endpoint of that edge, and sets tau = t. It never moves backward in time. If the current node has no events after tau, the walk restarts at a new random node and a new uniformly random time. Every recorded event and every restart costs one unit of the fixed budget.""",
    "time_agnostic_t": """The crawling procedure is known: it is a random walk on the static graph obtained by collapsing all events, so timestamps play no role in choosing the next step. From its current node it moves to a uniformly random neighbor. Each time it traverses an edge, it records one event of that edge, with the timestamp drawn uniformly at random from that edge's events. Every recorded event costs one unit of the fixed budget.""",
}

TASK = """Your task: estimate rho = the fraction of all edges of the underlying temporal graph that were truly active in at least 2 of the 5 windows. The crawler observed only part of the graph; account for the observation process as you see fit.

Think briefly (a few sentences at most), then give your final answer on the last line in exactly this format:
FINAL: <number between 0 and 1>"""


def build_prompt(condition, table_txt, strategy, calib=None):
    parts = [SETUP]
    if condition in ("disclosed", "disclosed_calib"):
        parts.append(MECHANISMS[strategy])
    if condition == "disclosed_calib":
        parts.append(
            "Calibration example: a different temporal graph of the same kind was "
            "crawled with exactly the same procedure and a similar budget. Its "
            "observed table was:\n\n" + calib["table"] +
            f"\n\nThe true rho of that graph was {calib['rho_true']:.3f}. "
            "Use this example to calibrate how the crawl distorts the table.")
    parts.append("Now the graph to estimate:\n\n" + table_txt)
    parts.append(TASK)
    return "\n\n".join(parts)


def select_cases(cases):
    """Stratified selection: N_PER_CELL per (strategy x band), spread over
    families and rho by taking evenly spaced rows of the sorted cell."""
    c = cases[~cases["case_id"].duplicated(keep=False)].copy()
    n_dup = len(cases) - len(c)
    c = c[c["family"] != RESERVED_FAMILY]
    rng = np.random.default_rng(SELECT_SEED)
    picks = []
    for strat in WALK_STRATS:
        for bnd in ["lo(<.02)", "mid", "hi(>.15)"]:
            cell = c[(c["strategy"] == strat) & (c["band"] == bnd)]
            cell = cell.sort_values(["family", "rho_true", "budget"]).reset_index(drop=True)
            k = min(N_PER_CELL, len(cell))
            if k == 0:
                continue
            pos = np.unique(np.round(np.linspace(0, len(cell) - 1, k)).astype(int))
            # tiny jitter so repeated rho values do not always pick the same rep
            take = cell.iloc[pos]
            picks.append(take)
    sel = pd.concat(picks, ignore_index=True)
    return sel, n_dup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="../results/phase3/reference_band_cases.csv")
    ap.add_argument("--out-dir", default="../results/phase3")
    args = ap.parse_args()

    t0 = time.time()
    cases = pd.read_csv(args.cases)
    for col in ["case_id", "family", "strategy", "band", "budget", "rho_true",
                "coverage", "n_obs_edges", "r0.0", "mle_uniform",
                "mle_transfer_beta", "mle_betacal_lofo", "floor_lofo", "oracle_lofo"]:
        assert col in cases.columns, f"column {col} missing in {args.cases}"

    sel, n_dup = select_cases(cases)
    print(f"selected {len(sel)} cases "
          f"({n_dup} rows excluded as duplicate case_ids, reserved family "
          f"{RESERVED_FAMILY} excluded)")
    print(sel.groupby(["strategy", "band"]).size().to_string(), "\n")

    # which (family, target-index, strategy, budget) do we need? map case_id back
    need = {}
    for _, r in sel.iterrows():
        need.setdefault(r["family"], set()).add(r["case_id"])
    need_calib_family = RESERVED_FAMILY

    DT0 = _dense(0.0)
    regen = {}          # case_id -> dict(na, wa, checks)
    calib_pool = []     # reserved-family candidates for the calibration example

    i = 0
    for kind in SUBSTRATES:
        for n in SIZES:
            for fr in range(FAMILIES):
                fid = f"{kind}{n}f{fr}"
                need_this = fid in need or fid == need_calib_family
                if need_this:
                    fam = generator.make_family(fid, kind, n=n, seed=100 + i)
                i += 1
                if not need_this:
                    continue
                rng_t = np.random.default_rng(zlib.crc32(f"{fid}|targets".encode()))
                targets = sorted(rng_t.uniform(0.02, 0.58,
                                               size=TARGETS_PER_FAMILY).tolist())
                for tgt in targets:
                    for rep in range(REPS):
                        cids_inst = [f"{fid}|t{tgt:.3f}|r{rep}|{s}|b{b}"
                                     for s in WALK_STRATS for b in BUDGETS]
                        wanted = [c for c in cids_inst
                                  if (fid in need and c in need[fid])]
                        calib_here = (fid == need_calib_family)
                        if not wanted and not calib_here:
                            continue
                        inst = generator.make_instance(fam, tgt, seed=500 + i * 7 + rep)
                        rt = inst.achieved["rho_headline"]
                        idx = walks.build_index(inst.events)
                        tot = len(idx.edge_times)
                        for strat in WALK_STRATS:
                            need_strat = any(f"|{strat}|" in c for c in wanted)
                            if not (need_strat or calib_here):
                                continue
                            log = walks.run_walk(idx, strat, max_budget=max(BUDGETS),
                                                 seed=900 + i * 13 + rep)
                            for b in BUDGETS:
                                cid = f"{fid}|t{tgt:.3f}|r{rep}|{strat}|b{b}"
                                if cid not in (need.get(fid) or set()) and not (
                                        calib_here and b == CALIB_BUDGET):
                                    continue
                                na, wa = obs_nw(log, b)
                                if len(na) == 0:
                                    continue
                                rec = {"na": na, "wa": wa, "rho_true": rt,
                                       "coverage": len(na) / tot,
                                       "n_obs_edges": len(na),
                                       "r00": rho_at_beta(na, wa, DT0)}
                                if cid in (need.get(fid) or set()):
                                    regen[cid] = rec
                                if calib_here and b == CALIB_BUDGET and rep == 0:
                                    calib_pool.append({"case_id": cid,
                                                       "strategy": strat, **rec})
                print(f"[{time.time()-t0:5.0f}s] regenerated family {fid}", flush=True)

    # ---- verification against the frozen CSV ----
    print("\n=== verification against frozen reference_band_cases.csv ===")
    failures = []
    sel = sel.set_index("case_id")
    for cid, rec in regen.items():
        row = sel.loc[cid]
        ok = (int(row["n_obs_edges"]) == rec["n_obs_edges"]
              and abs(row["coverage"] - rec["coverage"]) < 1e-9
              and abs(row["rho_true"] - rec["rho_true"]) < 1e-9
              and abs(row["r0.0"] - rec["r00"]) < 1e-6)
        if not ok:
            failures.append((cid,
                             f"n_obs {int(row['n_obs_edges'])} vs {rec['n_obs_edges']}, "
                             f"rho {row['rho_true']:.6f} vs {rec['rho_true']:.6f}, "
                             f"r0.0 {row['r0.0']:.6f} vs {rec['r00']:.6f}"))
    missing = [c for c in sel.index if c not in regen]
    print(f"verified {len(regen)-len(failures)}/{len(sel)} cases exactly; "
          f"{len(failures)} mismatches, {len(missing)} not regenerated")
    for cid, msg in failures[:10]:
        print(f"  MISMATCH {cid}: {msg}")
    if failures or missing:
        raise SystemExit(
            "ABORT: regeneration does not reproduce the frozen band on this "
            "machine. Do NOT run the pilot on these prompts. Did the venv / "
            "numpy version change since reference_band_cases.csv was created?")

    # verify calibration candidates against the frozen CSV too
    all_cases = pd.read_csv(args.cases).set_index("case_id")
    calib_by_strat = {}
    for c in calib_pool:
        row = all_cases.loc[c["case_id"]]
        assert abs(row["rho_true"] - c["rho_true"]) < 1e-9 and \
            int(row["n_obs_edges"]) == c["n_obs_edges"], \
            f"calibration case {c['case_id']} failed verification"
        cur = calib_by_strat.get(c["strategy"])
        if cur is None or abs(c["rho_true"] - CALIB_TARGET_RHO) < \
                abs(cur["rho_true"] - CALIB_TARGET_RHO):
            calib_by_strat[c["strategy"]] = c
    for s, c in calib_by_strat.items():
        print(f"calibration example [{s}]: {c['case_id']} "
              f"(rho_true={c['rho_true']:.3f}, edges={c['n_obs_edges']})")

    # ---- write prompts.jsonl (no labels!) and pilot_cases.csv (local) ----
    os.makedirs(args.out_dir, exist_ok=True)
    p_path = os.path.join(args.out_dir, "prompts.jsonl")
    n_written = 0
    with open(p_path, "w") as fh:
        for cid in sel.index:
            rec = regen[cid]
            row = sel.loc[cid]
            table_txt = hist_text(rec["na"], rec["wa"])
            cal = calib_by_strat[row["strategy"]]
            cal_arg = {"table": hist_text(cal["na"], cal["wa"]),
                       "rho_true": cal["rho_true"]}
            for cond in ["hidden", "disclosed", "disclosed_calib"]:
                prompt = build_prompt(cond, table_txt, row["strategy"],
                                      cal_arg if cond == "disclosed_calib" else None)
                fh.write(json.dumps({
                    "case_id": cid, "condition": cond,
                    "strategy": row["strategy"], "band": row["band"],
                    "budget": int(row["budget"]), "prompt": prompt}) + "\n")
                n_written += 1
    print(f"\nwrote {p_path} ({n_written} prompts = {len(sel)} cases x 3 conditions)")

    keep = ["family", "substrate", "n", "strategy", "band", "budget", "coverage",
            "n_obs_edges", "rho_true", "floor_lofo", "mle_uniform",
            "mle_transfer_beta", "mle_betacal_lofo", "oracle_lofo",
            "betacal_lofo_beta"]
    out_cases = sel[keep].reset_index()
    cpath = os.path.join(args.out_dir, "pilot_cases.csv")
    out_cases.to_csv(cpath, index=False)
    print(f"wrote {cpath} (KEEP LOCAL, contains labels)")

    rep_path = os.path.join(args.out_dir, "pilot_selection_report.txt")
    with open(rep_path, "w") as fh:
        fh.write(f"cases: {len(sel)}  prompts: {n_written}\n")
        fh.write(out_cases.groupby(['strategy', 'band']).size().to_string() + "\n")
        fh.write(f"calibration examples: "
                 f"{ {s: c['case_id'] for s, c in calib_by_strat.items()} }\n")
        fh.write("verification: all selected cases reproduced exactly.\n")
    print(f"wrote {rep_path}\ntotal {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
