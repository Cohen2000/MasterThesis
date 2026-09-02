#!/usr/bin/env python3
"""Phase 3 reference band: floor / uniform-MLE / beta-calibrated-MLE / oracle,
computed leave-one-family-out, on a fresh case pool spanning the three coverage
bands. This is the fixed yardstick that later LLM (hidden vs disclosed)
predictions get joined against on `case_id` -- it must exist BEFORE any LLM
call is made, and must not be re-tuned afterwards.

Design (matches HANDOFF.md sections 4.7 and 7):
  - walks: time_agnostic_t (clean control) and time_respecting (the opposed-channel
    forward-biased crawl). recency_biased is left out here to keep the case
    pool focused; add it back with --walks if needed.
  - budgets: 400 / 1600 / 3200, which is what naturally spreads coverage
    across the three bands together with the size sweep.
  - rho targets: drawn continuously per family in [0.02, 0.58] (not a shared
    grid), matching the cluster-grid philosophy in generator.py / handoff 4.4.
  - four estimators per case, ALL family-held-out (GroupKFold / leave-one-
    family-out by `family`, never trained on the family being scored):
      floor_lofo        mean of rho_true over the OTHER families in this band
      mle_uniform        occupancy MLE, beta=0 (corrected_estimator, no LOFO
                          needed: it is training-free)
      mle_transfer_beta  occupancy MLE at a FIXED beta taken from the prior
                          synthetic run (0.5 for time_agnostic_t, 2.0 for
                          time_respecting) -- NOT fit on this data at all,
                          the strongest label-free test of transfer
      mle_betacal_lofo   occupancy MLE with beta chosen by MAE-minimisation
                          on the other families in this band, applied to the
                          held-out family (LOFO, protocol A from the earlier
                          check)
      oracle_lofo        RandomForest on the (n,w) histogram, GroupKFold by
                          family

Run:
  python phase3_reference_band.py                    # ~3-5 min, ~70-100 instances
  python phase3_reference_band.py --quick             # smoke test, seconds

Requires: generator.py, walks.py, bias_identifiability.py, corrected_estimator.py
alongside (same src/ folder).
"""
import argparse
import os
import time
import warnings
import zlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")
import generator
import walks
from bias_identifiability import _dense, obs_nw, hist_nw, rho_at_beta, loglik_at_beta, _mae

W = 5
BETAS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
# beta values already validated (see chat history / oof_beta_calib.py output):
# LOFO-A picked exactly these values in 100% of held-out families on the prior grid.
CANON_BETA = {"time_agnostic_t": 0.5, "time_respecting": 2.0}


def band(c):
    return "lo(<.02)" if c < 0.02 else ("hi(>.15)" if c > 0.15 else "mid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1500,8000,20000")
    ap.add_argument("--substrates", default="ba,er")
    ap.add_argument("--families", type=int, default=2)
    ap.add_argument("--targets-per-family", type=int, default=6)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--budgets", default="400,1600,3200")
    ap.add_argument("--walks", default="time_agnostic_t,time_respecting")
    ap.add_argument("--out-cases", default="../results/phase3/reference_band_cases.csv")
    ap.add_argument("--out-summary", default="../results/phase3/reference_band_summary.csv")
    ap.add_argument("--quick", action="store_true", help="tiny grid, seconds")
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.families, args.reps = "1500,8000", 1, 1
        args.targets_per_family = 3

    sizes = [int(x) for x in args.sizes.split(",")]
    substrates = args.substrates.split(",")
    budgets = [int(x) for x in args.budgets.split(",")]
    walk_strats = args.walks.split(",")

    t0 = time.time()
    DT = {b: _dense(b) for b in BETAS}
    rc = [f"r{bb:.1f}" for bb in BETAS]

    records = []
    i = 0
    for kind in substrates:
        for n in sizes:
            for fr in range(args.families):
                fid = f"{kind}{n}f{fr}"
                fam = generator.make_family(fid, kind, n=n, seed=100 + i)
                i += 1
                rng_t = np.random.default_rng(zlib.crc32(f"{fid}|targets".encode()))
                targets = sorted(rng_t.uniform(0.02, 0.58, size=args.targets_per_family).tolist())
                for tgt in targets:
                    for rep in range(args.reps):
                        inst = generator.make_instance(fam, tgt, seed=500 + i * 7 + rep)
                        rt = inst.achieved["rho_headline"]
                        idx = walks.build_index(inst.events)
                        tot = len(idx.edge_times)
                        for strat in walk_strats:
                            log = walks.run_walk(idx, strat, max_budget=max(budgets),
                                                 seed=900 + i * 13 + rep)
                            for b in budgets:
                                na, wa = obs_nw(log, b)
                                if len(na) == 0:
                                    continue
                                cov = len(na) / tot
                                case_id = f"{fid}|t{tgt:.3f}|r{rep}|{strat}|b{b}"
                                rec = {"case_id": case_id, "family": fid, "substrate": kind,
                                       "n": n, "rho_target": round(tgt, 4), "rep": rep,
                                       "strategy": strat, "budget": b, "coverage": cov,
                                       "band": band(cov), "n_obs_edges": len(na),
                                       "rho_true": rt, "hist": hist_nw(na, wa)}
                                for bb in BETAS:
                                    rec[f"r{bb:.1f}"] = rho_at_beta(na, wa, DT[bb])
                                    rec[f"l{bb:.1f}"] = loglik_at_beta(na, wa, DT[bb])
                                records.append(rec)
                print(f"[{time.time()-t0:5.0f}s] done family {fid} "
                      f"({len(targets)} targets)", flush=True)

    d = pd.DataFrame(records)
    print(f"\nbuilt {len(d)} cases across {d['family'].nunique()} families "
          f"in {time.time()-t0:.0f}s")

    d["mle_uniform"] = d["r0.0"]
    d["mle_transfer_beta"] = np.nan
    d["mle_betacal_lofo"] = np.nan
    d["betacal_lofo_beta"] = np.nan
    d["floor_lofo"] = np.nan
    d["oracle_lofo"] = np.nan

    summary_rows = []
    for w in walk_strats:
        cb = CANON_BETA.get(w)
        if cb is not None:
            d.loc[d["strategy"] == w, "mle_transfer_beta"] = d.loc[d["strategy"] == w, f"r{cb:.1f}"]
        dw_idx = d.index[d["strategy"] == w]
        for bnd in ["lo(<.02)", "mid", "hi(>.15)"]:
            m = (d["strategy"] == w) & (d["band"] == bnd)
            e = d[m]
            if len(e) < 8 or e["family"].nunique() < 3:
                print(f"[skip] {w} {bnd}: only {len(e)} rows / "
                      f"{e['family'].nunique()} families (need >=3 families)")
                continue
            y = e["rho_true"].to_numpy()
            fam = e["family"].to_numpy()
            Rmat = e[rc].to_numpy(dtype=float)
            X = np.nan_to_num(np.array(list(e["hist"])))
            idxs = e.index.to_numpy()

            floor_p = np.full(len(e), np.nan)
            betacal_p = np.full(len(e), np.nan)
            betacal_b = np.full(len(e), np.nan)
            for f in np.unique(fam):
                tr_mask = fam != f
                te_mask = ~tr_mask
                if tr_mask.sum() < 4:
                    continue
                floor_p[te_mask] = y[tr_mask].mean()
                bidx = min(range(len(BETAS)),
                          key=lambda k: _mae(y[tr_mask], Rmat[tr_mask, k]))
                betacal_p[te_mask] = Rmat[te_mask, bidx]
                betacal_b[te_mask] = BETAS[bidx]

            oracle_p = np.full(len(e), np.nan)
            n_groups = len(np.unique(fam))
            gk = GroupKFold(min(4, n_groups))
            for tr_i, te_i in gk.split(X, y, fam):
                rf = RandomForestRegressor(150, random_state=0, n_jobs=-1)
                rf.fit(X[tr_i], y[tr_i])
                oracle_p[te_i] = rf.predict(X[te_i])

            d.loc[idxs, "floor_lofo"] = floor_p
            d.loc[idxs, "mle_betacal_lofo"] = betacal_p
            d.loc[idxs, "betacal_lofo_beta"] = betacal_b
            d.loc[idxs, "oracle_lofo"] = oracle_p

            uni = _mae(y, e["mle_uniform"])
            tr_ = _mae(y, e[f"r{cb:.1f}"]) if cb is not None else np.nan
            fl = _mae(y, floor_p)
            bc = _mae(y, betacal_p)
            orc = _mae(y, oracle_p)
            summary_rows.append({"walk": w, "band": bnd, "n": len(e),
                                 "n_families": e["family"].nunique(),
                                 "floor_lofo": fl, "mle_uniform": uni,
                                 "mle_transfer_beta": tr_,
                                 "mle_betacal_lofo": bc, "oracle_lofo": orc})

    summ = pd.DataFrame(summary_rows)
    print(f"\n{'walk':17s}{'band':10s}{'n':>5}{'floor':>8}{'uniform':>9}"
          f"{'transfer':>10}{'betacal':>9}{'oracle':>8}")
    print("-" * 78)
    for _, r in summ.iterrows():
        print(f"{r.walk:17s}{r.band:10s}{r.n:5.0f}{r.floor_lofo:8.3f}"
              f"{r.mle_uniform:9.3f}{r.mle_transfer_beta:10.3f}"
              f"{r.mle_betacal_lofo:9.3f}{r.oracle_lofo:8.3f}")

    os.makedirs(os.path.dirname(args.out_cases) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.out_summary) or ".", exist_ok=True)
    d.drop(columns=["hist"] + [f"l{bb:.1f}" for bb in BETAS]).to_csv(args.out_cases, index=False)
    summ.to_csv(args.out_summary, index=False)
    print(f"\nwrote {args.out_cases} ({len(d)} rows, {len(d.columns)-1} cols after dropping hist/loglik)")
    print(f"wrote {args.out_summary}")
    print(f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
