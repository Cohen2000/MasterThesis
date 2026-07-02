#!/usr/bin/env python3
"""Phase 2: is the forward-time sampling bias correctable label-free?

Extends the uniform-occupancy MLE (corrected_estimator.rho_mle) to a bias-aware
MLE with a single bias parameter beta: instead of assuming the n observations of
an edge fall UNIFORMLY on its k active windows, they fall with weights
proportional to exp(beta * rank), rank = 1..k (later active windows favoured).
beta = 0 reproduces the uniform occupancy MLE exactly.

Three ways to set beta, compared per walk and per coverage band against the
information-free floor and the supervised oracle ceiling (RandomForest on the
(n,w) histogram, GroupKFold by family):

  uniform     beta = 0                          (= corrected_estimator.rho_mle)
  beta-fit    beta chosen per instance by max profile log-likelihood  (LABEL-FREE)
  beta-calib  one beta per walk-band chosen to minimise MAE           (uses labels)

Headline result (see results/bias_identifiability/RESULTS.md):
  - uniform fails under forward bias at ALL coverages (MAE > floor).
  - beta-fit ALSO fails: the profile likelihood is maximised at beta=0 (the wrong
    value), so fitting beta from the biased sample does not help. The occupancy
    MLE is INCONSISTENT under forward sampling, not merely inefficient.
  - beta-calib (bias strength supplied externally) recovers rho: it reaches the
    oracle at mid/high coverage and captures most of the gap at low coverage.
  => The bias is correctable only when its strength comes from outside the sample
     (calibration on labels, or disclosure). This is what motivates Phase 3.

Run:  python bias_identifiability.py                # default grid (~1-2 min)
      python bias_identifiability.py --quick        # tiny grid (seconds)
Requires: census.py, generator.py, walks.py, corrected_estimator.py alongside.
"""
import argparse
import warnings
from itertools import combinations

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import census
import generator
import walks
import corrected_estimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold

W = 5
NCAP = 60


# ---- generalized occupancy: P(observe w distinct windows | n draws, k active, beta) ----
def _window_probs(k, beta):
    r = np.arange(1, k + 1)
    w = np.exp(beta * r)
    return w / w.sum()


def _P_w(n, k, beta):
    p = _window_probs(k, beta)
    out = np.zeros(k + 1)
    for w in range(1, k + 1):
        tot = 0.0
        for S in combinations(range(k), w):
            for r in range(0, w + 1):
                sign = (-1) ** (w - r)
                for R in combinations(S, r):
                    pR = p[list(R)].sum() if r > 0 else 0.0
                    tot += sign * (pR ** n if pR > 0 else 0.0)
        out[w] = tot
    return out


def _dense(beta):
    """Dense table B[n, w, k-1] = P(w | n, k, beta), for fast vectorized lookup."""
    B = np.zeros((NCAP, W + 1, W))
    for n in range(1, NCAP):
        for k in range(1, W + 1):
            pw = _P_w(n, k, beta)
            for w in range(1, k + 1):
                B[n, w, k - 1] = pw[w]
    return B


def _em(Lik, iters=150):
    pi = np.ones(W) / W
    for _ in range(iters):
        num = Lik * pi[None, :]
        den = num.sum(1, keepdims=True)
        den[den == 0] = 1.0
        pi = (num / den).mean(0)
        s = pi.sum()
        pi = pi / s if s > 0 else np.ones(W) / W
    return pi


def _lik(na, wa, dense_beta):
    na = np.asarray(na)
    wa = np.asarray(wa)
    cap = np.minimum(na, NCAP - 1)
    big = na >= NCAP
    L = dense_beta[cap, wa, :].copy()
    if big.any():
        L[big] = 0.0
        L[big, wa[big] - 1] = 1.0
    return L


def rho_at_beta(na, wa, dense_beta):
    return float(1 - _em(_lik(na, wa, dense_beta))[0])


def loglik_at_beta(na, wa, dense_beta):
    L = _lik(na, wa, dense_beta)
    pi = _em(L)
    return float(np.log(np.clip((L * pi[None, :]).sum(1), 1e-300, None)).sum())


# ---- observation model: extract per-edge (n, w) and the (n,w) histogram ----
_N_BINS = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 10), (11, 10 ** 9)]


def obs_nw(log, budget):
    Lg = log.iloc[:budget]
    s = Lg[Lg["kind"] == 1]
    tv = s["t"].to_numpy(float)
    if len(s) == 0 or not np.isfinite(tv).any():
        return np.array([]), np.array([])
    win = census.window_index(tv, 0.0, 1.0 / W, W)
    nw = {}
    for a, b, w in zip(s["u"], s["v"], win):
        d = nw.setdefault((int(a), int(b)), [0, set()])
        d[0] += 1
        d[1].add(int(w))
    return (np.array([d[0] for d in nw.values()]),
            np.array([len(d[1]) for d in nw.values()]))


def hist_nw(na, wa):
    M = np.zeros((7, W))
    for n, w in zip(na, wa):
        for bi, (lo, hi) in enumerate(_N_BINS):
            if lo <= n <= hi:
                M[bi, w - 1] += 1
                break
    return list(M.flatten()) + [len(na)]


def _mae(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    return float(np.abs(a[m] - b[m]).mean()) if m.sum() else np.nan


def _oracle(d):
    if d["fid"].nunique() < 2:
        return np.nan
    X = np.nan_to_num(np.array(list(d["hist"])))
    y = d["rho"].to_numpy()
    g = d["fid"].to_numpy()
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(min(4, d["fid"].nunique())).split(X, y, g):
        pred[te] = RandomForestRegressor(150, random_state=0, n_jobs=-1).fit(
            X[tr], y[tr]).predict(X[te])
    return _mae(pred, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="1500,15000")
    ap.add_argument("--substrates", default="ba,er")
    ap.add_argument("--families", type=int, default=2)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--targets", default="0.10,0.25,0.40,0.55")
    ap.add_argument("--budgets", default="200,800,3200")
    ap.add_argument("--betas", default="0.0,0.5,1.0,1.5,2.0,2.5,3.0")
    ap.add_argument("--out", default="../results/bias_identifiability/bias_band.csv")
    ap.add_argument("--quick", action="store_true", help="tiny grid for a smoke run")
    args = ap.parse_args()
    if args.quick:
        args.sizes, args.families, args.reps = "1500,8000", 1, 1
        args.targets, args.budgets = "0.15,0.45", "400,3200"

    betas = [float(x) for x in args.betas.split(",")]
    DT = {b: _dense(b) for b in betas}
    walk_strats = ["time_agnostic_t", "recency_biased", "time_respecting"]

    rows = {w: [] for w in walk_strats}
    i = 0
    for kind in args.substrates.split(","):
        for n in [int(x) for x in args.sizes.split(",")]:
            for fr in range(args.families):
                fid = f"{kind}{n}f{fr}"
                fam = generator.make_family(fid, kind, n=n, seed=100 + i)
                i += 1
                for tgt in [float(x) for x in args.targets.split(",")]:
                    for ir in range(args.reps):
                        inst = generator.make_instance(fam, tgt, seed=500 + i * 7 + ir)
                        rt = inst.achieved["rho_headline"]
                        idx = walks.build_index(inst.events)
                        tot = len(idx.edge_times)
                        for strat in walk_strats:
                            log = walks.run_walk(idx, strat, max_budget=max(
                                int(x) for x in args.budgets.split(",")),
                                seed=900 + i * 13 + ir)
                            for b in [int(x) for x in args.budgets.split(",")]:
                                na, wa = obs_nw(log, b)
                                if len(na) == 0:
                                    continue
                                rec = {"fid": fid, "rho": rt, "cov": len(na) / tot,
                                       "hist": hist_nw(na, wa)}
                                for bb in betas:
                                    rec[f"r{bb:.1f}"] = rho_at_beta(na, wa, DT[bb])
                                    rec[f"l{bb:.1f}"] = loglik_at_beta(na, wa, DT[bb])
                                rows[strat].append(rec)

    def band(c):
        return "lo(<.02)" if c < 0.02 else ("hi(>.15)" if c > 0.15 else "mid")

    out_rows = []
    print(f"{'walk':16s}{'band':10s}{'floor':>7}{'uniform':>9}"
          f"{'beta-fit':>10}{'beta-calib':>12}{'(cal b)':>8}{'oracle':>8}")
    print("-" * 80)
    for w in walk_strats:
        d = pd.DataFrame(rows[w])
        d["band"] = d["cov"].map(band)
        for bnd in ["lo(<.02)", "mid", "hi(>.15)"]:
            e = d[d["band"] == bnd]
            if len(e) < 4:
                continue
            y = e["rho"].to_numpy()
            fl = _mae(y, np.full_like(y, y.mean()))
            uni = _mae(y, e["r0.0"])
            llc = [f"l{bb:.1f}" for bb in betas]
            rc = [f"r{bb:.1f}" for bb in betas]
            amax = e[llc].to_numpy().argmax(1)
            bf = np.array([e.iloc[j][rc[amax[j]]] for j in range(len(e))])
            fit = _mae(y, bf)
            best = min((_mae(y, e[f"r{bb:.1f}"]), bb) for bb in betas)
            orc = _oracle(e)
            print(f"{w:16s}{bnd:10s}{fl:7.3f}{uni:9.3f}{fit:10.3f}"
                  f"{best[0]:12.3f}{best[1]:8.1f}{orc:8.3f}")
            out_rows.append({"walk": w, "band": bnd, "floor": fl, "uniform": uni,
                             "beta_fit": fit, "beta_calib": best[0],
                             "calib_beta": best[1], "oracle": orc, "n": len(e)})
    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print("claim: beta-fit ~ uniform (both fail under forward); beta-calib ~ oracle.")


if __name__ == "__main__":
    main()
