#!/usr/bin/env python3
"""Tests for walks.py. Run: python test_walks.py"""

import numpy as np
import pandas as pd

from generator import make_family, make_instance
from walks import build_index, run_walk, summarize

PASS = 0

def check(name, cond):
    global PASS
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    PASS += 1


# --- 1) hand-built micro graph ----------------------------------------------
# nodes 0,1,2; events: (0,1,t=0.1), (1,2,t=0.3), (0,1,t=0.7)   T=1, W=5
ev = pd.DataFrame({"u": [0, 1, 0], "v": [1, 2, 1], "t": [0.1, 0.3, 0.7]})
idx = build_index(ev, T=1.0, W=5)

check("index: node 1 has 3 incident events", len(idx.nbr_times[1]) == 3)
check("index: collapsed degree of node 1 is 2", idx.coll_deg[1] == 2)

# time-respecting walker placed manually: from (node 2, tau=0.4) the only
# future incident event of node 2 is none (its single event is at 0.3)
times2 = idx.nbr_times[2]
check("hand: node 2 has no future event after tau=0.4",
      np.searchsorted(times2, 0.4, side="right") >= len(times2))

# from (node 0, tau=0.0): future events at node 0 are t=0.1 and t=0.7
times0 = idx.nbr_times[0]
j0 = np.searchsorted(times0, 0.0, side="right")
check("hand: node 0 sees exactly two future events at tau=0",
      len(times0) - j0 == 2)

# --- 2) log mechanics on the micro graph -------------------------------------
log = run_walk(idx, "time_respecting", max_budget=50, seed=3)
check("budget: log length equals max_budget", len(log) == 50)
check("budget: first entry is a placement", log.iloc[0]["kind"] == 0)

# within each segment (between restarts) time must strictly increase
ok = True
last_t = -np.inf
for _, row in log.iterrows():
    if row["kind"] == 0:
        last_t = -np.inf
    else:
        ok &= row["t"] > last_t
        last_t = row["t"]
check("time-respecting: t strictly increases within segments", ok)

# every observed event must exist in the instance
ev_set = set(zip(ev["u"], ev["v"], ev["t"]))
steps = log[log["kind"] == 1]
ok = all((int(r["u"]), int(r["v"]), float(r["t"])) in ev_set
         for _, r in steps.iterrows())
check("every observed event exists in the data", ok)

# micro graph forces frequent dead ends -> restarts must occur
check("dead ends trigger restarts", (log["kind"] == 0).sum() > 1)

# --- 3) summarize on a crafted log (hand-computed features) ------------------
craft = pd.DataFrame({
    "kind": [0, 1, 1, 1, 1],
    "node": [0, 1, 0, 1, 2],
    "u":    [-1, 0, 0, 0, 1],
    "v":    [-1, 1, 1, 1, 2],
    "t":    [np.nan, 0.05, 0.45, 0.75, 0.85],   # windows 0, 2, 3, 4
    "dt":   [np.nan, 0.05, 0.40, 0.30, 0.10],
})
f = summarize(craft, idx, budget=5)
check("crafted: 4 observed events", f["n_observed_events"] == 4)
check("crafted: 2 unique edges", f["unique_edges"] == 2)
# edge (0,1) seen in windows {0,2,3}; edge (1,2) in {4} -> plugin = 1/2
check("crafted: walk_rho_plugin = 0.5", abs(f["walk_rho_plugin"] - 0.5) < 1e-12)
check("crafted: mean windows per edge = 2.0",
      abs(f["mean_windows_per_observed_edge"] - 2.0) < 1e-12)
check("crafted: window gap of revisited edge = 3",
      abs(f["mean_window_gap_revisits"] - 3.0) < 1e-12)
check("crafted: windows covered = 4/5",
      abs(f["windows_covered_frac"] - 0.8) < 1e-12)

# --- 4) twin blindness of the negative control -------------------------------
fam = make_family("fam_t", "ba", n=250, seed=11)
lo = make_instance(fam, 0.05, seed=1)
hi = make_instance(fam, 0.55, seed=2)
ilo = build_index(lo.events, T=fam.T, W=fam.W)
ihi = build_index(hi.events, T=fam.T, W=fam.W)

la = run_walk(ilo, "time_agnostic", 800, seed=99)
lb = run_walk(ihi, "time_agnostic", 800, seed=99)
check("time_agnostic: identical log across twins (same seed)",
      la.drop(columns=["t", "dt"]).equals(lb.drop(columns=["t", "dt"]))
      and la["t"].isna().all() and lb["t"].isna().all())

sa = summarize(la, ilo, 800)
check("time_agnostic: no temporal features in summary",
      "walk_rho_plugin" not in sa and "step_dt_mean" not in sa)

# --- 5) signal smoke test: time-respecting separates the twins ---------------
def mean_plugin(inst, index, seeds, budget=1500):
    vals = []
    for s in seeds:
        lg = run_walk(index, "time_respecting", budget, seed=s)
        vals.append(summarize(lg, index, budget).get("walk_rho_plugin", np.nan))
    return float(np.nanmean(vals))

p_lo = mean_plugin(lo, ilo, seeds=[1, 2, 3])
p_hi = mean_plugin(hi, ihi, seeds=[1, 2, 3])
print(f"   walk_rho_plugin: low-rho instance {p_lo:.3f}  vs  high-rho {p_hi:.3f}")
check("time-respecting walks separate low vs high rho twins",
      p_hi > p_lo + 0.10)

# time_agnostic_t baseline records times but identical transitions
lt = run_walk(ilo, "time_agnostic_t", 800, seed=99)
check("time_agnostic_t: same transitions as pure control, but with timestamps",
      lt[["kind", "node", "u", "v"]].equals(la[["kind", "node", "u", "v"]])
      and np.isfinite(lt.loc[lt["kind"] == 1, "t"]).all())

print(f"\nAll {PASS} walk checks passed.")

# --- 6) new bias-correction features on the crafted log -----------------------
f2 = summarize(craft, idx, budget=5)
# edge (0,1): 3 observations, windows {0,2,3}; edge (1,2): 1 observation
check("crafted: share of multi-observed edges = 0.5",
      abs(f2["share_edges_multi_observed"] - 0.5) < 1e-12)
check("crafted: conditional plugin = 1.0",
      abs(f2["walk_rho_conditional"] - 1.0) < 1e-12)
