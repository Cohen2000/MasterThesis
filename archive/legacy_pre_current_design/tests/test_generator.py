#!/usr/bin/env python3
"""Tests for the twin generator. Run: python test_generator.py"""

import numpy as np
import pandas as pd

from generator import make_family, make_instance

PASS = 0

def check(name, cond):
    global PASS
    print(("PASS  " if cond else "FAIL  ") + name)
    assert cond, name
    PASS += 1


fam = make_family("fam_ba_0", "ba", n=300, seed=7)

# --- 1) twin invariants: same family, two different assignments -------------
a = make_instance(fam, rho_target=0.10, seed=1)
b = make_instance(fam, rho_target=0.55, seed=2)

def pair_counts(df):
    return df.groupby(["u", "v"]).size().sort_index()

check("identical collapsed edge set",
      pair_counts(a.events).index.equals(pair_counts(b.events).index))
check("identical per-edge event counts",
      (pair_counts(a.events).to_numpy() == pair_counts(b.events).to_numpy()).all())
check("identical timestamp multiset",
      np.allclose(np.sort(a.events["t"].to_numpy()),
                  np.sort(b.events["t"].to_numpy())))
check("assignments actually differ",
      not a.events.sort_values(["u", "v", "t"]).reset_index(drop=True)
      .equals(b.events.sort_values(["u", "v", "t"]).reset_index(drop=True)))
check("event total equals family multiset size",
      len(a.events) == len(fam.t_all))

# --- 2) rho control ----------------------------------------------------------
targets = [0.05, 0.15, 0.25, 0.40, 0.55]
achieved = [make_instance(fam, r, seed=10 + i).achieved["rho_headline"]
            for i, r in enumerate(targets)]
print("   targets :", targets)
print("   achieved:", [round(x, 3) for x in achieved])
check("achieved rho close to targets (|dev| <= 0.05)",
      all(abs(t - a_) <= 0.05 for t, a_ in zip(targets, achieved)))
check("achieved rho strictly increasing across targets",
      all(x < y for x, y in zip(achieved, achieved[1:])))

# --- 3) determinism ----------------------------------------------------------
c1 = make_instance(fam, 0.25, seed=42)
c2 = make_instance(fam, 0.25, seed=42)
check("same seed gives identical instance", c1.events.equals(c2.events))

# --- 4) bursty edges really have span 1 at rho ~ 0 ---------------------------
low = make_instance(fam, 0.0, seed=5)
check("rho_target 0 yields near-zero achieved rho",
      low.achieved["rho_headline"] <= 0.02)

# --- 5) hub bias shifts persistence toward high-degree edges -----------------
hb = make_instance(fam, 0.30, seed=8, hub_bias=True)
rnd = make_instance(fam, 0.30, seed=8, hub_bias=False)

def mean_degprod_of_persistent(inst):
    df = inst.events.copy()
    win = fam.T / fam.W
    df["w"] = np.minimum((df["t"] / win).astype(int), fam.W - 1)
    span = df.drop_duplicates(["u", "v", "w"]).groupby(["u", "v"]).size()
    pers = span[span >= 2].index
    degs = {i: d for i, d in enumerate(fam.deg)}
    return np.mean([degs[u] * degs[v] for u, v in pers])

check("hub bias raises mean degree product of persistent edges",
      mean_degprod_of_persistent(hb) > mean_degprod_of_persistent(rnd))

# --- 6) achieved GT comes from the census code path --------------------------
check("achieved dict carries full profile",
      all(f"rho_W5_k{k}" in a.achieved for k in range(1, 6)))
check("profile monotone in k",
      all(a.achieved[f"rho_W5_k{k}"] >= a.achieved[f"rho_W5_k{k+1}"]
          for k in range(1, 5)))

print(f"\nAll {PASS} generator checks passed.")

# --- 7) bursty mode: twin invariants hold, per-pair burstiness reaches census range
famb = make_family("fam_bursty", "ba", n=300, seed=7, timestamps="bursty")
x1 = make_instance(famb, 0.25, seed=1)
x2 = make_instance(famb, 0.55, seed=2)
check("bursty: identical timestamp multiset across twins",
      np.allclose(np.sort(x1.events["t"].to_numpy()), np.sort(x2.events["t"].to_numpy())))
check("bursty: identical per-edge event counts",
      (pair_counts(x1.events).to_numpy() == pair_counts(x2.events).to_numpy()).all())
b_burst = x1.achieved["burstiness_pooled"]
b_unif = c1.achieved["burstiness_pooled"]
print(f"   burstiness pooled: uniform {b_unif:.3f}  vs  bursty {b_burst:.3f}")
check("bursty mode raises per-pair burstiness clearly", b_burst > b_unif + 0.15)
check("bursty burstiness lands in census range (0.3 .. 0.9)", 0.3 <= b_burst <= 0.9)
check("bursty: rho control still works", abs(x1.achieved["rho_headline"] - 0.25) <= 0.05)
