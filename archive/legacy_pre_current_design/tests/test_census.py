#!/usr/bin/env python3
"""Ground-truth tests for census.py.

This is the foundational test: test_generator.py and test_walks.py both call
census helpers (spans / census_row) to compute the ground-truth rho they check
against, so a silent bug in the census would pass *their* checks undetected.
Here we pin the census against a hand-computed micro-graph where every number
was worked out by hand on paper, independently of the implementation.

Micro-graph (5 pairs, 9 events), already normalized so t_min = 0.0, t_max = 1.0
=> T = 1.0, W = 5, window width = 0.2, windows
[0,.2) [.2,.4) [.4,.6) [.6,.8) [.8,1.0]:

  pair A (0-1): t = 0.00, 0.25, 0.85   -> windows {0,1,4}   3 events
  pair B (1-2): t = 0.10, 0.12         -> windows {0}       2 events
  pair C (2-3): t = 0.65, 1.00         -> windows {3,4}     2 events
  pair D (0-3): t = 0.50               -> windows {2}       1 event
  pair E (1-3): t = 0.95               -> windows {4}       1 event

Active-window counts a_e: A=3, B=1, C=2, D=1, E=1.
"""

import math

import numpy as np
import pandas as pd

import census


SEP = "\x1f"


def approx(a, b, tol=1e-9):
    if b is None or (isinstance(b, float) and math.isnan(b)):
        return a is None or (isinstance(a, float) and math.isnan(a))
    return abs(float(a) - float(b)) <= tol


def build():
    """Raw (u, v, t) events -> census.normalize -> census_row, full path."""
    rows = [
        ("0", "1", 0.00), ("0", "1", 0.25), ("0", "1", 0.85),  # A
        ("1", "2", 0.10), ("1", "2", 0.12),                    # B
        ("2", "3", 0.65), ("2", "3", 1.00),                    # C
        ("0", "3", 0.50),                                      # D
        ("1", "3", 0.95),                                      # E
    ]
    raw = pd.DataFrame(rows, columns=["u", "v", "t"])
    raw["t"] = raw["t"].astype(np.float64)
    df = census.normalize(raw)
    return df, census.census_row(df, label="micro")


def main():
    df, row = build()
    checks = []

    def check(name, got, want, tol=1e-9):
        ok = approx(got, want, tol)
        checks.append(ok)
        flag = "PASS" if ok else "FAIL"
        print(f"{flag}  {name}: got {got!r}  want {want!r}")

    # --- normalization sanity: 5 distinct pairs, 9 events, undirected ---------
    check("normalize: n distinct pairs", df["pair"].nunique(), 5)
    check("normalize: n events kept", len(df), 9)

    # --- structural scalars ---------------------------------------------------
    check("n_pairs", row["n_pairs"], 5)
    check("n_events", row["n_events"], 9)
    check("horizon_T", row["horizon_T"], 1.0)
    check("events_per_pair_median", row["events_per_pair_median"], 2.0)
    # single-event pairs: D and E -> 2 / 5
    check("share_single_event_pairs", row["share_single_event_pairs"], 0.4)

    # --- headline persistence profile (W=5) -----------------------------------
    # a_e = [3,1,2,1,1]; rho_k = share with a_e >= k
    check("rho_W5_k1", row["rho_W5_k1"], 1.0)        # all pairs active >=1 window
    check("rho_W5_k2", row["rho_W5_k2"], 0.4)        # A, C
    check("rho_W5_k3", row["rho_W5_k3"], 0.2)        # A only
    check("rho_W5_k4", row["rho_W5_k4"], 0.0)
    check("rho_W5_k5", row["rho_W5_k5"], 0.0)
    check("rho_headline (==rho_W5_k2)", row["rho_headline"], 0.4)

    # mean normalized span: mean(a_e)/W = (3+1+2+1+1)/5 / 5 = 1.6/5
    check("mean_span_frac", row["mean_span_frac"], 0.32)

    # event-weighted rho: events on persistent pairs (A:3 + C:2) / total 9
    check("rho_event_weighted", row["rho_event_weighted"], 5.0 / 9.0)

    # one-step pair persistence C:
    # active (pair,w) with w<=3: (A,0)(A,1)(B,0)(C,3)(D,2) = 5 denominators;
    # hits where (pair,w+1) active: (A,0)->(A,1) yes, (C,3)->(C,4) yes => 2/5
    check("C_one_step", row["C_one_step"], 0.4)

    # half-window shifted grid (W+1=6 bins, offset 0.1): a_s = [3,1,2,1,1]
    # (A{0,1,4}, B{1}, C{3,5}, D{3}, E{5}); >=2 -> A,C -> 2/5
    check("rho_shifted", row["rho_shifted"], 0.4)

    # equal-event windows (9 events -> bins by quantile): B's two events split
    # into different event-windows, so a_ee = [3,2,2,1,1] -> >=2: A,B,C -> 3/5
    check("rho_equal_event", row["rho_equal_event"], 0.6)

    # censoring: pairs first seen in final window w4 -> only E -> 1/5
    check("censoring_share_lastwin", row["censoring_share_lastwin"], 0.2)

    # --- timing scalars -------------------------------------------------------
    # pooled inter-event gaps: A[0.25,0.60], B[0.02], C[0.35] -> [0.25,.6,.02,.35]
    check("median_iet", row["median_iet"], 0.30, tol=1e-9)
    # burstiness_pooled = (sigma - mu)/(sigma + mu), mu=0.305, sigma~0.208147
    mu = np.mean([0.25, 0.60, 0.02, 0.35])
    sg = np.std([0.25, 0.60, 0.02, 0.35])
    check("burstiness_pooled", row["burstiness_pooled"], (sg - mu) / (sg + mu), tol=1e-9)
    # per-pair burstiness median: only A has >=2 gaps -> single value
    muA, sgA = np.mean([0.25, 0.60]), np.std([0.25, 0.60])
    check("burstiness_pair_median", row["burstiness_pair_median"],
          (sgA - muA) / (sgA + muA), tol=1e-9)

    # lifetimes (first->last): A=0.85,B=0.02,C=0.35,D=0,E=0
    check("lifetime_mean", row["lifetime_mean"], (0.85 + 0.02 + 0.35) / 5.0, tol=1e-9)
    check("lifetime_mean_over_T", row["lifetime_mean_over_T"], (0.85 + 0.02 + 0.35) / 5.0, tol=1e-9)
    check("lifetime_median", row["lifetime_median"], 0.02, tol=1e-9)

    # --- boundary-robustness case: events exactly on window boundaries --------
    # With the WINDOW_EPS guard in window_index, events sitting exactly on the
    # boundaries 0.2/0.4/0.6/0.8 must land in the correct (upper) window despite
    # IEEE float error, so the full pair reaches all 5 windows. This is the
    # case that previously exposed the float bug; it is now asserted directly
    # rather than side-stepped.
    rows2 = [("a", "b", x) for x in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]] + [("c", "d", 0.5)]
    raw2 = pd.DataFrame(rows2, columns=["u", "v", "t"])
    raw2["t"] = raw2["t"].astype(np.float64)
    r2 = census.census_row(census.normalize(raw2), label="micro2")
    check("boundary: rho_W5_k2 (one full, one single)", r2["rho_W5_k2"], 0.5)
    check("boundary: full pair reaches k=5 (events on grid lines)", r2["rho_W5_k5"], 0.5)
    check("boundary: censoring (none first in last win)", r2["censoring_share_lastwin"], 0.0)

    print()
    if all(checks):
        print(f"All {len(checks)} census checks passed.")
    else:
        print(f"{sum(checks)}/{len(checks)} census checks passed -- FAILURES ABOVE.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
