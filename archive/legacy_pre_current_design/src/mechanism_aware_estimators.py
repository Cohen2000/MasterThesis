"""Mechanism-aware estimators for the two clean arms, from prompt content only.

The question this answers is whether "correctly corrected" is a fair target at
all: if the correction magnitude is not identifiable from what the prompt
contains, then asking a model for it is asking for something no estimator could
supply either.

The answer differs by channel, and the arm's own mechanism text is what decides
it.

`event_sample_then_full_history` -- pure selection. Events are examined in
uniformly random order and a pair enters the sample if one of *its* records is
reached, so a pair with n events has inclusion probability
1 - (1 - f)^n for the realized sampling fraction f. The prompt does not state f;
the budget was deliberately removed from every prompt by the leakage audit. But
in the small-f regime the probability is proportional to n, and in the Hajek
(ratio) form the constant f cancels:

    rho_k = sum_d (1/n_d) 1[K_d >= k] / sum_d (1/n_d)

so the estimator needs no external parameter. `n_d` is in the prompt, and under
this arm the full history is retrieved, so the observed window mask is the true
one.

**Corrected 2026-09-03: this is an approximation, not an identification.** The
cancellation needs `n*f << 1`. On the 32 primary cases the realized fraction has
median 0.270 and `max(n)*f > 1` in 29 of 32, so the regime the argument assumes
is the exception here rather than the rule. The estimator still removes 94% of
the bias, which is an empirical result about this panel; it is not a guarantee,
and the most active pairs are systematically under-corrected.

`time_agnostic_t` -- near-pure censoring. Its mechanism text states the
observation model outright: each traversal returns one timestamp drawn
uniformly at random from the pair's complete history, independently and with
replacement, and n is the number of traversals. That is exactly the occupancy
model `corrected_estimator.rho_mle` inverts, so the existing label-free
occupancy MLE *is* the mechanism-aware estimator for this arm, with no bias
parameter to supply. It is already in the frozen table as `est__occ_mle_rho_k*`.

`time_respecting` -- not identifiable label-free, and this is not a gap in the
present work. `src/bias_identifiability.py` and
`results/bias_identifiability/RESULTS.md` establish it: a bias-aware MLE with a
single forward-bias parameter has its profile likelihood maximised at beta = 0,
the wrong value, so maximum likelihood chooses the no-correction answer. The
strength has to come from outside the sample.

Specificity matters as much as accuracy here, so the Hajek weighting is also run
on `node_panel_full_history`, where it does not belong. It should make that arm
worse. It does.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
KS = (2, 3, 4, 5)


def parse_nw(blob: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(n, w, count) from the exact `"n,w" -> count` histogram."""
    hist = json.loads(blob)
    n, w, c = [], [], []
    for key, count in hist.items():
        a, b = key.split(",")
        n.append(float(a))
        w.append(float(b))
        c.append(float(count))
    return np.asarray(n), np.asarray(w), np.asarray(c)


def hajek_profile(blob: str) -> dict:
    """Inverse-activity Hajek estimator, one value per k.

    Exact only while f * n stays small; as a pair's inclusion probability
    approaches 1 the true weight 1/(1 - (1 - f)^n) stops being proportional to
    1/n and this under-corrects the most active pairs. Without f in the prompt
    that residual cannot be removed, so it is a stated limit, not a bug.
    """
    n, w, c = parse_nw(blob)
    weight = c / n
    denom = weight.sum()
    return {f"hajek_rho_k{k}": float((weight * (w >= k)).sum() / denom)
            for k in KS}


def plugin_profile(blob: str) -> dict:
    n, w, c = parse_nw(blob)
    return {f"plugin_rho_k{k}": float((c * (w >= k)).sum() / c.sum())
            for k in KS}


def evaluate(cases: pd.DataFrame) -> pd.DataFrame:
    truth = [f"rho_W5_k{k}" for k in KS]
    rows = []
    for _, case in cases.iterrows():
        row = {"case_id": case.case_id, "strategy": case.strategy,
               "group_id": case.group_id}
        row.update(plugin_profile(case.input__nw_exact_json))
        row.update(hajek_profile(case.input__nw_exact_json))
        for k in KS:
            row[f"true_rho_k{k}"] = float(case[f"rho_W5_k{k}"])
            row[f"occmle_rho_k{k}"] = float(case[f"est__occ_mle_rho_k{k}"])
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    """Signed rho_2 bias and ProfileMAE over k = 2..5, per arm per estimator."""
    rows = []
    for arm, part in sorted(frame.groupby("strategy")):
        for name in ("plugin", "hajek", "occmle"):
            errors = np.column_stack(
                [part[f"{name}_rho_k{k}"] - part[f"true_rho_k{k}"] for k in KS])
            rows.append({
                "strategy": arm, "estimator": name, "n": len(part),
                "rho2_bias": float(errors[:, 0].mean()),
                "rho2_mae": float(np.abs(errors[:, 0]).mean()),
                "profile_mae": float(np.abs(errors).mean()),
            })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(REPO / "results/final_run_g2/final_cases.csv.gz"))
    ap.add_argument("--seed-slot", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g3"))
    args = ap.parse_args()

    cols = (["case_id", "strategy", "group_id", "seed_slot",
             "input__nw_exact_json"]
            + [f"rho_W5_k{k}" for k in KS]
            + [f"est__occ_mle_rho_k{k}" for k in KS])
    cases = pd.read_csv(args.cases, usecols=cols)
    cases = cases[cases.seed_slot == args.seed_slot]

    frame = evaluate(cases)
    table = summarize(frame)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "mechanism_aware_by_case.csv", index=False)
    table.to_csv(out / "mechanism_aware_by_arm.csv", index=False)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
