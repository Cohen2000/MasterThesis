"""Freeze (j): the time_agnostic_t / time_respecting twin contrast.

The two arms require nearly the same correction -- -0.280 and -0.301 -- from
very different channel compositions: 97% censoring against 35% censoring with an
opposing selection term. They are a matched pair in the required answer and a
contrast in the stated mechanism, which separates two explanations that the
main slope cannot.

  responds to the implied correction  ->  shifts nearly equally on both arms
  responds to text surface            ->  shifts differently, although the
                                          correct answer is nearly identical

Reported as the paired difference of `Delta_i` between the two arms on the same
instance, with a cluster bootstrap over graph groups, against the difference in
their required corrections as the reference. A model tracking the requirement
should land near that reference, not near zero: the requirements themselves
differ slightly, and pretending they do not would make a small real difference
look like a failure.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = 4000
SEED = 20260901
A, B = "time_agnostic_t", "time_respecting"


def _boot(values, groups, n=BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in uniq}
    draws = np.empty(n)
    for d in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        draws[d] = np.nanmean(values[idx])
    return float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))


def twins(paired: pd.DataFrame, seed_slot: int = 0) -> pd.DataFrame:
    """`paired` carries one row per (model, case) with delta_i and Delta_i.

    Restricted to one seed slot. The thinking prompt set contains the
    seed-replication subset, so eight of the graphs appear at four slots and
    the rest at one; pooling them would weight those eight four times as
    heavily in a contrast that is supposed to be per instance.
    """
    if "seed_slot" in paired:
        paired = paired[paired.seed_slot == seed_slot]
    rows = []
    for model, part in sorted(paired.groupby("model")):
        wide = part[part.strategy.isin((A, B))].pivot_table(
            index=["instance_id", "group_id"], columns="strategy",
            values=["Delta_i", "delta_i"])
        wide = wide.dropna()
        if len(wide) < 4:
            continue
        model_gap = (wide[("Delta_i", A)] - wide[("Delta_i", B)]).to_numpy(float)
        required_gap = (wide[("delta_i", A)] - wide[("delta_i", B)]).to_numpy(float)
        groups = wide.index.get_level_values("group_id").to_numpy()
        lo, hi = _boot(model_gap, groups)
        req_lo, req_hi = _boot(required_gap, groups)
        # How much of the arm gap the model reproduces. 1 = it tracks the
        # requirement, 0 = it treats the arms identically despite the texts.
        ratio = (float(np.mean(model_gap) / np.mean(required_gap))
                 if abs(np.mean(required_gap)) > 1e-6 else np.nan)
        rows.append({
            "model": model, "seed_slot": seed_slot, "instances": len(wide),
            "graph_groups": len(np.unique(groups)),
            "model_gap_mean": float(np.mean(model_gap)),
            "model_gap_lo": lo, "model_gap_hi": hi,
            "required_gap_mean": float(np.mean(required_gap)),
            "required_gap_lo": req_lo, "required_gap_hi": req_hi,
            "gap_ratio": ratio,
            "required_inside_model_ci": bool(lo <= np.mean(required_gap) <= hi),
            "zero_inside_model_ci": bool(lo <= 0 <= hi),
        })
    return pd.DataFrame(rows)


def coverage_confound(paired: pd.DataFrame, seed_slot: int = 0) -> pd.DataFrame:
    """Does a coverage rule of thumb explain the twin gap?

    The twins differ in dyad coverage (0.138 against 0.076) at nearly identical
    required correction, so "saw less, correct more" is a rival explanation
    that would produce a gap without any reading of the mechanism.

    It makes a directional prediction. `time_respecting` is the *lower*
    coverage arm, so the heuristic corrects more there, which is a **negative**
    model gap under this module's A - B convention. One number decides it: the
    sign of the observed gap.
    """
    part = paired[paired.seed_slot == seed_slot] if "seed_slot" in paired else paired
    rows = []
    for model, sub in sorted(part.groupby("model")):
        wide = sub[sub.strategy.isin((A, B))].pivot_table(
            index="instance_id", columns="strategy",
            values=["Delta_i", "coverage"])
        wide = wide.dropna()
        if wide.empty:
            continue
        gap = float((wide[("Delta_i", A)] - wide[("Delta_i", B)]).mean())
        cov_gap = float((wide[("coverage", A)] - wide[("coverage", B)]).mean())
        # The heuristic corrects more where coverage is lower, so it predicts
        # a gap with the opposite sign to the coverage gap.
        predicted = -np.sign(cov_gap)
        rows.append({
            "model": model, "instances": len(wide),
            "coverage_A": float(wide[("coverage", A)].mean()),
            "coverage_B": float(wide[("coverage", B)].mean()),
            "coverage_gap_A_minus_B": cov_gap,
            "model_gap_A_minus_B": gap,
            "heuristic_predicts_sign": float(predicted),
            "observed_sign": float(np.sign(gap)),
            "consistent_with_heuristic": bool(np.sign(gap) == predicted),
        })
    return pd.DataFrame(rows)


def coverage_association(paired: pd.DataFrame, seed_slot: int = 0
                         ) -> pd.DataFrame:
    """The same rival explanation across all five arms, as a supporting check.

    Arm-level: mean model correction against mean coverage over the five arms.
    Case-level: the within-arm correlation, pooled after centring each arm, so
    the arm means cannot manufacture it.
    """
    part = paired[paired.seed_slot == seed_slot] if "seed_slot" in paired else paired
    rows = []
    for model, sub in sorted(part.groupby("model")):
        arm = sub.groupby("strategy").agg(Delta_i=("Delta_i", "mean"),
                                          coverage=("coverage", "mean"))
        centred = sub.copy()
        for col in ("Delta_i", "coverage"):
            centred[col] = centred[col] - centred.groupby("strategy")[col].transform("mean")
        rows.append({
            "model": model, "arms": len(arm), "cases": len(sub),
            "arm_level_corr": float(arm.Delta_i.corr(arm.coverage))
            if len(arm) > 2 else np.nan,
            "within_arm_corr": float(centred.Delta_i.corr(centred.coverage)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired", required=True,
                    help="CSV with model, instance_id, group_id, strategy, "
                         "delta_i, Delta_i")
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paired = pd.read_csv(args.paired)
    table = twins(paired)
    table.to_csv(out / "twin_arms.csv", index=False)
    print(table.round(4).to_string(index=False))

    confound = coverage_confound(paired)
    confound.to_csv(out / "twin_coverage_confound.csv", index=False)
    print("\n== coverage rival explanation, directional test\n",
          confound.round(4).to_string(index=False))

    assoc = coverage_association(paired)
    assoc.to_csv(out / "coverage_association.csv", index=False)
    print("\n== correction against coverage, all five arms\n",
          assoc.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
