"""Split the rho_2 bias of each arm into selection and censoring.

The naive plug-in differs from the truth for two reasons that are usually
discussed together and are separable here exactly.

    rho2_true  = mean of 1[K_true >= 2] over population dyads with K_true >= 1
    rho2_seen  = mean of 1[K_true >= 2] over the dyads the sample contains
    rho2_naive = mean of 1[K_obs  >= 2] over those same dyads

    selection = rho2_seen  - rho2_true      which dyads are in the sample
    censoring = rho2_naive - rho2_seen      how much of their history is seen
    total     = rho2_naive - rho2_true = selection + censoring = -delta_i

`rho2_seen` is `oracle__seen_label_rho_k2`, already in the frozen case table:
true labels on the sampled dyad set, computed from the full event stream. So
this is the numerator/denominator split in the precise sense -- selection moves
the set the ratio is taken over, censoring moves the labels within it -- and it
is additive to machine precision rather than approximately.

**The order is forced, not chosen.** A decomposition through two stages is in
general path-dependent, and the usual complaint applies: why sample-then-censor
rather than censor-then-sample. Here the alternative path does not exist. A
dyad outside the sample has no observed history at all, so "the population
under censored labels" is not a defined quantity, and there is exactly one
route from the truth to the plug-in. Nothing is being held fixed by choice.

Analytic throughout: ground truth and sample composition only, no model output.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BOOTSTRAP = 4000
SEED = 20260901

TRUE = "rho_W5_k2"
SEEN = "oracle__seen_label_rho_k2"
NAIVE = "est__plugin_rho_k2"


def decompose(cases: pd.DataFrame) -> pd.DataFrame:
    out = cases.copy()
    out["selection"] = out[SEEN] - out[TRUE]
    out["censoring"] = out[NAIVE] - out[SEEN]
    out["total_bias"] = out[NAIVE] - out[TRUE]
    residual = (out.selection + out.censoring - out.total_bias).abs().max()
    if residual > 1e-9:
        raise RuntimeError(f"decomposition is not additive: {residual:.3e}")
    return out


def _boot_mean(values: np.ndarray, groups: np.ndarray,
               n: int = BOOTSTRAP) -> tuple[float, float]:
    """Cluster bootstrap over graph groups, as everywhere else in this suite."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    index = {g: np.flatnonzero(groups == g) for g in unique}
    draws = np.empty(n)
    for d in range(n):
        pick = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        draws[d] = values[idx].mean()
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def by_arm(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-arm channel table.

    Two share columns, because neither alone is honest.

    `share_of_total` is the textbook one, channel over total bias. It is only
    interpretable when the two channels have the same sign; where they oppose
    it leaves the unit interval, and where the total is near zero it is
    numerically meaningless. Both happen on this panel, so it is reported and
    flagged rather than quietly used.

    `share_of_movement` is |channel| / (|selection| + |censoring|). It stays in
    [0, 1] whatever the signs do, and answers the question the figure needs:
    how much of the total movement each channel accounts for.
    """
    rows = []
    for arm, part in sorted(frame.groupby("strategy")):
        groups = part.group_id.to_numpy()
        entry = {"strategy": arm, "n": len(part)}
        for name in ("selection", "censoring", "total_bias"):
            values = part[name].to_numpy(float)
            lo, hi = _boot_mean(values, groups)
            entry[f"{name}_mean"] = float(values.mean())
            entry[f"{name}_lo"] = lo
            entry[f"{name}_hi"] = hi
            entry[f"{name}_median"] = float(np.median(values))
        sel, cens = entry["selection_mean"], entry["censoring_mean"]
        total = entry["total_bias_mean"]
        movement = abs(sel) + abs(cens)
        # "opposed" is the interesting case and must not be hidden behind a
        # boolean that also swallows the single-channel arms, where one
        # component is exactly zero by construction.
        if abs(sel) < 1e-12 or abs(cens) < 1e-12:
            entry["channels"] = "single"
        else:
            entry["channels"] = "same_sign" if sel * cens > 0 else "opposed"
        entry["share_of_total_selection"] = sel / total if abs(total) > 1e-3 else np.nan
        entry["share_of_total_censoring"] = cens / total if abs(total) > 1e-3 else np.nan
        entry["share_of_movement_selection"] = abs(sel) / movement if movement else np.nan
        entry["share_of_movement_censoring"] = abs(cens) / movement if movement else np.nan
        rows.append(entry)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(REPO / "results/final_run_g2/final_cases.csv.gz"))
    ap.add_argument("--seed-slot", type=int, default=0)
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g3"))
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    cases = cases[cases.seed_slot == args.seed_slot][
        ["case_id", "instance_id", "group_id", "strategy", "coverage",
         TRUE, SEEN, NAIVE]]
    missing = cases[SEEN].isna().sum()
    if missing:
        print(f"WARNING: {missing} cases have no oracle seen-label column")

    frame = decompose(cases.dropna(subset=[SEEN]))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "bias_channels_by_case.csv", index=False)

    table = by_arm(frame)
    table.to_csv(out / "bias_channels_by_arm.csv", index=False)

    show = table[["strategy", "n", "total_bias_mean", "selection_mean",
                  "selection_lo", "selection_hi", "censoring_mean",
                  "censoring_lo", "censoring_hi", "channels",
                  "share_of_movement_selection"]]
    print(show.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
