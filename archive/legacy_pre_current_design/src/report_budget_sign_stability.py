"""Does the net bias sign on the opposed arms survive a budget change?

On `recent_history_k20` and `time_respecting` the two channels disagree --
selection positive, censoring negative and larger -- so the required correction
direction is a difference, not a mechanism. Freeze (j) reports those arms as an
extension on exactly that ground. If a moderate budget change flipped the net
sign, "the required direction" would not be a stable property of the arm and
that belongs in the text as a limitation.

Method. The frozen walk logs were not retained, but `walk_rng_seed` is in the
case table and `src/walks.py` builds a sweep from prefixes of one log: replaying
a seed reproduces the identical walk, and every shorter budget is a prefix of
it. Nothing frozen is touched -- this reads the panel's event files and writes
to `results_summary/`.

**The replay is verified before it is used.** The 800-step reproduction must
match the frozen case row's plug-in and oracle values to 1e-9, otherwise the
shorter budgets are meaningless and the script refuses to report them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from benchmark_features import build_case_features  # noqa: E402
from run_benchmark_walks import _strategy_spec  # noqa: E402
from walks import build_index, run_walk  # noqa: E402

OPPOSED = ("recent_history_k20", "time_respecting")
REFERENCE = "time_agnostic_t"          # the clean arm, as a contrast
FROZEN_BUDGET = 800
TRUE, SEEN, NAIVE = "rho_W5_k2", "oracle__seen_label_rho_k2", "est__plugin_rho_k2"


def replay(idx, strategy, seed, budgets, W, recent_limit):
    spec = _strategy_spec(strategy, 20)
    if spec["starts"] != 1:
        raise RuntimeError(f"{strategy} is multistart; prefixes are not valid")
    log = run_walk(idx, spec["base"], max_budget=max(budgets), seed=int(seed),
                   history_k=spec["history_k"])
    out = {}
    for budget in budgets:
        out[budget] = build_case_features(log, budget=budget, W=W, idx=idx,
                                          recent_limit=recent_limit)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default=str(REPO / "results/final_run_g2/final_cases.csv.gz"))
    ap.add_argument("--panel", default=str(REPO / "results/final_target_panel/panel32_final.csv"))
    ap.add_argument("--budgets", default="200,400,600,800,1200,1600")
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g3"))
    ap.add_argument("--tolerance", type=float, default=1e-9)
    ap.add_argument("--W", type=int, default=5)
    ap.add_argument("--recent-limit", type=int, default=100)
    args = ap.parse_args()

    budgets = sorted({int(b) for b in args.budgets.split(",")} | {FROZEN_BUDGET})
    arms = list(OPPOSED) + [REFERENCE]

    cases = pd.read_csv(args.cases, usecols=[
        "case_id", "instance_id", "group_id", "strategy", "seed_slot",
        "walk_rng_seed", TRUE, SEEN, NAIVE])
    cases = cases[(cases.seed_slot == 0) & (cases.strategy.isin(arms))]

    panel = pd.read_csv(args.panel)
    manifest_dir = Path(args.panel).parent
    paths = dict(zip(panel.instance_id, panel.path))

    rows, mismatches = [], []
    for instance, part in cases.groupby("instance_id"):
        events = pd.read_csv(manifest_dir / paths[instance])
        idx = build_index(events, T=1.0, W=args.W)
        for _, case in part.iterrows():
            features = replay(idx, case.strategy, case.walk_rng_seed, budgets,
                              args.W, args.recent_limit)
            check = features[FROZEN_BUDGET]
            for column in (SEEN, NAIVE):
                delta = abs(float(check[column]) - float(case[column]))
                if delta > args.tolerance:
                    mismatches.append({"case_id": case.case_id, "column": column,
                                       "delta": delta})
            for budget, f in features.items():
                seen, naive = float(f[SEEN]), float(f[NAIVE])
                rows.append({
                    "instance_id": instance, "group_id": case.group_id,
                    "strategy": case.strategy, "budget": budget,
                    "selection": seen - float(case[TRUE]),
                    "censoring": naive - seen,
                    "total_bias": naive - float(case[TRUE]),
                    "observed_dyads": float(f.get("wcnt__log_observed_dyads",
                                                  np.nan)),
                })

    if mismatches:
        frame = pd.DataFrame(mismatches)
        print(frame.head(10).to_string(index=False), file=sys.stderr)
        sys.exit(f"replay does not reproduce the frozen run: {len(frame)} "
                 f"mismatches above {args.tolerance:g}; shorter budgets are "
                 f"not interpretable and nothing was written")

    frame = pd.DataFrame(rows)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "budget_sign_stability_by_case.csv", index=False)

    summary = (frame.groupby(["strategy", "budget"], as_index=False)
               .agg(n=("total_bias", "size"),
                    selection=("selection", "mean"),
                    censoring=("censoring", "mean"),
                    total_bias=("total_bias", "mean"),
                    log_observed_dyads=("observed_dyads", "mean"),
                    share_cases_positive=("total_bias", lambda s: float((s > 0).mean()))))
    summary["selection_share"] = (summary.selection.abs()
                                  / (summary.selection.abs() + summary.censoring.abs()))
    summary.to_csv(out / "budget_sign_stability.csv", index=False)
    print(f"replay verified against the frozen run at budget {FROZEN_BUDGET} "
          f"for all {len(cases)} cases\n")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
