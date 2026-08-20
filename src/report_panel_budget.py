#!/usr/bin/env python3
"""Coverage and estimator report for panel walk cases across budgets.

Answers what a walk budget buys on a given panel: how much of each graph the
sampler actually sees, how the prespecified coverage bands fill, how large the
prompt histogram becomes, and whether the classical estimators still separate.
Read-only; it consumes case files produced by `run_benchmark_walks.py`.

Generate the cases first (see `scripts/run_panel_budget_probe.sh`). Nested
budgets are worth requesting in one pass: standard strategies draw a single
trajectory at the largest budget and every smaller budget is an exact prefix
of it, so the whole ladder costs one run and the budgets are perfectly paired.

The supervised row here is fitted on the panel cases alone with
leave-one-group-out. On a 32-graph panel that is thin, and it shows: the row
saturates and the analytical mask MLE catches up at larger budgets. A ceiling
meant to bound what is achievable wants the much larger frozen benchmark as
its training set instead, with the panel backbones held out.

Example:
  PYTHONPATH=src python src/report_panel_budget.py \
      --cases results/panel_budget_probe/cases.csv.gz \
      --panel results/final_target_panel/panel32_final.csv --budget 800
"""

import argparse
import glob
import json

import numpy as np
import pandas as pd

KS = [2, 3, 4, 5]
TRUTH = [f"rho_W5_k{k}" for k in KS]
BANDS = [-1.0, 0.01, 0.05, 0.2, 1.01]
BAND_LABELS = ["very_low(<.01)", "low(.01-.05)", "mid(.05-.20)", "high(>.20)"]

ANALYTICAL = {
    "naive read-off": [f"est__plugin_rho_k{k}" for k in KS],
    "occupancy MLE": [f"est__occ_mle_rho_k{k}" for k in KS],
    "mask MLE": [f"est__mask_mle_rho_k{k}" for k in KS],
    "oracle: seen labels": [f"oracle__seen_label_rho_k{k}" for k in KS],
    "oracle: traversal-weighted": [f"oracle__hh_label_rho_k{k}" for k in KS],
}
FEATURE_PREFIXES = ("occ__", "pat__", "crawl__", "est__")


def metrics(frame, columns):
    """ProfileMAE, group-macro, worst group, rho2 error and bias."""
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        return None
    pred = frame[columns].to_numpy(float)
    truth = frame[TRUTH].to_numpy(float)
    usable = np.isfinite(pred).all(axis=1)
    if not usable.any():
        return None
    errors = np.abs(pred[usable] - truth[usable])
    per_case = errors.mean(axis=1)
    grouped = pd.DataFrame({"g": frame.loc[usable, "group_id"].to_numpy(),
                            "v": per_case}).groupby("g").v.mean()
    return {"n": int(usable.sum()), "valid": float(usable.mean()),
            "pmae": float(per_case.mean()), "macro": float(grouped.mean()),
            "worst": float(grouped.max()), "rho2": float(errors[:, 0].mean()),
            "bias": float((pred[usable, 0] - truth[usable, 0]).mean())}


def add_floor(frame):
    """Leave-one-group-out mean of the truth: the simplest supervised row."""
    groups = frame["group_id"].to_numpy()
    truth = frame[TRUTH].to_numpy(float)
    for j, k in enumerate(KS):
        frame[f"floor_k{k}"] = [
            truth[groups != g].mean(axis=0)[j] if (groups != g).any()
            else truth.mean(axis=0)[j] for g in groups]
    return [f"floor_k{k}" for k in KS]


def add_supervised(frame, seed=0):
    """ExtraTrees profile model, leave-one-group-out over the panel groups."""
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.model_selection import LeaveOneGroupOut

    features = [c for c in frame.columns
                if c.startswith(FEATURE_PREFIXES) and frame[c].dtype != object]
    x = np.nan_to_num(frame[features].to_numpy(float),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = frame[TRUTH].to_numpy(float)
    groups = frame["group_id"].to_numpy()
    if len(np.unique(groups)) < 2:
        return None, 0
    pred = np.zeros_like(y)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = ExtraTreesRegressor(n_estimators=300, random_state=seed,
                                    n_jobs=-1)
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    for j, k in enumerate(KS):
        frame[f"ml_k{k}"] = pred[:, j]
    return [f"ml_k{k}" for k in KS], len(features)


def row(label, m, width=26):
    if m is None:
        return f"{label:<{width}}  (not available)"
    return (f"{label:<{width}}{m['n']:>5}{m['valid']:>8.2f}{m['pmae']:>9.4f}"
            f"{m['macro']:>9.4f}{m['worst']:>9.4f}{m['rho2']:>9.4f}"
            f"{m['bias']:>+9.4f}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True,
                    help="case file(s) from run_benchmark_walks.py; glob ok")
    ap.add_argument("--panel", default="results/final_target_panel/panel32_final.csv")
    ap.add_argument("--budget", type=int, default=800,
                    help="budget to inspect in detail")
    ap.add_argument("--no-supervised", action="store_true",
                    help="skip the ExtraTrees row")
    args = ap.parse_args()

    paths = sorted(glob.glob(args.cases))
    if not paths:
        raise SystemExit(f"no case files matched {args.cases!r}")
    data = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    panel = pd.read_csv(args.panel)
    extra = [c for c in ["graph_category", "panel_role", "matched_backbone",
                         "n_pairs"] if c in panel.columns]
    data = data.merge(panel[["instance_id"] + extra], on="instance_id",
                      how="left")
    data["band"] = pd.cut(data["coverage"], bins=BANDS, labels=BAND_LABELS)
    budgets = sorted(data.budget.unique())

    print(f"{len(data)} cases, {data.instance_id.nunique()} instances, "
          f"{data.group_id.nunique()} groups, "
          f"{data.strategy.nunique()} strategies, budgets {budgets}")

    print("\n=== coverage by budget ===")
    print(f"{'budget':>8}{'median':>9}{'mean':>9}{'p10':>9}{'p90':>9}"
          f"{'min':>9}{'max':>9}{'obs dyads':>11}")
    for b in budgets:
        c = data[data.budget == b]["coverage"]
        obs = data[data.budget == b]["observed_walk_edges"].median()
        print(f"{b:>8}{c.median():>9.3f}{c.mean():>9.3f}{c.quantile(.1):>9.3f}"
              f"{c.quantile(.9):>9.3f}{c.min():>9.4f}{c.max():>9.3f}{obs:>11.0f}")

    print("\n=== prespecified coverage bands ===")
    print(f"{'budget':>8}" + "".join(f"{l:>17}" for l in BAND_LABELS))
    for b in budgets:
        sub = data[data.budget == b]
        cells = []
        for label in BAND_LABELS:
            n = int((sub["band"] == label).sum())
            cells.append(f"{n} ({n / len(sub) * 100:.0f}%)".rjust(17))
        print(f"{b:>8}" + "".join(cells))

    if "input__nmask_exact_json" in data.columns:
        print("\n=== prompt histogram size (rows in the (n,mask) JSON) ===")
        print(f"{'budget':>8}{'median':>9}{'p90':>8}{'max':>8}{'median chars':>14}")
        for b in budgets:
            sub = data[data.budget == b]["input__nmask_exact_json"]
            rows = sub.map(lambda s: len(json.loads(s)))
            print(f"{b:>8}{rows.median():>9.0f}{rows.quantile(.9):>8.0f}"
                  f"{rows.max():>8.0f}{sub.map(len).median():>14.0f}")

    print(f"\n=== estimator ladder by budget (ProfileMAE) ===")
    names = ["floor (LOGO mean)", "naive read-off", "occupancy MLE", "mask MLE"]
    if not args.no_supervised:
        names.append("supervised (panel only)")
    print(f"{'budget':>8}" + "".join(f"{n[:22]:>24}" for n in names))
    detail = {}
    for b in budgets:
        sub = data[data.budget == b].reset_index(drop=True)
        cells, entry = [], {}
        floor_cols = add_floor(sub)
        entry["floor (LOGO mean)"] = metrics(sub, floor_cols)
        for name in ["naive read-off", "occupancy MLE", "mask MLE"]:
            entry[name] = metrics(sub, ANALYTICAL[name])
        if not args.no_supervised:
            ml_cols, _ = add_supervised(sub)
            entry["supervised (panel only)"] = (metrics(sub, ml_cols)
                                                if ml_cols else None)
        for name in names:
            m = entry[name]
            cells.append(("-" if m is None else f"{m['pmae']:.4f}").rjust(24))
        print(f"{b:>8}" + "".join(cells))
        detail[b] = (sub, entry)

    if args.budget not in detail:
        print(f"\nbudget {args.budget} not present; detail sections skipped")
        return
    sub, entry = detail[args.budget]

    print(f"\n=== budget {args.budget} in detail ===")
    print(f"{'estimator':<26}{'n':>5}{'valid':>8}{'pmae':>9}{'macro':>9}"
          f"{'worst':>9}{'rho2':>9}{'bias':>9}")
    for name in names:
        print(row(name, entry[name]))
    for name in ["oracle: seen labels", "oracle: traversal-weighted"]:
        m = metrics(sub, ANALYTICAL[name])
        if m is not None:
            print(row(name, m))
    print("\nThe oracle rows isolate selection bias from window censoring; they "
          "only exist\non freshly generated case files, not on the frozen "
          "shards.")

    print(f"\n=== budget {args.budget} by slice (ProfileMAE) ===")
    slices = [("strategy", sorted(sub.strategy.unique()))]
    if "graph_category" in sub.columns:
        slices.append(("graph_category", sorted(sub.graph_category.dropna().unique())))
    slices.append(("band", [b for b in BAND_LABELS if (sub["band"] == b).any()]))
    shown = [n for n in names if entry[n] is not None]
    for column, values in slices:
        print(f"\n{column:<24}{'n':>5}" + "".join(f"{n[:14]:>16}" for n in shown))
        for value in values:
            part = sub[sub[column] == value]
            if part.empty:
                continue
            cells = []
            for name in shown:
                cols = ({"floor (LOGO mean)": [f"floor_k{k}" for k in KS],
                         "supervised (panel only)": [f"ml_k{k}" for k in KS]}
                        .get(name, ANALYTICAL.get(name)))
                m = metrics(part, cols)
                cells.append(("-" if m is None else f"{m['pmae']:.4f}").rjust(16))
            print(f"{str(value)[:24]:<24}{len(part):>5}" + "".join(cells))
    print("\nRaw error tends to rise with coverage here because the "
          "high-coverage graphs are\nthe small dense ones with large targets; "
          "skill against the floor reads differently.")

    if "panel_role" in sub.columns:
        twins = sub[sub.panel_role.isin(["controlled_low", "controlled_high"])]
        if not twins.empty:
            lo = twins[twins.panel_role == "controlled_low"]
            hi = twins[twins.panel_role == "controlled_high"]
            true_delta = hi["rho_W5_k2"].mean() - lo["rho_W5_k2"].mean()
            print(f"\n=== controlled twins at budget {args.budget}: recovered "
                  f"contrast in rho2 ===")
            print(f"true contrast {true_delta:.4f}")
            print(f"{'estimator':<26}{'low':>10}{'high':>10}{'recovered':>11}"
                  f"{'share':>8}")
            for name in ["naive read-off", "occupancy MLE", "mask MLE"]:
                col = ANALYTICAL[name][0]
                if col not in twins.columns:
                    continue
                a, b = lo[col].mean(), hi[col].mean()
                print(f"{name:<26}{a:>10.4f}{b:>10.4f}{b - a:>11.4f}"
                      f"{(b - a) / true_delta:>8.0%}")


if __name__ == "__main__":
    main()
