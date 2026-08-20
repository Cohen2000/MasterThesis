#!/usr/bin/env python3
"""Where does the noise in a panel result live: graphs, or walk seeds?

The supervisor question "more graphs or more seeds?" is answerable by
measurement rather than argument. A per-case error decomposes over three
nested levels of the panel design -- group, instance within group, walk seed
within instance -- and only the bottom level is what an extra seed buys down.

The primary statistic of the main comparison is the group-macro mean error, so
that is what this reports a standard error for:

    SE^2(G, I, S) = s2_group / G + s2_inst / (G*I) + s2_seed / (G*I*S)

Adding seeds shrinks only the last term, which is already divided by G*I. That
makes the trade-off explicit: an extra seed and an extra graph are not
competing on equal footing, and the ratio is a property of the data.

Two kinds of comparison are reported separately, because they behave very
differently:

* absolute level of one method -- full variance applies;
* paired difference between two methods scored on the SAME walk sample --
  seed noise is common-mode and largely cancels, so seeds matter much less
  than the absolute numbers suggest. This is the case that governs "is model A
  better than model B", which is the thesis' actual claim.

Read-only. Consumes case files from `run_benchmark_walks.py`; generate them
with several walk seeds first:

  SEEDS=8 BUDGETS="[400, 800, 1600]" OUT_DIR=results/panel_seed_probe \
      bash scripts/run_panel_budget_probe.sh

  PYTHONPATH=src python src/report_seed_variance.py \
      --cases results/panel_seed_probe/cases.csv.gz --budget 800
"""

import argparse
import glob

import numpy as np
import pandas as pd

KS = [2, 3, 4, 5]
TRUTH = [f"rho_W5_k{k}" for k in KS]

METHODS = {
    "naive read-off": [f"est__plugin_rho_k{k}" for k in KS],
    "occupancy MLE": [f"est__occ_mle_rho_k{k}" for k in KS],
    "mask MLE": [f"est__mask_mle_rho_k{k}" for k in KS],
}
FEATURE_PREFIXES = ("occ__", "pat__", "crawl__", "est__")


def profile_mae(frame, columns):
    """Per-case ProfileMAE, NaN where the method produced no finite profile."""
    pred = frame[columns].to_numpy(float)
    truth = frame[TRUTH].to_numpy(float)
    out = np.abs(pred - truth).mean(axis=1)
    out[~np.isfinite(pred).all(axis=1)] = np.nan
    return out


def add_floor(frame):
    """Leave-one-group-out mean of truth: the simplest supervised predictor.

    Constant across walk seeds by construction, which makes it a useful anchor:
    its seed variance must come out at zero.
    """
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
        return None
    pred = np.zeros_like(y)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = ExtraTreesRegressor(n_estimators=300, random_state=seed,
                                    n_jobs=-1)
        model.fit(x[train], y[train])
        pred[test] = model.predict(x[test])
    for j, k in enumerate(KS):
        frame[f"ml_k{k}"] = pred[:, j]
    return [f"ml_k{k}" for k in KS]


def decompose(frame, values):
    """Nested variance components for group > instance > walk seed.

    Balanced-design moment estimators: the seed component is the mean
    within-instance variance, the instance component the mean within-group
    variance of instance means with the seed term removed, and the group
    component the variance of group means with the lower terms removed.
    Components are floored at zero, which is what an unbiased moment estimator
    can otherwise undershoot into when the true component is near zero.
    """
    work = pd.DataFrame({"group": frame["group_id"].to_numpy(),
                         "inst": frame["instance_id"].to_numpy(),
                         "v": values}).dropna()
    if work.empty:
        return None

    per_inst = work.groupby(["group", "inst"]).v.agg(["mean", "var", "size"])
    n_seed = float(per_inst["size"].mean())
    s2_seed = float(per_inst["var"].mean(skipna=True) or 0.0)

    per_group = per_inst.groupby("group")["mean"].agg(["mean", "var", "size"])
    n_inst = float(per_group["size"].mean())
    raw_inst = float(per_group["var"].mean(skipna=True) or 0.0)
    s2_inst = max(raw_inst - s2_seed / max(n_seed, 1.0), 0.0)

    n_group = float(len(per_group))
    raw_group = float(per_group["mean"].var(ddof=1))
    s2_group = max(raw_group - s2_inst / max(n_inst, 1.0)
                   - s2_seed / max(n_inst * n_seed, 1.0), 0.0)

    return {"mean": float(work.v.mean()), "n": int(len(work)),
            "n_group": n_group, "n_inst": n_inst, "n_seed": n_seed,
            "s2_group": s2_group, "s2_inst": s2_inst, "s2_seed": s2_seed}


def se_macro(parts, n_group=None, n_inst=None, n_seed=None):
    """SE of the group-macro mean under a hypothetical (G, I, S) design."""
    g = n_group or parts["n_group"]
    i = n_inst or parts["n_inst"]
    s = n_seed or parts["n_seed"]
    return float(np.sqrt(parts["s2_group"] / g
                         + parts["s2_inst"] / (g * i)
                         + parts["s2_seed"] / (g * i * s)))


def share_line(parts):
    total = parts["s2_group"] + parts["s2_inst"] + parts["s2_seed"]
    if total <= 0:
        return "  (no variance)"
    return ("  variance share:  group {:.0%}   graph-in-group {:.0%}   "
            "walk seed {:.0%}".format(parts["s2_group"] / total,
                                      parts["s2_inst"] / total,
                                      parts["s2_seed"] / total))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", required=True, help="case file(s); glob ok")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--strategies", default=None,
                    help="comma-separated subset; default: all present")
    ap.add_argument("--no-supervised", action="store_true")
    args = ap.parse_args()

    paths = sorted(set(sum((glob.glob(p) for p in args.cases.split(",")), [])))
    frame = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    frame = frame[frame["budget"] == args.budget].copy()
    if args.strategies:
        keep = [s.strip() for s in args.strategies.split(",")]
        frame = frame[frame["strategy"].isin(keep)].copy()
    if frame.empty:
        raise SystemExit(f"no cases at budget {args.budget}")

    seeds = sorted(frame["walk_seed"].unique())
    strategies = sorted(frame["strategy"].unique())
    print(f"cases {len(frame)} | graphs {frame.instance_id.nunique()} | "
          f"groups {frame.group_id.nunique()} | strategies {len(strategies)} | "
          f"walk seeds {len(seeds)} | budget {args.budget}")
    if len(seeds) < 2:
        raise SystemExit("need at least 2 walk seeds; rerun with SEEDS>1")

    methods = dict(METHODS)
    methods["mean floor (LOGO)"] = add_floor(frame)
    if not args.no_supervised:
        cols = add_supervised(frame)
        if cols:
            methods["supervised ExtraTrees"] = cols

    scores = {}
    for name, cols in methods.items():
        if all(c in frame.columns for c in cols):
            scores[name] = profile_mae(frame, cols)

    print("\n" + "=" * 78)
    print("1. Coverage of the same graph across walk seeds")
    print("=" * 78)
    cov = frame.groupby(["strategy", "instance_id"]).coverage.agg(
        ["mean", "std"])
    cov["cv"] = cov["std"] / cov["mean"].replace(0, np.nan)
    print(f"  within-graph coverage CV across {len(seeds)} seeds: "
          f"median {cov.cv.median():.3f}, p90 {cov.cv.quantile(.9):.3f}")
    print("  (how much the same walk budget on the same graph varies at all)")

    print("\n" + "=" * 78)
    print(f"2. Where the error variance lives (per strategy, budget {args.budget})")
    print("=" * 78)
    rows = []
    for strategy in strategies:
        mask = (frame["strategy"] == strategy).to_numpy()
        print(f"\n-- {strategy} --")
        for name, value in scores.items():
            parts = decompose(frame[mask], value[mask])
            if parts is None:
                continue
            print(f"{name:<24} macro-mean {parts['mean']:.4f}")
            print(share_line(parts))
            base = se_macro(parts)
            one = se_macro(parts, n_seed=1)
            many = se_macro(parts, n_seed=1e9)
            print(f"  SE of group-macro mean:  1 seed {one:.4f}   "
                  f"{int(parts['n_seed'])} seeds {base:.4f}   "
                  f"infinite seeds {many:.4f}")
            rows.append({"strategy": strategy, "method": name, **parts,
                         "se_1seed": one, "se_inf": many})

    print("\n" + "=" * 78)
    print("3. What an extra seed buys, and what an extra graph buys")
    print("=" * 78)
    print("SE of the group-macro ProfileMAE under hypothetical designs.")
    print("G = groups, I = graphs per group, S = walk seeds per graph.\n")
    summary = pd.DataFrame(rows)
    if not summary.empty:
        pooled = summary.groupby("method")[
            ["s2_group", "s2_inst", "s2_seed", "n_group", "n_inst"]].mean()
        header = (f"{'method':<24}{'S=1':>9}{'S=2':>9}{'S=4':>9}{'S=8':>9}"
                  f"{'2xG,S=1':>10}")
        print(header)
        print("-" * len(header))
        for name, r in pooled.iterrows():
            parts = {"s2_group": r.s2_group, "s2_inst": r.s2_inst,
                     "s2_seed": r.s2_seed, "n_group": r.n_group,
                     "n_inst": r.n_inst, "n_seed": 1.0}
            cells = [se_macro(parts, n_seed=s) for s in (1, 2, 4, 8)]
            doubled = se_macro(parts, n_group=2 * r.n_group, n_seed=1)
            print(f"{name:<24}" + "".join(f"{c:>9.4f}" for c in cells)
                  + f"{doubled:>10.4f}")

    print("\n" + "=" * 78)
    print("4. Paired comparisons: the case the thesis actually makes")
    print("=" * 78)
    print("Two methods scored on the SAME walk sample. Seed noise is")
    print("common-mode and cancels in the difference, so this is the SE that")
    print("governs 'is A better than B'.\n")
    names = list(scores)
    ref = "mask MLE" if "mask MLE" in scores else names[0]
    header = (f"{'A - B (B = {})'.format(ref):<34}{'diff':>9}{'SE S=1':>9}"
              f"{'SE S=8':>9}{'MDD S=1':>10}")
    print(header)
    print("-" * len(header))
    for name in names:
        if name == ref:
            continue
        diff = scores[name] - scores[ref]
        per_strategy = []
        for strategy in strategies:
            mask = (frame["strategy"] == strategy).to_numpy()
            parts = decompose(frame[mask], diff[mask])
            if parts:
                per_strategy.append(parts)
        if not per_strategy:
            continue
        agg = {k: float(np.mean([p[k] for p in per_strategy]))
               for k in ("mean", "s2_group", "s2_inst", "s2_seed",
                         "n_group", "n_inst", "n_seed")}
        se1 = se_macro(agg, n_seed=1)
        se8 = se_macro(agg, n_seed=8)
        print(f"{name:<34}{agg['mean']:>+9.4f}{se1:>9.4f}{se8:>9.4f}"
              f"{2.8 * se1:>10.4f}")
    print("\nMDD = smallest true difference detectable at 80% power, "
          "two-sided 5%.")

    print("\n" + "=" * 78)
    print("5. Strategy comparisons: independent samples, no cancellation")
    print("=" * 78)
    print("Access strategies draw different walks, so seed noise does NOT")
    print("cancel when comparing them.\n")
    if len(strategies) >= 2 and ref in scores:
        wide = pd.DataFrame({
            "group": frame["group_id"].to_numpy(),
            "inst": frame["instance_id"].to_numpy(),
            "seed": frame["walk_seed"].to_numpy(),
            "strategy": frame["strategy"].to_numpy(),
            "v": scores[ref]}).dropna()
        pivot = wide.pivot_table(index=["group", "inst", "seed"],
                                 columns="strategy", values="v")
        base = strategies[0]
        header = (f"{'strategy - ' + base:<34}{'diff':>9}{'SE S=1':>9}"
                  f"{'SE S=8':>9}{'MDD S=1':>10}")
        print(f"(scored with {ref})")
        print(header)
        print("-" * len(header))
        for strategy in strategies[1:]:
            if strategy not in pivot or base not in pivot:
                continue
            sub = pivot[[base, strategy]].dropna().reset_index()
            parts = decompose(
                sub.rename(columns={"group": "group_id",
                                    "inst": "instance_id"}),
                (sub[strategy] - sub[base]).to_numpy())
            if not parts:
                continue
            se1 = se_macro(parts, n_seed=1)
            se8 = se_macro(parts, n_seed=8)
            print(f"{strategy:<34}{parts['mean']:>+9.4f}{se1:>9.4f}"
                  f"{se8:>9.4f}{2.8 * se1:>10.4f}")


if __name__ == "__main__":
    main()
