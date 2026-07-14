#!/usr/bin/env python3
"""Leakage-aware estimator screen, ablations, and transfer evaluations."""

import argparse
import copy
import glob
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="Skipping features without any observed values")
warnings.filterwarnings("ignore", message="An input array is constant.*")


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_evaluation_config(path, preset):
    with open(path) as fh:
        cfg = yaml.safe_load(fh)
    base = cfg.get("evaluation", {})
    override = cfg.get("presets", {}).get(preset, {}).get("evaluation", {})
    return _deep_merge(base, override)


def read_cases(specs):
    paths = []
    for spec in specs:
        found = sorted(glob.glob(spec))
        paths.extend(found or [spec])
    if not paths:
        raise FileNotFoundError("no case files matched")
    frames = [pd.read_csv(p) for p in paths]
    d = pd.concat(frames, ignore_index=True)
    if d["case_id"].duplicated().any():
        dup = d.loc[d["case_id"].duplicated(), "case_id"].iloc[0]
        raise ValueError(f"duplicate case_id after merging shards: {dup}")
    return d, paths


def input_columns(d, name):
    mapping = {
        "occupancy": ("occ__",),
        "patterns": ("pat__",),
        "crawl": ("crawl__",),
        "combined": ("occ__", "pat__", "crawl__"),
        "combined_plus_estimators": ("occ__", "pat__", "crawl__", "est__"),
    }
    if name not in mapping:
        raise KeyError(f"unknown input set {name!r}")
    cols = [c for c in d.columns if c.startswith(mapping[name])]
    if not cols:
        raise ValueError(f"input set {name!r} has no columns")
    return cols


def make_model(name, jobs=-1, n_outputs=1):
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                             StandardScaler(), Ridge(alpha=10.0))
    if name == "random_forest":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
            RandomForestRegressor(n_estimators=180, min_samples_leaf=2, max_features=0.7,
                                  random_state=17, n_jobs=jobs))
    if name == "extra_trees":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
            ExtraTreesRegressor(n_estimators=180, min_samples_leaf=2, max_features=0.8,
                                random_state=23, n_jobs=jobs))
    if name == "hist_gradient_boosting":
        base = HistGradientBoostingRegressor(
            max_iter=140, max_leaf_nodes=15, learning_rate=0.06,
            l2_regularization=0.5, random_state=31)
        estimator = (
            base if int(n_outputs) == 1
            else MultiOutputRegressor(base, n_jobs=jobs)
        )
        return make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            estimator,
        )
    raise KeyError(name)


def analytical_column(target, estimator):
    if target.startswith("rho_W5_k"):
        k = int(target.rsplit("k", 1)[1])
        return f"est__{estimator}_rho_k{k}"
    if target == "C_one_step":
        return f"est__{estimator}_C_one_step"
    if target == "mean_span_frac":
        return {
            "plugin": "est__plugin_mean_occupancy",
            "occ_mle": "est__occ_mle_mean_occupancy",
            "mask_mle": "est__mask_mle_mean_occupancy",
        }.get(estimator)
    if target == "rho_event_weighted" and estimator == "plugin":
        return "est__plugin_rho_event_weighted"
    return None


def _prediction_records(test, target, model, input_name, pred, fold,
                        selected_beta=np.nan, protocol="group_kfold"):
    keep = ["case_id", "group_id", "instance_id", "strategy", "data_block",
            "source", "domain", "substrate", "generator", "budget", "coverage",
            "walk_seed", target]
    keep = [c for c in keep if c in test.columns]
    x = test[keep].copy().rename(columns={target: "y_true"})
    x["target"] = target; x["model"] = model; x["input"] = input_name
    x["prediction"] = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    x["fold"] = str(fold); x["protocol"] = protocol
    x["selected_beta"] = selected_beta
    return x


def _useful_columns(train, cols):
    Xtr = train[cols].to_numpy(dtype=float)
    finite = np.isfinite(Xtr)
    counts = finite.sum(axis=0)
    means = np.divide(np.nansum(Xtr, axis=0), counts,
                      out=np.zeros(Xtr.shape[1]), where=counts > 0)
    filled = np.where(finite, Xtr, means)
    useful = np.nanstd(filled, axis=0) > 1e-12
    return Xtr, useful


def _fit_models(train, test, targets, inputs, model_names, jobs,
                model_suffix="", fold="0", protocol="group_kfold",
                model_inputs=None, headline_only_pairs=None):
    if train.empty or test.empty:
        return []
    out = []
    if model_inputs:
        pairs = [(m, inp) for m in model_names
                 for inp in model_inputs.get(m, []) if inp in inputs]
    else:
        pairs = [(m, inp) for inp in inputs for m in model_names]
    headline_only_pairs = set(headline_only_pairs or [])
    for model_name, input_name in pairs:
        pair_targets = (["rho_W5_k2"] if f"{model_name}:{input_name}" in headline_only_pairs
                        and "rho_W5_k2" in targets else list(targets))
        cols = input_columns(train, input_name)
        Xtr, useful = _useful_columns(train, cols)
        if not useful.any():
            continue
        Xte = test[cols].to_numpy(float)
        ytr = train[pair_targets].to_numpy(float)
        fit_y = ytr.ravel() if ytr.shape[1] == 1 else ytr
        model = make_model(
            model_name,
            jobs=jobs,
            n_outputs=len(pair_targets),
        )
        model.fit(Xtr[:, useful], fit_y)
        pred = np.asarray(model.predict(Xte[:, useful]))
        if pred.ndim == 1:
            pred = pred[:, None]
        for j, target in enumerate(pair_targets):
            out.append(_prediction_records(
                test, target, model_name + model_suffix, input_name,
                pred[:, j], fold, protocol=protocol))
    return out


def _select_beta(train, target="rho_W5_k2"):
    beta_cols = [c for c in train.columns if c.startswith("est__beta_rho_b")]
    if target not in train or not beta_cols:
        return None, np.nan
    y = train[target].to_numpy(float)
    errs = []
    for c in beta_cols:
        pv = train[c].to_numpy(float)
        ok = np.isfinite(pv) & np.isfinite(y)
        errs.append(float(np.mean(np.abs(pv[ok] - y[ok]))) if ok.any() else np.nan)
    if not np.isfinite(errs).any():
        return None, np.nan
    col = beta_cols[int(np.nanargmin(errs))]
    return col, float(col.rsplit("b", 1)[1])


def evaluate_strategy(d, strategy, targets, inputs, model_names, folds, jobs, cfg):
    x = d[d["strategy"] == strategy].reset_index(drop=True)
    groups = x["group_id"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        print(f"[skip] {strategy}: only {n_groups} independent group(s)")
        return []
    n_splits = min(int(folds), n_groups)
    y = x[targets].to_numpy(dtype=float)
    if not np.isfinite(y).all():
        raise ValueError(f"non-finite target in strategy {strategy}")
    records = []
    split = GroupKFold(n_splits=n_splits)
    for fold, (tr, te) in enumerate(split.split(x, y, groups)):
        train, test = x.iloc[tr], x.iloc[te]
        ytr = train[targets].to_numpy(float)
        for j, target in enumerate(targets):
            records.append(_prediction_records(
                test, target, "mean_floor", "none",
                np.full(len(test), float(np.mean(ytr[:, j]))), fold))
            estimators = ["plugin", "occ_mle", "mask_mle"]
            if cfg.get("conditional_baseline", False):
                estimators.insert(1, "conditional")
            for est in estimators:
                col = analytical_column(target, est)
                if col and col in x:
                    records.append(_prediction_records(
                        test, target, est, "analytical", test[col].to_numpy(), fold))

        if "rho_W5_k2" in targets:
            col, beta = _select_beta(train)
            if col:
                records.append(_prediction_records(
                    test, "rho_W5_k2", "beta_mle_lofo", "analytical",
                    test[col].to_numpy(float), fold, selected_beta=beta))
            # Family-specific calibration: select beta independently within each
            # test block, using only training groups from that block.
            if cfg.get("block_beta", False):
                for block, block_test in test.groupby("data_block"):
                    block_train = train[train["data_block"] == block]
                    bcol, bbeta = _select_beta(block_train)
                    if bcol:
                        records.append(_prediction_records(
                            block_test, "rho_W5_k2", "beta_mle_block_lofo", "analytical",
                            block_test[bcol].to_numpy(float), fold,
                            selected_beta=bbeta))

        records.extend(_fit_models(
            train, test, targets, inputs, model_names, jobs, fold=fold,
            protocol="group_kfold", model_inputs=cfg.get("model_inputs"),
            headline_only_pairs=cfg.get("headline_only_pairs")))
        print(f"[{strategy}] fold {fold+1}/{n_splits} complete", flush=True)
    return records


def evaluate_strategy_blind(d, targets, cfg, folds, jobs):
    if not cfg.get("enabled", False):
        return []
    target_use = [t for t in cfg.get("targets", targets) if t in targets]
    x = d[d["strategy"] != "time_agnostic"].reset_index(drop=True)
    groups = x["group_id"].astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    if n_groups < 2:
        return []
    split = GroupKFold(n_splits=min(int(folds), n_groups))
    records = []
    y = x[targets].to_numpy(float)
    for fold, (tr, te) in enumerate(split.split(x, y, groups)):
        records.extend(_fit_models(
            x.iloc[tr], x.iloc[te], target_use,
            cfg.get("inputs", ["combined"]),
            cfg.get("models", ["extra_trees"]), jobs,
            model_suffix="_strategy_blind", fold=fold,
            protocol="strategy_blind_group_kfold"))
        print(f"[strategy-blind] fold {fold+1} complete", flush=True)
    return records


def evaluate_leave_one_block_out(d, targets, cfg, jobs):
    if not cfg.get("enabled", False):
        return []
    records = []
    target_use = [t for t in cfg.get("targets", targets) if t in targets]
    min_groups = int(cfg.get("min_train_groups", 5))
    for strategy in sorted(d["strategy"].unique()):
        x = d[d["strategy"] == strategy]
        for block in sorted(x["data_block"].dropna().unique()):
            test = x[x["data_block"] == block]
            held_groups = set(test["group_id"].astype(str))
            train = x[(x["data_block"] != block) &
                      (~x["group_id"].astype(str).isin(held_groups))]
            if train["group_id"].nunique() < min_groups or test.empty:
                continue
            records.extend(_fit_models(
                train, test, target_use,
                cfg.get("inputs", ["combined"]),
                cfg.get("models", ["extra_trees"]), jobs,
                model_suffix="_lobo", fold=block,
                protocol="leave_one_block_and_group_out"))
            print(f"[LOBO {strategy}] held out {block}", flush=True)
    return records


def evaluate_sim2real(d, targets, cfg, jobs):
    if not cfg.get("enabled", False):
        return []
    train_blocks = set(cfg.get("train_blocks", []))
    test_blocks = set(cfg.get("test_blocks", ["real_empirical"]))
    records = []
    target_use = [t for t in cfg.get("targets", targets) if t in targets]
    for strategy in sorted(d["strategy"].unique()):
        x = d[d["strategy"] == strategy]
        train = x[x["data_block"].isin(train_blocks)]
        test = x[x["data_block"].isin(test_blocks)]
        if train["group_id"].nunique() < 2 or test.empty:
            continue
        records.extend(_fit_models(
            train, test, target_use,
            cfg.get("inputs", ["combined"]),
            cfg.get("models", ["extra_trees"]), jobs,
            model_suffix="_sim2real", fold="synthetic_to_real",
            protocol="synthetic_to_real"))
        print(f"[sim2real {strategy}] {len(train)} train -> {len(test)} test", flush=True)
    return records


def _metric_row(g, slice_type, slice_value):
    y = g["y_true"].to_numpy(float); p = g["prediction"].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if not len(y):
        return None
    z = g.loc[ok].copy()
    z["ae"] = np.abs(y - p)
    group_mae = z.groupby("group_id")["ae"].mean()
    return {
        "target": g["target"].iloc[0], "strategy": g["strategy"].iloc[0],
        "model": g["model"].iloc[0], "input": g["input"].iloc[0],
        "protocol": g["protocol"].iloc[0],
        "slice_type": slice_type, "slice_value": str(slice_value),
        "n": len(y), "n_groups": z["group_id"].nunique(),
        "mae": mean_absolute_error(y, p),
        "mae_group_macro": float(group_mae.mean()),
        "mae_worst_group": float(group_mae.max()),
        "mae_p90_group": float(group_mae.quantile(0.90)),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "bias": float(np.mean(p - y)),
        "spearman": float(pd.Series(p).corr(pd.Series(y), method="spearman")),
    }


def compute_metrics(pred, cfg):
    p = pred.copy()
    bins = cfg.get("coverage_bins", [-np.inf, 0.02, 0.15, np.inf])
    labels = cfg.get("coverage_labels", ["low(<.02)", "mid", "high(>.15)"])
    # User-friendly config may use finite sentinels; expand boundaries safely.
    bins = list(map(float, bins)); bins[0] = -np.inf; bins[-1] = np.inf
    p["coverage_band"] = pd.cut(p["coverage"], bins, labels=labels,
                                 include_lowest=True).astype(str)
    keys = ["target", "strategy", "model", "input", "protocol"]
    rows = []
    for _, g in p.groupby(keys, dropna=False):
        rows.append(_metric_row(g, "overall", "all"))
        for col in ("data_block", "domain", "coverage_band", "budget", "walk_seed"):
            if col not in g:
                continue
            for value, sub in g.groupby(col, dropna=False):
                rows.append(_metric_row(sub, col, value))
    return pd.DataFrame([r for r in rows if r is not None])


def make_rankings(metrics):
    m = metrics.copy()
    m["model_label"] = np.where(
        m["input"].isin(["analytical", "none"]), m["model"],
        m["model"] + " [" + m["input"] + "]")
    group = ["target", "strategy", "protocol", "slice_type", "slice_value"]
    m["rank"] = m.groupby(group)["mae_group_macro"].rank(method="average")
    cols = group + ["model_label", "rank", "mae", "mae_group_macro",
                    "mae_worst_group", "mae_p90_group", "n", "n_groups"]
    return m[cols].sort_values(group + ["rank"])


def write_summary(out_dir, cases, metrics, rankings, case_paths):
    primary = rankings[(rankings.target == "rho_W5_k2") &
                       (rankings.protocol == "group_kfold") &
                       (rankings.slice_type == "overall")]
    lines = [
        "# Estimator-screen summary", "",
        f"- Cases: {len(cases):,}",
        f"- Independent source/family groups: {cases.group_id.nunique()}",
        f"- Case files: {', '.join(map(str, case_paths))}",
        "- Main split: GroupKFold by source/family; no variant-family leakage.",
        "- Extra protocols: strategy-blind, block+group holdout, and synthetic-to-real when enabled.",
        "- True coverage and EdgeBank AUC diagnostics are metadata, never model inputs.", "",
        "## Best headline estimators by access model", "",
        "| Access | Rank | Model | Group-macro MAE | Worst-group MAE |",
        "|---|---:|---|---:|---:|",
    ]
    for strategy, g in primary.groupby("strategy"):
        for _, r in g.nsmallest(5, "rank").iterrows():
            lines.append(f"| {strategy} | {r['rank']:.1f} | {r['model_label']} | "
                         f"{r['mae_group_macro']:.4f} | {r['mae_worst_group']:.4f} |")
    transfer = rankings[(rankings.target == "rho_W5_k2") &
                        (rankings.slice_type == "overall") &
                        (rankings.protocol != "group_kfold")]
    if not transfer.empty:
        lines += ["", "## Transfer protocols", "",
                  "| Protocol | Access | Model | Group-macro MAE |",
                  "|---|---|---|---:|"]
        for (protocol, strategy), g in transfer.groupby(["protocol", "strategy"]):
            for _, r in g.nsmallest(2, "rank").iterrows():
                lines.append(f"| {protocol} | {strategy} | {r['model_label']} | "
                             f"{r['mae_group_macro']:.4f} |")
    (out_dir / "SCREEN_SUMMARY.md").write_text("\n".join(lines) + "\n")


def make_plot(metrics, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    m = metrics[(metrics.target == "rho_W5_k2") &
                (metrics.protocol == "group_kfold") &
                (metrics.slice_type == "overall")].copy()
    if m.empty:
        return
    m["label"] = np.where(m["input"].isin(["analytical", "none"]), m["model"],
                          m["model"] + "\n" + m["input"])
    strategies = list(m["strategy"].unique())
    fig, axes = plt.subplots(len(strategies), 1,
                             figsize=(11, max(3.5, 3.2 * len(strategies))),
                             squeeze=False)
    for ax, strategy in zip(axes[:, 0], strategies):
        g = m[m.strategy == strategy].nsmallest(10, "mae_group_macro")
        ax.barh(g["label"][::-1], g["mae_group_macro"][::-1])
        ax.set_title(strategy); ax.set_xlabel("group-macro MAE (rho k=2)")
        ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "headline_ranking.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/benchmark.yaml")
    ap.add_argument("--preset", default="smoke")
    ap.add_argument("--cases", nargs="+", default=None)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--jobs", type=int, default=-1)
    args = ap.parse_args()
    cfg = load_evaluation_config(args.config, args.preset)
    case_specs = args.cases or [f"results/benchmark_{args.preset}/cases*.csv.gz"]
    out_dir = args.out_dir or f"results/benchmark_{args.preset}"
    cases, paths = read_cases(case_specs)
    targets = [t for t in cfg["targets"] if t in cases.columns]
    inputs, models = list(cfg["inputs"]), list(cfg["models"])
    print(f"{len(cases)} cases, {cases.group_id.nunique()} groups, "
          f"{len(targets)} targets, strategies={sorted(cases.strategy.unique())}")
    metric_parts = []
    saved_predictions = []
    save_targets = set(cfg.get("prediction_targets", targets))

    def consume(records):
        if not records:
            return
        part = pd.concat(records, ignore_index=True)
        metric_parts.append(compute_metrics(part, cfg))
        keep = part[part["target"].isin(save_targets)]
        if not keep.empty:
            saved_predictions.append(keep)

    for strategy in sorted(cases.strategy.unique()):
        consume(evaluate_strategy(
            cases, strategy, targets, inputs, models,
            folds=int(cfg.get("group_folds", 5)), jobs=args.jobs, cfg=cfg))
    consume(evaluate_strategy_blind(
        cases, targets, cfg.get("strategy_blind", {}),
        folds=int(cfg.get("group_folds", 5)), jobs=args.jobs))
    consume(evaluate_leave_one_block_out(
        cases, targets, cfg.get("leave_one_block_out", {}), jobs=args.jobs))
    consume(evaluate_sim2real(
        cases, targets, cfg.get("sim2real", {}), jobs=args.jobs))
    if not metric_parts:
        raise RuntimeError("no evaluable protocol had at least two groups")
    metrics = pd.concat(metric_parts, ignore_index=True)
    rankings = make_rankings(metrics)
    pred = (pd.concat(saved_predictions, ignore_index=True) if saved_predictions
            else pd.DataFrame())
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "metrics.csv", index=False)
    rankings.to_csv(out / "rankings.csv", index=False)
    # Compact case-level diagnostics for direct result inspection.
    diag_cols = [c for c in cases if c.startswith("diag__")]
    if diag_cols:
        keep = ["case_id", "group_id", "instance_id", "strategy", "data_block",
                "budget", "coverage"] + diag_cols
        cases[keep].to_csv(out / "case_diagnostics.csv.gz", index=False,
                           compression="gzip")
    write_summary(out, cases, metrics, rankings, paths)
    make_plot(metrics, out)
    print(f"wrote {out}/predictions.csv.gz, metrics.csv, rankings.csv, "
          "case_diagnostics.csv.gz, SCREEN_SUMMARY.md, headline_ranking.png")


if __name__ == "__main__":
    main()
