#!/usr/bin/env python3
"""Evaluate analytical, prompt-parity, and observable-ceiling baselines."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold

from design_baselines import discovery_diagnostics, parse_nmask_hist
from evaluate_benchmark import (
    _prediction_records,
    _useful_columns,
    compute_metrics,
    load_evaluation_config,
    make_model,
    make_rankings,
    read_cases,
)
from nonwalk_prompt_contract import metadata_only_columns, prompt_parity_columns
from reservoir_deconvolution import temporal_mask_rate_factorized_eb


RHO_TARGETS = [f"rho_W5_k{k}" for k in range(2, 6)]


def _long_predictions(cases, name, columns, input_name="analytical",
                      protocol="nonwalk_screen"):
    records = []
    for target, column in zip(RHO_TARGETS, columns):
        if column not in cases:
            return []
        records.append(_prediction_records(
            cases, target, name, input_name, cases[column].to_numpy(float),
            fold="none", protocol=protocol))
    return records


def _fit_one_reservoir_eb(case_id, raw, p, grid_size, max_iter):
    estimate, diag = temporal_mask_rate_factorized_eb(
        raw, sampling_fraction=p, W=5, grid_size=grid_size,
        max_iter=max_iter)
    diag["retry_from_iteration_limit"] = False
    if not diag.get("converged", False) and int(max_iter) < 2000:
        first_iterations = int(diag.get("iterations", max_iter))
        estimate, diag = temporal_mask_rate_factorized_eb(
            raw, sampling_fraction=p, W=5, grid_size=grid_size,
            max_iter=2000)
        diag["retry_from_iteration_limit"] = True
        diag["initial_iteration_limit"] = first_iterations
    return case_id, estimate, diag


def add_reservoir_eb(cases, grid_size=28, max_iter=300, jobs=1):
    """Attach observable EB estimates to reservoir rows only."""
    frame = cases.copy()
    reservoir = frame.strategy == "uniform_event_reservoir"
    diagnostics = []
    columns = [f"est__reservoir_eb_rho_k{k}" for k in range(2, 6)]
    for column in columns:
        frame[column] = np.nan
    subset = frame.loc[reservoir]
    tasks = [
        (row.case_id, row.input__window_counts_exact_json,
         min(1.0, float(row.budget) / float(row.n_events_true)))
        for row in subset.itertuples()
    ]
    fitted = Parallel(n_jobs=jobs, verbose=5)(
        delayed(_fit_one_reservoir_eb)(case_id, raw, p, grid_size, max_iter)
        for case_id, raw, p in tasks)
    index_by_case = dict(zip(frame.case_id, frame.index))
    for case_id, estimate, diag in fitted:
        idx = index_by_case[case_id]
        for k in range(2, 6):
            frame.at[idx, f"est__reservoir_eb_rho_k{k}"] = estimate.get(
                f"rho_k{k}", np.nan)
        diagnostics.append({"case_id": case_id, **diag})
    return frame, pd.DataFrame(diagnostics)


def analytical_records(cases):
    records = []
    for estimator in ("plugin", "occ_mle", "mask_mle"):
        records += _long_predictions(
            cases, estimator,
            [f"est__{estimator}_rho_k{k}" for k in range(2, 6)])
    records += _long_predictions(
        cases[cases.strategy == "uniform_event_reservoir"],
        "oracle_reservoir_ht_true_label",
        [f"oracle__reservoir_ht_true_label_rho_k{k}" for k in range(2, 6)],
        input_name="oracle", protocol="truth_diagnostic")
    records += _long_predictions(
        cases[cases.strategy == "uniform_event_reservoir"],
        "reservoir_factorized_temporal_eb",
        [f"est__reservoir_eb_rho_k{k}" for k in range(2, 6)],
        input_name="window_count_frequency",
        protocol="bernoulli_thinning_approximation")
    return records


def discovery_table(cases):
    rows = []
    for _, row in cases.iterrows():
        n, _, counts = parse_nmask_hist(row["input__nmask_exact_json"])
        rows.append({"case_id": row.case_id, "strategy": row.strategy,
                     "target_budget": row.get("target_budget", row["budget"]),
                     **discovery_diagnostics(n, counts)})
    return pd.DataFrame(rows)


def _feature_oracle_columns(frame):
    prefixes = ("occ__", "pat__", "crawl__", "est__")
    cols = [c for c in frame if c.startswith(prefixes)]
    cols += [c for c in frame if c.startswith("sample__")
             and pd.api.types.is_numeric_dtype(frame[c])]
    cols += metadata_only_columns(frame)
    return list(dict.fromkeys(cols))


def _fit_extra_trees(train, test, cols, name, fold, jobs):
    Xtr, useful = _useful_columns(train, cols)
    if not useful.any():
        return []
    Xte = test[cols].to_numpy(float)
    ytr = train[RHO_TARGETS].to_numpy(float)
    model = make_model(
        "extra_trees", jobs=jobs, n_outputs=len(RHO_TARGETS),
        n_features=int(useful.sum()), n_samples=len(train))
    model.fit(Xtr[:, useful], ytr)
    pred = np.asarray(model.predict(Xte[:, useful]))
    return [_prediction_records(
        test, target, name, name.removeprefix("ET_"), pred[:, j],
        fold=str(fold), protocol="group_kfold_profile")
        for j, target in enumerate(RHO_TARGETS)]


def learned_records(cases, folds, jobs, input_kind):
    records = []
    for strategy in sorted(cases.strategy.unique()):
        frame = cases[cases.strategy == strategy].reset_index(drop=True)
        groups = frame.group_id.astype(str).to_numpy()
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            continue
        splitter = GroupKFold(min(folds, n_groups))
        feature_sets = {
            "ET_metadata_only": metadata_only_columns(frame),
            "ET_prompt_parity": prompt_parity_columns(frame, input_kind),
            "ET_feature_oracle": _feature_oracle_columns(frame),
        }
        y = frame[RHO_TARGETS].to_numpy(float)
        for fold, (train_idx, test_idx) in enumerate(splitter.split(frame, y, groups)):
            train, test = frame.iloc[train_idx], frame.iloc[test_idx]
            for name, cols in feature_sets.items():
                if cols:
                    records += _fit_extra_trees(
                        train, test, cols, name, fold, jobs)
            print(f"[baseline {strategy}] fold {fold + 1}", flush=True)
    return records


def profile_table(predictions, target_budget=800):
    rho = predictions[predictions.target.isin(RHO_TARGETS)].copy()
    rho["target_budget"] = rho.case_id.str.extract(
        r"\|b(\d+)$")[0].astype(int)
    rho = rho[rho.target_budget == target_budget]
    if rho.empty:
        return pd.DataFrame()
    keys = ["case_id", "group_id", "strategy", "model", "input", "protocol"]
    wide_p = rho.pivot_table(index=keys, columns="target", values="prediction")
    wide_y = rho.pivot_table(index=keys, columns="target", values="y_true")
    if not set(RHO_TARGETS).issubset(wide_p.columns):
        return pd.DataFrame()
    common = wide_p.dropna(subset=RHO_TARGETS).index.intersection(
        wide_y.dropna(subset=RHO_TARGETS).index)
    if len(common) == 0:
        return pd.DataFrame()
    loss = np.abs(wide_p.loc[common, RHO_TARGETS].to_numpy() -
                  wide_y.loc[common, RHO_TARGETS].to_numpy()).mean(axis=1)
    result = pd.DataFrame([dict(zip(keys, idx)) for idx in common])
    result["profile_ae"] = loss
    by_group = (result.groupby(
        ["strategy", "model", "input", "protocol", "group_id"], as_index=False)
        .profile_ae.mean())
    return (by_group.groupby(
        ["strategy", "model", "input", "protocol"], as_index=False)
        .agg(n_groups=("group_id", "size"),
             profile_mae_group_macro=("profile_ae", "mean")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+",
                        default=["results/nonwalk_screen/panel32_cases.csv.gz"])
    parser.add_argument("--out-dir", default="results/nonwalk_screen/baselines")
    parser.add_argument("--config", default="config/benchmark_v21.yaml")
    parser.add_argument("--preset", default="v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=-1)
    parser.add_argument("--input-kind", default="window_counts_crawl_temporal")
    parser.add_argument("--eb-grid-size", type=int, default=28)
    parser.add_argument("--eb-max-iter", type=int, default=500)
    parser.add_argument("--eb-jobs", type=int, default=1)
    parser.add_argument("--max-cases", type=int, default=0)
    parser.add_argument(
        "--strategies", nargs="+", default=None,
        help="restrict the evaluation to these strategy names; the default "
             "keeps every strategy present in the case files")
    args = parser.parse_args()

    cases, paths = read_cases(args.cases)
    if args.strategies:
        wanted = set(args.strategies)
        missing = sorted(wanted - set(cases.strategy.unique()))
        if missing:
            raise SystemExit(f"strategies not present in the cases: {missing}")
        cases = cases[cases.strategy.isin(wanted)].reset_index(drop=True)
    if args.max_cases:
        cases = cases.head(args.max_cases).copy()
    print(f"{len(cases)} cases from {len(paths)} file(s)", flush=True)
    cases, eb_diagnostics = add_reservoir_eb(
        cases, grid_size=args.eb_grid_size, max_iter=args.eb_max_iter,
        jobs=args.eb_jobs)
    records = analytical_records(cases)
    records += learned_records(cases, args.folds, args.jobs, args.input_kind)
    predictions = pd.concat(records, ignore_index=True)
    cfg = load_evaluation_config(args.config, args.preset)
    metrics = compute_metrics(predictions, cfg)
    rankings = make_rankings(metrics)
    profiles = profile_table(predictions)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "predictions.csv.gz", index=False,
                       compression="gzip")
    metrics.to_csv(out / "metrics.csv", index=False)
    rankings.to_csv(out / "rankings.csv", index=False)
    profiles.to_csv(out / "profile_mae_b800.csv", index=False)
    discovery_table(cases).to_csv(out / "discovery_diagnostics.csv.gz",
                                  index=False, compression="gzip")
    eb_diagnostics.to_csv(out / "reservoir_eb_diagnostics.csv.gz",
                          index=False, compression="gzip")
    note = [
        "# Non-walk baseline screen", "",
        "`ET_prompt_parity` uses only prompt-recoverable quantities; "
        "`ET_feature_oracle` is an observable feature ceiling; "
        "`ET_metadata_only` mirrors the mandatory no-sample control.", "",
        "`reservoir_factorized_temporal_eb` is an observable active-mask/rate "
        "mixture with population mask-rate independence. It uses a Bernoulli-"
        "thinning approximation to the exact fixed-size reservoir; both "
        "assumptions must be reported.", "",
    ]
    if not profiles.empty:
        note += [profiles.sort_values(["strategy", "profile_mae_group_macro"])
                 .to_markdown(index=False, floatfmt=".4f"), ""]
    (out / "SUMMARY.md").write_text("\n".join(note))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
