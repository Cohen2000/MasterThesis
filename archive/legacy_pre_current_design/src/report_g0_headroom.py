#!/usr/bin/env python3
"""G0 headroom audit for mechanism-description conditions.

The frozen V2 case shards contain exact observed ``(n, mask)`` histograms but
predate dyad-level oracle labels.  This script therefore replays only the
requested frozen cases against separately regenerated event streams, verifies
that every replayed histogram equals the frozen histogram, and aggregates the
joint counts needed for ``P(observed mask | true K, arm)``.  Frozen inputs are
read-only; every generated artifact is written below ``--out-dir`` and the
requested gate report is written to ``--report``.

The empirical mechanism model contains mask zero (the dyad was not observed).
For an observed mask histogram, the label-assisted arm-likelihood estimator fits the mixture
among included dyads and divides out the arm- and K-specific inclusion
probability.  It is label-assisted and is reported only as a ceiling.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import glob
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from benchmark_features import edge_observations
from census import window_index
from evaluate_benchmark import input_columns, make_model, _useful_columns
from walks import build_index, run_walk


ARMS = ("time_agnostic_t", "time_respecting", "recent_history_k20")
KS = (1, 2, 3, 4, 5)
PROFILE_KS = (2, 3, 4, 5)
TRUTH = tuple(f"rho_W5_k{k}" for k in PROFILE_KS)
MASKS = tuple(range(32))


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def parse_nmask_hist(raw: str) -> Counter:
    """Parse the frozen ``n,hexmask -> count`` JSON representation."""
    obj = json.loads(raw)
    out = Counter()
    for key, count in obj.items():
        n_text, mask_text = key.split(",", 1)
        out[(int(n_text), int(mask_text, 16))] += int(count)
    return out


def mask_histogram(raw: str) -> np.ndarray:
    """Collapse the exact (n, mask) histogram to counts by nonzero mask."""
    out = np.zeros(32, dtype=float)
    for (_, mask), count in parse_nmask_hist(raw).items():
        out[mask] += count
    return out


def true_window_counts(idx, W: int = 5) -> dict[tuple[int, int], int]:
    out = {}
    for edge, times in idx.edge_times.items():
        wi = window_index(np.asarray(times, dtype=float), 0.0, idx.T / W, W)
        out[tuple(edge)] = int(len(np.unique(wi)))
    return out


def joint_counts_for_log(log: pd.DataFrame, idx, budget: int,
                         W: int = 5,
                         true_k: dict[tuple[int, int], int] | None = None) -> np.ndarray:
    """Count population dyads by true K and observed mask, including mask 0."""
    true_k = true_k if true_k is not None else true_window_counts(idx, W=W)
    counts = np.zeros((W + 1, 1 << W), dtype=np.int64)
    for k in true_k.values():
        counts[k, 0] += 1
    for obs in edge_observations(log, budget=budget, W=W, T=idx.T):
        edge = tuple(obs["edge"])
        if edge not in true_k:
            raise AssertionError(f"observed edge {edge} absent from population")
        k = true_k[edge]
        mask = int(obs["mask"])
        counts[k, 0] -= 1
        counts[k, mask] += 1
    if np.any(counts < 0):
        raise AssertionError("negative joint count after observation replay")
    if int(counts.sum()) != len(true_k):
        raise AssertionError("joint counts do not cover the population dyads")
    return counts


def _read_cases(specs: Iterable[str], columns: list[str] | None = None) -> pd.DataFrame:
    paths = []
    for spec in specs:
        found = sorted(glob.glob(spec))
        paths.extend(found or [spec])
    if not paths:
        raise FileNotFoundError("no case files matched")
    parts = [pd.read_csv(path, usecols=columns) for path in paths]
    return pd.concat(parts, ignore_index=True)


def _event_path(manifest_path: Path, rel: str) -> Path:
    path = Path(str(rel))
    return path if path.is_absolute() else manifest_path.parent / path


def _replay_instance(task):
    """Worker: replay all selected frozen cases for one event stream."""
    instance_id, group_id, event_path, rows, decay_scale = task
    events = pd.read_csv(event_path)
    idx = build_index(events, T=1.0, W=5)
    aggregate = {arm: np.zeros((6, 32), dtype=np.int64) for arm in ARMS}
    checked = 0
    for row in rows:
        if int(row["n_starts"]) != 1:
            raise AssertionError(f"G0 replay does not accept multistart case {row['case_id']}")
        log = run_walk(
            idx,
            str(row["base_strategy"]),
            max_budget=int(row["budget"]),
            seed=int(row["walk_rng_seed"]),
            decay_scale=float(decay_scale),
            history_k=int(row["history_k"]),
        )
        replay = Counter((int(x["n"]), int(x["mask"]))
                         for x in edge_observations(log, int(row["budget"]), W=5))
        frozen = parse_nmask_hist(str(row["input__nmask_exact_json"]))
        if replay != frozen:
            raise AssertionError(
                f"replay mismatch for {row['case_id']}: "
                f"replay={sum(replay.values())} frozen={sum(frozen.values())}")
        aggregate[str(row["strategy"])] += joint_counts_for_log(
            log, idx, int(row["budget"]), W=5)
        checked += 1
    records = []
    for arm, matrix in aggregate.items():
        for k in KS:
            for mask in MASKS:
                records.append({
                    "group_id": group_id,
                    "arm": arm,
                    "K": k,
                    "mask": mask,
                    "count": int(matrix[k, mask]),
                })
    return instance_id, checked, records


def collect_observation_counts(benchmark_cases: pd.DataFrame,
                               event_manifest_path: str | Path,
                               jobs: int = 1,
                               decay_scale: float = 0.1):
    """Replay frozen cases and return group-level joint count totals."""
    manifest_path = Path(event_manifest_path)
    manifest = pd.read_csv(manifest_path)
    event_lookup = {
        str(r.instance_id): _event_path(manifest_path, str(r.path))
        for r in manifest.itertuples(index=False)
    }
    missing = sorted(set(benchmark_cases.instance_id) - set(event_lookup))
    if missing:
        raise ValueError(f"regenerated manifest misses {len(missing)} instances")

    tasks = []
    for instance_id, frame in benchmark_cases.groupby("instance_id", sort=True):
        groups = frame["group_id"].astype(str).unique()
        if len(groups) != 1:
            raise AssertionError(f"instance {instance_id} has multiple groups")
        tasks.append((
            str(instance_id), str(groups[0]), str(event_lookup[str(instance_id)]),
            frame.to_dict("records"), float(decay_scale),
        ))

    totals = defaultdict(lambda: np.zeros((6, 32), dtype=np.int64))
    checked = 0
    if jobs == 1:
        iterator = map(_replay_instance, tasks)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        iterator = pool.map(_replay_instance, tasks, chunksize=1)
    try:
        for pos, (instance_id, n_checked, records) in enumerate(iterator, start=1):
            checked += n_checked
            for row in records:
                totals[(row["group_id"], row["arm"])][row["K"], row["mask"]] += row["count"]
            if pos % 25 == 0 or pos == len(tasks):
                print(f"[G0 replay] {pos}/{len(tasks)} instances; "
                      f"{checked}/{len(benchmark_cases)} frozen cases verified", flush=True)
    finally:
        if jobs != 1:
            pool.shutdown()

    rows = []
    for (group_id, arm), matrix in sorted(totals.items()):
        for k in KS:
            for mask in MASKS:
                rows.append({"group_id": group_id, "arm": arm, "K": k,
                             "mask": mask, "count": int(matrix[k, mask])})
    return pd.DataFrame(rows), checked


def observation_model(counts: pd.DataFrame,
                      excluded_groups: Iterable[str] = (),
                      arms: Iterable[str] = ARMS) -> dict[str, np.ndarray]:
    """Group-balanced empirical q[m|K,arm], including mask zero."""
    excluded = set(map(str, excluded_groups))
    use = counts[~counts["group_id"].astype(str).isin(excluded)].copy()
    models = {}
    for arm in tuple(arms):
        q = np.zeros((6, 32), dtype=float)
        sub = use[use["arm"] == arm]
        for k in KS:
            distributions = []
            for _, group in sub[sub["K"] == k].groupby("group_id"):
                values = np.zeros(32, dtype=float)
                values[group["mask"].to_numpy(int)] = group["count"].to_numpy(float)
                total = values.sum()
                if total > 0:
                    distributions.append(values / total)
            if not distributions:
                raise ValueError(f"no observation counts for {arm}, K={k}")
            q[k] = np.mean(distributions, axis=0)
            q[k] /= q[k].sum()
        models[arm] = q
    return models


def group_macro_k_weights(counts: pd.DataFrame, arm: str | None = None) -> np.ndarray:
    """K weights aligned with group-macro evaluation (one distribution/group)."""
    one_arm = counts[counts["arm"] == (arm or ARMS[0])]
    values = []
    for _, group in one_arm.groupby("group_id"):
        by_k = group.groupby("K")["count"].sum().reindex(KS, fill_value=0).to_numpy(float)
        if by_k.sum() > 0:
            values.append(by_k / by_k.sum())
    weights = np.mean(values, axis=0)
    return weights / weights.sum()


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p, float) - np.asarray(q, float)).sum())


def tv_table(models: dict[str, np.ndarray], k_weights: np.ndarray,
             conditional_observed: bool = False,
             arms: Iterable[str] | None = None) -> pd.DataFrame:
    rows = []
    arm_names = tuple(arms or models.keys())
    for i, arm_a in enumerate(arm_names):
        for arm_b in arm_names[i + 1:]:
            values = []
            for k in KS:
                a = models[arm_a][k].copy()
                b = models[arm_b][k].copy()
                if conditional_observed:
                    a, b = a[1:], b[1:]
                    a = a / a.sum(); b = b / b.sum()
                value = total_variation(a, b)
                values.append(value)
                rows.append({"arm_a": arm_a, "arm_b": arm_b,
                             "K": str(k), "tv": value})
            rows.append({"arm_a": arm_a, "arm_b": arm_b,
                         "K": "K-weighted", "tv": float(np.dot(k_weights, values))})
    return pd.DataFrame(rows)


def arm_likelihood_profile(observed_masks: np.ndarray, q: np.ndarray,
                           prior: float = 1e-6,
                           iters: int = 1000) -> np.ndarray:
    """MLE of population K proportions under empirical q[m|K], m=0..31.

    Only nonzero masks are supplied.  EM first fits alpha, the K mixture among
    included dyads, using q[m|K, observed].  Population pi follows from
    pi_k proportional to alpha_k / P(observed|K).
    """
    h = np.asarray(observed_masks, dtype=float)
    if h.shape != (32,) or h[0] != 0 or h[1:].sum() <= 0:
        raise ValueError("observed mask histogram must have nonzero mass only in masks 1..31")
    qk = np.asarray(q, dtype=float)[1:6, :]
    inclusion = qk[:, 1:].sum(axis=1)
    if np.any(inclusion <= 0):
        raise ValueError("every K state needs positive inclusion probability")
    likelihood = (qk[:, 1:] / inclusion[:, None]).T
    # Avoid an empirical zero turning a possible panel observation into an
    # impossible case.  The floor is far below the sampling resolution and is
    # renormalized within each K state.
    allowed = np.array([[int(mask).bit_count() <= k for k in KS]
                        for mask in range(1, 32)], dtype=bool)
    likelihood = np.where(allowed, np.maximum(likelihood, 1e-12), 0.0)
    likelihood /= likelihood.sum(axis=0, keepdims=True)
    counts = h[1:]
    alpha = np.ones(5, dtype=float) / 5.0
    for _ in range(iters):
        weighted = likelihood * alpha[None, :]
        denom = weighted.sum(axis=1, keepdims=True)
        active = counts > 0
        if np.any((denom[:, 0] <= 0) & active):
            raise ValueError("observation model assigns zero probability to an observed mask")
        denom[denom <= 0] = 1.0
        new = ((weighted / denom) * counts[:, None]).sum(axis=0) + prior
        new /= new.sum()
        if np.max(np.abs(new - alpha)) < 1e-12:
            alpha = new
            break
        alpha = new
    pi = alpha / inclusion
    pi /= pi.sum()
    return np.array([pi[k - 1:].sum() for k in PROFILE_KS], dtype=float)


# Compatibility for archived analyses.  The more precise name avoids
# conflating this label-assisted arm likelihood with the repository's uniform,
# mechanism-agnostic occupancy and mask MLEs.
design_aware_profile = arm_likelihood_profile


def metric_summary(frame: pd.DataFrame, pred_cols: Iterable[str]) -> dict:
    pred_cols = list(pred_cols)
    truth = frame[list(TRUTH)].to_numpy(float)
    pred = frame[pred_cols].to_numpy(float)
    valid = np.isfinite(pred).all(axis=1) & np.isfinite(truth).all(axis=1)
    if not valid.any():
        return {"n": 0, "profile_mae": np.nan, "rho2_bias": np.nan,
                "pooled_profile_mae": np.nan, "pooled_rho2_bias": np.nan}
    use = frame.loc[valid, ["group_id"]].copy()
    err = pred[valid] - truth[valid]
    use["profile_ae"] = np.abs(err).mean(axis=1)
    use["rho2_error"] = err[:, 0]
    grouped = use.groupby("group_id")[["profile_ae", "rho2_error"]].mean()
    return {
        "n": int(valid.sum()),
        "profile_mae": float(grouped["profile_ae"].mean()),
        "rho2_bias": float(grouped["rho2_error"].mean()),
        "pooled_profile_mae": float(use["profile_ae"].mean()),
        "pooled_rho2_bias": float(use["rho2_error"].mean()),
    }


def add_panel_logo_floor(panel: pd.DataFrame) -> list[str]:
    groups = panel["group_id"].astype(str).to_numpy()
    truth = panel[list(TRUTH)].to_numpy(float)
    columns = []
    for j, k in enumerate(PROFILE_KS):
        col = f"g0__floor_rho_k{k}"
        panel[col] = [float(truth[groups != group, j].mean()) for group in groups]
        columns.append(col)
    return columns


def extra_trees_transfer(train: pd.DataFrame, panel: pd.DataFrame,
                         jobs: int = -1,
                         arms: Iterable[str] = ARMS) -> list[str]:
    """Benchmark-trained profile ExtraTrees with each panel group held out."""
    out_cols = [f"g0__extra_trees_rho_k{k}" for k in PROFILE_KS]
    for col in out_cols:
        panel[col] = np.nan
    for arm in tuple(arms):
        tr_arm = train[train["strategy"] == arm].reset_index(drop=True)
        te_arm = panel[panel["strategy"] == arm]
        cols = input_columns(tr_arm, "combined")
        missing = [c for c in cols if c not in te_arm.columns]
        if missing:
            raise ValueError(f"panel misses {len(missing)} benchmark features")
        for held_group, test in te_arm.groupby("group_id"):
            fit = tr_arm[tr_arm["group_id"].astype(str) != str(held_group)]
            if fit.empty:
                raise ValueError(f"no ExtraTrees training data after holding out {held_group}")
            Xtr, useful = _useful_columns(fit, cols)
            model = make_model(
                "extra_trees", jobs=jobs, n_outputs=4,
                n_features=int(useful.sum()), n_samples=len(fit))
            model.fit(Xtr[:, useful], fit[list(TRUTH)].to_numpy(float))
            prediction = np.asarray(
                model.predict(test[cols].to_numpy(float)[:, useful]), dtype=float)
            panel.loc[test.index, out_cols] = prediction
            print(f"[G0 ExtraTrees] {arm}: held out {held_group}", flush=True)
    if panel[out_cols].isna().any().any():
        raise AssertionError("missing ExtraTrees transfer prediction")
    return out_cols


def _format_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def build_report(*, tv: pd.DataFrame, tv_observed: pd.DataFrame,
                 matrix: pd.DataFrame, penalties: pd.DataFrame,
                 ladder: pd.DataFrame, counts: pd.DataFrame,
                 benchmark_cases: pd.DataFrame, panel: pd.DataFrame,
                 verified_cases: int, mismatch_survives: bool,
                 mismatch_pair: tuple[str, str] | None,
                 flagged_arms: list[str], data_paths: dict[str, str]) -> str:
    tv_wide = tv.pivot(index=["arm_a", "arm_b"], columns="K", values="tv").reset_index()
    tv_obs_wide = tv_observed.pivot(
        index=["arm_a", "arm_b"], columns="K", values="tv").reset_index()
    pmae_matrix = matrix.pivot(index="sample_arm", columns="assumed_arm",
                               values="profile_mae").reset_index()
    bias_matrix = matrix.pivot(index="sample_arm", columns="assumed_arm",
                              values="rho2_bias").reset_index()
    ladder_show = ladder[["arm", "estimator", "profile_mae", "rho2_bias",
                          "pooled_profile_mae", "pooled_rho2_bias"]]
    median_penalty = float(penalties["profile_mae_penalty"].median())
    median_bias_shift = float(penalties["abs_rho2_bias_shift"].median())
    decision = "SURVIVES" if mismatch_survives else "CUT"
    pair_text = (f"`{mismatch_pair[0]}` ↔ `{mismatch_pair[1]}`"
                 if mismatch_pair else "none")
    flag_text = ", ".join(f"`{x}`" for x in flagged_arms) or "none"
    group_counts = counts.groupby("arm")["group_id"].nunique().to_dict()
    present_panel_groups = len(set(panel.group_id.astype(str)) &
                               set(benchmark_cases.group_id.astype(str)))
    return f"""# G0 headroom audit: does arm identity change the correction?

Prepared: **2026-08-31** for the September 2026 freeze  
Gate status: **G0 complete; blocked pending confirmation. No G1 work has begun.**

## Scope and provenance

This audit used no LLM calls and did not modify a frozen artifact. The frozen
V2 shards have no `oracle__` columns and retain only aggregated observed
histograms, which cannot identify the true K of individual dyads. The event
streams were therefore regenerated in a new directory from the archived V2
plan and the eight raw datasets present when the 745-instance archive was
built. The exact archived walk RNG seeds were replayed at budget 800.

- frozen benchmark cases: `{data_paths['benchmark_cases']}`
- frozen benchmark manifest: `{data_paths['benchmark_manifest']}`
- regenerated event manifest: `{data_paths['event_manifest']}`
- panel probe (read-only, walk seed 0): `{data_paths['panel_cases']}`
- compact joint counts and numerical tables: `{data_paths['out_dir']}`
- frozen cases selected: {len(benchmark_cases):,} ({benchmark_cases.instance_id.nunique()} instances, {benchmark_cases.group_id.nunique()} groups, four seeds, three arms)
- replayed `(n,mask)` histograms verified exactly: {verified_cases:,}/{len(benchmark_cases):,}
- observation-model groups by arm: {json.dumps(group_counts, sort_keys=True)}
- panel cases: {len(panel)} (32 graphs × 3 arms; {panel.group_id.nunique()} groups)
- panel groups also present in benchmark training universe and held out: {present_panel_groups}/{panel.group_id.nunique()}

The observation model includes mask `0`, meaning that a population dyad was
not observed. Probabilities are normalized within group first and then averaged
over groups, matching the primary group-macro estimand. The reported K-weighted
TV uses the corresponding group-macro distribution of true K.

## G0.1 — Empirical observation models

Total-variation distance for the full `P(m | K, arm)`, including mask 0:

{_format_table(tv_wide)}

Supplementary diagnostic conditional on the dyad being observed (`m != 0`):

{_format_table(tv_obs_wide)}

The first table is the requested population observation model. The conditional
table separates differences in the masks of discovered dyads from differences
in arm-specific inclusion probabilities.

## Parameterization check

`src/corrected_estimator.py:rho_mle` is parameterized only by `(n, w, W,
iters)` and uses one uniform-occupancy likelihood. `src/mask_estimator.py:mask_mle`
is parameterized only by `(n, mask, W, iters, prior, weights)` and likewise uses
one uniform-within-active-windows likelihood. Neither accepts an access arm or
an arm-specific propensity model. Evaluating those columns on a particular arm
therefore does **not** make them mechanism-parameterized.

G0.2 consequently uses the empirical label-assisted `P(m | K, arm)` above.
For each panel group and assumed arm, the model is refitted after excluding the
entire matching benchmark source/family. The MLE conditions on observing a
nonzero mask and divides out the learned `P(observed | K, arm)`.

## G0.2 — Wrong-mechanism estimator matrix

Group-macro ProfileMAE (rows: sample-producing arm; columns: assumed arm):

{_format_table(pmae_matrix)}

Group-macro signed bias on `rho_2`:

{_format_table(bias_matrix)}

Off-diagonal penalties relative to the correct diagonal assumption:

{_format_table(penalties)}

**This matrix uses true dyad labels to fit each observation model and is a
ceiling, not a label-free estimator.** It measures whether arm identity offers
potentially usable headroom; it is not a proposed deployable correction.

## G0.3 — Estimator ladder on the panel

All rows use the same 96 seed-0 panel cases at budget 800. The mean floor is
leave-one-panel-group-out. ExtraTrees uses the frozen benchmark V2 cases at
budget 800 and the observable `combined = occ + pat + crawl` feature set; for
every panel prediction its complete source/family is excluded from training.
The occupancy and mask MLE rows are the existing uniform likelihoods described
above, not newly mechanism-parameterized estimators.

{_format_table(ladder_show)}

Naive-to-mask-MLE ProfileMAE headroom by arm:

{_format_table(ladder.attrs['headroom'])}

## G0.4 — Mechanical decisions

- Median off-diagonal ProfileMAE penalty: **{median_penalty:.4f}** (survival threshold: > 0.02).
- Median absolute off-diagonal shift in signed `rho_2` bias: **{median_bias_shift:.4f}** (survival threshold: > 0.03).
- `mismatched`: **{decision}**.
- Prespecified bidirectional mismatch pair if retained: **{pair_text}**.
- Arms with naive-to-mask-MLE headroom below 0.02: **{flag_text}**.

Conditions surviving G0: **`hidden`, `mechanism`, and
`mechanism_direction`**{'; also `mismatched` for the pair ' + pair_text if mismatch_survives else '; `mismatched` is cut'}.

## What this shows, and what it does not

The arm comparison is estimated from the archived benchmark population and
evaluated on a separate fixed panel with matching source/family exclusions. It
directly tests whether substituting one learned arm likelihood for another can
matter at the planned budget.

It does not prove that an LLM can recover the correction. The empirical model
uses labels, collapses the exact `(n,mask)` sample to mask frequencies, and
averages over heterogeneous coverage, topology and graph size within an arm.
Fixed-budget walk observations are dependent, whereas this MLE uses their
marginal empirical distribution. The 3×3 matrix is therefore a deliberately
optimistic diagnostic with model misspecification, not an information-theoretic
bound. G0 also uses an existing seed-0 development-panel sample; G2, if
authorized later, is still responsible for the fresh final sample.

## Decision

**STOP at G0. Await explicit confirmation before writing the G1 prompt
contract or generating any final prompt.**
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--benchmark-cases", nargs="+", required=True)
    ap.add_argument("--benchmark-manifest", required=True)
    ap.add_argument("--event-manifest", required=True,
                    help="manifest of separately regenerated archived events")
    ap.add_argument("--panel-cases", required=True)
    ap.add_argument("--panel-seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--out-dir", default="results/g0_headroom_2026_09")
    ap.add_argument("--report", default="docs/HEADROOM_2026-09.md")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--model-jobs", type=int, default=-1)
    ap.add_argument("--decay-scale", type=float, default=0.1)
    ap.add_argument("--rebuild-counts", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_path = out_dir / "observation_counts_by_group.csv.gz"

    index_cols = ["case_id", "instance_id", "group_id", "strategy",
                  "base_strategy", "history_k", "n_starts", "walk_seed",
                  "walk_rng_seed", "budget", "input__nmask_exact_json"]
    benchmark_index = _read_cases(args.benchmark_cases, columns=index_cols)
    benchmark_index = benchmark_index[
        (benchmark_index["budget"] == args.budget) &
        benchmark_index["strategy"].isin(ARMS)
    ].reset_index(drop=True)
    if benchmark_index["case_id"].duplicated().any():
        raise ValueError("duplicate frozen benchmark case_id")
    expected = 745 * len(ARMS) * 4
    if len(benchmark_index) != expected:
        raise ValueError(f"expected {expected} frozen G0 cases, found {len(benchmark_index)}")

    frozen_manifest = pd.read_csv(args.benchmark_manifest).sort_values("instance_id").reset_index(drop=True)
    regenerated_manifest = pd.read_csv(args.event_manifest).sort_values("instance_id").reset_index(drop=True)
    if list(frozen_manifest.instance_id) != list(regenerated_manifest.instance_id):
        raise ValueError("regenerated manifest instance IDs differ from frozen archive")
    compare_cols = [c for c in frozen_manifest.columns
                    if c in regenerated_manifest.columns and c != "path"]
    for col in compare_cols:
        a, b = frozen_manifest[col], regenerated_manifest[col]
        if pd.api.types.is_numeric_dtype(a):
            if not np.allclose(a.to_numpy(float), b.to_numpy(float),
                               equal_nan=True, rtol=1e-10, atol=1e-12):
                raise ValueError(f"regenerated manifest differs in numeric column {col}")
        else:
            if not a.fillna("").astype(str).equals(b.fillna("").astype(str)):
                raise ValueError(f"regenerated manifest differs in column {col}")

    if counts_path.exists() and not args.rebuild_counts:
        counts = pd.read_csv(counts_path)
        verified_cases = len(benchmark_index)
        print(f"using cached {counts_path}", flush=True)
    else:
        counts, verified_cases = collect_observation_counts(
            benchmark_index, args.event_manifest, jobs=args.jobs,
            decay_scale=args.decay_scale)
        counts.to_csv(counts_path, index=False, compression="gzip")
        print(f"wrote {counts_path}", flush=True)

    models = observation_model(counts)
    k_weights = group_macro_k_weights(counts)
    tv = tv_table(models, k_weights, conditional_observed=False)
    tv_observed = tv_table(models, k_weights, conditional_observed=True)
    tv.to_csv(out_dir / "tv_full.csv", index=False)
    tv_observed.to_csv(out_dir / "tv_conditional_observed.csv", index=False)

    panel = pd.read_csv(args.panel_cases)
    panel = panel[(panel["budget"] == args.budget) &
                  (panel["walk_seed"] == args.panel_seed) &
                  panel["strategy"].isin(ARMS)].copy().reset_index(drop=True)
    if len(panel) != 32 * len(ARMS):
        raise ValueError(f"expected 96 panel cases, found {len(panel)}")

    cross_models = {}
    for group_id in sorted(panel.group_id.astype(str).unique()):
        cross_models[group_id] = observation_model(counts, excluded_groups=[group_id])

    matrix_records = []
    prediction_records = []
    for sample_arm in ARMS:
        sample = panel[panel["strategy"] == sample_arm]
        for assumed_arm in ARMS:
            pred_cols = [f"g0__design_rho_k{k}" for k in PROFILE_KS]
            scored = sample.copy()
            predictions = []
            for _, row in scored.iterrows():
                hist = mask_histogram(str(row["input__nmask_exact_json"]))
                profile = arm_likelihood_profile(
                    hist, cross_models[str(row["group_id"])][assumed_arm])
                predictions.append(profile)
            scored[pred_cols] = np.asarray(predictions)
            summary = metric_summary(scored, pred_cols)
            matrix_records.append({"sample_arm": sample_arm,
                                   "assumed_arm": assumed_arm, **summary})
            for idx, row in scored.iterrows():
                prediction_records.append({
                    "case_id": row["case_id"], "group_id": row["group_id"],
                    "sample_arm": sample_arm, "assumed_arm": assumed_arm,
                    **{f"prediction_rho_k{k}": row[f"g0__design_rho_k{k}"]
                       for k in PROFILE_KS},
                })
    matrix = pd.DataFrame(matrix_records)
    matrix.to_csv(out_dir / "wrong_mechanism_matrix.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(
        out_dir / "wrong_mechanism_predictions.csv.gz", index=False,
        compression="gzip")

    diagonal = matrix[matrix["sample_arm"] == matrix["assumed_arm"]].set_index("sample_arm")
    penalty_rows = []
    for _, row in matrix[matrix["sample_arm"] != matrix["assumed_arm"]].iterrows():
        base = diagonal.loc[row["sample_arm"]]
        penalty_rows.append({
            "sample_arm": row["sample_arm"],
            "assumed_arm": row["assumed_arm"],
            "profile_mae_penalty": float(row["profile_mae"] - base["profile_mae"]),
            "rho2_bias_shift": float(row["rho2_bias"] - base["rho2_bias"]),
            "abs_rho2_bias_shift": float(abs(row["rho2_bias"] - base["rho2_bias"])),
        })
    penalties = pd.DataFrame(penalty_rows)
    penalties.to_csv(out_dir / "wrong_mechanism_penalties.csv", index=False)

    median_pmae_penalty = float(penalties["profile_mae_penalty"].median())
    median_abs_bias_shift = float(penalties["abs_rho2_bias_shift"].median())
    mismatch_survives = (median_pmae_penalty > 0.02 or
                         median_abs_bias_shift > 0.03)
    mismatch_pair = None
    if mismatch_survives:
        criterion = ("profile_mae_penalty" if median_pmae_penalty > 0.02
                     else "abs_rho2_bias_shift")
        best = penalties.loc[penalties[criterion].idxmax()]
        mismatch_pair = (str(best["sample_arm"]), str(best["assumed_arm"]))

    # Load only budget-800 cases for the supervised transfer reference.
    all_columns = pd.read_csv(sorted(glob.glob(args.benchmark_cases[0]))[0], nrows=0).columns
    transfer_cols = [c for c in all_columns if c.startswith(("occ__", "pat__", "crawl__"))]
    transfer_usecols = ["instance_id", "group_id", "strategy", "budget", *TRUTH,
                        *transfer_cols]
    benchmark_train = _read_cases(args.benchmark_cases, columns=transfer_usecols)
    benchmark_train = benchmark_train[
        (benchmark_train["budget"] == args.budget) &
        benchmark_train["strategy"].isin(ARMS)
    ].reset_index(drop=True)
    floor_cols = add_panel_logo_floor(panel)
    et_cols = extra_trees_transfer(benchmark_train, panel, jobs=args.model_jobs)

    methods = {
        "mean floor (panel LOGO)": floor_cols,
        "naive read-off": [f"est__plugin_rho_k{k}" for k in PROFILE_KS],
        "occupancy MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__occ_mle_rho_k{k}" for k in PROFILE_KS],
        "mask MLE (uniform; censoring-aware, mechanism-agnostic)":
            [f"est__mask_mle_rho_k{k}" for k in PROFILE_KS],
        "supervised ExtraTrees (benchmark transfer)": et_cols,
    }
    ladder_rows = []
    for arm in ARMS:
        sample = panel[panel["strategy"] == arm]
        for name, columns in methods.items():
            ladder_rows.append({"arm": arm, "estimator": name,
                                **metric_summary(sample, columns)})
    ladder = pd.DataFrame(ladder_rows)
    naive = ladder[ladder.estimator == "naive read-off"].set_index("arm")
    mask = ladder[ladder.estimator.str.startswith("mask MLE")].set_index("arm")
    headroom = pd.DataFrame({
        "arm": list(ARMS),
        "naive_profile_mae": [naive.loc[a, "profile_mae"] for a in ARMS],
        "mask_mle_profile_mae": [mask.loc[a, "profile_mae"] for a in ARMS],
        "naive_to_mask_headroom": [naive.loc[a, "profile_mae"] - mask.loc[a, "profile_mae"]
                                    for a in ARMS],
    })
    ladder.attrs["headroom"] = headroom
    flagged_arms = headroom.loc[
        headroom["naive_to_mask_headroom"] < 0.02, "arm"].tolist()
    ladder.to_csv(out_dir / "estimator_ladder.csv", index=False)
    headroom.to_csv(out_dir / "censoring_headroom.csv", index=False)

    report = build_report(
        tv=tv, tv_observed=tv_observed, matrix=matrix, penalties=penalties,
        ladder=ladder, counts=counts, benchmark_cases=benchmark_index,
        panel=panel, verified_cases=verified_cases,
        mismatch_survives=mismatch_survives, mismatch_pair=mismatch_pair,
        flagged_arms=flagged_arms,
        data_paths={
            "benchmark_cases": " ".join(args.benchmark_cases),
            "benchmark_manifest": args.benchmark_manifest,
            "event_manifest": args.event_manifest,
            "panel_cases": args.panel_cases,
            "out_dir": str(out_dir),
        })
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(f"wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
