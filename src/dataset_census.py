"""Audited empirical census, W sensitivity and bounded timing feasibility.

Only local files are read. Each retained row is one event, with no deduplication,
edge-weight expansion, sampling, largest-component restriction or downloads.
"""

from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
import math
from pathlib import Path
import zlib

import numpy as np
import pandas as pd

from census import count_nodes, load_registry, normalize, window_index

WINDOWS = tuple(range(2, 21))
THRESHOLDS = ("0.2", "0.4", "0.6", "0.8", "1.0")
TARGETS = ("0.15", "0.55")
INTEGER_COLUMNS = ("data_rows", "invalid_rows_removed", "header_rows_skipped",
                   "comment_or_blank_rows_skipped", "n_nodes", "n_edges_full", "n_events",
                   "self_events_removed", "duplicate_dyad_timestamp_events")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_audited(path: Path, fmt: dict, audit: dict | None = None):
    """Parse declared columns and count exclusions without guessing a layout."""
    audit = audit or {}
    columns = fmt["columns"]
    delimiter = fmt.get("delimiter", "whitespace")
    skip = fmt.get("skiprows", 0)
    comments = tuple(fmt.get("comment", "#%"))
    widths = Counter()
    counts = dict(data_rows=0, invalid_rows_removed=0, header_rows_skipped=0,
                  comment_or_blank_rows_skipped=0)
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as stream:
        for line_number, line in enumerate(stream):
            text = line.strip()
            if line_number < skip:
                if line_number == 0 and audit.get("expected_header"):
                    if text != audit["expected_header"]:
                        raise ValueError("header differs from reviewed column layout")
                counts["header_rows_skipped"] += 1
                continue
            if not text or text.startswith(comments):
                counts["comment_or_blank_rows_skipped"] += 1
                continue
            counts["data_rows"] += 1
            fields = text.split() if delimiter == "whitespace" else text.split(delimiter)
            widths[len(fields)] += 1
            try:
                if len(fields) <= max(columns.values()):
                    raise ValueError("too few columns")
                if audit.get("expected_fields") and len(fields) != audit["expected_fields"]:
                    raise ValueError("row width differs from reviewed layout")
                u, v = fields[columns["u"]].strip(), fields[columns["v"]].strip()
                t = float(fields[columns["t"]])
                if not u or not v or not math.isfinite(t):
                    raise ValueError("empty endpoint or non-finite timestamp")
                if fmt.get("bipartite", False):
                    u, v = "u:" + u, "i:" + v
                rows.append((u, v, t))
            except (ValueError, IndexError):
                counts["invalid_rows_removed"] += 1
    counts["observed_row_widths"] = json.dumps(dict(sorted(widths.items())))
    raw = pd.DataFrame(rows, columns=["u", "v", "t"])
    counts["raw_min_timestamp"] = float(raw.t.min()) if len(raw) else None
    counts["raw_max_timestamp"] = float(raw.t.max()) if len(raw) else None
    return raw, counts


def prepare_complete(raw):
    """Canonical undirected dyads after finite-time and self-event exclusion."""
    x = raw[["u", "v", "t"]].copy()
    missing = x.u.isna() | x.v.isna()
    x["t"] = pd.to_numeric(x.t, errors="coerce")
    missing |= ~np.isfinite(x.t.to_numpy(float))
    x = x.loc[~missing].copy()
    x["u"], x["v"] = x.u.astype(str), x.v.astype(str)
    invalid = int(missing.sum())
    self_events = int((x.u == x.v).sum())
    x = x.loc[x.u != x.v].reset_index(drop=True)
    if x.empty:
        raise ValueError("no valid non-self interaction events")
    pairs = normalize(x)
    times = pairs.t.to_numpy(float)
    lo, hi = float(times.min()), float(times.max())
    if hi <= lo:
        raise ValueError("degenerate complete observation horizon")
    pair_ids = pairs.pair.to_numpy(np.int64)
    event_counts = np.bincount(pair_ids)
    # Affine normalization; x=1 is assigned to the final window, explicitly.
    normalized_times = (times - lo) / (hi - lo)
    # Retained as events (no deduplication); reported because they inflate m_e.
    duplicates = len(times) - int(distinct_timestamp_counts(pair_ids, times).sum())
    return x, pair_ids, normalized_times, event_counts, {
        "invalid_rows_removed": invalid, "self_events_removed": self_events,
        "duplicate_dyad_timestamp_events": duplicates,
        "observation_start": lo, "observation_end": hi,
        "observation_span_source_units": hi - lo,
    }


def distinct_timestamp_counts(pair_ids, times):
    """Distinct timestamps per dyad; inputs are sorted by (pair, time)."""
    first = np.ones(len(pair_ids), dtype=bool)
    first[1:] = (pair_ids[1:] != pair_ids[:-1]) | (times[1:] != times[:-1])
    return np.bincount(pair_ids[first], minlength=int(pair_ids.max()) + 1)


def occupancy_counts(pair_ids, normalized_times, W):
    """K per dyad using the shared EPS boundary guard and final endpoint rule."""
    if W < 2:
        raise ValueError("W must be at least two")
    windows = window_index(normalized_times, 0.0, 1.0 / W, W)
    distinct = np.unique(pair_ids * W + windows)
    return np.bincount(distinct // W, minlength=int(pair_ids.max()) + 1)


def survival(K, W):
    histogram = np.bincount(K, minlength=W + 1)
    return np.cumsum(histogram[::-1])[::-1] / len(K)


def relative_cutoff(tau, W):
    """Use exact decimal fractions so floating multiplication cannot shift ceil."""
    threshold = Fraction(str(tau))
    if not 0 < threshold <= 1:
        raise ValueError("tau must lie in (0,1]")
    return math.ceil(threshold * W)


def _main_summary(x, pair_ids, times, counts, quality, label):
    K = occupancy_counts(pair_ids, times, 5)
    rho = survival(K, 5)
    return {
        "label": label, "n_nodes": count_nodes(x),
        "n_edges_full": int(len(counts)), "n_events": int(len(x)),
        "events_per_edge": float(len(x) / len(counts)),
        "fraction_m_eq_1": float(np.mean(counts == 1)),
        **{f"p_m_ge_{k}": float(np.mean(counts >= k)) for k in range(2, 6)},
        **{f"rho_{k}": float(rho[k]) for k in range(2, 6)},
        **{f"q{k}": float(np.mean(K == k)) for k in range(1, 6)},
        "mean_occupancy_derived": float((1 + rho[2:].sum()) / 5),
        **quality,
        "window_duration_source_units": quality["observation_span_source_units"] / 5,
    }


def summarize_events(raw: pd.DataFrame, label: str = "") -> dict:
    """Current W=5 census over a complete event table, with equal dyad weights."""
    return _main_summary(*prepare_complete(raw), label)


def window_sensitivity(dataset, pair_ids, times, counts):
    """Long table: one row per rho, occupancy summary or relative threshold."""
    rows = []
    for W in WINDOWS:
        K = occupancy_counts(pair_ids, times, W)
        rho = survival(K, W)
        common = {"dataset": dataset, "W": W, "n_edges_full": len(K)}
        for k in range(2, W + 1):
            rows.append({**common, "statistic": "rho", "k": k, "tau": None,
                         "value": float(rho[k]),
                         "event_count_upper_bound": float(np.mean(counts >= k))})
        for name, value in (("mean_P", np.mean(K / W)), ("median_P", np.median(K / W))):
            rows.append({**common, "statistic": name, "k": None, "tau": None,
                         "value": float(value), "event_count_upper_bound": None})
        for tau in THRESHOLDS:
            k = relative_cutoff(tau, W)
            rows.append({**common, "statistic": "relative_survival", "k": k,
                         "tau": float(tau), "value": float(rho[k]),
                         "event_count_upper_bound": float(np.mean(counts >= k))})
    return rows


def count_feasibility(dataset, counts, empirical_rho_2=None, distinct_counts=None):
    """Necessary event-count bound only; it does not show the allocator can hit a target."""
    upper = float(np.mean(counts >= 2))
    row = {"dataset": dataset, "n_edges_full": int(len(counts)),
           "empirical_rho_2": empirical_rho_2, "p_m_ge_2": upper,
           **{f"rho_{k}_upper_bound": float(np.mean(counts >= k))
              for k in range(2, 6)}}
    for text in TARGETS:
        target = Fraction(text)
        label = text.replace(".", "_")
        row[f"target_{label}_count_feasible"] = (
            int(np.sum(counts >= 2)) * target.denominator >= len(counts) * target.numerator)
        row[f"target_{label}_margin"] = upper - float(target)
    # Sensitivity only: the bound if same-dyad same-timestamp rows were one event.
    row["p_distinct_timestamps_ge_2"] = (
        None if distinct_counts is None else float(np.mean(distinct_counts >= 2)))
    return row


def realized_feasibility(dataset, raw, seed_base=20260911):
    """Exactly one call per target to retained functions; save no event streams."""
    from benchmark_generators import family_from_events, normalize_event_stream
    from generator import make_instance

    original = normalize_event_stream(raw)
    family = family_from_events(original, name=dataset, W=5, T=1.0)
    # Keep the retained generator's empirical timestamp pool mode explicit.
    family.timestamps = "empirical"
    expected_counts = original.groupby(["u", "v"]).size().sort_index()
    expected_nodes = set(original.u) | set(original.v)
    rows = []
    for target in TARGETS:
        seed = zlib.crc32(f"{seed_base}|census_twin|{dataset}|{target}".encode()) & 0xFFFFFFFF
        row = {"dataset": dataset, "requested_rho_2": float(target), "seed": seed,
               "W": 5, "span_layout": "contiguous", "timestamp_mode": "empirical",
               "hub_bias": False}
        try:
            instance = make_instance(family, float(target), seed=seed,
                                     hub_bias=False, span_layout="contiguous")
            observed = instance.events.groupby(["u", "v"]).size().sort_index()
            x, pairs, times, counts, quality = prepare_complete(instance.events)
            achieved = float(np.mean(occupancy_counts(pairs, times, 5) >= 2))
            row.update(status="ok", achieved_rho_2=achieved,
                       absolute_deviation=abs(achieved - float(target)),
                       nodes_preserved=(set(x.u.astype(int)) | set(x.v.astype(int))) == expected_nodes,
                       topology_preserved=expected_counts.index.equals(observed.index),
                       per_dyad_counts_preserved=expected_counts.equals(observed),
                       timestamps_preserved=np.array_equal(np.sort(original.t),
                                                           np.sort(instance.events.t)),
                       allocation_deviations=instance.deviations, error="")
        except (ValueError, RuntimeError, AssertionError) as exc:
            row.update(status="error", error=str(exc))
        rows.append(row)
    return rows


def compute_census(registry_path, raw_dir, dataset_keys=None, realized=False, progress=None):
    registry = load_registry(Path(registry_path))
    keys = list(registry) if dataset_keys is None else list(dataset_keys)
    if set(keys) - set(registry):
        raise ValueError(f"unknown dataset keys: {sorted(set(keys) - set(registry))}")
    main, sensitivity, feasible, twins = [], [], [], []
    for key in keys:
        spec = registry[key]
        audit = spec.get("census_audit", {})
        path = Path(raw_dir) / spec["file"]
        warnings = audit.get("warnings", "")
        row = {"dataset": key, "file": spec["file"], "label": spec["label"],
               "domain": spec.get("domain", "unknown"),
               "bipartite": "yes" if spec.get("format", {}).get("bipartite") else "no",
               "originally_directed": audit.get("originally_directed", "unknown"),
               "timestamp_unit": audit.get("timestamp_unit", "unknown"),
               "parser_note": audit.get("parser_note", ""),
               "warnings": warnings,
               "metadata_source": spec.get("page_url", "") if audit.get("metadata_source") == "registry_page"
                                  else audit.get("metadata_source", ""),
               "time_unit_evidence": audit.get("time_unit_evidence", "")}
        if not path.is_file():
            row.update(status="absent", warnings="Registered but unavailable; not downloaded.")
            main.append(row)
            continue
        if progress:
            progress(key, "census")
        try:
            raw, parser_quality = parse_audited(path, spec["format"], audit)
            x, pairs, times, counts, quality = prepare_complete(raw)
            quality["invalid_rows_removed"] += parser_quality["invalid_rows_removed"]
            row.update(parser_quality)
            row.update(_main_summary(x, pairs, times, counts, quality, spec["label"]))
            row["raw_sha256"] = sha256_file(path)
            row["status"] = "ok"
            if row["invalid_rows_removed"]:
                row["warnings"] += " Invalid data rows excluded; see removal count."
            seconds = row["timestamp_unit"] == "seconds"
            row["physical_observation_span_seconds"] = (
                quality["observation_span_source_units"] if seconds else "UNKNOWN")
            row["physical_W5_window_seconds"] = (
                quality["observation_span_source_units"] / 5 if seconds else "UNKNOWN")
            sensitivity.extend(window_sensitivity(key, pairs, times, counts))
            feasible.append(count_feasibility(key, counts, row["rho_2"],
                                              distinct_timestamp_counts(pairs, times)))
            if realized:
                if progress:
                    progress(key, "one low/high timing attempt")
                twins.extend(realized_feasibility(key, x))
        except (OSError, ValueError) as exc:
            row.update(status="error", warnings=(warnings + " " + str(exc)).strip())
        main.append(row)
    tables = [pd.DataFrame(rows) for rows in (main, sensitivity, feasible, twins)]
    # Nullable integers keep counts integral in CSV despite absent/error rows.
    for column in INTEGER_COLUMNS:
        if column in tables[0]:
            tables[0][column] = tables[0][column].astype("Int64")
    for column in ("k",):
        if column in tables[1]:
            tables[1][column] = tables[1][column].astype("Int64")
    return tuple(tables)


def census_datasets(registry_path: Path, raw_dir: Path,
                    dataset_keys: list[str] | None = None) -> pd.DataFrame:
    """Compatibility entry point returning the main table without timing variants."""
    return compute_census(registry_path, raw_dir, dataset_keys)[0]
