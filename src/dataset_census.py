"""Current census columns over the preserved empirical parsing/window helpers.

No data are read at import time. Physical durations retain the source timestamp
unit; the registry does not yet contain a machine-readable unit for every file.
The inherited endpoint/EPS convention is documented in docs/REPRODUCIBILITY.md.
"""

from pathlib import Path

import pandas as pd

from census import census_row, count_nodes, load_registry, normalize, parse_events


def summarize_events(raw: pd.DataFrame, label: str = "") -> dict:
    """Census of a complete stream, with one vote per undirected dyad.

    Reuse the existing preprocessing exactly: self loops are dropped; duplicate
    event rows are retained. No sample is re-normalized here.
    """
    pairs = normalize(raw)
    if pairs.empty:
        raise ValueError("no non-self interaction events")
    old = census_row(pairs, label=label)
    counts = pairs.groupby("pair").size()
    # Count nodes on the same retained interaction population as E_full.
    retained = raw.loc[raw["u"] != raw["v"]]
    profile = {f"rho_{k}": old[f"rho_W5_k{k}"] for k in range(2, 6)}
    return {
        "label": label,
        "n_nodes": count_nodes(retained),
        "n_edges_full": int(len(counts)),
        "n_events": int(len(pairs)),
        "events_per_edge": float(len(pairs) / len(counts)),
        **{f"p_m_ge_{k}": float((counts >= k).mean()) for k in range(2, 6)},
        **profile,
        "mean_occupancy_derived": (1 + sum(profile.values())) / 5,
        "observation_start": float(pairs.t.min()),
        "observation_end": float(pairs.t.max()),
        "observation_span_source_units": old["horizon_T"],
        "window_duration_source_units": old["horizon_T"] / 5,
    }


def census_datasets(registry_path: Path, raw_dir: Path,
                    dataset_keys: list[str] | None = None) -> pd.DataFrame:
    """Read only locally present registered inputs; retain absent/error rows.

    Selecting a subset for inspection does not select the empirical panel.
    This function never downloads, modifies or writes an empirical input.
    """
    registry = load_registry(registry_path)
    keys = list(registry) if dataset_keys is None else list(dataset_keys)
    unknown = set(keys) - set(registry)
    if unknown:
        raise ValueError(f"unknown dataset keys: {sorted(unknown)}")
    rows = []
    for key in keys:
        spec = registry[key]
        path = raw_dir / spec["file"]
        row = {"dataset": key, "file": spec["file"], "label": spec["label"]}
        if not path.is_file():
            row["status"] = "absent"
        else:
            try:
                raw = parse_events(path, spec["format"])
                row.update(summarize_events(raw, spec["label"]))
                row["status"] = "ok"
            except (OSError, ValueError) as exc:
                row.update(status="error", error=str(exc))
        rows.append(row)
    return pd.DataFrame(rows)
