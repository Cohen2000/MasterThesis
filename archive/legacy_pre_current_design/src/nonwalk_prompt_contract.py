#!/usr/bin/env python3
"""Prompt/no-sample contract and prompt-parity feature definitions."""

import json

import numpy as np

from make_llm_prompts_v2 import DEFINITIONS, TASK, render_input


WINDOW_COUNTS = """EXACT PER-WINDOW EVENT-COUNT FREQUENCIES
For every observed pair, the vector [x1,x2,x3,x4,x5] gives the number of
recorded events in each of the five windows. The JSON maps the comma-separated
count vector to the number of observed pairs with that vector. This is the
same sampled information as the event records, aggregated without loss of
per-pair window counts."""


MECHANISMS = {
    "uniform_event_reservoir": """SAMPLING MECHANISM
Exactly B distinct interaction records were selected uniformly without
replacement from the complete event stream. Event inclusion is uniform, but a
pair with many full-stream events is more likely to appear at least once than
a pair with few events.""",
    "time_prefix_events": """SAMPLING MECHANISM
The observation contains the first B interaction records in chronological
order. The requested targets concern the full observation horizon, including
the unobserved future. This is a full-horizon extrapolation task, not a random
sample from the horizon.""",
    "time_random_window_events": """SAMPLING MECHANISM
The observation is a randomly anchored contiguous chronological block of B
interaction records.
The requested targets concern the complete horizon outside the block as well
as the observed block; this is a full-horizon extrapolation task.""",
    "node_panel_full_history": """SAMPLING MECHANISM
A uniformly random panel of active nodes was selected. Every interaction from
the complete observation horizon whose two endpoints both belong to the panel
was retained. Panel size was calibrated so that the expected number of retained
events was close to B; the realized event count can therefore differ from B.
Dyad histories inside the panel are complete, but dyads touching nodes outside
the panel are entirely absent.""",
    "ego_recent_k1": """SAMPLING MECHANISM
Nodes were queried at the end of the observation period. Each query returned
the single most recent incident interaction. The newest event expanded a FIFO
snowball frontier; random restarts selected another active node when needed.
Duplicate event records were removed.""",
    "ego_recent_k5": """SAMPLING MECHANISM
Nodes were queried at the end of the observation period. Each query returned
up to its 5 most recent incident interactions. Only the newest event expanded
a FIFO snowball frontier; random restarts selected another active node when
needed. Duplicate event records were removed. The final query response may
have been truncated when the total event budget B was reached.""",
    "ego_recent_k20": """SAMPLING MECHANISM
Nodes were queried at the end of the observation period. Each query returned
up to its 20 most recent incident interactions. Only the newest event expanded
a FIFO snowball frontier; random restarts selected another active node when
needed. Duplicate event records were removed. The final query response may
have been truncated when the total event budget B was reached.""",
    "ego_recent_kall": """SAMPLING MECHANISM
Nodes were queried at the end of the observation period. Each query returned
its complete incident interaction history over the observation horizon. Only
the newest event expanded a FIFO snowball frontier; random restarts selected
another active node when needed. Duplicate event records were removed. The
final query response may have been truncated when the total event budget B was
reached.""",
}


def public_metadata(row) -> dict:
    """The identical non-event metadata supplied in full and no-sample arms."""
    return {
        "W": 5,
        "sampling_strategy": str(row["strategy"]),
        "requested_event_budget": int(row["target_budget"]),
        "realized_unique_event_records": int(row["budget"]),
        "declared_number_of_nodes": int(row["n_nodes_true"]),
    }


def render_nonwalk_prompt(row, include_sample: bool,
                          input_kind: str = "window_counts_crawl_temporal") -> str:
    """Render paired full-sample or metadata-only prompts."""
    strategy = str(row["strategy"])
    if strategy not in MECHANISMS:
        raise KeyError(f"no LLM mechanism contract for {strategy!r}")
    metadata = json.dumps(public_metadata(row), separators=(",", ":"))
    parts = [DEFINITIONS, MECHANISMS[strategy],
             "PUBLIC NON-EVENT METADATA\n" + metadata]
    if include_sample:
        if input_kind == "window_counts_crawl_temporal":
            rendered = (render_input(row, "mask_crawl_temporal") + "\n" +
                        WINDOW_COUNTS + "\n" +
                        str(row["input__window_counts_exact_json"]))
        else:
            rendered = render_input(row, input_kind)
        parts += ["SAMPLED OBSERVATION", rendered]
    else:
        parts += ["SAMPLED OBSERVATION\nNo sampled events, histograms, observed "
                  "node/pair counts, or sample-derived diagnostics are "
                  "provided in this control condition."]
    parts.append(TASK)
    return "\n\n".join(parts)


def prompt_parity_columns(frame, input_kind="mask_crawl_temporal"):
    """Columns mechanically recoverable from the corresponding prompt.

    This intentionally excludes raw metadata labels, truth diagnostics,
    oracle columns, and engineered quantities absent from the rendered JSON.
    """
    cols = [c for c in frame.columns if c.startswith("occ__")]
    if input_kind != "nw":
        mask_visible = (
            "pat__mask_", "pat__n", "pat__first_w", "pat__last_w",
            "pat__adjacent_observed_C", "pat__noncontiguous_edge_share",
            "pat__mean_mask_width",
        )
        cols += [c for c in frame.columns if c.startswith(mask_visible)]
    if input_kind in {"mask_crawl_full", "mask_crawl_temporal",
                      "mask_crawl_temporal_recent",
                      "window_counts_crawl_temporal"}:
        crawl_exact = {
            "crawl__log_budget", "crawl__step_fraction",
            "crawl__restart_fraction", "crawl__log_unique_nodes",
            "crawl__log_unique_edges", "crawl__edge_revisit_rate",
            "crawl__discovery_010", "crawl__discovery_050",
            "crawl__discovery_100", "crawl__observed_degree_mean",
            "crawl__observed_degree_max", "crawl__observed_time_span",
            "crawl__first_node_collision_frac",
        }
        crawl_exact.update(
            f"crawl__{stem}_q{q}" for stem in ("edge_hits", "node_hits", "dt")
            for q in (25, 50, 75, 90))
        cols += [c for c in frame.columns if c in crawl_exact]
    if input_kind in {"mask_crawl_temporal", "mask_crawl_temporal_recent",
                      "window_counts_crawl_temporal"}:
        # The temporal JSON serializes every aggregate pat__ field, including
        # event-window shares and IET/lifetime summaries.
        cols += [c for c in frame.columns if c.startswith("pat__")]
    if input_kind == "window_counts_crawl_temporal":
        # This condition contains the same mask/crawl/temporal blocks plus the
        # exact per-window count-vector frequency table.
        cols += [c for c in frame.columns if c.startswith(
            ("pat__", "wcnt__"))]
    # Public metadata is present verbatim in both full and no-sample prompts.
    cols += [c for c in ("target_budget", "budget", "n_nodes_true")
             if c in frame.columns]
    return list(dict.fromkeys(cols))


def metadata_only_columns(frame):
    return [c for c in ("target_budget", "budget", "n_nodes_true")
            if c in frame.columns]


def assert_prompt_metadata_parity(full_prompt: str, control_prompt: str,
                                  row) -> None:
    metadata = json.dumps(public_metadata(row), separators=(",", ":"))
    if metadata not in full_prompt or metadata not in control_prompt:
        raise AssertionError("public metadata differs between paired prompts")
    forbidden = [str(row.get("input__nmask_exact_json", "")),
                 str(row.get("input__nw_exact_json", "")),
                 str(row.get("input__window_counts_exact_json", ""))]
    for value in forbidden:
        if value and value != "nan" and value in control_prompt:
            raise AssertionError("sample histogram leaked into no-sample prompt")
