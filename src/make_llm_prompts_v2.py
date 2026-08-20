#!/usr/bin/env python3
"""Build prompts.jsonl for the v2 LLM benchmark from the selected cases.

Design (maps the supervisor feedback onto concrete conditions):
* conditions: "hidden" (input + definitions only), "disclosed" (+ faithful
  mechanism description of the access model), "disclosed_examples"
  (+ three worked low/mid/high examples from leakage-free groups), and
  "method" (tool-use subset: disclosed + explicit method requirement; the
  code-execution harness variant is added on the runner side later);
* input: one base representation ("nw" exact (n,w) histogram, or "mask"
  exact (n, window-mask) histogram) plus any subset of three independently
  switchable add-on blocks -- "crawl" (observable crawl diagnostics),
  "temporal" (aggregate observed temporal patterns), "recent" (most recent
  anonymized event tuples). Two ablation designs are selectable:
  "ladder" reproduces the historical cumulative chain nw -> mask ->
  +crawl -> +temporal -> +recent, in which no single add-on is ever isolated;
  "ofat" varies one factor at a time around the `mask` reference cell
  (nw, mask, mask_crawl, mask_temporal, mask_recent, mask_all), so each cell
  answers about exactly one block. `mask_crawl` and `mask_all` render the same
  text as the ladder's `mask_crawl_full` and `mask_crawl_temporal_recent`,
  so those two ladder cells carry over;
* historical nine-key target contract: survival profile rho_k2..k5,
  mean occupancy, adjacent-window persistence, mean lifetime, and a 90%
  interval for rho_k2;
* no "think briefly", no thinking constraints of any kind; the final line
  must be one JSON object.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCHEMA_KEYS = ["rho_k2", "rho_k3", "rho_k4", "rho_k5", "mean_occupancy",
               "C_one_step", "lifetime_mean_over_T", "lo90", "hi90"]

DEFINITIONS = """DEFINITIONS
A temporal network was observed over a time period normalized to [0, 1).
The period is split into W = 5 equal windows: window 1 = [0.0, 0.2),
window 2 = [0.2, 0.4), ..., window 5 = [0.8, 1.0).
A node pair is "active in a window" if at least one interaction event between
the two nodes falls into that window. For every pair that has at least one
event in the FULL stream, let K = the number of distinct windows in which the
pair is active (K in 1..5).
Targets, always defined over ALL pairs of the full network with K >= 1
(not only the pairs you were shown):
  rho_k = share of pairs with K >= k, for k = 2, 3, 4, 5.
  mean_occupancy = the average of K/5 over these pairs.
  C_one_step = probability that a pair active in window w is also active in
  window w+1 (pooled over w = 1..4 and over all pairs).
  lifetime_mean_over_T = the average over these pairs of (time of the pair's
  last event minus time of its first event), in the normalized time units
  (a pair whose events all share one timestamp has lifetime 0).
You only saw a small sampled subset of the network (described below).
Estimate the values for the FULL network, correcting for how the sample was
collected if you can."""

INPUT_NW = """OBSERVED DATA: exact (n, w) histogram
Each observed pair is summarized by n = how many times the sampling process
recorded an event of this pair, and w = how many DISTINCT windows those
recorded events fell into. The JSON below maps "n,w" -> number of observed
pairs with that combination."""

INPUT_MASK = """OBSERVED DATA: exact (n, window-mask) histogram
Each observed pair is summarized by n = how many times the sampling process
recorded an event of this pair, and a 5-bit window mask of the windows in
which recorded events fell. The mask is written in hexadecimal; bit i
(value 2^i) set means the pair was observed in window i+1 (bit 0 = earliest
window). Example: mask 11 (hex) = binary 10001 = observed in windows 1 and 5.
The JSON below maps "n,mask" -> number of observed pairs with that
combination."""

INPUT_CRAWL = """SAMPLING DIAGNOSTICS (all computed only from the sampling log itself):"""

INPUT_TEMPORAL = """TEMPORAL PROFILE OF THE OBSERVED PAIRS (computed only from the
recorded events; times in the normalized [0,1) units; window indices 1..5):
  observed_adjacent_C: among observed pairs and windows 1..4, the share of
    (pair, window)-slots observed active whose NEXT window was also observed
    active.
  noncontiguous_pair_share: share of observed pairs whose observed windows do
    not form one contiguous run.
  mean_window_gap_first_to_last: average (last observed window - first
    observed window) per pair; 0 = single window.
  observed_lifetime_*: per-pair (last - first recorded time), only pairs with
    >= 2 recorded events.
  observed_IET_*: gaps between consecutive recorded events of the same pair.
  first/last_observed_window_distribution: share of observed pairs whose
    first/last observed event fell into window 1..5.
  event_share_per_window: share of all recorded events per window 1..5."""

INPUT_RECENT = """LAST RECORDED EVENT TUPLES
The last event tuples the sampling process recorded, as
[node_a, node_b, time], in RECORDING ORDER (the order the process traversed
them, which is not necessarily chronological). Node ids are anonymized by
order of first appearance."""

MECHANISM = {
    "time_agnostic_t": """SAMPLING MECHANISM
A random walker moved on the time-collapsed graph (all events of a pair merged
into one edge): from its current node it stepped to a uniformly random
neighbor. Each traversal of an edge revealed ONE uniformly random historical
event timestamp of that edge (repeated traversals draw independently, with
replacement). Every step cost 1 budget unit; n counts the traversals of a
pair. This access has full, unbiased access to each visited edge's history,
but visits edges in proportion to random-walk traffic, not uniformly.""",
    "time_respecting": """SAMPLING MECHANISM
A causal forward-in-time walker: at node x with clock tau it chose uniformly
among all events at x with time strictly greater than tau, moved across that
event, and set its clock to the event time. When no later event existed
(temporal dead end), it restarted at a uniformly random node with a uniformly
random clock in [0, 1). Every move AND every restart cost 1 budget unit; n
counts the traversals of a pair. Consequence: for each visited pair, only
events after the arrival clock could be recorded, so later windows are easier
to record than earlier ones, and long-lived pairs are reached differently
than short-lived ones.""",
    "recent_history_k20": """SAMPLING MECHANISM
A reverse-time retrieval walker: its clock started at the END of the
observation period. At node x it retrieved the up to 20 most recent events at
x strictly before its clock, picked ONE uniformly, moved across it, and set
its clock to that event's time. When no earlier event existed, it restarted
at a uniformly random node with the clock reset to the end. Every retrieval
AND every restart cost 1 budget unit; n counts the traversals of a pair.
Consequence: recording is biased toward recent activity and moves strictly
backwards in time between restarts.""",
}

TASK = """TASK
Estimate, for the FULL network:
  rho_k2, rho_k3, rho_k4, rho_k5 (each in [0, 1], non-increasing in k),
  mean_occupancy in [0, 1],
  C_one_step in [0, 1] (use null if it cannot be inferred from your input),
  lifetime_mean_over_T in [0, 1],
  and a 90% interval [lo90, hi90] for rho_k2.
Reason as much as you need. The LAST line of your reply must be exactly one
JSON object with the keys
  {"rho_k2": ..., "rho_k3": ..., "rho_k4": ..., "rho_k5": ...,
   "mean_occupancy": ..., "C_one_step": ..., "lifetime_mean_over_T": ...,
   "lo90": ..., "hi90": ...}
and nothing after it."""

METHOD_ADDON = """METHOD REQUIREMENT
You may design and carry out any estimation procedure, including deriving a
likelihood-based correction for the sampling mechanism described above and
working through the computation step by step. Directly before the final JSON
line, state in one short line which method you used (e.g. "method: plug-in",
"method: occupancy MLE", "method: custom correction")."""


def _clean(x):
    """NaN/Inf -> None recursively, so every emitted block is strict JSON."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not np.isfinite(x):
        return None
    return x


def jdump(d):
    return json.dumps(_clean(d), separators=(",", ":"), allow_nan=False)


def crawl_block(row):
    def fnum(x, nd=4):
        v = float(x)
        return round(v, nd) if np.isfinite(v) else None

    def fint(x):
        v = float(x)
        return int(v) if np.isfinite(v) else None

    d = {
        "budget_units_spent": fint(row["budget"]),
        "unique_nodes_visited": fint(row["observed_walk_nodes"]),
        "unique_pairs_traversed": fint(row["observed_walk_edges"]),
        "pairs_with_recorded_timestamps": fint(row["observed_timed_edges"]),
        "restart_fraction": fnum(row["crawl__restart_fraction"]),
        "pair_revisit_rate": fnum(row["crawl__edge_revisit_rate"]),
        "new_pair_rate_first10pct": fnum(row["crawl__discovery_010"]),
        "new_pair_rate_first50pct": fnum(row["crawl__discovery_050"]),
        "new_pair_rate_total": fnum(row["crawl__discovery_100"]),
        "node_visit_count_quantiles_q25_q50_q75_q90": [
            fnum(row[f"crawl__node_hits_q{q}"], 3) for q in (25, 50, 75, 90)],
        "pair_visit_count_quantiles_q25_q50_q75_q90": [
            fnum(row[f"crawl__edge_hits_q{q}"], 3) for q in (25, 50, 75, 90)],
        "observed_subgraph_degree_mean": fnum(
            row["crawl__observed_degree_mean"], 3),
        "observed_subgraph_degree_max": fint(
            row["crawl__observed_degree_max"]),
        "step_time_gap_quantiles_q25_q50_q75_q90": [
            fnum(row[f"crawl__dt_q{q}"]) for q in (25, 50, 75, 90)],
        "observed_time_span": fnum(row["crawl__observed_time_span"]),
        "first_node_revisit_position_frac": fnum(
            row["crawl__first_node_collision_frac"]),
    }
    return jdump(d)


def temporal_block(row):
    def r4(x):
        v = float(x)
        return round(v, 4) if np.isfinite(v) else None
    d = {
        "observed_adjacent_C": r4(row["pat__adjacent_observed_C"]),
        "noncontiguous_pair_share": r4(row["pat__noncontiguous_edge_share"]),
        "mean_window_gap_first_to_last": r4(row["pat__mean_mask_width"]),
        "observed_lifetime_mean": r4(row["pat__lifetime_mean"]),
        "observed_lifetime_q25_q50_q75_q90": [
            r4(row[f"pat__lifetime_q{q}"]) for q in (25, 50, 75, 90)],
        "observed_IET_mean": r4(row["pat__iet_mean"]),
        "observed_IET_q25_q50_q75_q90": [
            r4(row[f"pat__iet_q{q}"]) for q in (25, 50, 75, 90)],
        "first_observed_window_distribution": [
            r4(row[f"pat__first_w{w}"]) for w in range(5)],
        "last_observed_window_distribution": [
            r4(row[f"pat__last_w{w}"]) for w in range(5)],
        "event_share_per_window": [
            r4(row[f"pat__event_share_w{w}"]) for w in range(5)],
    }
    return jdump(d)


# The input is one base representation plus an independently switchable set of
# add-on blocks. Rendering order is fixed and never depends on the order the
# factors were requested, so a factor set names exactly one input text.
INPUT_BASES = ("nw", "mask")
INPUT_FACTORS = ("crawl", "temporal", "recent")

# Historical cumulative ladder. Kept so the frozen V2.1 prompts can still be
# regenerated byte-identically; it is a naming alias, not a second renderer.
INPUT_LADDER = ["nw", "mask", "mask_crawl_full", "mask_crawl_temporal",
                "mask_crawl_temporal_recent"]

# One-factor-at-a-time design: a reference cell plus one cell per add-on and
# one full cell. `mask_crawl` and `mask_all` are the same input text as the
# ladder's `mask_crawl_full` and `mask_crawl_temporal_recent`.
INPUT_OFAT = ["nw", "mask", "mask_crawl", "mask_temporal", "mask_recent",
              "mask_all"]

_LEGACY_KINDS = {
    "mask_crawl_full": ("mask", ("crawl",)),
    "mask_crawl_temporal": ("mask", ("crawl", "temporal")),
    "mask_crawl_temporal_recent": ("mask", ("crawl", "temporal", "recent")),
}


def parse_input_kind(kind):
    """Resolve an input_kind name into (base, ordered factor tuple)."""
    if kind in _LEGACY_KINDS:
        return _LEGACY_KINDS[kind]
    if kind in INPUT_BASES:
        return kind, ()
    base, _, rest = kind.partition("_")
    if base not in INPUT_BASES or not rest:
        raise ValueError(f"unknown input_kind: {kind}")
    if rest == "all":
        return base, INPUT_FACTORS
    requested = rest.split("_")
    unknown = [f for f in requested if f not in INPUT_FACTORS]
    if unknown or len(set(requested)) != len(requested):
        raise ValueError(f"unknown input_kind: {kind}")
    return base, tuple(f for f in INPUT_FACTORS if f in requested)


def canonical_input_kind(base, factors):
    """Inverse of parse_input_kind, using the short `_all` name when full."""
    factors = tuple(f for f in INPUT_FACTORS if f in set(factors))
    if not factors:
        return base
    if factors == INPUT_FACTORS:
        return f"{base}_all"
    return "_".join((base,) + factors)


def render_input(row, kind):
    base, factors = parse_input_kind(kind)
    parts = []
    if base == "nw":
        parts += [INPUT_NW, row["input__nw_exact_json"]]
    else:
        parts += [INPUT_MASK, row["input__nmask_exact_json"]]
    if "crawl" in factors:
        parts += [INPUT_CRAWL, crawl_block(row)]
    if "temporal" in factors:
        parts += [INPUT_TEMPORAL, temporal_block(row)]
    if "recent" in factors:
        parts += [INPUT_RECENT, row["input__recent_events_json"]]
    return "\n".join(parts)


def example_block(ex_rows, kind):
    out = ["WORKED EXAMPLES (different networks, same access mechanism and "
           "same input format; the correct full-network answers are shown; "
           "your final answer must additionally contain lo90 and hi90):"]
    for i, r in enumerate(ex_rows.itertuples(), 1):
        ans = {"rho_k2": round(float(r.rho_W5_k2), 4),
               "rho_k3": round(float(r.rho_W5_k3), 4),
               "rho_k4": round(float(r.rho_W5_k4), 4),
               "rho_k5": round(float(r.rho_W5_k5), 4),
               "mean_occupancy": round(float(r.mean_span_frac), 4),
               "C_one_step": round(float(r.C_one_step), 4),
               "lifetime_mean_over_T": round(
                   float(r.lifetime_mean_over_T), 4)}
        out.append(f"--- Example {i} ---")
        out.append(render_input(r._asdict() if hasattr(r, "_asdict")
                                else r, kind))
        out.append("Correct answer: " + jdump(ans))
    return "\n".join(out)


def build_prompt(row, condition, kind, examples):
    parts = [DEFINITIONS]
    if condition in ("disclosed", "disclosed_examples", "method"):
        parts.append(MECHANISM[row["strategy"]])
    else:
        parts.append("SAMPLING MECHANISM\nNot disclosed. The data below was "
                     "collected by a budget-limited sampling process on the "
                     "network; the process is not described.")
    if condition == "disclosed_examples":
        ex = examples[examples.strategy == row["strategy"]]
        parts.append(example_block(ex, kind))
    parts.append("NOW THE ACTUAL OBSERVATION TO EVALUATE:")
    parts.append(render_input(row, kind))
    if condition == "method":
        parts.append(METHOD_ADDON)
    parts.append(TASK)
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", default="results/llm_v2")
    ap.add_argument("--out", default=None,
                    help="default: <cases-dir>/prompts.jsonl")
    ap.add_argument("--main-conditions",
                    default="hidden,disclosed,disclosed_examples")
    ap.add_argument("--main-input", default="mask")
    ap.add_argument("--ablation-design", choices=["ladder", "ofat", "none"],
                    default="ladder",
                    help="ladder: historical cumulative chain (no add-on is "
                         "isolated). ofat: one factor at a time around the "
                         "--main-input reference cell. none: no input "
                         "ablation cells at all.")
    ap.add_argument("--ablation-inputs", default=None,
                    help="explicit comma-separated input kinds; overrides "
                         "--ablation-design")
    ap.add_argument("--ablation-condition", default="disclosed")
    ap.add_argument("--tool-input", default="mask_crawl_temporal")
    ap.add_argument("--emit-tool-subset", action="store_true", default=True)
    ap.add_argument("--no-tool-subset", dest="emit_tool_subset",
                    action="store_false")
    args = ap.parse_args()

    if args.ablation_inputs is not None:
        ablation_inputs = [k.strip() for k in args.ablation_inputs.split(",")
                           if k.strip()]
    elif args.ablation_design == "ladder":
        ablation_inputs = [k for k in INPUT_LADDER if k != args.main_input]
    elif args.ablation_design == "ofat":
        ablation_inputs = [k for k in INPUT_OFAT if k != args.main_input]
    else:
        ablation_inputs = []
    for kind in [args.main_input, args.tool_input] + ablation_inputs:
        parse_input_kind(kind)

    cdir = Path(args.cases_dir)
    cases = pd.read_csv(cdir / "llm_cases.csv")
    examples = pd.read_csv(cdir / "llm_examples.csv")
    out_path = Path(args.out or cdir / "prompts.jsonl")

    records, seen = [], set()

    def emit(row, condition, kind):
        key = (row["case_id"], condition, kind)
        if key in seen:
            return
        seen.add(key)
        prompt = build_prompt(row, condition, kind, examples)
        pid = hashlib.sha1("|".join(key).encode()).hexdigest()[:12]
        base, factors = parse_input_kind(kind)
        records.append({
            "id": pid, "prompt_id": pid, "case_id": row["case_id"],
            "condition": condition, "input_kind": kind,
            "input_base": base, "input_factors": ",".join(factors),
            "strategy": row["strategy"], "block_group": row["block_group"],
            "coverage_band": row["coverage_band"], "budget": int(row["budget"]),
            "prompt": prompt,
            # content identity: two differently named cells that render the
            # same text share this hash, so answers stay poolable across a
            # renaming and the frozen-prompt gate has something to pin.
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })

    for _, row in cases.iterrows():
        for cond in [c.strip() for c in args.main_conditions.split(",")]:
            emit(row, cond, args.main_input)
        if int(row.get("ablation_subset", 0)) == 1:
            for kind in ablation_inputs:
                emit(row, args.ablation_condition, kind)
        if args.emit_tool_subset and int(row.get("tool_use_subset", 0)) == 1:
            emit(row, "method", args.tool_input)

    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    lens = pd.Series([len(r["prompt"]) for r in records])
    per = pd.DataFrame(records).groupby(["condition", "input_kind"]).size()
    print(f"wrote {out_path}: {len(records)} prompts")
    print(per.to_string())
    print(f"prompt chars p50={int(lens.median())} p95={int(lens.quantile(.95))}"
          f" max={int(lens.max())} (~tokens: /4)")
    leak = [r for r in records
            if f'"{r["strategy"]}"' in r["prompt"] or "rho_W5" in r["prompt"]
            or "pat__" in r["prompt"] or "crawl__" in r["prompt"]
            or "est__" in r["prompt"]]
    assert not leak, "internal column names leaked into a prompt"


if __name__ == "__main__":
    main()
