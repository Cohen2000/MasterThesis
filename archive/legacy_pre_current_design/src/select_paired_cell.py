#!/usr/bin/env python3
"""Select prompt IDs for a new prompt cell, paired to an existing answer file.

A second cell for a model that already ran one is only interpretable when both
cells cover the SAME cases: otherwise the contrast between conditions is
confounded with case difficulty. The case set is therefore taken from what a
run actually answered completely, not from the prompt list.

``--limit`` subsamples when a full paired cell does not fit a usage budget.
The subsample is stratified round-robin over (strategy, coverage_band) and
fully deterministic, so a smaller cell stays balanced across the design
instead of drifting towards whichever cases happen to sort first.

``--prefer-from`` puts the cases another run also answered at the front of
that ordering. Two screens sized with the same stratification then produce
nested subsets, which makes their cells comparable to each other and not only
each to its own baseline.
"""

import argparse
import csv
import glob
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

from run_llm_v2 import is_complete_record


def complete_cases(patterns, condition=None, input_kind=None):
    """case_ids whose newest record in these files is structurally complete."""
    latest = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        prompt_id = record["prompt_id"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    latest[prompt_id] = record
    out = set()
    for record in latest.values():
        if condition and record.get("condition") != condition:
            continue
        if input_kind and record.get("input_kind") != input_kind:
            continue
        if is_complete_record(record):
            out.add(record["case_id"])
    return out


def stratified(case_ids, strata, limit):
    """Deterministic round-robin over strata, then over cases within a stratum.

    Cases whose stratum is unknown share one bucket rather than being dropped:
    a missing metadata column must not silently shrink the cell.
    """
    buckets = defaultdict(list)
    for case_id in sorted(case_ids):
        buckets[strata.get(case_id, ("?", "?"))].append(case_id)
    order = []
    keys = sorted(buckets)
    index = 0
    while len(order) < len(case_ids):
        progressed = False
        for key in keys:
            if index < len(buckets[key]):
                order.append(buckets[key][index])
                progressed = True
        if not progressed:
            break
        index += 1
    return order[:limit] if limit else order


def select(prompts_path, cases_path, answer_patterns, condition, input_kind,
           base_condition, base_input_kind, limit, prefer_patterns):
    base = complete_cases(answer_patterns, base_condition, base_input_kind)
    if not base:
        raise SystemExit("no complete records in the baseline answer file(s)")

    strata = {}
    with open(cases_path) as fh:
        for row in csv.DictReader(fh):
            strata[row["case_id"]] = (row.get("strategy", "?"),
                                      row.get("coverage_band", "?"))

    with open(prompts_path) as fh:
        target = OrderedDict()
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if (row["condition"] == condition
                    and row["input_kind"] == input_kind
                    and row["case_id"] in base):
                target[row["case_id"]] = row["prompt_id"]
    missing = base - set(target)
    if missing:
        raise SystemExit(
            f"{len(missing)} answered cases have no {condition}/{input_kind} "
            f"prompt; the cell cannot be paired")

    preferred = set()
    if prefer_patterns:
        preferred = complete_cases(prefer_patterns, base_condition,
                                   base_input_kind) & set(target)
    head = stratified(preferred, strata, 0)
    tail = stratified(set(target) - preferred, strata, 0)
    chosen = (head + tail)[:limit] if limit else head + tail
    return [target[c] for c in chosen], len(base), len(preferred)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--cases", required=True,
                    help="llm_cases.csv, for the stratification columns")
    ap.add_argument("--answers", action="append", required=True,
                    help="baseline answer-file glob; may be repeated")
    ap.add_argument("--condition", required=True, help="target cell condition")
    ap.add_argument("--input-kind", required=True, help="target cell input kind")
    ap.add_argument("--base-condition", default="disclosed")
    ap.add_argument("--base-input-kind", default="mask")
    ap.add_argument("--limit", type=int, default=0,
                    help="subsample to N cases (0 = all)")
    ap.add_argument("--prefer-from", action="append", default=None,
                    help="answer-file glob whose cases are selected first")
    ap.add_argument("--min-cases", type=int, default=10,
                    help="fail rather than run a cell below this size")
    ap.add_argument("--report", action="store_true",
                    help="write the selection summary to stderr")
    args = ap.parse_args()

    ids, n_base, n_pref = select(
        args.prompts, args.cases, args.answers, args.condition,
        args.input_kind, args.base_condition, args.base_input_kind,
        args.limit, args.prefer_from)
    if len(ids) < args.min_cases:
        raise SystemExit(f"only {len(ids)} cases selected, minimum is "
                         f"{args.min_cases}")
    if args.report:
        import sys
        print(f"baseline complete cases: {n_base}; preferred pool: {n_pref}; "
              f"selected: {len(ids)}", file=sys.stderr)
    print(",".join(ids))


if __name__ == "__main__":
    main()
