#!/usr/bin/env python3
"""Evaluation helpers implementing the frozen scoring rule.

`docs/TARGET_EVALUATION_FREEZE.md` asks for: parse only the final JSON object,
never clip or monotonically repair a raw prediction, treat a missing,
non-numeric, non-finite or out-of-range requested component as invalid, and
report both a failure-penalized metric (invalid component costs absolute loss
1) and complete-case metrics beside the validity rate.

Answer parsing and structural completeness come from `run_llm_v2` rather than
being reimplemented here, so that what a run counts as "done" and what an
evaluation counts as an answer cannot drift apart.

The historical `eval_llm_v2.py` clips predictions to [0,1] and predates the
failure penalty; it is a convenience view, not this rule.
"""

import csv
import glob as _glob
import json
import math
from pathlib import Path

from run_llm_v2 import PRED_KEYS, extract_last_json, is_complete_record

# Profile components in prediction order, and their truth columns.
PROFILE_PRED = ["rho_k2", "rho_k3", "rho_k4", "rho_k5"]
PROFILE_TRUTH = ["rho_W5_k2", "rho_W5_k3", "rho_W5_k4", "rho_W5_k5"]


def valid_unit(value):
    """Whether a raw prediction is usable: numeric, finite, inside [0,1]."""
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and 0.0 <= value <= 1.0)


def load_answers(patterns, root="."):
    """Merge answer files into {prompt_id: record}, last complete record wins.

    Accepts globs so sharded runs and their retry/escalation files collapse
    into one view. Escalation records replace earlier truncated attempts for
    the same prompt; they are not extra observations. Smoke files are skipped.
    A prompt with only incomplete records is still returned, so callers can
    tell "attempted and failed" from "never attempted" (absent entirely).
    """
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    paths = []
    for pattern in patterns:
        paths += [p for p in _glob.glob(str(Path(root) / pattern))
                  if "smoke" not in Path(p).name]
    complete, latest = {}, {}
    for path in sorted(paths):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                prompt_id = record.get("prompt_id")
                if prompt_id is None:
                    continue
                latest[prompt_id] = record
                if is_complete_record(record):
                    complete[prompt_id] = record
    return {pid: complete.get(pid, rec) for pid, rec in latest.items()}


def load_prompts(path):
    with open(path) as fh:
        return {d["prompt_id"]: d
                for d in (json.loads(l) for l in fh if l.strip())}


def load_cases(path):
    with open(path) as fh:
        return {row["case_id"]: row for row in csv.DictReader(fh)}


def cell_index(prompts):
    """{(condition, input_kind): {case_id: prompt_id}}.

    Each input format has its own prompt, so pairing across formats runs over
    case_id; prompt_id only identifies a cell.
    """
    cells = {}
    for prompt_id, meta in prompts.items():
        key = (meta["condition"], meta["input_kind"])
        cells.setdefault(key, {})[meta["case_id"]] = prompt_id
    return cells


def _mean(values):
    return sum(values) / len(values) if values else float("nan")


def score(answers, mapping, case_ids, cases):
    """Frozen-rule metrics for one cell restricted to `case_ids`.

    Only cases with a record are scored; cases the run never attempted are
    reported separately as `missing` instead of counting as failures.
    """
    scoped = [(c, mapping[c]) for c in case_ids
              if c in mapping and mapping[c] in answers]
    missing = sum(1 for c in case_ids
                  if c in mapping and mapping[c] not in answers)
    if not scoped:
        return None

    penalized, complete, rho2_abs, rho2_signed, c_abs, covered = [], [], [], [], [], []
    per_case, predicted_rho2, truth_rho2 = {}, [], []
    n_response = n_valid = n_c_valid = n_violation = 0

    for case_id, prompt_id in scoped:
        obj = extract_last_json(answers[prompt_id].get("answer"))
        truth = cases[case_id]
        if isinstance(obj, dict):
            n_response += 1

        losses, usable, values = [], [], []
        for pred_key, truth_key in zip(PROFILE_PRED, PROFILE_TRUTH):
            raw = obj.get(pred_key) if isinstance(obj, dict) else None
            if valid_unit(raw):
                losses.append(abs(float(raw) - float(truth[truth_key])))
                usable.append(True)
                values.append(float(raw))
            else:
                losses.append(1.0)
                usable.append(False)
                values.append(None)

        penalized.append(sum(losses) / len(losses))
        per_case[case_id] = penalized[-1]
        if all(usable):
            n_valid += 1
            complete.append(sum(losses) / len(losses))
            if any(values[i] < values[i + 1] - 1e-12 for i in range(3)):
                n_violation += 1
        if usable[0]:
            rho2_abs.append(losses[0])
            rho2_signed.append(values[0] - float(truth["rho_W5_k2"]))
            predicted_rho2.append(values[0])
            truth_rho2.append(float(truth["rho_W5_k2"]))

        c_raw = obj.get("C_one_step") if isinstance(obj, dict) else None
        if valid_unit(c_raw):
            n_c_valid += 1
            c_abs.append(abs(float(c_raw) - float(truth["C_one_step"])))

        lo = obj.get("lo90") if isinstance(obj, dict) else None
        hi = obj.get("hi90") if isinstance(obj, dict) else None
        if valid_unit(lo) and valid_unit(hi):
            covered.append(
                1.0 if float(lo) <= float(truth["rho_W5_k2"]) <= float(hi) else 0.0)

    n = len(scoped)
    return {
        "n": n, "missing": missing,
        "response_rate": n_response / n, "validity": n_valid / n,
        "profile_mae_penalized": _mean(penalized),
        "profile_mae_complete": _mean(complete),
        "rho2_mae": _mean(rho2_abs), "rho2_bias": _mean(rho2_signed),
        "rho2_spearman": spearman(predicted_rho2, truth_rho2),
        "c_mae": _mean(c_abs), "c_validity": n_c_valid / n,
        "cover90": _mean(covered),
        "violation_rate": n_violation / n_valid if n_valid else float("nan"),
        "per_case": per_case,
    }


def spearman(a, b):
    """Rank correlation with average ranks for ties."""
    n = len(a)
    if n < 3:
        return float("nan")

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            average = (i + j) / 2 + 1
            for k in range(i, j + 1):
                out[order[k]] = average
            i = j + 1
        return out

    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return num / (da * db) if da and db else float("nan")


def readoff_rho2(case_row):
    """Observed rho2 implied by the mask histogram: the naive read-off.

    Shares the sampling bias and window censoring of the observation, so it is
    the quantity a model reproduces when it reports what it saw instead of
    extrapolating.
    """
    histogram = json.loads(case_row["input__nmask_exact_json"])
    total = multi = 0
    for key, count in histogram.items():
        mask = int(key.split(",")[1], 16)
        total += count
        if bin(mask).count("1") >= 2:
            multi += count
    return multi / total if total else 0.0


def example_anchor(examples_path):
    """Per-strategy mean profile of the shown few-shot examples.

    Worth having as a reference row: examples are picked to span low/mid/high,
    so their mean sits near the population mean, and a constant prediction at
    that level is a surprisingly strong competitor whenever a method's ranking
    is weak.
    """
    with open(examples_path) as fh:
        rows = list(csv.DictReader(fh))
    anchors = {}
    for strategy in {r["strategy"] for r in rows}:
        group = [r for r in rows if r["strategy"] == strategy]
        anchors[strategy] = [
            sum(float(r[col]) for r in group) / len(group)
            for col in PROFILE_TRUTH]
    return anchors


__all__ = [
    "PRED_KEYS", "PROFILE_PRED", "PROFILE_TRUTH",
    "extract_last_json", "is_complete_record", "valid_unit",
    "load_answers", "load_prompts", "load_cases", "cell_index",
    "score", "spearman", "readoff_rho2", "example_anchor",
]
