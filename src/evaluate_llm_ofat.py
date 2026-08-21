#!/usr/bin/env python3
"""Evaluate the paired six-cell input OFAT, including repeated generations.

The primary view is replicate 1 on the same 36 cases for every cell.  This is
the direct extension of the frozen suite.  A second view pools only replicate
numbers for which all six cells were attempted, averaging repeats within a
case before averaging over cases.  Thus a case remains the statistical unit;
three stochastic replies are not misreported as three independent graphs.

Invalid/malformed replies receive absolute profile loss 1 under the frozen
rule.  Missing jobs are reported as missing, not converted into model errors.
Paired deltas and bootstrap intervals are computed over the 36 cases.
"""

import argparse
import csv
import hashlib
import math
from pathlib import Path
import statistics

import numpy as np

from llm_eval_frozen import (PROFILE_PRED, PROFILE_TRUTH, extract_last_json,
                             load_answers, load_cases, valid_unit)


ROOT = Path(__file__).resolve().parents[1]
OFAT = ROOT / "results/llm_v21_ofat"
CELLS = ["nw", "mask", "mask_crawl", "mask_temporal", "mask_recent",
         "mask_all"]

RUNS = [
    ("codex-5.6 notools/high", [
        "results/codex_screen_snapshot/answers_codex-gpt-5.6-sol_notools_high*.jsonl",
        "results/llm_v21_ofat/answers_codex*.jsonl",
    ]),
    ("gemini-3.1-lite minimal", [
        "results/llm_v21/answers_gemini-3.1-flash-lite_minimal.jsonl",
        "results/llm_v21_ofat/answers_gemini-3.1-flash-lite_minimal.shard*.jsonl",
    ]),
    ("deepseek-v4-flash-0731", [
        "results/llm_v21_ofat/answers_deepseek-v4-flash-0731_nothink.shard*.jsonl",
    ]),
    ("qwen3.6-27b nothink", [
        "results/llm_v21/cluster_snapshot/answers_qwen36-27b_nothink.shard*.jsonl",
        "results/llm_v21/cluster_snapshot/answers_qwen36-27b_nothink_16k.shard*.jsonl",
        "results/llm_v21/cluster_snapshot/answers_qwen36-27b_nothink_32k.shard*.jsonl",
        "results/llm_v21_ofat/answers_ofat_qwen36_nothink*.jsonl",
    ]),
    ("qwen3.6-27b think", [
        "results/llm_v21/cluster_snapshot/answers_qwen36-27b_think.shard*.jsonl",
        "results/llm_v21/cluster_snapshot/answers_qwen36-27b_think_64k.shard*.jsonl",
        "results/llm_v21/cluster_snapshot/answers_qwen36-27b_think_128k.shard*.jsonl",
        "results/llm_v21_ofat/answers_ofat_qwen36_think*.jsonl",
    ]),
]


def load_prompt_rows(path):
    import json
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def record_score(record, truth):
    obj = extract_last_json(record.get("answer")) if record else None
    losses = []
    usable = []
    for pred, target in zip(PROFILE_PRED, PROFILE_TRUTH):
        raw = obj.get(pred) if isinstance(obj, dict) else None
        good = valid_unit(raw)
        usable.append(good)
        losses.append(abs(float(raw) - float(truth[target])) if good else 1.0)
    return {
        "responded": isinstance(obj, dict),
        "valid": all(usable),
        "penalized": statistics.mean(losses),
        "complete": statistics.mean(losses) if all(usable) else None,
    }


def index_prompts(rows):
    out = {}
    for row in rows:
        out[(row["input_kind"], int(row["rep"]), row["case_id"])] = row["prompt_id"]
    return out


def attempted_per_cell_rep(answers, prompt_index, cases, rep):
    return {
        cell: sum(prompt_index.get((cell, rep, case)) in answers for case in cases)
        for cell in CELLS
    }


def balanced_reps(answers, prompt_index, cases):
    reps = sorted({key[1] for key in prompt_index})
    return [rep for rep in reps
            if min(attempted_per_cell_rep(
                answers, prompt_index, cases, rep).values()) == len(cases)]


def cell_metrics(answers, prompt_index, truths, cases, cell, reps):
    per_case = {}
    complete_per_case = {}
    attempted = responded = valid = 0
    expected = len(cases) * len(reps)
    for case in cases:
        scores = []
        complete = []
        for rep in reps:
            pid = prompt_index.get((cell, rep, case))
            record = answers.get(pid) if pid else None
            if record is None:
                continue
            attempted += 1
            s = record_score(record, truths[case])
            responded += int(s["responded"])
            valid += int(s["valid"])
            scores.append(s["penalized"])
            if s["complete"] is not None:
                complete.append(s["complete"])
        if scores:
            per_case[case] = statistics.mean(scores)
        if complete:
            complete_per_case[case] = statistics.mean(complete)
    return {
        "expected": expected,
        "attempted": attempted,
        "response_rate": responded / attempted if attempted else math.nan,
        "validity": valid / attempted if attempted else math.nan,
        "penalized": (statistics.mean(per_case.values())
                      if per_case else math.nan),
        "complete": (statistics.mean(complete_per_case.values())
                     if complete_per_case else math.nan),
        "per_case": per_case,
    }


def paired_delta(other, base, n_boot, seed_text):
    shared = sorted(set(other) & set(base))
    values = np.asarray([other[c] - base[c] for c in shared], dtype=float)
    if not len(values):
        return math.nan, math.nan, math.nan, 0
    delta = float(values.mean())
    if len(values) < 2 or n_boot <= 0:
        return delta, math.nan, math.nan, len(values)
    seed = int(hashlib.sha1(seed_text.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    # Chunking avoids a large temporary allocation if --bootstrap is raised.
    draws = []
    left = n_boot
    while left:
        size = min(left, 2000)
        idx = rng.integers(0, len(values), size=(size, len(values)))
        draws.append(values[idx].mean(axis=1))
        left -= size
    boot = np.concatenate(draws)
    lo, hi = np.quantile(boot, [0.025, 0.975])
    return delta, float(lo), float(hi), len(values)


def repeat_stability(answers, prompt_index, cases, cell, reps=(1, 2, 3)):
    """Within-case answer SD, used only as a diagnostic for extra replies."""
    spreads = []
    repeated_cases = set()
    for case in cases:
        by_key = {key: [] for key in PROFILE_PRED}
        for rep in reps:
            pid = prompt_index.get((cell, rep, case))
            obj = extract_last_json(answers.get(pid, {}).get("answer")) if pid else None
            if not isinstance(obj, dict):
                continue
            for key in PROFILE_PRED:
                if valid_unit(obj.get(key)):
                    by_key[key].append(float(obj[key]))
        for values in by_key.values():
            if len(values) >= 2:
                spreads.append(statistics.pstdev(values))
                repeated_cases.add(case)
    if not spreads:
        return None
    return {"cell": cell, "repeated_cases": len(repeated_cases),
            "case_components": len(spreads),
            "mean_prediction_sd": statistics.mean(spreads),
            "median_prediction_sd": statistics.median(spreads)}


def fmt(value, digits=4):
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def evaluate_view(label, reps, answers, prompt_index, truths, cases, n_boot):
    cells = {cell: cell_metrics(answers, prompt_index, truths, cases, cell, reps)
             for cell in CELLS}
    base = cells["mask"]["per_case"]
    rows = []
    for cell in CELLS:
        m = cells[cell]
        delta, lo, hi, paired_n = paired_delta(
            m["per_case"], base, n_boot, f"{label}|{reps}|{cell}")
        rows.append({
            "view": label, "reps": ",".join(map(str, reps)), "cell": cell,
            **{k: m[k] for k in ("expected", "attempted", "response_rate",
                                  "validity", "penalized", "complete")},
            "delta_vs_mask": delta, "delta_ci_lo": lo, "delta_ci_hi": hi,
            "paired_cases": paired_n,
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompts", default=str(OFAT / "prompts_ofat.jsonl"))
    ap.add_argument("--cases", default=str(ROOT / "results/llm_v2/llm_cases.csv"))
    ap.add_argument("--out-dir", default=str(OFAT / "eval"))
    ap.add_argument("--bootstrap", type=int, default=10000)
    args = ap.parse_args()

    prompt_rows = load_prompt_rows(args.prompts)
    prompt_index = index_prompts(prompt_rows)
    truths = load_cases(args.cases)
    cases = sorted({r["case_id"] for r in prompt_rows})
    if len(cases) != 36:
        raise SystemExit(f"expected 36 paired cases, found {len(cases)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    stability_rows = []
    status = []
    for run, patterns in RUNS:
        answers = load_answers(patterns, root=ROOT)
        if label.startswith("codex-"):
            # The notools treatment excludes structurally valid answers that
            # nevertheless used a hosted tool. The runner also retries these.
            answers = {pid: rec for pid, rec in answers.items()
                       if not rec.get("n_tool_events")}
        matched = {pid: rec for pid, rec in answers.items()
                   if pid in {r["prompt_id"] for r in prompt_rows}}
        reps = balanced_reps(matched, prompt_index, cases)
        status.append((run, len(matched), reps))
        for cell in CELLS:
            diag = repeat_stability(matched, prompt_index, cases, cell)
            if diag:
                stability_rows.append({"run": run, **diag})
        if matched:
            primary = evaluate_view("rep1", [1], matched, prompt_index,
                                    truths, cases, args.bootstrap)
            for row in primary:
                row["run"] = run
            all_rows.extend(primary)
        if reps and reps != [1]:
            pooled = evaluate_view("balanced_reps", reps, matched, prompt_index,
                                   truths, cases, args.bootstrap)
            for row in pooled:
                row["run"] = run
            all_rows.extend(pooled)

    fieldnames = ["run", "view", "reps", "cell", "expected", "attempted",
                  "response_rate", "validity", "penalized", "complete",
                  "delta_vs_mask", "delta_ci_lo", "delta_ci_hi",
                  "paired_cases"]
    with open(out_dir / "ofat_metrics.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    stability_fields = ["run", "cell", "repeated_cases", "case_components",
                        "mean_prediction_sd", "median_prediction_sd"]
    with open(out_dir / "repeat_stability.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=stability_fields)
        writer.writeheader()
        writer.writerows(stability_rows)

    lines = [
        "# LLM input OFAT",
        "",
        "Six input cells on the same 36 cases. Lower ProfileMAE is better. ",
        "`pen` applies loss 1 to an invalid profile component; `cc` scores ",
        "only structurally valid profiles. Deltas are paired against `mask`; ",
        "negative means the input variant helped. Bootstrap intervals resample ",
        "cases, not repeated replies.",
        "",
        "## Coverage",
        "",
        "| run | matched answers | fully attempted reps across all cells |",
        "|---|---:|---|",
    ]
    for run, matched, reps in status:
        lines.append(f"| {run} | {matched} | {','.join(map(str, reps)) or '-'} |")

    for view, title in (("rep1", "Primary paired comparison (replicate 1)"),
                        ("balanced_reps", "All repeated, case-balanced comparison")):
        subset = [r for r in all_rows if r["view"] == view]
        if not subset:
            continue
        lines += ["", f"## {title}", "",
                  "| run | cell | attempted | valid | pen | cc | delta vs mask (95% CI) | paired cases |",
                  "|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in subset:
            delta = fmt(r["delta_vs_mask"])
            ci = (f"{delta} [{fmt(r['delta_ci_lo'])}, {fmt(r['delta_ci_hi'])}]"
                  if math.isfinite(r["delta_ci_lo"]) else delta)
            lines.append(
                f"| {r['run']} | {r['cell']} | {r['attempted']}/{r['expected']} | "
                f"{fmt(r['validity'], 2)} | {fmt(r['penalized'])} | "
                f"{fmt(r['complete'])} | {ci} | {r['paired_cases']} |")

    if stability_rows:
        lines += ["", "## Extra-reply stability diagnostic", "",
                  "This does not enter the OFAT factor estimate. It only shows "
                  "how much repeated replies to the newly generated cells move.",
                  "", "| run | cell | repeated cases | mean prediction SD | median prediction SD |",
                  "|---|---|---:|---:|---:|"]
        for r in stability_rows:
            lines.append(f"| {r['run']} | {r['cell']} | {r['repeated_cases']} | "
                         f"{fmt(r['mean_prediction_sd'])} | "
                         f"{fmt(r['median_prediction_sd'])} |")

    lines += ["", "The report is provisional whenever `attempted < expected` ",
              "or fewer than 36 paired cases enter a delta. Missing jobs are ",
              "not silently scored as model failures.", ""]
    (out_dir / "SUMMARY.md").write_text("\n".join(lines))
    print(f"{len(all_rows)} metric rows -> {out_dir / 'ofat_metrics.csv'}")
    print(f"{len(stability_rows)} stability rows -> "
          f"{out_dir / 'repeat_stability.csv'}")
    print(f"report -> {out_dir / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
