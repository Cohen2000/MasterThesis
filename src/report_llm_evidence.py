#!/usr/bin/env python3
"""Cross-run report over every LLM answer file, scored by the frozen rule.

Covers the API runs, the partial Open-Weights cluster snapshot and the two CLI
product screens in one pass. Read-only.

Comparisons are restricted to shared case_ids before anything is ranked: each
input format has its own prompt, the ablation cells cover 36 cases while
`mask` covers 84, and the screens ran on their own subsets. An aggregate row
from a leaderboard is not comparable across those without that restriction.

Sections (choose with --section, default all):
  inventory   how much of the 420-prompt suite each run actually has
  inputs      the input ladder on the paired cases, plus per-case deltas
  context     hidden -> disclosed -> disclosed_examples on identical cases
  quality     per-run validity, error, bias, calibration on disclosed/mask
  screens     every run restricted to a screen's own case set
  anchor      does a run track the examples' level, the data, or neither

Example:
  PYTHONPATH=src python src/report_llm_evidence.py --section inputs context
"""

import argparse
from collections import defaultdict
from pathlib import Path
import statistics

from llm_eval_frozen import (cell_index, example_anchor, extract_last_json,
                             is_complete_record, load_answers, load_cases,
                             load_prompts, readoff_rho2, score, spearman,
                             valid_unit)

# A run needs several input cells before its ordering says anything about the
# ladder; a run with a single format would otherwise hand that format rank 1.
MIN_CELLS_FOR_RANK = 3

# What exists locally. Globs collapse shards and retry/escalation files.
DEFAULT_RUNS = [
    ("gemini-lite minimal", "API",
     ["results/llm_v21/eval_input/answers_gemini-3.1-flash-lite_minimal.jsonl"]),
    ("gemini-lite think", "API",
     ["results/llm_v21/eval_input/answers_gemini-3.1-flash-lite_think.jsonl"]),
    ("deepseek-v4 nothink", "API",
     ["results/llm_v21/eval_input/answers_deepseek-v4-pro_nothink.jsonl"]),
    ("mistral-s4 none", "API",
     ["results/llm_v21/eval_input/answers_mistral-small-4_none.jsonl"]),
    ("mistral-s4 high", "API",
     ["results/llm_v21/eval_input/answers_mistral-small-4_high.jsonl"]),
    ("qwen3.6 nothink", "OpenW",
     ["results/llm_v21/cluster_snapshot/answers_qwen36-27b_nothink*.jsonl"]),
    ("qwen3.6 think", "OpenW",
     ["results/llm_v21/cluster_snapshot/answers_qwen36-27b_think*.jsonl"]),
    ("r1-distill-32b", "OpenW",
     ["results/llm_v21/cluster_snapshot/answers_r1-distill-32b*.jsonl"]),
    ("codex-5.6 notools", "Screen",
     ["results/codex_screen_snapshot/answers_codex-gpt-5.6-sol_notools_high*.jsonl"]),
    ("codex-5.6 tools", "Screen",
     ["results/codex_screen_snapshot/answers_codex-gpt-5.6-sol_tools_high*.jsonl"]),
    ("claude-code notools", "Screen",
     ["results/cc_screen_snapshot/answers_claude-code-opus_notools*.jsonl"]),
    ("claude-code tools", "Screen",
     ["results/cc_screen_snapshot/answers_claude-code-opus_tools*.jsonl"]),
]

INPUT_ORDER = ["nw", "mask", "mask_crawl_full", "mask_crawl_temporal",
               "mask_crawl_temporal_recent"]
SHORT = {"nw": "nw", "mask": "mask", "mask_crawl_full": "+crawl",
         "mask_crawl_temporal": "+temporal",
         "mask_crawl_temporal_recent": "+recent"}
CONTEXTS = ["hidden", "disclosed", "disclosed_examples"]


def fmt(value, width=10, digits=3):
    if value is None or value != value:
        return "-".rjust(width)
    return f"{value:.{digits}f}".rjust(width)


def section_inventory(ctx):
    print("\n=== inventory: coverage of the 420 frozen prompts ===")
    print(f"{'run':<22}{'family':<8}{'records':>9}{'complete':>10}"
          f"{'incomplete':>12}{'not attempted':>15}")
    for label, family, _ in ctx["runs"]:
        answers = ctx["answers"][label]
        complete = sum(1 for r in answers.values() if is_complete_record(r))
        print(f"{label:<22}{family:<8}{len(answers):>9}{complete:>10}"
              f"{len(answers) - complete:>12}{420 - len(answers):>15}")
    print("\nIncomplete records are mostly token-limit truncations on the open-"
          "weights runs.\nThey stay retryable and are a different thing from a "
          "prompt that was never attempted.")


def section_inputs(ctx):
    paired = ctx["paired36"]
    print(f"\n=== input ladder: failure-penalized ProfileMAE on the "
          f"{len(paired)} paired cases ===")
    print("condition=disclosed, `mask` restricted to the same cases. "
          "Lower is better.")
    header = f"{'run':<22}" + "".join(f"{SHORT[i]:>12}" for i in INPUT_ORDER)
    print(header + f"{'best':>13}")

    by_family = defaultdict(lambda: defaultdict(list))
    ranks = defaultdict(list)
    for label, family, _ in ctx["runs"]:
        row, values = f"{label:<22}", {}
        for kind in INPUT_ORDER:
            mapping = ctx["cells"].get(("disclosed", kind), {})
            metrics = score(ctx["answers"][label], mapping, paired, ctx["cases"])
            if metrics is None:
                row += "-".rjust(12)
                continue
            value = metrics["profile_mae_penalized"]
            tag = f"{value:.3f}" + (f"({metrics['n']})"
                                    if metrics["n"] != len(paired) else "")
            row += tag.rjust(12)
            values[kind] = value
            by_family[family][kind].append(value)
        if values:
            order = sorted(values, key=values.get)
            if len(values) >= MIN_CELLS_FOR_RANK:
                for position, kind in enumerate(order, start=1):
                    ranks[kind].append(position)
            row += f"{SHORT[order[0]]:>13}"
        print(row)
    print("(n) marks a cell scored on fewer cases because the run never "
          "attempted the rest.")

    print(f"\n{'mean rank':<22}" + "".join(
        fmt(statistics.mean(ranks[i]), 12, 2) if ranks[i] else "-".rjust(12)
        for i in INPUT_ORDER))
    print(f"{'runs ranked':<22}" + "".join(
        f"{len(ranks[i]):>12}" for i in INPUT_ORDER))
    print(f"only runs with at least {MIN_CELLS_FOR_RANK} input cells enter the "
          "rank tally")

    print("\nmean penalized ProfileMAE by family")
    for family in ["API", "OpenW", "Screen"]:
        if not by_family[family]:
            continue
        print(f"  {family:<8}" + "".join(
            fmt(statistics.mean(by_family[family][i]), 12)
            if by_family[family][i] else "-".rjust(12) for i in INPUT_ORDER))

    print("\nper-case deltas against `mask` (negative = the variant helped)")
    for label, _, _ in ctx["runs"]:
        base = score(ctx["answers"][label], ctx["cells"].get(("disclosed", "mask"), {}),
                     paired, ctx["cases"])
        if base is None:
            continue
        parts = []
        for kind in INPUT_ORDER:
            if kind == "mask":
                continue
            other = score(ctx["answers"][label],
                          ctx["cells"].get(("disclosed", kind), {}),
                          paired, ctx["cases"])
            if other is None:
                continue
            shared = sorted(set(base["per_case"]) & set(other["per_case"]))
            if not shared:
                continue
            deltas = [other["per_case"][c] - base["per_case"][c] for c in shared]
            parts.append(f"{SHORT[kind]} {statistics.mean(deltas):+.3f} "
                         f"({sum(d < 0 for d in deltas)}/{len(deltas)} better)")
        if parts:
            print(f"  {label:<22}" + " | ".join(parts))


def section_context(ctx):
    ids = ctx["all84"]
    width = 27
    print(f"\n=== context axis on `mask`, {len(ids)} identical cases ===")
    print(f"{'run':<22}" + "".join(
        f"{c[:12]:>{width}}" for c in CONTEXTS))
    print(f"{'':<22}" + "".join(
        f"{'pen / rho2 bias':>{width}}" for _ in CONTEXTS))
    for label, _, _ in ctx["runs"]:
        cells = {c: score(ctx["answers"][label],
                          ctx["cells"].get((c, "mask"), {}), ids, ctx["cases"])
                 for c in CONTEXTS}
        if not any(cells.values()):
            continue
        row = f"{label:<22}"
        for c in CONTEXTS:
            m = cells[c]
            if m is None:
                row += "-".rjust(width)
                continue
            value = (f"{m['profile_mae_penalized']:.3f} / "
                     f"{m['rho2_bias']:+.3f}")
            if m["n"] != len(ids):
                value += f" (n={m['n']})"
            row += value.rjust(width)
        print(row)
    print("\nBias moving toward zero across the row is the effect to watch; the "
          "error alone\nunderstates it. See the anchor section before reading a "
          "gain as learned skill.")


def section_quality(ctx):
    ids = ctx["all84"]
    print("\n=== per-run quality on condition=disclosed, input=mask ===")
    print(f"{'run':<22}{'n':>4}{'resp':>7}{'valid':>7}{'pen':>9}{'cc':>9}"
          f"{'rho2':>9}{'bias':>9}{'spear':>8}{'C-MAE':>9}{'cov90':>7}")
    for label, _, _ in ctx["runs"]:
        m = score(ctx["answers"][label], ctx["cells"].get(("disclosed", "mask"), {}),
                  ids, ctx["cases"])
        if m is None:
            continue
        print(f"{label:<22}{m['n']:>4}{m['response_rate']:>7.2f}"
              f"{m['validity']:>7.2f}"
              + fmt(m["profile_mae_penalized"], 9)
              + fmt(m["profile_mae_complete"], 9)
              + fmt(m["rho2_mae"], 9) + fmt(m["rho2_bias"], 9)
              + fmt(m["rho2_spearman"], 8, 2)
              + fmt(m["c_mae"], 9) + fmt(m["cover90"], 7, 2))
    print("\n`pen` and `cc` diverge exactly where a run failed to answer. "
          "Ranking on one\nalone hides either non-response or truncation "
          "budget effects.")


def section_screens(ctx):
    print("\n=== every run restricted to a screen's own case set ===")
    for owner in ["claude-code tools", "codex-5.6 notools"]:
        if owner not in ctx["answers"]:
            continue
        mapping = ctx["cells"].get(("disclosed", "mask"), {})
        owned = sorted(c for c, pid in mapping.items()
                       if pid in ctx["answers"][owner])
        if not owned:
            continue
        print(f"\n-- {len(owned)} cases answered by {owner} --")
        print(f"{'run':<22}{'n':>4}{'valid':>7}{'pen':>9}{'cc':>9}{'rho2':>9}"
              f"{'bias':>9}{'cov90':>7}")
        rows = []
        for label, _, _ in ctx["runs"]:
            m = score(ctx["answers"][label], mapping, owned, ctx["cases"])
            if m and m["n"] >= 0.8 * len(owned):
                rows.append((label, m))
        for label, m in sorted(rows, key=lambda r: r[1]["profile_mae_penalized"]):
            print(f"{label:<22}{m['n']:>4}{m['validity']:>7.2f}"
                  + fmt(m["profile_mae_penalized"], 9)
                  + fmt(m["profile_mae_complete"], 9)
                  + fmt(m["rho2_mae"], 9) + fmt(m["rho2_bias"], 9)
                  + fmt(m["cover90"], 7, 2))
        subset_cov = [float(ctx["cases"][c]["coverage"]) for c in owned]
        subset_truth = [float(ctx["cases"][c]["rho_W5_k2"]) for c in owned]
        full_cov = [float(ctx["cases"][c]["coverage"]) for c in ctx["all84"]]
        full_truth = [float(ctx["cases"][c]["rho_W5_k2"]) for c in ctx["all84"]]
        print(f"   subset: coverage median {statistics.median(subset_cov):.3f}, "
              f"truth rho2 mean {statistics.mean(subset_truth):.3f}"
              f"   |   all {len(ctx['all84'])}: "
              f"{statistics.median(full_cov):.3f}, "
              f"{statistics.mean(full_truth):.3f}")
    print("\nA screen carries a harness prompt that is not part of the frozen "
          "prompt and\ncannot be version-pinned; that bounds what its numbers "
          "can be claimed to show.")


def section_anchor(ctx):
    ids = ctx["all84"]
    anchors = ctx["anchors"]
    cases = ctx["cases"]
    print("\n=== anchor diagnostics: level transfer, or use of the data? ===")
    constant = [anchors[cases[c]["strategy"]][0] for c in ids]
    truth = [float(cases[c]["rho_W5_k2"]) for c in ids]
    observed = [readoff_rho2(cases[c]) for c in ids]
    print(f"example anchor per strategy: "
          f"{ {k: round(v[0], 3) for k, v in anchors.items()} }")
    print(f"truth rho2      mean {statistics.mean(truth):.3f} "
          f"sd {statistics.pstdev(truth):.3f}")
    print(f"observed rho2   mean {statistics.mean(observed):.3f} "
          f"sd {statistics.pstdev(observed):.3f}   "
          f"spearman vs truth {spearman(observed, truth):.3f}")
    print(f"\nA constant at the anchor scores rho2 MAE "
          f"{statistics.mean([abs(a - b) for a, b in zip(constant, truth)]):.3f} "
          f"without reading the input at all.")

    print(f"\n{'run':<22}{'condition':<20}{'sd(pred)':>10}{'|pred-anchor|':>15}"
          f"{'spear(truth)':>14}{'spear(obs)':>12}{'near anchor':>13}")
    for label, _, _ in ctx["runs"]:
        printed = False
        for condition in ["disclosed", "disclosed_examples"]:
            mapping = ctx["cells"].get((condition, "mask"), {})
            answers = ctx["answers"][label]
            pairs = []
            for case_id in ids:
                prompt_id = mapping.get(case_id)
                if prompt_id is None or prompt_id not in answers:
                    continue
                obj = extract_last_json(answers[prompt_id].get("answer"))
                raw = obj.get("rho_k2") if isinstance(obj, dict) else None
                if valid_unit(raw):
                    pairs.append((case_id, float(raw)))
            if len(pairs) < 0.5 * len(ids):
                continue
            preds = [v for _, v in pairs]
            tv = [float(cases[c]["rho_W5_k2"]) for c, _ in pairs]
            ov = [readoff_rho2(cases[c]) for c, _ in pairs]
            av = [anchors[cases[c]["strategy"]][0] for c, _ in pairs]
            distance = [abs(p - a) for p, a in zip(preds, av)]
            near = sum(1 for d in distance if d <= 0.05) / len(distance)
            print(f"{label:<22}{condition:<20}"
                  + fmt(statistics.pstdev(preds), 10)
                  + fmt(statistics.mean(distance), 15)
                  + fmt(spearman(preds, tv), 14)
                  + fmt(spearman(preds, ov), 12)
                  + f"{near:>12.0%}")
            printed = True
        if printed:
            print()
    print("A run that merely repeats the anchor would show sd near zero, "
          "distance near zero\nand almost everything near the anchor. Spread "
          "that survives, together with a rising\ncorrelation to the observed "
          "data, points at a level shift on top of real estimation\nrather "
          "than substitution for it.")


SECTIONS = {
    "inventory": section_inventory, "inputs": section_inputs,
    "context": section_context, "quality": section_quality,
    "screens": section_screens, "anchor": section_anchor,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--prompts", default="results/llm_v2/prompts.jsonl")
    ap.add_argument("--cases", default="results/llm_v2/llm_cases.csv")
    ap.add_argument("--examples", default="results/llm_v2/llm_examples.csv")
    ap.add_argument("--section", nargs="+", default=list(SECTIONS),
                    choices=list(SECTIONS))
    args = ap.parse_args()

    root = Path(args.root)
    prompts = load_prompts(root / args.prompts)
    cases = load_cases(root / args.cases)
    cells = cell_index(prompts)

    runs = [(label, family, pats) for label, family, pats in DEFAULT_RUNS
            if load_answers(pats, root=root)]
    ctx = {
        "runs": runs,
        "answers": {label: load_answers(pats, root=root)
                    for label, _, pats in runs},
        "prompts": prompts, "cases": cases, "cells": cells,
        "paired36": sorted(cells.get(("disclosed", "nw"), {})),
        "all84": sorted(cells.get(("disclosed", "mask"), {})),
        "anchors": example_anchor(root / args.examples),
    }
    print(f"{len(runs)} runs, {sum(len(a) for a in ctx['answers'].values())} "
          f"answered prompts, {len(ctx['all84'])} cases, "
          f"{len(ctx['paired36'])} paired across input formats")
    for name in args.section:
        SECTIONS[name](ctx)


if __name__ == "__main__":
    main()
