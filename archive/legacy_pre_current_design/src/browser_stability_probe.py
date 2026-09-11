#!/usr/bin/env python3
"""Prepare and score the manual DeepSeek thinking stability extension.

The first browser sheet produced one answer on each of twelve graphs.  This
extension reuses each answer as the first observation in both arms and asks
for four additional answers per graph:

* two fresh generations of the identical prompt (three responses total),
* one generation for each of two redrawn walks (three walk samples total).

The reported quantities are direct and deliberately simple: the mean absolute
pairwise difference between rho_k2 estimates on the same graph.  They fill the
"same prompt" and "new walk" columns used in the supervisor-meeting table.

    PYTHONPATH=src python src/browser_stability_probe.py sheet
    # Fill stability_answers_pasted.txt, then:
    PYTHONPATH=src python src/browser_stability_probe.py ingest
"""
from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from browser_probe import ingest, parse_pasted
from llm_eval_frozen import load_answers, load_prompts, valid_unit
from run_llm_v2 import extract_last_json, is_complete_record


PROMPTS = "results/llm_noise_probe/prompts.jsonl"
OUT_DIR = "results/llm_noise_probe/browser_think"
BASE_SELECTED = "selected.csv"
BASE_ANSWERS = "answers_deepseek-v4-pro_think_browser.jsonl"
SELECTED = "stability_selected.csv"
SHEET = "stability_prompts_sheet.md"
ANSWERS_IN = "stability_answers_pasted.txt"
ANSWERS_OUT = "answers_deepseek-v4-pro_think_browser_stability.jsonl"
MARKER = "### "


def build_selection(prompts, base_selected):
    """Select two identical-prompt repeats and two new walks per graph."""
    meta = pd.DataFrame(prompts.values())
    rows = []
    for graph_no, base in enumerate(base_selected.itertuples(), start=1):
        original = prompts[base.prompt_id]
        graph = meta[meta.instance_id == base.instance_id].copy()

        response = graph[
            graph.arms.str.contains("response") &
            (graph.walk_seed == 0) &
            (graph.prompt_id != base.prompt_id)
        ].sort_values(["rep", "prompt_id"]).head(2)
        walk = graph[
            graph.arms.str.contains("input") &
            (graph.walk_seed != 0)
        ].sort_values(["walk_seed", "rep", "prompt_id"]).head(2)

        if len(response) != 2 or len(walk) != 2:
            raise ValueError(f"{base.instance_id}: incomplete probe records")
        if not all(text == original["prompt"] for text in response.prompt):
            raise ValueError(f"{base.instance_id}: response prompts differ")
        if any(text == original["prompt"] for text in walk.prompt):
            raise ValueError(f"{base.instance_id}: redrawn walks are identical")

        for slot, row in enumerate(response.itertuples(), start=2):
            rows.append({**row._asdict(), "measurement": "same_prompt",
                         "slot": slot, "graph_no": graph_no,
                         "base_prompt_id": base.prompt_id})
        for slot, row in enumerate(walk.itertuples(), start=2):
            rows.append({**row._asdict(), "measurement": "new_walk",
                         "slot": slot, "graph_no": graph_no,
                         "base_prompt_id": base.prompt_id})

    selected = pd.DataFrame(rows)
    if selected.prompt_id.duplicated().any():
        raise ValueError("stability selection contains duplicate prompt ids")
    return selected.sort_values(["graph_no", "measurement", "slot"],
                                ascending=[True, False, True])


def write_sheet(selected, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graphs = selected.graph_no.nunique()
    lines = [
        "# DeepSeek V4 Pro Thinking: stability and walk-seed probe",
        "",
        f"{len(selected)} additional prompts on {graphs} graphs. The answer "
        "from the first browser sheet is already repetition 1 for both arms.",
        "",
        "Rules:",
        "",
        "1. Use **Expert + DeepThink** for every prompt.",
        "2. Open a **fresh chat for every prompt**.",
        "3. Paste the response verbatim into `stability_answers_pasted.txt`.",
        "4. Do not correct, round, or retry an answer because it looks wrong.",
        "5. If rate-limited, stop and continue later; do not change settings.",
        "",
    ]
    template = ["# Paste each raw reply under its prompt id.", ""]
    for graph_no, block in selected.groupby("graph_no", sort=True):
        instance = block.instance_id.iloc[0]
        lines += ["---", "", f"## Graph {graph_no}: `{instance}`", ""]
        for row in block.itertuples():
            label = (f"Same prompt — generation {row.slot}/3"
                     if row.measurement == "same_prompt"
                     else f"New walk — sample {row.slot}/3")
            lines += [f"### {row.prompt_id} — {label}", "", "```",
                      row.prompt.strip(), "```", ""]
            template += [f"{MARKER}{row.prompt_id}", "", ""]

    (out / SHEET).write_text("\n".join(lines))
    selected.to_csv(out / SELECTED, index=False)
    answer_path = out / ANSWERS_IN
    if answer_path.exists():
        print(f"kept existing {answer_path} (not overwritten)")
    else:
        answer_path.write_text("\n".join(template))
    return out


def rho_value(record):
    obj = extract_last_json(record.get("answer"))
    value = obj.get("rho_k2") if isinstance(obj, dict) else None
    return float(value) if valid_unit(value) else np.nan


def mean_pairwise_gap(groups):
    gaps = []
    for values in groups:
        gaps.extend(abs(a - b) for a, b in itertools.combinations(values, 2))
    return float(np.mean(gaps)) if gaps else np.nan


def score(out_dir):
    out = Path(out_dir)
    base_selected = pd.read_csv(out / BASE_SELECTED)
    selected = pd.read_csv(out / SELECTED)
    base = load_answers(str(out / BASE_ANSWERS))
    extra = load_answers(str(out / ANSWERS_OUT))

    same_groups, walk_groups = [], []
    complete = 0
    for row in base_selected.itertuples():
        base_value = rho_value(base.get(row.prompt_id, {}))
        block = selected[selected.base_prompt_id == row.prompt_id]
        same = [base_value] + [rho_value(extra.get(pid, {})) for pid in
                               block[block.measurement == "same_prompt"].prompt_id]
        walk = [base_value] + [rho_value(extra.get(pid, {})) for pid in
                               block[block.measurement == "new_walk"].prompt_id]
        if all(np.isfinite(same)) and all(np.isfinite(walk)):
            complete += 1
            same_groups.append(same)
            walk_groups.append(walk)

    print(f"complete graphs: {complete}/{len(base_selected)}")
    print(f"complete additional answers: "
          f"{sum(is_complete_record(r) for r in extra.values())}/{len(selected)}")
    if not complete:
        return
    print(f"same prompt, mean |rho_k2 difference|: "
          f"{mean_pairwise_gap(same_groups):.4f}")
    print(f"new walk, mean |rho_k2 difference|:   "
          f"{mean_pairwise_gap(walk_groups):.4f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["sheet", "ingest", "score"])
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    out = Path(args.out_dir)
    if args.command == "sheet":
        prompts = load_prompts(args.prompts)
        base_selected = pd.read_csv(out / BASE_SELECTED)
        selected = build_selection(prompts, base_selected)
        write_sheet(selected, out)
        print(f"wrote {out / SHEET} with {len(selected)} prompts")
        print(f"paste replies into {out / ANSWERS_IN}")
        return

    if args.command == "score":
        score(out)
        return

    selected = pd.read_csv(out / SELECTED)
    prompts = load_prompts(args.prompts)
    pasted = parse_pasted(out / ANSWERS_IN, set(selected.prompt_id))
    if not pasted:
        raise SystemExit(f"nothing pasted into {out / ANSWERS_IN} yet")
    out_path = out / ANSWERS_OUT
    existing = load_answers(str(out_path)) if out_path.exists() else {}
    pending = {pid: text for pid, text in pasted.items()
               if pid not in existing or not is_complete_record(existing[pid])}
    if pending:
        written = ingest(prompts, pending, out_path)
        print(f"appended {written} records to {out_path}")
    else:
        print("no new answers to append")
    score(out)


if __name__ == "__main__":
    main()
