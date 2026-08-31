#!/usr/bin/env python3
"""Run a probe arm by hand in a chat product, then score it like any other arm.

The DeepSeek V4 Pro thinking arm of the noise probe died at 15 of 288 prompts
on `HTTP 402 Insufficient Balance`.  This gives the same frozen prompts to a
human with a browser instead, so the thinking mode gets a tendency rather than
nothing.

What that costs in claim strength is the Codex objection again, and it should
be stated the same way: a chat product carries its own system prompt, its
version is not pinnable, and its sampling parameters are not visible.  Answers
collected this way document the product, not `deepseek-v4-pro` at
`reasoning_effort=high`, and belong beside the API rows rather than inside
them.  `--anchor` exists to bound that gap: some of the selected prompts are
ones the API arm did answer, so the two paths can be compared on identical
input before the rest of the sheet is believed.

    PYTHONPATH=src python src/browser_probe.py sheet
    # paste each prompt into a *fresh* chat, paste the reply back into the
    # answers file under its prompt id
    PYTHONPATH=src python src/browser_probe.py ingest
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_eval_frozen import load_answers, load_prompts
from report_llm_noise import build_frame, load_by_model
from run_llm_v2 import is_complete_record

PROMPTS = "results/llm_noise_probe/prompts.jsonl"
CASES = "results/panel_seed_probe/cases.csv.gz"
API_THINK = "results/llm_noise_probe/answers_deepseek-v4-pro_think_high_*.jsonl"
OUT_DIR = "results/llm_noise_probe/browser_think"
SHEET = "prompts_sheet.md"
ANSWERS_IN = "answers_pasted.txt"
ANSWERS_OUT = "answers_deepseek-v4-pro_think_browser.jsonl"
MARKER = "### "

REQUIRED_KEYS = ["rho_k2", "rho_k3", "rho_k4", "rho_k5", "mean_occupancy",
                 "C_one_step", "lifetime_mean_over_T", "lo90", "hi90"]


def _api_answered(pattern):
    answered = set()
    for prompt_id, record in load_answers(pattern).items():
        if is_complete_record(record):
            answered.add(prompt_id)
    return answered


def select(prompts, answered, n_anchor, n_new):
    """Anchors first, then one shared-cell record per uncovered graph.

    The shared cell (`arms == "response,input"`, walk seed 0, rep 1) is the one
    record per graph that both probe arms have in common, so it is the natural
    single case to spend a manual generation on.
    """
    meta = pd.DataFrame(prompts.values())
    anchors = (meta[meta.prompt_id.isin(answered)]
               .drop_duplicates("instance_id")
               .sort_values("prompt_id").head(n_anchor))
    shared = meta[(meta.arms == "response,input") &
                  (~meta.instance_id.isin(anchors.instance_id)) &
                  (~meta.prompt_id.isin(answered))]
    step = max(1, len(shared) // max(1, n_new))
    new = shared.sort_values("instance_id").iloc[::step].head(n_new)
    anchors = anchors.assign(role="anchor")
    new = new.assign(role="new")
    return pd.concat([anchors, new], ignore_index=True)


def write_sheet(selected, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Browser sheet: DeepSeek V4 Pro, thinking on",
        "",
        f"{len(selected)} prompts, "
        f"{int((selected.role == 'anchor').sum())} of them anchors the API arm "
        "already answered.",
        "",
        "Rules that keep this scorable:",
        "",
        "1. **A fresh chat per prompt.** Chat products carry context and memory "
        "across turns; two prompts in one thread are not two independent cases.",
        "2. **Thinking/reasoning mode on**, and the same setting for all of them.",
        "3. **Paste the reply verbatim** into the answers file, including any "
        "prose around the JSON. Do not fix, sort or round anything -- the "
        "scorer takes the last JSON object and an inconsistent profile is a "
        "result, not a defect to repair.",
        "4. If a reply is unusable, leave the section empty rather than "
        "retrying until it looks good.",
        "",
        "---",
        "",
    ]
    for row in selected.itertuples():
        lines += [f"## {row.prompt_id}  ({row.role}, graph `{row.instance_id}`)",
                  "", "```", row.prompt.strip(), "```", "", "---", ""]
    (out / SHEET).write_text("\n".join(lines))

    template = ["# Paste each raw reply under its prompt id. Keep the ### lines.",
                ""]
    for row in selected.itertuples():
        template += [f"{MARKER}{row.prompt_id}", "", ""]
    path = out / ANSWERS_IN
    if path.exists():
        print(f"kept existing {path} (not overwritten)")
    else:
        path.write_text("\n".join(template))
    selected.to_csv(out / "selected.csv", index=False)
    return out


def parse_pasted(path, valid_ids):
    """{prompt_id: raw reply}; empty sections are dropped, not scored as wrong.

    A line counts as a section header only when the text after the marker is a
    known prompt id.  Models write markdown headings, and a reply containing
    `### Result` must land inside its answer rather than starting a new one.
    """
    answers, current, buffer = {}, None, []
    for line in Path(path).read_text().splitlines():
        candidate = line[len(MARKER):].strip() if line.startswith(MARKER) else None
        if candidate in valid_ids:
            if current:
                answers[current] = "\n".join(buffer).strip()
            current, buffer = candidate, []
        elif current is not None:
            buffer.append(line)
    if current:
        answers[current] = "\n".join(buffer).strip()
    return {k: v for k, v in answers.items() if v}


def ingest(prompts, pasted, out_path):
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    written = 0
    with open(out_path, "a") as fh:
        for prompt_id, text in sorted(pasted.items()):
            meta = prompts.get(prompt_id)
            if meta is None:
                raise SystemExit(f"unknown prompt id in the paste file: {prompt_id}")
            fh.write(json.dumps({
                "id": prompt_id,
                "prompt_id": prompt_id,
                "case_id": meta["case_id"],
                "condition": meta["condition"],
                "input_kind": meta["input_kind"],
                "strategy": meta["strategy"],
                "model": "deepseek-v4-pro-browser",
                "backend": "browser",
                "harness": "deepseek chat product, version not pinnable",
                "thinking": "on",
                "reasoning_effort": None,
                "temperature": None,
                "top_p": None,
                "seed": None,
                "rep": meta["rep"],
                "prompt_sha256": meta["prompt_sha256"],
                "required_keys": REQUIRED_KEYS,
                "est_cost_usd": 0.0,
                "ts": stamp,
                "answer": text,
                "finish_reason": "stop",
                "usage": None,
            }) + "\n")
            written += 1
    return written


def score(prompts, cases, out_path, api_pattern):
    browser = load_answers(out_path)
    if not browser:
        raise SystemExit("no answers to score yet")
    subset = {pid: prompts[pid] for pid in browser}
    frame = build_frame(subset, browser, cases)
    print(f"\nbrowser arm: {len(frame)} answers on "
          f"{frame.instance_id.nunique()} graphs")
    print(f"  validity                  {frame.valid.mean():.2f}")
    print(f"  ProfileMAE (penalized)    {frame.penalized.mean():.4f}")
    complete = frame[frame.valid == 1]
    if len(complete):
        print(f"  ProfileMAE (complete)     {complete.profile_mae.mean():.4f}"
              f"   on {len(complete)} answers")

    api = {}
    for records in load_by_model(api_pattern).values():
        api.update(records)
    # Failed API attempts may exist for prompts selected as new browser cases.
    # They are not anchors: comparing their failure penalty against a browser
    # answer would measure billing/transport failure rather than harness drift.
    shared = sorted(pid for pid in set(api) & set(browser)
                    if is_complete_record(api[pid]))
    if not shared:
        print("\nno anchor overlap with the API arm yet")
        return
    a = build_frame({p: prompts[p] for p in shared}, api, cases).set_index("prompt_id")
    b = build_frame({p: prompts[p] for p in shared}, browser, cases).set_index("prompt_id")
    delta = a.loc[shared].penalized - b.loc[shared].penalized
    print(f"\nanchors, identical prompts on both paths (n={len(shared)}):")
    print(f"  API ProfileMAE            {a.loc[shared].penalized.mean():.4f}")
    print(f"  browser ProfileMAE        {b.loc[shared].penalized.mean():.4f}")
    print(f"  paired difference         {delta.mean():+.4f}   (API minus browser)")
    gap = float(np.abs(a.loc[shared].rho_k2_pred - b.loc[shared].rho_k2_pred).mean())
    print(f"  mean |rho_k2 difference|  {gap:.4f}")
    print("\n  A small anchor gap makes the rest of the sheet usable as a "
          "tendency for\n  the thinking mode; a large one means the sheet "
          "documents the product.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["sheet", "ingest"])
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--cases", default=CASES)
    ap.add_argument("--api-answers", default=API_THINK)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--anchor", type=int, default=4)
    ap.add_argument("--new", type=int, default=8)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    out_dir = Path(args.out_dir)
    if args.command == "sheet":
        selected = select(prompts, _api_answered(args.api_answers),
                          args.anchor, args.new)
        write_sheet(selected, out_dir)
        print(f"wrote {out_dir / SHEET} with {len(selected)} prompts "
              f"({int((selected.role == 'anchor').sum())} anchors)")
        print(f"paste replies into {out_dir / ANSWERS_IN}")
        return

    pasted = parse_pasted(out_dir / ANSWERS_IN, set(prompts))
    if not pasted:
        raise SystemExit(f"nothing pasted into {out_dir / ANSWERS_IN} yet")
    out_path = out_dir / ANSWERS_OUT
    written = ingest(prompts, pasted, out_path)
    print(f"appended {written} records to {out_path}")
    score(prompts, pd.read_csv(args.cases).set_index("case_id"),
          str(out_path), args.api_answers)


if __name__ == "__main__":
    main()
