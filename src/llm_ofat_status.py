#!/usr/bin/env python3
"""Read-only progress report for the six-cell LLM OFAT runs."""

import argparse
import collections
import glob
import json
import os
from pathlib import Path

from run_llm_v2 import is_complete_record


ROOT = Path(__file__).resolve().parents[1]
OFAT = ROOT / "results/llm_v21_ofat"


def read_jsonl(path):
    rows = []
    try:
        fh = open(path)
    except FileNotFoundError:
        return rows
    with fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def expand(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(glob.glob(str(pattern)))
    return sorted(set(paths))


def latest_records(paths, require_no_tools=False):
    latest = {}
    complete = {}
    attempts = collections.Counter()
    for path in paths:
        for row in read_jsonl(path):
            pid = row.get("prompt_id")
            if not pid:
                continue
            attempts[pid] += 1
            latest[pid] = row
            if (is_complete_record(row)
                    and not (require_no_tools and row.get("n_tool_events"))):
                complete[pid] = row
    # As in the evaluator: an earlier complete retry beats a later failed one.
    merged = dict(latest)
    merged.update(complete)
    return merged, complete, attempts


def alive(pid_path):
    try:
        pid = int(Path(pid_path).read_text().strip())
        os.kill(pid, 0)
        return str(pid)
    except (OSError, ValueError):
        return None


def report(label, prompt_path, answer_patterns, pid_glob=None,
           require_no_tools=False):
    prompts = read_jsonl(prompt_path)
    planned = {r["prompt_id"]: r for r in prompts}
    paths = expand(answer_patterns)
    merged, complete, attempts = latest_records(paths, require_no_tools)
    attempted_ids = set(merged) & set(planned)
    complete_ids = set(complete) & set(planned)
    reasons = collections.Counter(
        str(merged[pid].get("finish_reason")) for pid in attempted_ids)
    cells = collections.Counter(planned[pid]["input_kind"]
                                for pid in complete_ids)
    proc = []
    if pid_glob:
        proc = [p for p in (alive(x) for x in glob.glob(str(pid_glob))) if p]

    total = len(planned)
    pct = 100.0 * len(complete_ids) / total if total else 0.0
    print(f"\n{label}")
    print(f"  complete {len(complete_ids):>4}/{total:<4} ({pct:5.1f}%) | "
          f"attempted {len(attempted_ids):>4} | files {len(paths):>2}"
          + (f" | live PIDs {','.join(proc)}" if proc else ""))
    if cells:
        print("  complete by cell: " + ", ".join(
            f"{cell}={cells[cell]}" for cell in sorted(cells)))
    if reasons:
        print("  latest finish reasons: " + ", ".join(
            f"{reason}={count}" for reason, count in sorted(reasons.items())))
    if require_no_tools:
        leaks = sum(bool(row.get("n_tool_events")) for row in merged.values())
        if leaks:
            print(f"  excluded tool-contaminated prompts: {leaks}")
    retries = sum(max(0, n - 1) for n in attempts.values())
    if retries:
        print(f"  appended retries: {retries}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    home_codex = (Path.home() / "Dokumente/codex_ofat" /
                  "answers_codex-gpt-5.6-sol_notools_high_ofat.jsonl")
    report(
        "Codex gpt-5.6-sol notools/high",
        OFAT / "prompts_ofat_codex.jsonl",
        [OFAT / "answers_codex*.jsonl", home_codex],
        require_no_tools=True,
    )
    report(
        "Gemini 3.1 Flash Lite minimal",
        OFAT / "prompts_ofat_gemini.jsonl",
        [OFAT / "answers_gemini-3.1-flash-lite_minimal.shard*.jsonl"],
        OFAT / "pids/gemini.shard*.pid",
    )
    report(
        "DeepSeek V4 Flash 0731 non-thinking",
        OFAT / "prompts_ofat_deepseek.jsonl",
        [OFAT / "answers_deepseek-v4-flash-0731_nothink.shard*.jsonl"],
        OFAT / "pids/deepseek.shard*.pid",
    )
    qprompt = OFAT / "prompts_ofat_qwen.jsonl"
    for mode in ("nothink", "think"):
        report(
            f"Qwen3.6-27B {mode} (downloaded)",
            qprompt,
            [OFAT / f"answers_ofat_qwen36_{mode}*.jsonl"],
        )


if __name__ == "__main__":
    main()
