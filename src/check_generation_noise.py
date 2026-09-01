#!/usr/bin/env python3
"""Assert that two generations of the same prompt actually differ.

`run_llm_v2.HFModel` reseeds torch before every generation.  If `gen_seed` is
not varied, two generations of one prompt are byte-identical and every
response-noise number downstream is a property of the runner, not the model.
That failure is silent: the run completes, the files look right, and the
variance decomposition reports zero response noise.

This exits non-zero when it finds that pattern, so a job can fail loudly rather
than produce confident nonsense.  It is meant to run on a smoke before the real
submission and again over the finished answers.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path


def load(path: str) -> dict[str, dict]:
    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        out[record["prompt_id"]] = record
    return out


def compare(files: list[str]) -> tuple[list[dict], dict]:
    generations = [load(p) for p in files]
    shared = set(generations[0])
    for gen in generations[1:]:
        shared &= set(gen)
    rows = []
    for prompt_id in sorted(shared):
        answers = [g[prompt_id].get("answer", "") for g in generations]
        seeds = [g[prompt_id].get("seed") for g in generations]
        distinct_answers = len(set(answers))
        rows.append({
            "prompt_id": prompt_id,
            "generations": len(answers),
            "distinct_answers": distinct_answers,
            "identical": distinct_answers == 1,
            "distinct_seeds": len(set(s for s in seeds if s is not None)),
            "seeds": seeds,
        })
    identical = [r for r in rows if r["identical"]]
    same_seed = [r for r in rows if r["distinct_seeds"] < len(files)]
    summary = {
        "prompts_compared": len(rows),
        "identical_across_generations": len(identical),
        "prompts_with_repeated_seed": len(same_seed),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", nargs="+", required=True,
                        help="one answers file per generation (globs allowed)")
    parser.add_argument("--allow-identical-fraction", type=float, default=0.02,
                        help="tolerance for genuinely degenerate prompts")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    files = []
    for spec in args.answers:
        matched = sorted(glob.glob(spec))
        files.append(matched if len(matched) > 1 else [spec])
    # Each --answers entry is one generation; a glob inside it is that
    # generation's shards, which are merged before comparing.
    merged_paths = []
    for group in files:
        if len(group) == 1:
            merged_paths.append(group[0])
        else:
            combined = Path(group[0]).with_suffix(".merged.jsonl")
            with combined.open("w") as handle:
                for path in group:
                    handle.write(Path(path).read_text())
            merged_paths.append(str(combined))

    if len(merged_paths) < 2:
        print("need at least two generations to compare", file=sys.stderr)
        return 2

    rows, summary = compare(merged_paths)
    if args.out:
        Path(args.out).write_text(json.dumps(
            {"summary": summary, "rows": rows}, indent=2))
    print(json.dumps(summary, indent=2))

    if not summary["prompts_compared"]:
        print("FAIL: no prompt was answered in every generation",
              file=sys.stderr)
        return 1
    if summary["prompts_with_repeated_seed"]:
        print(f"FAIL: {summary['prompts_with_repeated_seed']} prompts reused a "
              "seed across generations; gen_seed is not being varied",
              file=sys.stderr)
        return 1
    share = summary["identical_across_generations"] / summary["prompts_compared"]
    if share > args.allow_identical_fraction:
        print(f"FAIL: {share:.1%} of prompts produced byte-identical answers "
              "across generations; the response-noise measurement would be a "
              "runner artifact", file=sys.stderr)
        return 1
    print(f"OK: {share:.1%} identical, within the "
          f"{args.allow_identical_fraction:.1%} tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
