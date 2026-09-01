#!/usr/bin/env python3
"""Response, validity and consistency rates for the Qwen branches.

These are outcomes, not housekeeping. Whether mechanism information improves
the numbers while degrading internal consistency is itself reportable, and
differential dropout across conditions would bias the primary contrast, so the
rates are tracked per arm *and* per condition rather than pooled.

Answer files are append-only and a raised token cap retries truncated prompts,
so a prompt can carry several records. The complete one wins per (prompt_id,
generation); nothing is repaired, and an out-of-range value stays a recorded
outcome rather than a retry reason.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_eval_frozen import PROFILE_PRED, valid_unit
from run_llm_v2 import extract_last_json, is_complete_record


def load(patterns: list[str]) -> pd.DataFrame:
    best: dict[tuple[str, int], dict] = {}
    for pattern in patterns:
        for path in glob.glob(pattern):
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                key = (record["prompt_id"], record.get("rep"))
                previous = best.get(key)
                if previous is None or (is_complete_record(record)
                                        and not is_complete_record(previous)):
                    best[key] = record

    rows = []
    for record in best.values():
        parsed = extract_last_json(record.get("answer")) or {}
        values = [parsed.get(k) for k in PROFILE_PRED]
        valid = all(valid_unit(v) for v in values)
        rows.append({
            "prompt_id": record["prompt_id"],
            "generation": record.get("rep"),
            "arm": record["strategy"],
            "condition": record["condition"],
            "complete": is_complete_record(record),
            "truncated": record.get("finish_reason") == "length",
            "valid": valid,
            "monotone": (bool(np.all(np.diff(values) <= 1e-12))
                         if valid else None),
            "completion_tokens": (record.get("usage") or {}).get(
                "completion_tokens"),
        })
    return pd.DataFrame(rows)


def rates(frame: pd.DataFrame, by: str) -> pd.DataFrame:
    rows = []
    for key, part in frame.groupby(by):
        valid = part[part.valid]
        rows.append({
            by: key, "answers": int(len(part)),
            "complete_rate": float(part.complete.mean()),
            "truncation_rate": float(part.truncated.mean()),
            "validity_rate": float(part.valid.mean()),
            "monotone_rate": (float(valid.monotone.mean())
                              if len(valid) else np.nan),
            "median_completion_tokens": float(
                part.completion_tokens.median()),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", nargs="+", default=[
        "results/final_run_g2/answers/qwen/answers_vllm_qwen36-27b_*.jsonl"])
    parser.add_argument("--tag", default="all")
    parser.add_argument("--summary-dir", default="results_summary/g3")
    args = parser.parse_args()

    frame = load(args.answers)
    if frame.empty:
        print("no answers found")
        return
    out = Path(args.summary_dir)
    out.mkdir(parents=True, exist_ok=True)
    for by in ("arm", "condition"):
        table = rates(frame, by)
        table.to_csv(out / f"qwen_rates_by_{by}_{args.tag}.csv", index=False)
        print(f"=== by {by} ===")
        print(table.to_markdown(index=False, floatfmt=".4f"))
        print()
    overall = rates(frame.assign(all="all"), "all")
    overall.to_csv(out / f"qwen_rates_overall_{args.tag}.csv", index=False)
    print(overall.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
    main()
