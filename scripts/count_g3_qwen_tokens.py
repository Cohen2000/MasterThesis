#!/usr/bin/env python3
"""Exact Qwen3.6 token counts for the frozen G3 prompts.

Runs on the cluster, where the tokenizer is cached.  It replaces the calibrated
estimates (portable count x 1.664) used up to G2, which were explicitly marked
as estimates rather than measurements.

Counts the whole prompt as sent, not just the data block, because what matters
operationally is context fit.
"""

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    rows = []
    for line in Path(args.prompts).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        rows.append({
            "prompt_id": record["prompt_id"],
            "case_id": record["case_id"],
            "arm": record["strategy"],
            "condition": record["condition"],
            "subset": record["subset"],
            "qwen36_tokens": len(tokenizer.encode(record["prompt"],
                                                  add_special_tokens=False)),
        })
    out = Path(args.out)
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = sorted(r["qwen36_tokens"] for r in rows)
    n = len(counts)
    print(f"wrote {out}: {n} prompts")
    print(f"median {counts[n // 2]}  p90 {counts[int(n * .9)]}  max {counts[-1]}")


if __name__ == "__main__":
    main()
