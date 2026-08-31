#!/usr/bin/env python3
"""Count cached Qwen3.6 tokens for exported G0c mask-input blocks."""

import argparse
import csv
import json
from pathlib import Path

from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True)
    rows = []
    with open(args.input) as handle:
        for line in handle:
            record = json.loads(line)
            rows.append({
                "case_id": record["case_id"],
                "arm": record["arm"],
                "qwen36_tokens": len(tokenizer.encode(
                    record["text"], add_special_tokens=False)),
            })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["case_id", "arm", "qwen36_tokens"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {output}: {len(rows)} rows")


if __name__ == "__main__":
    main()
