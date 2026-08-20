#!/usr/bin/env python3
"""Select prompt IDs still incomplete after one or more answer files.

The newest record per prompt_id wins.  Missing prompts and records that fail
``run_llm_v2.is_complete_record`` are selected, which includes length
truncations, errors, empty answers and malformed/incomplete final JSON.
"""

import argparse
import glob
import json
from pathlib import Path

from run_llm_v2 import is_complete_record


def load_latest(patterns):
    latest = {}
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            with open(path) as fh:
                for line in fh:
                    try:
                        record = json.loads(line)
                        prompt_id = record["prompt_id"]
                    except (json.JSONDecodeError, KeyError):
                        continue
                    latest[prompt_id] = record
    return latest


def select_ids(prompts_path, answer_patterns):
    with open(prompts_path) as fh:
        prompt_ids = sorted({
            json.loads(line)["prompt_id"] for line in fh if line.strip()
        })
    latest = load_latest(answer_patterns)
    return [
        prompt_id for prompt_id in prompt_ids
        if prompt_id not in latest or not is_complete_record(latest[prompt_id])
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--answers", action="append", required=True,
                    help="answer-file glob; may be supplied more than once")
    ap.add_argument("--output", choices=["csv", "lines"], default="csv")
    args = ap.parse_args()

    selected = select_ids(args.prompts, args.answers)
    separator = "," if args.output == "csv" else "\n"
    print(separator.join(selected))


if __name__ == "__main__":
    main()
