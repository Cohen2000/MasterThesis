#!/usr/bin/env python3
"""Merge validated non-walk case shards into one deterministic case file."""

import argparse
from pathlib import Path

from evaluate_benchmark import read_cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", required=True)
    parser.add_argument("--out", default="results/nonwalk_screen/panel32_cases.csv.gz")
    args = parser.parse_args()
    cases, paths = read_cases(args.cases)
    cases = cases.sort_values("case_id", kind="mergesort").reset_index(drop=True)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    cases.to_csv(out, index=False, compression="gzip")
    print(f"wrote {out}: {len(cases)} cases from {len(paths)} shards")


if __name__ == "__main__":
    main()
