#!/usr/bin/env python3
"""Future local dataset census; output must be new and outside raw/archive."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_census import census_datasets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "config/datasets.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--only", nargs="+", help="registry keys; default: all entries")
    parser.add_argument("--out", type=Path, required=True, help="new CSV path")
    args = parser.parse_args()
    destination = args.out.resolve()
    for protected in (args.raw_dir.resolve(), ROOT / "data/raw", ROOT / "archive"):
        if destination == protected or protected in destination.parents:
            parser.error("output must be outside raw data and the historical archive")
    if destination.exists():
        parser.error("output already exists; choose a new path")
    rows = census_datasets(args.registry, args.raw_dir, args.only)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents overwrites if the path appears mid-run.
    with destination.open("x", encoding="utf-8", newline="") as stream:
        rows.to_csv(stream, index=False)
    print(rows[["dataset", "status"]].to_string(index=False))
    if rows.status.eq("error").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
