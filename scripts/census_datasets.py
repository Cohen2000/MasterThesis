#!/usr/bin/env python3
"""Local empirical census and W=2..20 sensitivity; no downloads or sampling."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset_census import compute_census


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=ROOT / "config/datasets.yaml")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--only", nargs="+", help="registry keys; default: all entries")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results/dataset_census")
    parser.add_argument("--realized-twins", action="store_true",
                        help="also attempt one deterministic low/high timing variant per dataset")
    args = parser.parse_args()
    destination = args.out_dir.resolve()
    for protected in (args.raw_dir.resolve(), ROOT / "data/raw", ROOT / "archive"):
        if destination == protected or protected in destination.parents:
            parser.error("output must be outside raw data and the historical archive")
    names = ["empirical_census.csv", "window_sensitivity.csv", "twin_feasibility.csv"]
    if args.realized_twins:
        names.append("twin_realized_feasibility.csv")
    if any((destination / name).exists() for name in names):
        parser.error("a requested output already exists; choose a new output directory")
    tables = compute_census(args.registry, args.raw_dir, args.only,
                            realized=args.realized_twins,
                            progress=lambda key, step: print(f"{key}: {step}", flush=True))
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in zip(names, tables):
        with (destination / name).open("x", encoding="utf-8", newline="") as stream:
            # Default float repr is the shortest string that round-trips exactly.
            table.to_csv(stream, index=False, na_rep="NA")
    print(tables[0][["dataset", "status"]].to_string(index=False))
    if tables[0].status.eq("error").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
