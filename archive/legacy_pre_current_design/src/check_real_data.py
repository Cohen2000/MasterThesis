#!/usr/bin/env python3
"""List the real benchmark files that are present or still need downloading."""

import argparse
from pathlib import Path
import sys

import yaml

import census


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-config", default="config/benchmark.yaml")
    ap.add_argument("--registry", default="config/datasets.yaml")
    ap.add_argument("--preset", default="full", help="Preset whose real-data registry should be checked")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--no-fail", action="store_true")
    args = ap.parse_args()
    config = yaml.safe_load(open(args.benchmark_config))
    if args.preset not in config["presets"]:
        raise KeyError(f"unknown preset: {args.preset}")
    plan = config["presets"][args.preset]
    registry = census.load_registry(Path(args.registry))
    real_cfg = plan.get("real", {})
    keys = list(real_cfg.get("datasets", []))
    found = 0
    for key in keys:
        spec = registry[key]; p = Path(args.raw_dir) / spec["file"]
        if p.exists():
            found += 1
            print(f"[OK]      {key:24s} {p} ({p.stat().st_size / 1e6:.1f} MB)")
        else:
            print(f"[MISSING] {key:24s} expected {p}")
            print(f"          {spec.get('page_url', '')}")
    minimum = int(real_cfg.get("min_datasets", 0))
    print(f"\n{found}/{len(keys)} present; minimum for preset {args.preset}: {minimum}")
    if found < minimum and not args.no_fail:
        sys.exit(2)


if __name__ == "__main__":
    main()
