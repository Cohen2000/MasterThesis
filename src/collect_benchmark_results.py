#!/usr/bin/env python3
"""Create the small result bundle to return after the estimator screen."""

import argparse
from pathlib import Path
import zipfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="full")
    ap.add_argument("--data-dir")
    ap.add_argument("--results-dir")
    ap.add_argument("--out")
    ap.add_argument("--logs", nargs="*", default=[])
    args = ap.parse_args()
    data = Path(args.data_dir or f"data/benchmark_{args.preset}")
    results = Path(args.results_dir or f"results/benchmark_{args.preset}")
    out = Path(args.out or f"benchmark_{args.preset}_results_to_share.zip")
    wanted = [
        data / "manifest.csv", data / "effective_plan.yaml",
        data / "missing_real_datasets.tsv",
        results / "cases.csv.gz", results / "predictions.csv.gz",
        results / "metrics.csv", results / "rankings.csv",
        results / "case_diagnostics.csv.gz",
        results / "target_distribution_by_block.csv",
        results / "edgebank_auc_summary.csv",
        results / "seed_noise_floor.csv",
        results / "headline_all_protocols.csv",
        results / "V2_DIAGNOSTICS.md",
        results / "SCREEN_SUMMARY.md", results / "headline_ranking.png",
    ]
    wanted.extend(sorted(results.glob("cases_shard_*.csv.gz")))
    for pattern in args.logs:
        wanted.extend(sorted(Path().glob(pattern)))
    files, seen = [], set()
    for p in wanted:
        if p.exists() and p.resolve() not in seen:
            files.append(p); seen.add(p.resolve())
    required = [data / "manifest.csv", results / "metrics.csv",
                results / "rankings.csv", results / "SCREEN_SUMMARY.md"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("run is incomplete; missing: " + ", ".join(missing))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as zf:
        for p in files:
            if p.is_relative_to(data):
                arc = Path("data") / p.relative_to(data)
            elif p.is_relative_to(results):
                arc = Path("results") / p.relative_to(results)
            else:
                arc = Path("logs") / p.name
            zf.write(p, arcname=str(arc))
    print(f"wrote {out} with {len(files)} files ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
