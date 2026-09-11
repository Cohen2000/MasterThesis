#!/usr/bin/env python3
"""Build the pilot dataset: SIZES x substrates x families x rho targets x reps.

Change vs the original: a SIZE SWEEP. The original took a single --n; this takes
--sizes (a comma list of node counts) and loops over them, so one run produces
graphs spanning a wide range of node/edge counts. This is what spreads "walk
coverage" (= fraction of edges a fixed walk budget can see): tiny graphs ->
coverage near 1 (plug-in is trivially good), huge graphs -> coverage near 0
(plug-in must extrapolate from a thin sample). Coverage, not node count, is the
axis the downstream analysis lives on; the size sweep is just how we generate a
spread of coverage values together with the walk-budget ladder.

Each instance is tagged with its node count n; the family name embeds n so the
twin invariant (instances of ONE family share an identical collapsed graph,
per-edge event counts, and timestamp multiset) is preserved per (substrate, n).

Writes:
  out/<substrate>/<substrate>_n<n>_fam<f>/rho<target>_rep<r>.csv.gz
  out/manifest.csv          one row per instance incl. n and achieved ground truth
  out/calibration_report.md target-vs-achieved summary + invariant checks

Usage:
  # full landscape (heavy at the top size -> cluster):
  python make_pilot_data.py --out data_grid --sizes 400,2000,10000,50000,250000 \
      --substrates ba,er --families 4 --reps 3
  # quick local smoke test (seconds):
  python make_pilot_data.py --out data_smoke --sizes 400,2000 \
      --substrates ba,er --families 2 --reps 2 --targets 0.1,0.3,0.5
"""

import argparse
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from generator import make_family, make_instance

TARGETS = [0.05, 0.15, 0.25, 0.40, 0.55]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_grid")
    ap.add_argument("--substrates", default="ba,er",
                    help="ba,er recommended for the large sizes; lfr is slow/"
                         "fragile above ~50k nodes (add it only at moderate sizes)")
    ap.add_argument("--sizes", default="400,2000,10000,50000,250000",
                    help="comma list of node counts; the size sweep")
    ap.add_argument("--families", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--targets", default=",".join(map(str, TARGETS)))
    ap.add_argument("--timestamps", default="uniform", choices=["uniform", "bursty"])
    ap.add_argument("--burst-sigma", type=float, default=1.6)
    ap.add_argument("--n-targets", type=int, default=0,
                    help="number of continuous targets per family (default: len of grid)")
    ap.add_argument("--targets-mode", default="grid", choices=["grid", "continuous"],
                    help="grid: shared target list; continuous: per-family uniform draws in [0.02, 0.58]")
    ap.add_argument("--with-hub-bias", action="store_true",
                    help="additionally generate one hub-biased instance per (family, target)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    targets = [float(x) for x in args.targets.split(",")]
    sizes = [int(x) for x in args.sizes.split(",")]

    rows = []
    for n in sizes:
        for kind in args.substrates.split(","):
            for f in range(args.families):
                # seed depends on (kind, n, f) so different sizes get different graphs
                fam_seed = abs(zlib.crc32(f"{kind}|{n}|{f}".encode())) % 10_000
                name = f"{kind}_n{n}_fam{f}"
                try:
                    fam = make_family(name, kind, n=n, seed=fam_seed,
                                      timestamps=args.timestamps, burst_sigma=args.burst_sigma)
                except RuntimeError as err:
                    print(f"[skip] {name}: {err}")
                    continue
                if fam.max_rho < max(targets):
                    print(f"[warn] {name}: max reachable rho {fam.max_rho:.2f} "
                          f"< top target {max(targets)}")
                fam_dir = out / kind / name
                fam_dir.mkdir(parents=True, exist_ok=True)

                if args.targets_mode == "continuous":
                    rng_t = np.random.default_rng(zlib.crc32(f"{name}|targets".encode()))
                    fam_targets = sorted(rng_t.uniform(
                        0.02, 0.58, size=(args.n_targets or len(targets))).tolist())
                else:
                    fam_targets = targets

                multisets = []
                variants = [(False, r) for r in range(args.reps)]
                if args.with_hub_bias:
                    variants += [(True, 0)]
                for tgt in fam_targets:
                    for hub, rep in variants:
                        seed = zlib.crc32(
                            f"{name}|{tgt}|{rep}|{hub}".encode()) % 1_000_000
                        inst = make_instance(fam, tgt, seed=seed, hub_bias=hub)
                        tag = f"rho{tgt:.2f}_rep{rep}" + ("_hub" if hub else "")
                        path = fam_dir / f"{tag}.csv.gz"
                        inst.events.to_csv(path, index=False, compression="gzip")
                        multisets.append(np.sort(inst.events["t"].to_numpy()))
                        rows.append({
                            "substrate": kind, "n": n, "family": name,
                            "rho_target": tgt, "rep": rep, "hub_bias": hub,
                            "seed": seed, "path": str(path),
                            "n_events": len(inst.events),
                            "deviations": inst.deviations, **inst.achieved,
                        })
                ok = all(np.allclose(multisets[0], m) for m in multisets[1:])
                print(f"[fam ] {name}: {len(multisets)} instances, "
                      f"multiset invariant {'OK' if ok else 'BROKEN'}")
                if not ok:
                    raise RuntimeError(f"invariant broken in {name}")

    mf = pd.DataFrame(rows)
    mf.to_csv(out / "manifest.csv", index=False)

    # calibration report: target vs achieved rho, per (substrate, n, target)
    mf2 = mf.assign(dev=(mf.rho_headline - mf.rho_target).abs())
    rep = (mf2.groupby(["substrate", "n", "rho_target"])
              .agg(n_inst=("rho_headline", "size"),
                   achieved_mean=("rho_headline", "mean"),
                   achieved_sd=("rho_headline", "std"),
                   abs_dev_mean=("dev", "mean"),
                   occ_mean=("mean_span_frac", "mean"),
                   deviations_max=("deviations", "max"))
              .reset_index())
    md = ["# Pilot calibration report", "",
          f"instances: {len(mf)}, families: {mf.family.nunique()}, "
          f"substrates: {sorted(mf.substrate.unique())}, "
          f"sizes: {sorted(mf.n.unique())}", "",
          rep.round(4).to_markdown(index=False), "",
          "## Achieved secondary stats (mean over all instances)", "",
          mf[["share_single_event_pairs", "burstiness_pooled",
              "rho_event_weighted", "mean_span_frac", "C_one_step"]]
          .mean().round(3).to_markdown(), ""]
    (out / "calibration_report.md").write_text("\n".join(md))
    print(f"wrote {out/'manifest.csv'} and {out/'calibration_report.md'}")


if __name__ == "__main__":
    main()
