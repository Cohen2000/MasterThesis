#!/usr/bin/env python3
"""Build the pilot dataset: substrates x families x rho targets x reps.

Writes:
  out/<substrate>/<family>/rho<target>_rep<r>.csv.gz   (event lists u, v, t)
  out/manifest.csv          one row per instance incl. achieved ground truth
  out/calibration_report.md target-vs-achieved summary + invariant checks

Usage:
  python make_pilot_data.py --out data_pilot --families 4 --reps 3
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
    ap.add_argument("--out", default="data_pilot")
    ap.add_argument("--substrates", default="er,ba,lfr")
    ap.add_argument("--n", type=int, default=300)
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

    rows = []
    for kind in args.substrates.split(","):
        for f in range(args.families):
            fam_seed = 1000 * (zlib.crc32(kind.encode()) % 97) + f
            name = f"{kind}_fam{f}"
            try:
                fam = make_family(name, kind, n=args.n, seed=abs(fam_seed) % 10_000,
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
                fam_targets = sorted(rng_t.uniform(0.02, 0.58, size=(args.n_targets or len(targets))).tolist())
            else:
                fam_targets = targets

            multisets = []
            variants = [(False, r) for r in range(args.reps)]
            if args.with_hub_bias:
                variants += [(True, 0)]
            for tgt in fam_targets:
                for hub, rep in variants:
                    seed = zlib.crc32(f"{name}|{tgt}|{rep}|{hub}".encode()) % 1_000_000
                    inst = make_instance(fam, tgt, seed=seed, hub_bias=hub)
                    tag = f"rho{tgt:.2f}_rep{rep}" + ("_hub" if hub else "")
                    path = fam_dir / f"{tag}.csv.gz"
                    inst.events.to_csv(path, index=False, compression="gzip")
                    multisets.append(np.sort(inst.events["t"].to_numpy()))
                    rows.append({
                        "substrate": kind, "family": name, "rho_target": tgt,
                        "rep": rep, "hub_bias": hub, "seed": seed,
                        "path": str(path), "n_events": len(inst.events),
                        "deviations": inst.deviations, **inst.achieved,
                    })
            ok = all(np.allclose(multisets[0], m) for m in multisets[1:])
            print(f"[fam ] {name}: {len(multisets)} instances, "
                  f"multiset invariant {'OK' if ok else 'BROKEN'}")
            if not ok:
                raise RuntimeError(f"invariant broken in {name}")

    mf = pd.DataFrame(rows)
    mf.to_csv(out / "manifest.csv", index=False)

    rep = (mf.groupby(["substrate", "rho_target"])
             .agg(n=("rho_headline", "size"),
                  achieved_mean=("rho_headline", "mean"),
                  achieved_sd=("rho_headline", "std"),
                  abs_dev_mean=("rho_headline",
                                lambda s: float(np.mean(np.abs(s - s.name if False else 0)))),
                  deviations_max=("deviations", "max"))
             .reset_index())
    rep["abs_dev_mean"] = (mf.assign(dev=(mf.rho_headline - mf.rho_target).abs())
                             .groupby(["substrate", "rho_target"])["dev"].mean()
                             .to_numpy())
    md = ["# Pilot calibration report", "",
          f"instances: {len(mf)}, families: {mf.family.nunique()}, "
          f"substrates: {sorted(mf.substrate.unique())}", "",
          rep.round(4).to_markdown(index=False), "",
          "## Achieved secondary stats (mean over all instances)", "",
          mf[["share_single_event_pairs", "burstiness_pooled",
              "rho_event_weighted", "C_one_step"]].mean().round(3).to_markdown(), ""]
    (out / "calibration_report.md").write_text("\n".join(md))
    print(f"wrote {out/'manifest.csv'} and {out/'calibration_report.md'}")


if __name__ == "__main__":
    main()
