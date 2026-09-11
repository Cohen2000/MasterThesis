"""Robustness of the slope interval to the resampling scheme.

Twelve graph groups is few for a cluster bootstrap, and a percentile interval
over twelve clusters is optimistic in a way that is hard to see from the
interval itself. This adds three alternatives beside the preregistered one. The
preregistered variant does not change; these are a robustness row.

  cluster_group      preregistered: percentile bootstrap over the 12 groups
  wild_group         wild cluster bootstrap, Rademacher weights on the group
                     residuals. It resamples signs rather than clusters, so it
                     does not need the cluster count to be large for the
                     resampled design matrix to stay full rank -- the standard
                     recommendation for few clusters (Cameron, Gelbach & Miller).
  cluster_instance   the same percentile bootstrap over the 32 graph instances.
                     A finer unit gives a narrower interval, and it is *wrong*
                     if instances inside a group are dependent -- which is why
                     the group is preregistered. Reported to show the size of
                     the assumption, not as a competitor.
  leave_one_group    the slope with each group deleted in turn. Not an interval:
                     it answers whether one group carries the estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
DRAWS = 4000
SEED = 20260901


def _slope(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 2 or np.allclose(x, x[0]):
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def cluster_percentile(frame, x, y, unit, draws=DRAWS, seed=SEED):
    rng = np.random.default_rng(seed)
    units = frame[unit].to_numpy()
    uniq = np.unique(units)
    index = {u: np.flatnonzero(units == u) for u in uniq}
    xv, yv = frame[x].to_numpy(float), frame[y].to_numpy(float)
    out = np.empty(draws)
    for d in range(draws):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([index[u] for u in pick])
        out[d] = _slope(xv[idx], yv[idx])
    return out


def wild_cluster(frame, x, y, unit, draws=DRAWS, seed=SEED):
    """Rademacher wild cluster bootstrap around the fitted line.

    The cluster keeps its design; only the sign of its residual block flips.
    With twelve clusters there are 2^12 = 4096 distinct sign vectors, so 4000
    draws is close to enumerating them and the interval is not limited by
    resampling noise.
    """
    rng = np.random.default_rng(seed)
    xv, yv = frame[x].to_numpy(float), frame[y].to_numpy(float)
    b, a = np.polyfit(xv, yv, 1)
    resid = yv - (a + b * xv)
    units = frame[unit].to_numpy()
    uniq = np.unique(units)
    index = {u: np.flatnonzero(units == u) for u in uniq}
    out = np.empty(draws)
    for d in range(draws):
        star = yv.copy()
        for u in uniq:
            i = index[u]
            star[i] = a + b * xv[i] + resid[i] * rng.choice((-1.0, 1.0))
        out[d] = _slope(xv, star)
    return out


def leave_one_out(frame, x, y, unit):
    rows = []
    for u in sorted(frame[unit].unique()):
        part = frame[frame[unit] != u]
        rows.append({"dropped": u, "n_left": len(part),
                     "slope": _slope(part[x], part[y])})
    return pd.DataFrame(rows).sort_values("slope")


def table(frame, x, y, group="group_id", instance="instance_id"):
    point = _slope(frame[x], frame[y])
    rows = [{"scheme": "cluster_group (preregistered)",
             "units": frame[group].nunique(), **_band(cluster_percentile(frame, x, y, group))},
            {"scheme": "wild_group_rademacher",
             "units": frame[group].nunique(), **_band(wild_cluster(frame, x, y, group))},
            {"scheme": "cluster_instance",
             "units": frame[instance].nunique(), **_band(cluster_percentile(frame, x, y, instance))}]
    out = pd.DataFrame(rows)
    out.insert(0, "slope", point)
    out["width"] = out.hi - out.lo
    return out


def _band(draws):
    return {"lo": float(np.nanpercentile(draws, 2.5)),
            "hi": float(np.nanpercentile(draws, 97.5)),
            "sd": float(np.nanstd(draws, ddof=1)),
            "share_positive": float(np.mean(np.asarray(draws) > 0))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paired-cases",
                    default=str(REPO / "results_summary/g3/step1_paired_cases.csv"))
    ap.add_argument("--x", default="delta_i")
    ap.add_argument("--y", default="Delta_i")
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g3"))
    args = ap.parse_args()

    frame = pd.read_csv(args.paired_cases)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    schemes = table(frame, args.x, args.y)
    schemes.insert(0, "source", Path(args.paired_cases).name)
    schemes.to_csv(out / "bootstrap_robustness.csv", index=False)
    print(schemes.round(4).to_string(index=False))

    logo = leave_one_out(frame, args.x, args.y, "group_id")
    logo.insert(0, "source", Path(args.paired_cases).name)
    logo.to_csv(out / "bootstrap_leave_one_group_out.csv", index=False)
    print()
    print(f"leave-one-group-out: slope ranges {logo.slope.min():.4f} to "
          f"{logo.slope.max():.4f} over {len(logo)} groups")
    print(logo.round(4).head(3).to_string(index=False))
    print(logo.round(4).tail(2).to_string(index=False))


if __name__ == "__main__":
    main()
