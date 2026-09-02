"""Is the G4 panel's dropout ignorable?

Roughly a dozen of the 976 thinking prompts never terminate in any generation,
at any token budget, and a few more return a structurally complete answer whose
profile falls outside [0, 1]. Both are dropped before any slope is fitted. That
is only harmless if the loss is unrelated to the quantity being estimated.

Two levels of loss are separated, because they have different causes and
different remedies:

* `structural` -- no parseable complete record for that (prompt, generation).
  These are token-cap burns, not answers.
* `invalid` -- structurally complete, but a profile component outside [0, 1]
  or a provider refusal. The frozen rules treat these as invalid and they are
  never repaired.

The bias-relevant question is not the overall rate but whether loss tracks
`delta_i`, the size of the correct correction: if the hard cases are the ones
that burn, the reported slope is fitted on the easy half of the panel. That is
tested by the difference in mean `delta_i` between lost and kept records, with
a cluster bootstrap over graph groups, and separately for the two legs of the
paired contrast -- losing `mechanism` more often than `hidden` on high-`delta`
cases would move `Delta_i` directly.

Read-only. Writes tables under results_summary/g4/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/final_run_g2"
SEED = 20260901
DRAWS = 4000
LEGS = ("hidden", "mechanism")


def load(prompts: str, cases: str) -> pd.DataFrame:
    from build_g4_cell import MODEL_GROUPS, load_raw

    frames = []
    for name, groups in MODEL_GROUPS.items():
        try:
            frames.append(load_raw(name, groups, prompts, cases))
        except SystemExit as exc:
            print(f"{name}: skipped -- {exc}")
    frame = pd.concat(frames, ignore_index=True)
    frame["lost_structural"] = ~frame.structurally_complete
    frame["lost_invalid"] = (frame.structurally_complete
                             & (~frame.all_components_valid
                                | frame.provider_refused))
    frame["lost"] = frame.lost_structural | frame.lost_invalid
    return frame


def rate_table(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(["model"] + by, dropna=False)
    out = grouped.agg(
        attempted=("lost", "size"),
        lost_structural=("lost_structural", "sum"),
        lost_invalid=("lost_invalid", "sum"),
        lost=("lost", "sum")).reset_index()
    out["loss_rate"] = out.lost / out.attempted
    return out.sort_values(["model"] + by)


def cluster_bootstrap_gap(part: pd.DataFrame, rng: np.random.Generator
                          ) -> tuple[float, float, float, float]:
    """Mean delta_i among lost minus among kept, resampling graph groups."""
    groups = sorted(part.group_id.dropna().unique())
    by_group = {g: part[part.group_id == g] for g in groups}
    draws = []
    for _ in range(DRAWS):
        pick = rng.choice(groups, size=len(groups), replace=True)
        sample = pd.concat([by_group[g] for g in pick], ignore_index=True)
        lost, kept = sample[sample.lost], sample[~sample.lost]
        if lost.empty or kept.empty:
            continue
        draws.append(lost.delta_i.mean() - kept.delta_i.mean())
    lost, kept = part[part.lost], part[~part.lost]
    point = float(lost.delta_i.mean() - kept.delta_i.mean())
    if not draws:
        return point, np.nan, np.nan, np.nan
    draws = np.asarray(draws)
    return (point, float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)), float((draws > 0).mean()))


def delta_gap_table(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for (model, scope), part in _scopes(frame):
        n_lost = int(part.lost.sum())
        if n_lost == 0 or n_lost == len(part):
            rows.append({"model": model, "scope": scope, "n": len(part),
                         "n_lost": n_lost, "delta_gap": np.nan,
                         "ci_lo": np.nan, "ci_hi": np.nan,
                         "share_positive_draws": np.nan,
                         "mean_delta_lost": np.nan,
                         "mean_delta_kept": float(part.delta_i.mean())})
            continue
        point, lo, hi, share = cluster_bootstrap_gap(part, rng)
        rows.append({"model": model, "scope": scope, "n": len(part),
                     "n_lost": n_lost, "delta_gap": point, "ci_lo": lo,
                     "ci_hi": hi, "share_positive_draws": share,
                     "mean_delta_lost": float(part[part.lost].delta_i.mean()),
                     "mean_delta_kept": float(part[~part.lost].delta_i.mean())})
    return pd.DataFrame(rows)


def _scopes(frame: pd.DataFrame):
    """Pooled scopes first, then one per arm.

    `delta_i` differs by an order of magnitude across arms and the loss is not
    spread evenly over them, so a pooled gap is confounded by which arm burned.
    The within-arm rows are the ones that carry the argument.
    """
    for model, part in frame.groupby("model"):
        yield (model, "all_conditions"), part
        legs = part[part.condition.isin(LEGS)]
        if len(legs):
            yield (model, "paired_legs"), legs
        for leg in LEGS:
            leg_part = part[part.condition == leg]
            if len(leg_part):
                yield (model, f"leg:{leg}"), leg_part
        for arm, arm_part in part.groupby("strategy"):
            yield (model, f"arm:{arm}"), arm_part


def paired_loss(frame: pd.DataFrame) -> pd.DataFrame:
    """Per (model, case, generation): which leg of the contrast survived.

    An asymmetric loss is the one that moves `Delta_i` without moving the
    predictor, so the two one-sided columns are reported separately rather
    than folded into a single "incomplete pair" count.
    """
    legs = frame[frame.condition.isin(LEGS)]
    wide = legs.pivot_table(index=["model", "case_id", "strategy", "generation",
                                   "delta_i"],
                            columns="condition", values="lost",
                            aggfunc="first")
    wide = wide.dropna(subset=list(LEGS)).reset_index()
    # pivot_table returns the boolean loss flags as numbers; `~` on those is a
    # bitwise complement, which is truthy for both 0 and 1 and would report
    # every pair as kept.
    for leg in LEGS:
        wide[leg] = wide[leg].astype(bool)
    wide["both_kept"] = ~wide.hidden & ~wide.mechanism
    wide["only_hidden_lost"] = wide.hidden & ~wide.mechanism
    wide["only_mechanism_lost"] = ~wide.hidden & wide.mechanism
    wide["both_lost"] = wide.hidden & wide.mechanism
    out = (wide.groupby(["model", "strategy"])
           .agg(pairs=("both_kept", "size"),
                both_kept=("both_kept", "sum"),
                only_hidden_lost=("only_hidden_lost", "sum"),
                only_mechanism_lost=("only_mechanism_lost", "sum"),
                both_lost=("both_lost", "sum"),
                mean_delta_pairs_kept=("delta_i", "mean")).reset_index())
    out["asymmetric"] = out.only_hidden_lost + out.only_mechanism_lost
    out["pair_loss_rate"] = 1 - out.both_kept / out.pairs
    return out


def coverage_bins(frame: pd.DataFrame) -> pd.DataFrame:
    part = frame.dropna(subset=["coverage"]).copy()
    part["coverage_quartile"] = pd.qcut(
        part.coverage, 4, labels=["q1_lowest", "q2", "q3", "q4_highest"])
    return rate_table(part, ["coverage_quartile"])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(BASE / "prompts.jsonl"))
    ap.add_argument("--cases", default=str(BASE / "final_cases.csv.gz"))
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    frame = load(args.prompts, args.cases)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    overall = rate_table(frame, [])
    by_arm = rate_table(frame, ["strategy"])
    by_condition = rate_table(frame, ["condition"])
    by_generation = rate_table(frame, ["generation"])
    by_coverage = coverage_bins(frame)
    gaps = delta_gap_table(frame)
    pairs = paired_loss(frame)

    overall.to_csv(out / "missingness_overall.csv", index=False)
    by_arm.to_csv(out / "missingness_by_arm.csv", index=False)
    by_condition.to_csv(out / "missingness_by_condition.csv", index=False)
    by_generation.to_csv(out / "missingness_by_generation.csv", index=False)
    by_coverage.to_csv(out / "missingness_by_coverage.csv", index=False)
    gaps.to_csv(out / "missingness_delta_gap.csv", index=False)
    pairs.to_csv(out / "missingness_paired.csv", index=False)

    lost = frame[frame.lost][["model", "generation", "prompt_id", "strategy",
                              "condition", "coverage", "delta_i",
                              "lost_structural", "lost_invalid"]]
    lost.sort_values(["model", "generation", "prompt_id"]).to_csv(
        out / "missingness_records.csv", index=False)

    pd.set_option("display.width", 200)
    print("== overall\n", overall.to_string(index=False))
    print("\n== by arm\n", by_arm.to_string(index=False))
    print("\n== by condition\n", by_condition.to_string(index=False))
    print("\n== by coverage quartile\n", by_coverage.to_string(index=False))
    print("\n== delta_i gap, lost minus kept (cluster bootstrap over groups)\n",
          gaps.to_string(index=False))
    print("\n== paired legs\n", pairs.to_string(index=False))
    print(f"\n{len(lost)} lost records -> {out / 'missingness_records.csv'}")


if __name__ == "__main__":
    main()
