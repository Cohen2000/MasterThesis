"""Every number on the slide sheet, checked against the table it came from.

A stale number in `docs/PRESENTATION_NUMBERS.md` is the cheapest way to lose an
examiner's trust, and one had already slipped in: the Codex twin required gap
read -0.002 in the document against -0.053 in the CSV, because the row was
updated for one column and not the other when Codex finished.

Rather than parse the markdown -- which would silently pass whatever it failed
to recognise -- each claim is written here as an explicit query against its
source table. A number that is not in this manifest is not checked, and adding
a number to the sheet means adding it here too. `--list-unchecked` reports
figures in the document that no claim accounts for, so the gap is visible.

Exit code 1 if any claim disagrees with its source.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
G4 = REPO / "results_summary/g4"
TOL = 0.0006          # the sheet quotes three decimals
SHEET = REPO / "docs/PRESENTATION_NUMBERS.md"


def cell(csv: str, where: dict, column: str) -> float:
    frame = pd.read_csv(G4 / csv)
    for key, value in where.items():
        frame = frame[frame[key] == value]
    if len(frame) != 1:
        raise LookupError(f"{csv} {where} matched {len(frame)} rows, need 1")
    return float(frame.iloc[0][column])


# (label, quoted value, source csv, row selector, column)
CLAIMS = [
    ("slope think clean", 0.826, "primary_slope.csv",
     {"model": "qwen36-27b_think", "scope": "clean_arms (primary)"}, "slope"),
    ("slope nothink clean", 0.685, "primary_slope.csv",
     {"model": "qwen36-27b_nothink", "scope": "clean_arms (primary)"}, "slope"),
    ("slope codex clean", 0.830, "primary_slope.csv",
     {"model": "codex-gpt-5.6-sol", "scope": "clean_arms (primary)"}, "slope"),
    ("N1 qwen clean", 0.888, "primary_slope.csv",
     {"model": "qwen36-27b_think", "scope": "clean_arms (primary)"}, "null_N1"),
    ("N2 qwen clean", 0.617, "primary_slope.csv",
     {"model": "qwen36-27b_think", "scope": "clean_arms (primary)"},
     "null_N2_lookup"),
    ("perm think clean", 0.516, "primary_slope.csv",
     {"model": "qwen36-27b_think", "scope": "clean_arms (primary)"},
     "null_perm_mean"),
    ("slope think time_agnostic", 1.058, "primary_slope.csv",
     {"model": "qwen36-27b_think", "scope": "time_agnostic_t"}, "slope"),
    ("N1 time_agnostic", 0.210, "primary_slope.csv",
     {"model": "qwen36-27b_think", "scope": "time_agnostic_t"}, "null_N1"),

    ("skill think ta hidden", -1.216, "skill_scores.csv",
     {"model": "qwen36-27b_think", "strategy": "time_agnostic_t",
      "condition": "hidden"}, "skill_score"),
    ("skill think ta mechanism", 0.811, "skill_scores.csv",
     {"model": "qwen36-27b_think", "strategy": "time_agnostic_t",
      "condition": "mechanism"}, "skill_score"),
    ("skill think es direction_only", 0.760, "skill_scores.csv",
     {"model": "qwen36-27b_think",
      "strategy": "event_sample_then_full_history",
      "condition": "direction_only"}, "skill_score"),
    ("skill think es mechanism", 0.613, "skill_scores.csv",
     {"model": "qwen36-27b_think",
      "strategy": "event_sample_then_full_history",
      "condition": "mechanism"}, "skill_score"),
    ("skill nothink es direction_only", 0.840, "skill_scores.csv",
     {"model": "qwen36-27b_nothink",
      "strategy": "event_sample_then_full_history",
      "condition": "direction_only"}, "skill_score"),
    ("skill nothink es mechanism", 0.601, "skill_scores.csv",
     {"model": "qwen36-27b_nothink",
      "strategy": "event_sample_then_full_history",
      "condition": "mechanism"}, "skill_score"),

    ("coverage think ta hidden", 0.031, "stated_intervals.csv",
     {"model": "qwen36-27b_think", "strategy": "time_agnostic_t",
      "condition": "hidden"}, "empirical_coverage"),
    ("coverage think ta mechanism", 0.625, "stated_intervals.csv",
     {"model": "qwen36-27b_think", "strategy": "time_agnostic_t",
      "condition": "mechanism"}, "empirical_coverage"),
    ("width think ta hidden", 0.016, "stated_intervals.csv",
     {"model": "qwen36-27b_think", "strategy": "time_agnostic_t",
      "condition": "hidden"}, "median_width"),
    ("width think ta mechanism", 0.160, "stated_intervals.csv",
     {"model": "qwen36-27b_think", "strategy": "time_agnostic_t",
      "condition": "mechanism"}, "median_width"),

    ("twin gap think", 0.207, "twin_arms.csv",
     {"model": "qwen36-27b_think"}, "model_gap_mean"),
    ("twin gap nothink", 0.221, "twin_arms.csv",
     {"model": "qwen36-27b_nothink"}, "model_gap_mean"),
    ("twin gap codex", 0.159, "twin_arms.csv",
     {"model": "codex-gpt-5.6-sol"}, "model_gap_mean"),
    ("twin required qwen", -0.021, "twin_arms.csv",
     {"model": "qwen36-27b_think"}, "required_gap_mean"),
    ("twin required codex", -0.053, "twin_arms.csv",
     {"model": "codex-gpt-5.6-sol"}, "required_gap_mean"),

    ("wrongdir gap nothink", 0.309, "wrong_direction_contrast.csv",
     {"model": "qwen36-27b_nothink"}, "between_arm_gap"),
    ("wrongdir gap think", 0.352, "wrong_direction_contrast.csv",
     {"model": "qwen36-27b_think"}, "between_arm_gap"),
    ("wrongdir shift nothink ta", 0.237, "wrong_direction.csv",
     {"model": "qwen36-27b_nothink", "arm": "time_agnostic_t"},
     "shift_toward_stated"),
    ("wrongdir evidence think ta", 0.344, "wrong_direction.csv",
     {"model": "qwen36-27b_think", "arm": "time_agnostic_t"},
     "follows_evidence_rate"),

    ("profile slope think k3", 0.641, "profile_component_slopes.csv",
     {"model": "qwen36-27b_think", "scope": "clean_arms", "k": 3}, "slope"),
    ("profile slope think k5", 0.421, "profile_component_slopes.csv",
     {"model": "qwen36-27b_think", "scope": "clean_arms", "k": 5}, "slope"),
    ("profile slope nothink k4", 0.366, "profile_component_slopes.csv",
     {"model": "qwen36-27b_nothink", "scope": "clean_arms", "k": 4}, "slope"),
]


def list_unchecked(quoted: set[float]) -> list[str]:
    """Numbers in the sheet that no claim accounts for -- visibility, not gate."""
    text = SHEET.read_text()
    found = set()
    for match in re.finditer(r"(?<![\w.])[−-]?\d+\.\d{3}(?![\d])", text):
        found.add(float(match.group().replace("−", "-")))
    return sorted(f"{v:+.3f}" for v in found - quoted)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-unchecked", action="store_true")
    args = ap.parse_args()

    bad = 0
    for label, quoted, csv, where, column in CLAIMS:
        try:
            actual = cell(csv, where, column)
        except (LookupError, KeyError, FileNotFoundError) as exc:
            print(f"UNRESOLVED  {label:34s} {exc}")
            bad += 1
            continue
        if abs(actual - quoted) > TOL:
            print(f"MISMATCH    {label:34s} sheet {quoted:+.4f}  "
                  f"source {actual:+.4f}  ({csv})")
            bad += 1
        else:
            print(f"ok          {label:34s} {actual:+.4f}")

    if args.list_unchecked:
        rest = list_unchecked({round(q, 3) for _, q, _, _, _ in CLAIMS})
        print(f"\n{len(rest)} three-decimal figures in the sheet are not in "
              f"the manifest:\n  " + " ".join(rest))

    print(f"\n{len(CLAIMS) - bad}/{len(CLAIMS)} claims agree with their source")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
