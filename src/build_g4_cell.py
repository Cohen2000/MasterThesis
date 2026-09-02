"""Assemble the per-(model, case, condition, generation) table for G4.

One table, two consumers: `report_dispersion_coverage.py` needs every
generation kept separately, `report_twin_arms.py` needs the paired contrast
after generations are averaged away. Both are written here so the two analyses
cannot silently disagree about which records were usable.

Validity follows the frozen rules and is not relaxed: a record enters only if
it is structurally complete, carries all four profile components in [0, 1], and
was not a provider refusal. Nothing is repaired.
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "results/final_run_g2"


def load_model(name: str, patterns: list[str], prompts: str,
               cases: str) -> pd.DataFrame:
    from report_step1_signal import load_answers, merge

    paths = []
    for pattern in patterns:
        hits = sorted(glob.glob(pattern))
        if not hits:
            raise SystemExit(f"no answer files match {pattern}")
        paths.extend(hits)
    frame = merge(load_answers(paths), prompts, cases)
    frame = frame[frame.all_components_valid & ~frame.provider_refused]
    frame["model"] = name
    return frame


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", default=str(BASE / "prompts.jsonl"))
    ap.add_argument("--cases", default=str(BASE / "final_cases.csv.gz"))
    ap.add_argument("--out-dir", default=str(REPO / "results_summary/g4"))
    args = ap.parse_args()

    q = str(BASE / "answers/qwen")
    models = {
        "qwen36-27b_think": [f"{q}/answers_vllm_qwen36-27b_think_g*.shard*.jsonl"],
        "qwen36-27b_nothink": [f"{q}/answers_vllm_qwen36-27b_nothink_g*.shard*.jsonl"],
        "codex-gpt-5.6-sol": [str(BASE / "answers/step1_codex_gen0.jsonl"),
                              str(BASE / "answers/step2_codex_gen0.jsonl")],
    }

    frames = []
    for name, patterns in models.items():
        try:
            frames.append(load_model(name, patterns, args.prompts, args.cases))
            print(f"{name}: {len(frames[-1])} valid records")
        except SystemExit as exc:
            print(f"{name}: skipped -- {exc}")
    cell = pd.concat(frames, ignore_index=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # The thinking prompt set carries the seed-replication subset, so some
    # instances appear at several seed slots and others at one. Any analysis
    # that indexes on the instance would weight those graphs more heavily
    # without saying so, which is why the slot travels with every row.
    cell["seed_slot"] = cell.case_id.str.extract(r"\|(?:ss|ws)(\d+)\|").astype(int)
    keep = ["model", "case_id", "instance_id", "group_id", "strategy",
            "condition", "coverage", "generation", "seed_slot", "rho_k2",
            "rho_W5_k2", "est__plugin_rho_k2", "delta_i"]
    cell[keep].to_csv(out / "g4_cell.csv", index=False)

    # Generations are not independent observations; the nesting is
    # graph -> seed -> condition -> generation, so the generation axis is
    # averaged away before any paired contrast, exactly as in Step 1.
    averaged = (cell.groupby(["model", "case_id", "instance_id", "group_id",
                              "strategy", "condition", "coverage", "seed_slot",
                              "delta_i"],
                             as_index=False)
                .agg(rho_k2=("rho_k2", "mean"),
                     generations=("generation", "nunique")))
    wide = averaged.pivot_table(
        index=["model", "case_id", "instance_id", "group_id", "strategy",
               "coverage", "seed_slot", "delta_i"],
        columns="condition", values="rho_k2").reset_index()
    wide = wide.dropna(subset=["hidden", "mechanism"])
    wide["Delta_i"] = wide.mechanism - wide.hidden
    wide.to_csv(out / "g4_paired.csv", index=False)

    print(f"\ncell     {len(cell):6d} records -> {out / 'g4_cell.csv'}")
    print(f"paired   {len(wide):6d} cases   -> {out / 'g4_paired.csv'}")
    print(wide.groupby(["model", "strategy"]).size().to_string())


if __name__ == "__main__":
    main()
