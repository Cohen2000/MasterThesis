#!/usr/bin/env python3
"""The missing cell: correct process description, false direction claim.

Added 2026-09-01. Recorded here because it matters for how the cell is read:
at the time it was written the Step 1 slice was still running and its numbers
had deliberately not been interpreted, and no Qwen answer had been analysed.
It is a prespecified addition, not a post-hoc one.

`mismatched` swaps the entire mechanism description, so a model that shifts
under it may be reacting to a process that plainly does not fit the sample it
can see. This cell keeps the description correct and inverts only the sentence
claiming which way the naive estimate errs. That isolates deference to an
explicit claim from derivation out of the described process.

Two arms, chosen so their correct directions are opposite: a walk needing an
upward correction and the one arm needing a downward one. An arm that is
approximately unbiased has no direction to invert and is excluded.

Its own file and its own hash. The frozen prompt set is not touched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

import prompt_contract_g1 as C

CONDITION = "mechanism_wrong_direction"
ARMS = ("time_agnostic_t", "event_sample_then_full_history")
ADDED = "2026-09-01"


def build(primary: pd.DataFrame) -> list[dict]:
    out = []
    for row in primary[primary.strategy.isin(ARMS)].to_dict("records"):
        text = C.build_prompt(row, CONDITION)
        out.append({
            "prompt_id": f"{row['case_id']}|{CONDITION}",
            "case_id": str(row["case_id"]),
            "instance_id": str(row["instance_id"]),
            "group_id": str(row["group_id"]),
            "strategy": str(row["strategy"]),
            "condition": CONDITION,
            "stated_direction": "inverted",
            "correct_direction": ("upward" if row["strategy"] != ARMS[1]
                                  else "downward"),
            "subset": "wrong_direction",
            "added": ADDED,
            "input_kind": "mask",
            "seed_slot": int(row.get("seed_slot", 0)),
            "budget": int(row["budget"]),
            "coverage": float(row["coverage"]),
            "prompt": text,
            "prompt_sha256": hashlib.sha256(text.encode()).hexdigest(),
        })
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary",
                        default="results/final_run_g2/primary_cases.csv.gz")
    parser.add_argument("--config", default="config/final_run_g2.yaml")
    parser.add_argument("--out-dir", default="results/final_run_g2/qwen")
    parser.add_argument("--summary-dir", default="results_summary/g3")
    args = parser.parse_args()

    import yaml
    from build_benchmark_data import stable_seed

    master = int(yaml.safe_load(open(args.config))["final_run"]["master_seed"])
    records = build(pd.read_csv(args.primary))
    if len(records) != 64:
        raise RuntimeError(f"expected 64 prompts, built {len(records)}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for model in ("qwen36-27b_think", "qwen36-27b_nothink"):
        for generation in range(3):
            rows = []
            for record in records:
                row = dict(record)
                row["gen_seed"] = stable_seed(
                    master, model, record["prompt_id"], generation)
                row["rep"] = generation
                rows.append(row)
            path = out_dir / f"prompts_wrongdir_{model}_g{generation}.jsonl"
            path.write_text("".join(
                json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
            written.append(path)

    digest = hashlib.sha256(
        "\n".join(sorted(r["prompt_sha256"] for r in records)).encode()
    ).hexdigest()
    summary = Path(args.summary_dir)
    summary.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "condition": CONDITION, "added": ADDED, "prompts": len(records),
        "arms": ", ".join(ARMS), "models": "qwen think + nothink, 3 gens each",
        "calls": len(records) * 3 * 2,
        "prompt_set_sha256": digest,
        "frozen_set_touched": False,
    }]).to_csv(summary / "wrong_direction_cell.csv", index=False)

    print(f"{len(records)} prompts, {len(written)} files")
    print(f"prompt set sha256: {digest}")
    for arm in ARMS:
        n = sum(1 for r in records if r["strategy"] == arm)
        print(f"  {arm}: {n}")


if __name__ == "__main__":
    main()
