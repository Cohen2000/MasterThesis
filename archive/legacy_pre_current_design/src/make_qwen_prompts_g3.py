#!/usr/bin/env python3
"""Per-generation Qwen prompt files with an explicit `gen_seed`.

`run_llm_v2.HFModel` reseeds torch before every generation from the record's
`gen_seed`, falling back to `--seed`.  Running the same file three times with
one `--seed` therefore yields three byte-identical answers, and the response
noise measurement becomes an artifact of the runner rather than a property of
the model.  This is the single most likely silent failure in G3, so the seed is
written into the record rather than left to a command-line flag.

Seeds derive from the study master seed, so the whole run stays reproducible
from the one number recorded in `config/final_run_g2.yaml`.

One file per (model, generation).  `prompt_id` stays stable across generations
because resume is per file; `rep` records which generation a record belongs to.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from build_benchmark_data import stable_seed

# The G3 roster. Codex carries every subset; the Qwen rows carry less.
SCOPE = {
    "qwen36-27b_think": ("factorial", "mismatched", "metadata_only",
                         "seed_replication"),
    "qwen36-27b_nothink": ("factorial", "mismatched", "metadata_only"),
}


def build(records: list[dict], model: str, generation: int,
          master_seed: int) -> list[dict]:
    keep = SCOPE[model]
    out = []
    for record in records:
        if record["subset"] not in keep:
            continue
        row = dict(record)
        row["gen_seed"] = stable_seed(
            master_seed, model, record["prompt_id"], generation)
        row["rep"] = generation
        row["model_scope"] = model
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts",
                        default="results/final_run_g2/prompts.jsonl")
    parser.add_argument("--config", default="config/final_run_g2.yaml")
    parser.add_argument("--out-dir", default="results/final_run_g2/qwen")
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    master = int(config["final_run"]["master_seed"])
    generations = {
        "qwen36-27b_think":
            int(config["final_run"]["models"]["qwen3.6-27b_think"]["generations"]),
        "qwen36-27b_nothink":
            int(config["final_run"]["models"]["qwen3.6-27b_nothink"]["generations"]),
    }
    records = [json.loads(l) for l in
               Path(args.prompts).read_text().splitlines() if l.strip()]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seen_seeds: dict[str, set[int]] = {}
    for model, n_gen in generations.items():
        for generation in range(n_gen):
            rows = build(records, model, generation, master)
            path = out_dir / f"prompts_{model}_g{generation}.jsonl"
            with path.open("w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            seeds = {r["gen_seed"] for r in rows}
            seen_seeds.setdefault(model, set())
            overlap = seen_seeds[model] & seeds
            if overlap:
                raise RuntimeError(
                    f"{model} generation {generation} reuses "
                    f"{len(overlap)} seeds from an earlier generation")
            seen_seeds[model] |= seeds
            print(f"{path.name}: {len(rows)} prompts, "
                  f"{len(seeds)} distinct gen_seeds")
    for model, n_gen in generations.items():
        print(f"{model}: {n_gen} generations, "
              f"{len(seen_seeds[model])} distinct seeds in total")


if __name__ == "__main__":
    main()
