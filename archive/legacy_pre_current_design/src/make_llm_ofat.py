#!/usr/bin/env python3
"""Assemble the OFAT input ablation: six cells, 36 paired cases, R replicates.

The design isolates one factor at a time around the `mask` reference cell
instead of walking a cumulative ladder, so every cell differs from `mask` in
exactly one respect:

    nw            no mask at all (the weakest input)
    mask          reference
    mask_crawl    + crawl statistics
    mask_temporal + temporal statistics
    mask_recent   + recent-event list
    mask_all      + all three

Four of those cells already exist in the frozen 420-prompt suite under their
historical ladder names, on the same 36 cases:

    mask_crawl == mask_crawl_full
    mask_all   == mask_crawl_temporal_recent

Their prompt text is copied verbatim rather than regenerated -- the frozen
prompts are immutable, and reusing the exact text is what lets the answers
already collected for Codex, Gemini, DeepSeek V4 Pro and Qwen count as cells
of this ablation instead of being thrown away.

The master file can represent replicates, but the execution plans remain
deliberately small: only Gemini gets three replies for the two new cells;
Codex and both Qwen modes get the missing cells once, while DeepSeek V4 Flash
gets all six cells once because its V4 Pro answers cannot be mixed in. Rep 1
keeps the original prompt_id so existing answers still match; later reps get
a suffixed id. Replicate 1 deliberately keeps generation seed 0 as well: the
existing Qwen answers used seed 0, so changing only the two newly rendered
cells to a hash seed would confound the input-factor comparison with a
decoding-seed change.

  PYTHONPATH=src python src/make_llm_ofat.py --reps 3
"""

import argparse
import hashlib
import json
from pathlib import Path

FROZEN_PROMPTS = "results/llm_v2/prompts.jsonl"
NEW_CELLS = "results/llm_v21_ofat/prompts_ofat_new_cells.jsonl"

# canonical OFAT name -> name the frozen suite stored it under
FROM_FROZEN = {
    "nw": "nw",
    "mask": "mask",
    "mask_crawl": "mask_crawl_full",
    "mask_all": "mask_crawl_temporal_recent",
}
FROM_NEW = ("mask_temporal", "mask_recent")
CELLS = list(FROM_FROZEN) + list(FROM_NEW)


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def generation_seed(prompt_id, rep):
    digest = hashlib.sha1(f"ofat|{prompt_id}|rep{rep}".encode())
    return int(digest.hexdigest()[:8], 16) % (2 ** 31 - 1)


def generation_seed_for_rep(prompt_id, rep):
    """Match the frozen seed on rep 1; vary deterministic later HF reps."""
    return 0 if rep == 1 else generation_seed(prompt_id, rep)


def include_in_plan(name, row):
    """Exact work inventory; existing answers are intentionally excluded."""
    is_new = row["input_kind"] in FROM_NEW
    if name == "codex":
        return is_new and row["rep"] == 1
    if name == "gemini":
        return is_new                    # three reps of the two new cells
    if name == "deepseek":
        return row["rep"] == 1           # all six cells fresh, once
    if name == "qwen":
        return is_new and row["rep"] == 1
    raise ValueError(f"unknown plan: {name}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--frozen", default=FROZEN_PROMPTS)
    ap.add_argument("--new-cells", default=NEW_CELLS)
    ap.add_argument("--out", default="results/llm_v21_ofat/prompts_ofat.jsonl")
    ap.add_argument("--deepseek-shards", type=int, default=8,
                    help="parallel NIM processes; more is faster "
                         "until the free tier answers with 429")
    ap.add_argument("--gemini-shards", type=int, default=3)
    args = ap.parse_args()

    if args.reps < 1:
        raise SystemExit("--reps must be at least 1")
    if args.deepseek_shards < 1 or args.gemini_shards < 1:
        raise SystemExit("shard counts must be at least 1")

    new_rows = read_jsonl(args.new_cells)
    cases = sorted({r["case_id"] for r in new_rows})
    if len(cases) != 36:
        raise SystemExit(f"expected 36 ablation cases, found {len(cases)}")

    frozen = read_jsonl(args.frozen)
    base = []
    for canonical, legacy in FROM_FROZEN.items():
        rows = [r for r in frozen
                if r["condition"] == "disclosed"
                and r["input_kind"] == legacy
                and r["case_id"] in set(cases)]
        if len(rows) != 36:
            raise SystemExit(
                f"cell {canonical}: expected 36 frozen prompts, got {len(rows)}")
        for r in rows:
            base.append({**r, "input_kind": canonical,
                         "legacy_input_kind": legacy, "source": "frozen"})
    for r in new_rows:
        if r["input_kind"] not in FROM_NEW:
            raise SystemExit(f"unexpected cell in new file: {r['input_kind']}")
        base.append({**r, "legacy_input_kind": r["input_kind"],
                     "source": "new"})

    if len(base) != len(CELLS) * 36:
        raise SystemExit(f"expected {len(CELLS) * 36} prompts, got {len(base)}")

    records = []
    for rep in range(1, args.reps + 1):
        for row in base:
            pid = row["prompt_id"]
            # Replicate 1 keeps the frozen identity so answers already
            # collected under that id still match; later reps must not.
            rep_id = pid if rep == 1 else f"{pid}__r{rep}"
            records.append({
                **row, "id": rep_id, "prompt_id": rep_id,
                "base_prompt_id": pid, "rep": rep,
                "gen_seed": generation_seed_for_rep(pid, rep),
                # The frozen suite predates this field; recomputing it here
                # gives every cell one content identity, so two cells that
                # render the same text stay poolable across the rename.
                "prompt_sha256": hashlib.sha256(
                    row["prompt"].encode()).hexdigest()})

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")
    print(f"{len(records)} prompts -> {out}")
    print(f"  {len(CELLS)} cells x 36 cases x {args.reps} reps")

    # Per-model prompt files, not id lists. The Gemini runner's completion
    # loop counts what is left in the prompt file it was given, so a file that
    # contains work the model is not supposed to do would never reach zero.
    # A file per model also means every runner is invoked the same way.
    # Exact intended workload: Codex 72, Gemini 216, DeepSeek 216 and Qwen 72
    # per mode.  Only DeepSeek is rebuilt across all six cells.
    plans = {name: (lambda r, n=name: include_in_plan(n, r))
             for name in ("codex", "gemini", "deepseek", "qwen")}
    # Shards are separate files, not --shard-index: the Gemini runner's
    # completion loop counts what is left in the file it was handed, and a
    # shard of a larger file would never reach zero. Round-robin keeps every
    # shard balanced across cells, reps and cases.
    shards = {"deepseek": args.deepseek_shards, "gemini": args.gemini_shards}
    for name, keep in plans.items():
        sel = [r for r in records if keep(r)]
        path = out.parent / f"prompts_ofat_{name}.jsonl"
        with open(path, "w") as fh:
            for r in sel:
                fh.write(json.dumps(r) + "\n")
        cells = sorted({r["input_kind"] for r in sel})
        reps = sorted({r["rep"] for r in sel})
        print(f"  {name:<9} {len(sel):>4} prompts  reps {reps}  "
              f"{len(cells)} cells -> {path}")
        n_shard = int(shards.get(name, 1) or 1)
        if n_shard > 1:
            for i in range(n_shard):
                part = sel[i::n_shard]
                sp = out.parent / f"prompts_ofat_{name}.shard{i}.jsonl"
                with open(sp, "w") as fh:
                    for r in part:
                        fh.write(json.dumps(r) + "\n")
            print(f"  {'':<9} split into {n_shard} shards of "
                  f"~{len(sel) // n_shard} prompts")

    # Remove only superseded generated prompt plans from the abandoned full-
    # replication proposal.  Answer files are never touched.
    for stale_name in ("prompts_ofat_codex_fresh.jsonl",
                       "prompts_ofat_qwen_core.jsonl",
                       "prompts_ofat_qwen_full.jsonl"):
        stale = out.parent / stale_name
        if stale.exists():
            stale.unlink()

    ids_dir = out.parent / "ids"
    ids_dir.mkdir(parents=True, exist_ok=True)
    id_sets = {
        "all": records,
        "new_cells": [r for r in records if r["input_kind"] in FROM_NEW],
        "new_cells_rep1": [r for r in records
                           if r["input_kind"] in FROM_NEW and r["rep"] == 1],
        "new_cells_rep12": [r for r in records
                            if r["input_kind"] in FROM_NEW and r["rep"] <= 2],
        "extra_reps_only": [r for r in records
                            if r["rep"] > 1
                            and r["input_kind"] in FROM_NEW],
    }
    for name, selected in id_sets.items():
        (ids_dir / f"{name}.txt").write_text(
            ",".join(r["prompt_id"] for r in selected) + "\n")


if __name__ == "__main__":
    main()
