#!/usr/bin/env python3
"""Prompts for separating an LLM's two noise sources on the panel.

`docs/SEED_VS_GRAPH_EVIDENCE.md` measured how much a redrawn walk moves a
*classical* estimator, and found the effect negligible. That result does not
transfer to a language model by assumption: a model consumes the same sample
but adds a second noise source the estimators do not have, namely its own
sampling at generation time. Both have to be measured before the replication
budget of the final run can be set.

Two arms, sharing one cell so they are on the same scale:

  response arm  one fixed walk (seed 0), the identical prompt generated R
                times. Within-graph spread here is the model's own response
                noise, s2_resp.
  input arm     R different walk seeds of the same graph, one generation
                each. Within-graph spread is s2_resp + s2_input.

The difference identifies s2_input. Prompt text is identical across the
response arm by construction, so only `prompt_id` distinguishes the repeats --
which is what makes `run_llm_v2.py` treat them as separate work units and
`llm_eval_frozen.load_answers` keep them apart.

This probe reports variance components only. Its error *levels* are not an
LLM result for the final panel: the panel's own comparison has not been frozen
yet, and reading a ranking off a variance probe would be exactly the kind of
design-after-the-fact this project keeps out of the main study.

  PYTHONPATH=src python src/make_llm_noise_probe.py \
      --cases results/panel_seed_probe/cases.csv.gz \
      --out results/llm_noise_probe/prompts.jsonl
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from make_llm_prompts_v2 import build_prompt, parse_input_kind

RESPONSE_ARM = "response"
INPUT_ARM = "input"


def probe_prompt_id(case_id, condition, kind, rep):
    """Distinct per repeat, and namespaced away from the frozen suite."""
    key = f"noise_probe|{case_id}|{condition}|{kind}|rep{rep}"
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def generation_seed(instance_id, rep):
    """Per-record decoding seed, varied in BOTH arms.

    `run_llm_v2.HFModel` reseeds torch before every generation, so on the
    cluster backend an identical prompt decodes identically unless the record
    supplies its own seed. Without this the response arm would report zero
    noise by construction. Both arms vary it, so the arms differ only in
    whether the walk behind the prompt was redrawn.
    """
    digest = hashlib.sha1(f"noise_probe|{instance_id}|rep{rep}".encode())
    return int(digest.hexdigest()[:8], 16) % (2 ** 31 - 1)


def build_records(cases, condition, kind, repeats):
    """One record per (graph, arm cell); the shared cell carries both arms."""
    records = []
    for instance_id, block in cases.groupby("instance_id", sort=True):
        block = block.sort_values("walk_seed")
        seeds = list(block["walk_seed"])[:repeats]
        if len(seeds) < repeats:
            raise SystemExit(
                f"{instance_id}: need {repeats} walk seeds, found {len(seeds)}")
        base = block[block.walk_seed == seeds[0]].iloc[0].to_dict()

        for rep in range(1, repeats + 1):
            arms = [RESPONSE_ARM] + ([INPUT_ARM] if rep == 1 else [])
            records.append((base, seeds[0], rep, arms))
        for position, seed in enumerate(seeds[1:], start=2):
            row = block[block.walk_seed == seed].iloc[0].to_dict()
            records.append((row, seed, position, [INPUT_ARM]))

    out = []
    for row, seed, rep, arms in records:
        prompt = build_prompt(row, condition, kind, None)
        pid = probe_prompt_id(row["case_id"], condition, kind, rep)
        base, factors = parse_input_kind(kind)
        out.append({
            "id": pid, "prompt_id": pid, "case_id": row["case_id"],
            "instance_id": row["instance_id"], "group_id": row["group_id"],
            "graph_category": row.get("graph_category", ""),
            "strategy": row["strategy"], "budget": int(row["budget"]),
            "walk_seed": int(seed), "rep": int(rep), "arms": ",".join(arms),
            "gen_seed": generation_seed(row["instance_id"], rep),
            "condition": condition, "input_kind": kind,
            "input_base": base, "input_factors": ",".join(factors),
            "coverage": float(row["coverage"]),
            "prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", default="results/panel_seed_probe/cases.csv.gz")
    ap.add_argument("--out", default="results/llm_noise_probe/prompts.jsonl")
    ap.add_argument("--strategy", default="time_agnostic_t")
    ap.add_argument("--budget", type=int, default=800)
    ap.add_argument("--condition", default="disclosed")
    ap.add_argument("--input-kind", default="mask")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--frozen-prompts", default="results/llm_v2/prompts.jsonl",
                    help="checked for prompt_id collisions only; never read "
                         "for content and never written to")
    args = ap.parse_args()

    cases = pd.read_csv(args.cases)
    cases = cases[(cases.budget == args.budget)
                  & (cases.strategy == args.strategy)].copy()
    if cases.empty:
        raise SystemExit(f"no cases for {args.strategy} at budget {args.budget}")

    records = build_records(cases, args.condition, args.input_kind,
                            args.repeats)

    ids = {r["prompt_id"] for r in records}
    if len(ids) != len(records):
        raise SystemExit("duplicate prompt_id in probe")
    frozen = Path(args.frozen_prompts)
    if frozen.exists():
        taken = {json.loads(l)["prompt_id"] for l in open(frozen) if l.strip()}
        if ids & taken:
            raise SystemExit("probe prompt_id collides with the frozen suite")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    frame = pd.DataFrame(records)
    resp = frame[frame.arms.str.contains(RESPONSE_ARM)]
    inp = frame[frame.arms.str.contains(INPUT_ARM)]
    texts = resp.groupby("instance_id").prompt_sha256.nunique()
    assert (texts == 1).all(), "response arm must repeat one identical prompt"
    texts_in = inp.groupby("instance_id").prompt_sha256.nunique()
    assert (texts_in == args.repeats).all(), "input arm needs distinct walks"

    for arm in (RESPONSE_ARM, INPUT_ARM):
        sub = frame[frame.arms.str.contains(arm)]
        per_graph = sub.groupby("instance_id").gen_seed.nunique()
        assert (per_graph == args.repeats).all(), \
            f"{arm} arm must vary the decoding seed across repeats"

    chars = frame.prompt.str.len()
    print(f"wrote {out_path}: {len(records)} prompts, "
          f"{frame.instance_id.nunique()} graphs, "
          f"{frame.group_id.nunique()} groups")
    print(f"  response arm {len(resp)} generations "
          f"({args.repeats} identical prompts per graph)")
    print(f"  input arm    {len(inp)} generations "
          f"({args.repeats} walk seeds per graph)")
    both = int((frame.arms == f"{RESPONSE_ARM},{INPUT_ARM}").sum())
    print(f"  shared cell  {both} prompts counted in both arms "
          f"(one per graph), so {len(records)} generations in total")
    print(f"  prompt chars p50={int(chars.median())} max={int(chars.max())} "
          f"(~tokens: /3.7 -> p50 {int(chars.median() / 3.7)})")


if __name__ == "__main__":
    main()
