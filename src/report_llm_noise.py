#!/usr/bin/env python3
"""Split an LLM's error variance into response noise and sampling noise.

Companion to `src/make_llm_noise_probe.py`. The probe runs two arms that share
a cell: one repeats an identical prompt, one varies the walk seed behind the
prompt. Within-graph spread is therefore

    response arm:  s2_resp
    input arm:     s2_resp + s2_input

and the difference identifies the part a redrawn walk contributes. Everything
is scored under the frozen rule from `docs/TARGET_EVALUATION_FREEZE.md`:
final JSON only, no clipping or monotone repair, an invalid component costs
absolute loss 1.

The design question it answers is what the final run should replicate. With
G groups, I graphs per group, S walk seeds and R generations per prompt:

    SE^2 = s2_group/G + s2_inst/(G*I) + s2_input/(G*I*S) + s2_resp/(G*I*S*R)

Walk seeds and repeated generations sit at different depths, and the data
decides which is worth paying for.

  PYTHONPATH=src python src/report_llm_noise.py \
      --answers 'results/llm_noise_probe/answers_*.jsonl' \
      --prompts results/llm_noise_probe/prompts.jsonl \
      --cases results/panel_seed_probe/cases.csv.gz
"""

import argparse
import collections
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_eval_frozen import PROFILE_PRED, PROFILE_TRUTH, load_answers, \
    load_prompts, valid_unit
from run_llm_v2 import extract_last_json, is_complete_record

RESPONSE_ARM = "response"
INPUT_ARM = "input"


def load_by_model(patterns, root="."):
    """{model: {prompt_id: record}}, never merged across models.

    `load_answers` keys on prompt_id alone, so a glob spanning several models
    would let whichever file sorts last silently overwrite the others -- the
    probe deliberately asks every model the same prompt_ids, so the collision
    is total rather than occasional. Bucketing by the record's own `model`
    field keeps each model's answers separate. Within a model the same rule as
    `load_answers` applies: smoke files are skipped and the last complete
    record for a prompt wins, so a successful retry replaces a failed attempt.
    """
    if isinstance(patterns, (str, Path)):
        patterns = [patterns]
    paths = []
    for pattern in patterns:
        paths += [q for q in glob.glob(str(Path(root) / pattern))
                  if "smoke" not in Path(q).name]
    complete = collections.defaultdict(dict)
    latest = collections.defaultdict(dict)
    for path in sorted(paths):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue          # a half-written last line, not evidence
                prompt_id = record.get("prompt_id")
                if prompt_id is None:
                    continue
                model = str(record.get("model") or Path(path).stem)
                latest[model][prompt_id] = record
                if is_complete_record(record):
                    complete[model][prompt_id] = record
    out = {}
    for model in latest:
        merged = dict(latest[model])
        merged.update(complete[model])
        out[model] = merged
    return out


def build_frame(prompts, answers, cases):
    rows = []
    for prompt_id, meta in prompts.items():
        record = answers.get(prompt_id)
        if record is None:
            continue
        truth = cases.loc[meta["case_id"]]
        rows.append({**{k: meta[k] for k in
                        ("case_id", "instance_id", "group_id", "walk_seed",
                         "rep", "arms", "coverage")},
                     "prompt_id": prompt_id,
                     **score_record(record.get("answer"), truth)})
    return pd.DataFrame(rows)


def score_record(answer_text, truth_row):
    """Penalized and complete-case ProfileMAE plus the raw rho_k2 prediction."""
    obj = extract_last_json(answer_text)
    losses, values = [], []
    for pred_key, truth_key in zip(PROFILE_PRED, PROFILE_TRUTH):
        raw = obj.get(pred_key) if isinstance(obj, dict) else None
        if valid_unit(raw):
            losses.append(abs(float(raw) - float(truth_row[truth_key])))
            values.append(float(raw))
        else:
            losses.append(1.0)
            values.append(np.nan)
    complete = all(np.isfinite(v) for v in values)
    return {
        "responded": isinstance(obj, dict),
        "valid": complete,
        "penalized": float(np.mean(losses)),
        "profile_mae": float(np.mean(losses)) if complete else np.nan,
        "rho_k2_pred": values[0],
    }


def within_graph_variance(frame, column):
    """Mean within-graph variance, and the between-graph part of the rest."""
    work = frame[["instance_id", "group_id", column]].dropna()
    per_graph = work.groupby(["group_id", "instance_id"])[column].agg(
        ["mean", "var", "size"])
    per_graph = per_graph[per_graph["size"] >= 2]
    if per_graph.empty:
        return None
    within = float(per_graph["var"].mean())
    n_rep = float(per_graph["size"].mean())
    per_group = per_graph.groupby("group_id")["mean"].agg(["mean", "var",
                                                           "size"])
    n_inst = float(per_group["size"].mean())
    inst = max(float(per_group["var"].mean(skipna=True) or 0.0)
               - within / max(n_rep, 1.0), 0.0)
    group = max(float(per_group["mean"].var(ddof=1))
                - inst / max(n_inst, 1.0)
                - within / max(n_inst * n_rep, 1.0), 0.0)
    return {"within": within, "inst": inst, "group": group,
            "n_rep": n_rep, "n_inst": n_inst,
            "n_group": float(len(per_group)),
            "mean": float(work[column].mean())}


def se_design(group, inst, s2_input, s2_resp, n_group, n_inst, seeds, reps):
    return float(np.sqrt(group / n_group
                         + inst / (n_group * n_inst)
                         + s2_input / (n_group * n_inst * seeds)
                         + s2_resp / (n_group * n_inst * seeds * reps)))


def report(frame, prompts, metric, label):
    """The full variance read for one model."""
    print("\n" + "#" * 76)
    print(f"# {label}")
    print("#" * 76)
    attempted, total = len(frame), len(prompts)
    print(f"probe generations: {attempted}/{total} answered")
    print(f"response rate {frame.responded.mean():.2f} | "
          f"validity rate {frame.valid.mean():.2f} | "
          f"graphs {frame.instance_id.nunique()} | "
          f"groups {frame.group_id.nunique()}")

    arms = {RESPONSE_ARM: frame[frame.arms.str.contains(RESPONSE_ARM)],
            INPUT_ARM: frame[frame.arms.str.contains(INPUT_ARM)]}
    for name, sub in arms.items():
        print(f"  {name:<9} n={len(sub):>4}  response {sub.responded.mean():.2f}"
              f"  valid {sub.valid.mean():.2f}")
    if abs(arms[RESPONSE_ARM].valid.mean()
           - arms[INPUT_ARM].valid.mean()) > 0.10:
        print("  WARNING: validity differs by arm; the variance comparison is "
              "confounded with non-response and should not be read as noise "
              "alone.")

    # metric arrives as a parameter now; the old `args` lookup lived in main.
    print("\n" + "=" * 76)
    print(f"Variance components on `{metric}`")
    print("=" * 76)
    parts = {}
    for name, sub in arms.items():
        parts[name] = within_graph_variance(sub, metric)
        if parts[name] is None:
            raise SystemExit(f"not enough repeats in the {name} arm")
        p = parts[name]
        print(f"  {name:<9} mean {p['mean']:.4f}  within-graph sd "
              f"{np.sqrt(p['within']):.4f}  ({int(p['n_rep'])} per graph)")

    s2_resp = parts[RESPONSE_ARM]["within"]
    s2_both = parts[INPUT_ARM]["within"]
    s2_input = max(s2_both - s2_resp, 0.0)
    total_within = s2_resp + s2_input
    print(f"\n  s2_resp  (same prompt, regenerated) sd {np.sqrt(s2_resp):.4f}")
    print(f"  s2_input (redrawn walk)             sd {np.sqrt(s2_input):.4f}")
    if total_within > 0:
        print(f"  share of within-graph noise: response "
              f"{s2_resp / total_within:.0%}, walk redraw "
              f"{s2_input / total_within:.0%}")
    if s2_both < s2_resp:
        print("  NOTE: the input arm was not noisier than the response arm; "
              "s2_input is reported as 0 rather than negative.")

    group = parts[INPUT_ARM]["group"]
    inst = parts[INPUT_ARM]["inst"]
    n_group = parts[INPUT_ARM]["n_group"]
    n_inst = parts[INPUT_ARM]["n_inst"]

    print("\n" + "=" * 76)
    print("What replication is worth buying")
    print("=" * 76)
    print(f"SE of the group-macro {metric}, G={int(n_group)} groups, "
          f"I={n_inst:.2f} graphs/group.")
    print("S = walk seeds per graph, R = generations per prompt.\n")
    header = f"{'design':<28}{'SE':>10}{'vs baseline':>14}"
    print(header)
    print("-" * len(header))
    base = se_design(group, inst, s2_input, s2_resp, n_group, n_inst, 1, 1)
    plans = [("S=1, R=1  (baseline)", 1, 1, n_group),
             ("S=1, R=3", 1, 3, n_group),
             ("S=1, R=5", 1, 5, n_group),
             ("S=3, R=1", 3, 1, n_group),
             ("S=5, R=1", 5, 1, n_group),
             ("S=3, R=3", 3, 3, n_group),
             ("2x graphs, S=1, R=1", 1, 1, 2 * n_group)]
    for label, seeds, reps, groups in plans:
        se = se_design(group, inst, s2_input, s2_resp, groups, n_inst,
                       seeds, reps)
        print(f"{label:<28}{se:>10.4f}{se / base - 1:>+13.0%}")
    print("\nCost multiplier of a design is S*R generations per case.")

    print("\n" + "=" * 76)
    print("Agreement between two generations of the identical prompt")
    print("=" * 76)
    pivot = arms[RESPONSE_ARM].pivot_table(
        index="instance_id", columns="rep", values="rho_k2_pred")
    if pivot.shape[1] >= 2:
        cols = list(pivot.columns)[:2]
        pair = pivot[cols].dropna()
        if len(pair) >= 3:
            pearson = float(np.corrcoef(pair[cols[0]], pair[cols[1]])[0, 1])
            spear = float(pd.Series(pair[cols[0]]).corr(
                pd.Series(pair[cols[1]]), method="spearman"))
            spread = float((pair[cols[0]] - pair[cols[1]]).abs().mean())
            between = float(pair[cols[0]].std())
            print(f"  rho_k2 across reps 1 vs 2 on {len(pair)} graphs: "
                  f"pearson {pearson:.3f}, spearman {spear:.3f}")
            print(f"  mean |difference| {spread:.4f} against a between-graph "
                  f"sd of {between:.4f} (ratio {spread / between:.2f})")
            print("  A ratio near or above 1 means asking twice moves the "
                  "answer about as much as changing the network.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--answers", required=True, help="glob of answer jsonl")
    ap.add_argument("--prompts", default="results/llm_noise_probe/prompts.jsonl")
    ap.add_argument("--cases", default="results/panel_seed_probe/cases.csv.gz")
    ap.add_argument("--metric", default="penalized",
                    choices=["penalized", "profile_mae", "rho_k2_pred"],
                    help="penalized: frozen primary. profile_mae: complete "
                         "cases only. rho_k2_pred: the raw prediction, which "
                         "separates 'the answer moves' from 'the error moves'")
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    cases = pd.read_csv(args.cases).set_index("case_id")
    per_model = load_by_model(args.answers)
    if not per_model:
        raise SystemExit("no answer files matched")

    frames = {}
    for model, answers in sorted(per_model.items()):
        frame = build_frame(prompts, answers, cases)
        if frame.empty:
            continue
        frames[model] = frame
    if not frames:
        raise SystemExit("no answers matched the probe prompts")

    print(f"models found: {len(frames)}")
    for model, frame in frames.items():
        print(f"  {model:<44} {len(frame):>4} answers")

    for model, frame in frames.items():
        # One model at a time. Merging them would average over machines with
        # different noise levels, which is the very thing under test.
        report(frame, prompts, args.metric, model)


if __name__ == "__main__":
    main()
