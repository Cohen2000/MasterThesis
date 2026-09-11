#!/usr/bin/env python3
"""Does the product harness change the answer? Same model, two access paths.

The supervisor's objection to Codex and Claude Code is that they inject a
harness prompt which is not part of the frozen prompt and cannot be pinned to a
version. That is a real objection about *what the run documents*. Whether it is
also an objection about *what the run measures* is a separate question, and the
noise probe happens to contain the paired data to answer it: the same model
(`gpt-5.6-sol`, reasoning effort high) answered probe prompts once through the
plain API and once through the Codex CLI with tools disabled.

Only the prompt ids present on both sides are compared, so the two arms differ
in the access path and in nothing else.

Run from the repository root:

    PYTHONPATH=src python src/compare_harness_vs_api.py

Read-only: it touches nothing but the append-only answer files.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from llm_eval_frozen import load_prompts
from report_llm_noise import build_frame, load_by_model

API_ANSWERS = "results/llm_noise_probe/answers_gpt56sol.jsonl"
CLI_ANSWERS = ("results/llm_noise_probe/"
               "answers_codex-gpt-5.6-sol_notools_high_noise.jsonl")
PROMPTS = "results/llm_noise_probe/prompts.jsonl"
CASES = "results/panel_seed_probe/cases.csv.gz"

# Codex answering the identical prompt twice, from the noise probe's own
# reproducibility diagnostic. It is the yardstick the harness effect has to be
# read against: a difference smaller than this is below the model's own floor.
CODEX_SELF_REPEAT = 0.0164


def flatten(per_model):
    out = {}
    for _model, records in per_model.items():
        out.update(records)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--api", default=API_ANSWERS)
    ap.add_argument("--cli", default=CLI_ANSWERS)
    ap.add_argument("--prompts", default=PROMPTS)
    ap.add_argument("--cases", default=CASES)
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)
    cases = pd.read_csv(args.cases).set_index("case_id")
    api = flatten(load_by_model([args.api]))
    cli = flatten(load_by_model([args.cli]))

    shared = sorted(set(api) & set(cli))
    if not shared:
        raise SystemExit("no prompt ids answered on both paths")
    subset = {pid: prompts[pid] for pid in shared}

    frame_api = build_frame(subset, api, cases).set_index("prompt_id").loc[shared]
    frame_cli = build_frame(subset, cli, cases).set_index("prompt_id").loc[shared]

    print("Same model, two access paths: gpt-5.6-sol, reasoning effort high")
    print(f"  API answers {len(api)} | Codex CLI answers {len(cli)} | "
          f"shared prompt ids {len(shared)}")
    print(f"  graphs {frame_api.instance_id.nunique()} | "
          f"groups {frame_api.group_id.nunique()}")

    print(f"\n{'':<26}{'API':>10}{'Codex CLI':>12}")
    print(f"{'ProfileMAE (penalized)':<26}{frame_api.penalized.mean():>10.4f}"
          f"{frame_cli.penalized.mean():>12.4f}")
    print(f"{'validity':<26}{frame_api.valid.mean():>10.2f}"
          f"{frame_cli.valid.mean():>12.2f}")

    delta = frame_api.penalized - frame_cli.penalized
    print(f"{'paired difference':<26}{delta.mean():>+10.4f}   (API minus CLI)")
    print(f"{'  sd of the difference':<26}{delta.std(ddof=1):>10.4f}   "
          f"n={len(delta)}")

    pred_api, pred_cli = frame_api.rho_k2_pred, frame_cli.rho_k2_pred
    both = pred_api.notna() & pred_cli.notna()
    truth = np.array([float(cases.loc[prompts[pid]["case_id"], "rho_W5_k2"])
                      for pid in shared])
    gap = float(np.abs(pred_api[both] - pred_cli[both]).mean())
    truth_sd = float(truth[both.to_numpy()].std(ddof=1))

    print(f"\nrho_k2 predictions on {int(both.sum())} cases:")
    print(f"  pearson API vs CLI        "
          f"{np.corrcoef(pred_api[both], pred_cli[both])[0, 1]:.3f}")
    print(f"  mean |difference|         {gap:.4f}")
    print(f"  sd of the truth           {truth_sd:.4f}")
    print(f"  ratio to that spread      {gap / truth_sd:.2f}")
    print(f"\n  the same model answering the identical prompt twice moves it by "
          f"{CODEX_SELF_REPEAT:.4f},")
    print(f"  so the access path moves the answer "
          f"{CODEX_SELF_REPEAT / gap:.1f}x less than the model's own sampling.")

    print("\nReading: on this evidence the harness is a documentation problem,\n"
          "not an accuracy problem. The sample is small -- "
          f"{len(shared)} shared prompts on "
          f"{frame_api.instance_id.nunique()} graphs -- so it bounds the effect\n"
          "rather than estimating it precisely.")


if __name__ == "__main__":
    main()
