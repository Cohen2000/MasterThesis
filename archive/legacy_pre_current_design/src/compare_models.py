#!/usr/bin/env python3
"""Compare multiple LLM pilot runs against the frozen reference band.

Takes one answers.jsonl per model plus the local pilot_cases.csv and produces:

  1. <prefix>_mae.csv + console table: MAE vs rho_true per model x condition,
     pooled and per strategy, next to floor / uniform-MLE / transfer-beta /
     betacal / oracle recomputed on exactly the joined pilot subset.
  2. <prefix>_diagnostics.csv: per model x condition x strategy
       spearman_true    rank information used (high = uses case signal)
       mean_pred        level of the answers
       plugin_gap       hidden only: mean |pred - plugin| (plugin = share of
                        observed edges with w>=2, reconstructed from prompts);
                        small = the model just reads off the naive share
       anchor_share     disclosed_calib only: share of answers within +/-0.05
                        of the calibration label; high + low spearman = the
                        model anchors on the example instead of calibrating
     plus the constant-predictor MAE (always answer the calib label) as the
     degenerate reference for disclosed_calib.
  3. <prefix>_models.png: grouped bar chart per strategy with the reference
     band as horizontal lines.

Usage:
  python compare_models.py \
      --pilot-cases ../results/phase3/pilot_cases.csv \
      --prompts ../results/phase3/prompts.jsonl \
      --answers ../results/phase3/answers_qwen14b.jsonl \
                ../results/phase3/answers_qwen32b.jsonl \
                ../results/phase3/answers_r1_32b.jsonl \
      --out-prefix ../results/phase3/compare
"""
import argparse
import json
import re

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

CONDS = ["hidden", "disclosed", "disclosed_calib"]
REFS = [("floor_lofo", "floor"), ("mle_uniform", "uniform MLE"),
        ("mle_transfer_beta", "transfer-beta MLE"),
        ("mle_betacal_lofo", "betacal MLE"), ("oracle_lofo", "oracle")]
STRATS = ["time_agnostic_t", "time_respecting"]


def mae(y, p):
    y = np.asarray(y, float); p = np.asarray(p, float)
    m = np.isfinite(y) & np.isfinite(p)
    return float(np.abs(y[m] - p[m]).mean()) if m.sum() else np.nan


def short_name(model_id):
    s = model_id.split("/")[-1]
    return (s.replace("-Instruct", "").replace("DeepSeek-", "")
             .replace("Distill-", "d"))


def plugin_from_prompts(prompts_path):
    """share of observed edges with w>=2, reconstructed from the hidden
    prompt's binned table (same numbers the models saw)."""
    plug = {}
    for r in (json.loads(l) for l in open(prompts_path) if l.strip()):
        if r["condition"] != "hidden":
            continue
        rows = re.findall(r"n [=>]+ ?[\d\-]+ +\|((?: +\d+){5})", r["prompt"])
        M = np.array([[int(x) for x in row.split()] for row in rows])
        plug[r["case_id"]] = M[:, 1:].sum() / M.sum() if M.sum() else np.nan
    return plug


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot-cases", default="../results/phase3/pilot_cases.csv")
    ap.add_argument("--prompts", default="../results/phase3/prompts.jsonl")
    ap.add_argument("--answers", nargs="+", required=True)
    ap.add_argument("--calib-rho", type=float, default=0.294,
                    help="label of the calibration example (anchor check)")
    ap.add_argument("--out-prefix", default="../results/phase3/compare")
    args = ap.parse_args()

    cases = pd.read_csv(args.pilot_cases)
    try:
        cases["plugin"] = cases["case_id"].map(plugin_from_prompts(args.prompts))
    except FileNotFoundError:
        print(f"[warn] {args.prompts} not found, plugin diagnostics skipped")
        cases["plugin"] = np.nan

    models = {}
    for path in args.answers:
        ans = pd.DataFrame([json.loads(l) for l in open(path) if l.strip()])
        ans = ans.drop_duplicates(subset=["case_id", "condition"], keep="last")
        name = short_name(ans["model"].iloc[0])
        wide = ans.pivot_table(index="case_id", columns="condition",
                               values="pred", aggfunc="last")
        wide.columns = [f"{name}|{c}" for c in wide.columns]
        models[name] = {
            "unparsed": int(ans["pred"].isna().sum()),
            "retried": int(ans.get("retried", pd.Series(dtype=bool)).sum()),
            "median_s": float(ans["gen_seconds"].median()),
        }
        cases = cases.merge(wide, on="case_id", how="left")
        print(f"{name:16s} {len(ans)} answers | unparsed "
              f"{models[name]['unparsed']} | retried {models[name]['retried']}"
              f" | median {models[name]['median_s']:.0f}s/prompt")

    # ---- 1. MAE table ------------------------------------------------------
    scopes = [("ALL", cases)] + [(s, cases[cases.strategy == s]) for s in STRATS]
    rows = []
    for col, lab in REFS:
        rows.append({"method": lab,
                     **{sc: mae(g.rho_true, g[col]) for sc, g in scopes}})
    rows.append({"method": f"constant {args.calib_rho:.3f}",
                 **{sc: mae(g.rho_true, np.full(len(g), args.calib_rho))
                    for sc, g in scopes}})
    for name in models:
        for c in CONDS:
            col = f"{name}|{c}"
            if col not in cases:
                continue
            rows.append({"method": f"{name} {c}",
                         **{sc: mae(g.rho_true, g[col]) for sc, g in scopes}})
    tab = pd.DataFrame(rows)
    print("\n=== MAE vs rho_true (pilot subset, pooled and per strategy) ===")
    print(tab.round(3).to_string(index=False))
    tab.to_csv(f"{args.out_prefix}_mae.csv", index=False)

    # per-band CSV for the appendix (not printed)
    band_rows = []
    for (s, b), g in cases.groupby(["strategy", "band"]):
        r = {"strategy": s, "band": b, "n": len(g)}
        for col, lab in REFS:
            r[lab] = mae(g.rho_true, g[col])
        for name in models:
            for c in CONDS:
                col = f"{name}|{c}"
                if col in g:
                    r[f"{name} {c}"] = mae(g.rho_true, g[col])
        band_rows.append(r)
    pd.DataFrame(band_rows).to_csv(f"{args.out_prefix}_mae_by_band.csv", index=False)

    # ---- 2. diagnostics ----------------------------------------------------
    diag = []
    print("\n=== diagnostics (per model x condition x strategy) ===")
    print(f"{'model':16s}{'cond':17s}{'strat':17s}{'sp_true':>8}{'meanpred':>9}"
          f"{'plug_gap':>9}{'anchor':>7}")
    for name in models:
        for c in CONDS:
            col = f"{name}|{c}"
            if col not in cases:
                continue
            for s in STRATS:
                g = cases[cases.strategy == s].dropna(subset=[col])
                if len(g) < 5:
                    continue
                sp = spearmanr(g[col], g.rho_true).statistic
                mp = float(g[col].mean())
                pg = (float((g[col] - g.plugin).abs().mean())
                      if c == "hidden" and g.plugin.notna().any() else np.nan)
                an = (float(((g[col] - args.calib_rho).abs() < 0.05).mean())
                      if c == "disclosed_calib" else np.nan)
                diag.append({"model": name, "condition": c, "strategy": s,
                             "n": len(g), "spearman_true": sp, "mean_pred": mp,
                             "plugin_gap": pg, "anchor_share": an})
                pgS = f"{pg:9.3f}" if np.isfinite(pg) else f"{'--':>9}"
                anS = f"{an:7.0%}" if np.isfinite(an) else f"{'--':>7}"
                print(f"{name:16s}{c:17s}{s:17s}{sp:+8.2f}{mp:9.3f}{pgS}{anS}")
    pd.DataFrame(diag).to_csv(f"{args.out_prefix}_diagnostics.csv", index=False)

    # ---- 3. figure ---------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mnames = list(models)
    colors = plt.cm.tab10(np.linspace(0, 0.5, max(len(mnames), 2)))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    width = 0.8 / len(mnames)
    for ax, s in zip(axes, STRATS):
        g = cases[cases.strategy == s]
        x = np.arange(len(CONDS))
        for j, name in enumerate(mnames):
            vals = [mae(g.rho_true, g[f"{name}|{c}"]) if f"{name}|{c}" in g
                    else np.nan for c in CONDS]
            ax.bar(x + (j - (len(mnames) - 1) / 2) * width, vals, width,
                   label=name, color=colors[j])
        for col, lab, ls, cc in [("floor_lofo", "floor", "--", "#888888"),
                                 ("mle_uniform", "uniform MLE", "-", "#d62728"),
                                 ("mle_transfer_beta", "transfer-beta", "-", "#9467bd"),
                                 ("oracle_lofo", "oracle", "-", "#2ca02c")]:
            ax.axhline(mae(g.rho_true, g[col]), ls=ls, color=cc, lw=1.6,
                       label=lab if s == STRATS[0] else None)
        ax.set_xticks(x); ax.set_xticklabels(CONDS, fontsize=9)
        ax.set_title(s); ax.grid(alpha=0.25, axis="y")
    axes[0].set_ylabel("MAE vs true rho")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=min(7, len(labels)),
               fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("LLM pilot: models vs reference band", y=1.09)
    fig.tight_layout()
    fig.savefig(f"{args.out_prefix}_models.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote {args.out_prefix}_mae.csv, _mae_by_band.csv, "
          f"_diagnostics.csv, _models.png")


if __name__ == "__main__":
    main()
