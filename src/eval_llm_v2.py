#!/usr/bin/env python3
"""Score LLM answers for the v2 benchmark and build the leaderboard.

Input: one or more answers-*.jsonl files. Each line needs an id matching
prompts.jsonl ("id"/"prompt_id"/"custom_id") and the raw model reply under one
of "answer"/"response"/"text"/"content"/"output". A "model" field is used when
present, otherwise the file stem.

The last JSON object in the reply is parsed (code fences tolerated). Scores:
rho_k2 MAE / group-macro / worst-group / bias / Spearman, profile MAE
(k3..k5), 90%-interval coverage and width, C MAE, parse failure rate; slices
by strategy, block group and coverage band. Classical baselines are
re-evaluated on exactly the answered cases so the leaderboard is
apples-to-apples.

--smoke fabricates two reference pseudo-models (oracle+noise, global mean)
over the main-condition prompts and runs the full pipeline once.
"""

import argparse
import warnings
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="An input array is constant")

PRED_KEYS = ["rho_k2", "rho_k3", "rho_k4", "rho_k5", "mean_occupancy",
             "C_one_step", "lifetime_mean_over_T", "lo90", "hi90"]
BASELINE_COLS = ["pred__floor", "pred__occ_mle", "pred__mask_mle",
                 "pred__beta_block", "pred__et_combined", "pred__et_stacked",
                 "pred_occ__plugin", "pred_occ__occ_mle", "pred_occ__mask_mle",
                 "pred_lt__plugin_zero", "pred_lt__plugin_cond",
                 "pred_C__plugin", "pred_C__mask_mle"]
# baseline model -> {answer key: source column or aggregate sentinel}
BASE_SPECS = [
    ("baseline_floor", {"rho_k2": "pred__floor",
                        "mean_occupancy": "__occ_set_mean__",
                        "lifetime_mean_over_T": "__lt_set_mean__"}),
    ("baseline_occ_mle", {"rho_k2": "pred__occ_mle",
                          "mean_occupancy": "pred_occ__occ_mle"}),
    ("baseline_mask_mle", {"rho_k2": "pred__mask_mle",
                           "mean_occupancy": "pred_occ__mask_mle",
                           "C_one_step": "pred_C__mask_mle"}),
    ("baseline_beta_block", {"rho_k2": "pred__beta_block"}),
    ("baseline_et_combined", {"rho_k2": "pred__et_combined"}),
    ("baseline_et_stacked", {"rho_k2": "pred__et_stacked"}),
    ("baseline_plugin_obs", {"mean_occupancy": "pred_occ__plugin",
                             "lifetime_mean_over_T": "pred_lt__plugin_zero",
                             "C_one_step": "pred_C__plugin"}),
    ("baseline_lt_plugin_cond", {"lifetime_mean_over_T":
                                 "pred_lt__plugin_cond"}),
]


def extract_last_json(text):
    """Last balanced {...} object that contains rho_k2."""
    if not isinstance(text, str):
        return None
    t = re.sub(r"```(?:json)?", "", text)
    end = len(t)
    while True:
        close = t.rfind("}", 0, end)
        if close < 0:
            return None
        depth, start = 0, None
        for i in range(close, -1, -1):
            if t[i] == "}":
                depth += 1
            elif t[i] == "{":
                depth -= 1
                if depth == 0:
                    start = i
                    break
        if start is not None:
            frag = t[start:close + 1]
            if "rho_k2" in frag:
                try:
                    obj = json.loads(frag)
                    if isinstance(obj, dict):
                        return obj
                except json.JSONDecodeError:
                    pass
        end = close
        if end <= 0:
            return None


def coerce(obj):
    out = {}
    for k in PRED_KEYS:
        v = obj.get(k)
        if v is None:
            out[k] = np.nan
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            out[k] = np.nan
            continue
        out[k] = float(np.clip(x, 0.0, 1.0))
    ks = [out[f"rho_k{k}"] for k in (2, 3, 4, 5)]
    out["profile_violation"] = int(any(
        np.isfinite(a) and np.isfinite(b) and b > a + 1e-9
        for a, b in zip(ks, ks[1:])))
    return out


def load_answers(specs):
    rows = []
    for spec in specs:
        for path in sorted(glob.glob(spec)) or [spec]:
            stem = Path(path).stem.replace("answers_", "")
            with open(path) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pid = r.get("id") or r.get("prompt_id") or r.get("custom_id")
                    text = None
                    for key in ("answer", "response", "text", "content",
                                "output", "raw"):
                        if isinstance(r.get(key), str):
                            text = r[key]
                            break
                    rows.append({"prompt_id": pid,
                                 "model": r.get("model", stem),
                                 "raw": text})
    if not rows:
        raise FileNotFoundError("no answer rows loaded")
    df = pd.DataFrame(rows)
    # retries append: keep the LAST record per (model, prompt_id)
    return df.groupby(["model", "prompt_id"], as_index=False).last()


def metric_rows(df, label_cols, y="rho_W5_k2", p="rho_k2"):
    out = []
    for keys, g in df.groupby(label_cols, dropna=False):
        keys = keys if isinstance(keys, tuple) else (keys,)
        k2ok = np.isfinite(g[p]) & np.isfinite(g[y])
        occok = np.isfinite(g["mean_occupancy"]) & np.isfinite(g["occupancy_true"])
        ltok = (np.isfinite(g["lifetime_mean_over_T"])
                & np.isfinite(g["lifetime_true"]))
        anyok = k2ok | occok | ltok
        row = dict(zip(label_cols, keys))
        row.update({"n": len(g), "parsed": int(anyok.sum()),
                    "parse_rate": float(anyok.mean()) if len(g) else np.nan})
        if k2ok.sum():
            e = g.loc[k2ok, p] - g.loc[k2ok, y]
            gm = g.loc[k2ok].assign(ae=e.abs()).groupby("group_id")["ae"].mean()
            row.update({
                "mae": float(e.abs().mean()),
                "mae_group_macro": float(gm.mean()),
                "mae_worst_group": float(gm.max()),
                "bias": float(e.mean()),
                "spearman": float(pd.Series(g.loc[k2ok, p]).corr(
                    g.loc[k2ok, y], method="spearman")),
            })
            prof = []
            for k in (3, 4, 5):
                pk, yk = g.loc[k2ok, f"rho_k{k}"], g.loc[k2ok, f"rho_W5_k{k}"]
                m = np.isfinite(pk)
                if m.any():
                    prof.append(float((pk[m] - yk[m]).abs().mean()))
            row["profile_mae_k3to5"] = float(np.mean(prof)) if prof else np.nan
            lo, hi = g.loc[k2ok, "lo90"], g.loc[k2ok, "hi90"]
            iv = np.isfinite(lo) & np.isfinite(hi)
            row["interval_cover90"] = float(
                ((lo[iv] <= g.loc[k2ok, y][iv]) &
                 (g.loc[k2ok, y][iv] <= hi[iv])).mean()) if iv.any() else np.nan
            row["interval_width_med"] = float(
                (hi[iv] - lo[iv]).median()) if iv.any() else np.nan
            row["profile_violation_rate"] = float(
                g.loc[k2ok, "profile_violation"].mean())
        if occok.sum():
            eo = g.loc[occok, "mean_occupancy"] - g.loc[occok, "occupancy_true"]
            row["occ_mae"] = float(eo.abs().mean())
            row["occ_bias"] = float(eo.mean())
        prof_ok = np.isfinite(g["occupancy_true"])
        for k in (2, 3, 4, 5):
            prof_ok &= np.isfinite(g[f"rho_k{k}"])
        if prof_ok.any():
            implied = (1.0 + sum(g.loc[prof_ok, f"rho_k{k}"]
                                 for k in (2, 3, 4, 5))) / 5.0
            row["occ_derived_mae"] = float(
                (implied - g.loc[prof_ok, "occupancy_true"]).abs().mean())
            both = prof_ok & np.isfinite(g["mean_occupancy"])
            if both.any():
                implied_b = (1.0 + sum(g.loc[both, f"rho_k{k}"]
                                       for k in (2, 3, 4, 5))) / 5.0
                row["occ_profile_gap"] = float(
                    (g.loc[both, "mean_occupancy"] - implied_b).abs().mean())
        if ltok.sum():
            el = (g.loc[ltok, "lifetime_mean_over_T"]
                  - g.loc[ltok, "lifetime_true"])
            row["lifetime_mae"] = float(el.abs().mean())
            row["lifetime_bias"] = float(el.mean())
        c_p, c_y = g["C_one_step"], g["C_one_step_true"]
        cm = np.isfinite(c_p) & np.isfinite(c_y)
        if cm.any():
            row["C_mae"] = float((c_p[cm] - c_y[cm]).abs().mean())
        out.append(row)
    return out


def baseline_frame(scored):
    """Re-evaluate classical baselines on exactly the answered case sets."""
    frames = []
    for (cond, kind), g in scored.groupby(["condition", "input_kind"]):
        cases = g.drop_duplicates("case_id")
        occ_mean = float(cases["occupancy_true"].mean())
        lt_mean = float(cases["lifetime_true"].mean())
        for name, spec in BASE_SPECS:
            b = cases.copy()
            b["model"] = name
            b["condition"] = cond
            b["input_kind"] = kind
            for key in PRED_KEYS:
                b[key] = np.nan
            for key, src in spec.items():
                if src == "__occ_set_mean__":
                    b[key] = occ_mean
                elif src == "__lt_set_mean__":
                    b[key] = lt_mean
                else:
                    b[key] = b[src]
            b["profile_violation"] = 0
            frames.append(b)
    return pd.concat(frames, ignore_index=True)


def smoke_answers(prompts, cases, out_path, seed=7):
    rng = np.random.default_rng(seed)
    truth = cases.set_index("case_id")
    gmean = float(cases["rho_W5_k2"].mean())
    rows = []
    for _, p in prompts.iterrows():
        y = truth.loc[p.case_id]
        noisy = {f"rho_k{k}": float(np.clip(
            y[f"rho_W5_k{k}"] + rng.normal(0, 0.03), 0, 1))
            for k in (2, 3, 4, 5)}
        occ_cons = (1.0 + sum(noisy.values())) / 5.0
        ans1 = {**noisy,
                "mean_occupancy": float(np.clip(
                    occ_cons + rng.normal(0, 0.005), 0, 1)),
                "C_one_step": float(np.clip(
                    y["C_one_step"] + rng.normal(0, 0.05), 0, 1)),
                "lifetime_mean_over_T": float(np.clip(
                    y["lifetime_mean_over_T"] + rng.normal(0, 0.03), 0, 1)),
                "lo90": max(0.0, noisy["rho_k2"] - 0.10),
                "hi90": min(1.0, noisy["rho_k2"] + 0.10)}
        rows.append({"id": p.prompt_id, "model": "demo_oracle_noise",
                     "answer": "Reasoning...\n" + json.dumps(ans1)})
        ans2 = {"rho_k2": gmean, "rho_k3": gmean * 0.7, "rho_k4": gmean * 0.5,
                "rho_k5": gmean * 0.3, "mean_occupancy": 0.5,
                "C_one_step": None, "lifetime_mean_over_T": 0.5,
                "lo90": 0.0, "hi90": 1.0}
        rows.append({"id": p.prompt_id, "model": "demo_global_mean",
                     "answer": "```json\n" + json.dumps(ans2) + "\n```"})
    with open(out_path, "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", default="results/llm_v2")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--answers", nargs="*", default=[])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="fabricate demo answers and run the full pipeline")
    args = ap.parse_args()

    cdir = Path(args.cases_dir)
    out = Path(args.out_dir or cdir); out.mkdir(parents=True, exist_ok=True)
    cases = pd.read_csv(cdir / "llm_cases.csv")
    prompts = pd.read_json(args.prompts or cdir / "prompts.jsonl", lines=True)

    answer_specs = list(args.answers)
    if args.smoke:
        main_prompts = prompts[prompts.condition.isin(
            ["hidden", "disclosed", "disclosed_examples"])]
        answer_specs.append(str(smoke_answers(
            main_prompts, cases, out / "smoke_answers.jsonl")))
    if not answer_specs:
        raise SystemExit("give --answers files (or --smoke)")

    ans = load_answers(answer_specs)
    parsed = []
    for _, r in ans.iterrows():
        obj = extract_last_json(r["raw"])
        rec = coerce(obj) if obj else {k: np.nan for k in PRED_KEYS}
        if not obj:
            rec["profile_violation"] = 0
        parsed.append({"prompt_id": r.prompt_id, "model": r.model, **rec})
    parsed = pd.DataFrame(parsed)

    meta_cols = ["prompt_id", "case_id", "condition", "input_kind",
                 "strategy", "block_group", "coverage_band", "budget"]
    truth_cols = ["case_id", "group_id", "rho_W5_k2", "rho_W5_k3",
                  "rho_W5_k4", "rho_W5_k5", "C_one_step", "mean_span_frac",
                  "lifetime_mean_over_T"] + BASELINE_COLS
    scored = (parsed.merge(prompts[meta_cols], on="prompt_id", how="left")
                    .merge(cases[truth_cols].rename(
                        columns={"C_one_step": "C_one_step_true",
                                 "mean_span_frac": "occupancy_true",
                                 "lifetime_mean_over_T": "lifetime_true"}),
                        on="case_id", how="left", suffixes=("", "_case")))
    unmatched = scored.case_id.isna().mean()
    if unmatched > 0:
        print(f"[warn] {unmatched:.1%} answers without matching prompt id")
        scored = scored.dropna(subset=["case_id"])

    base = baseline_frame(scored)
    both = pd.concat([scored, base], ignore_index=True)

    rows = metric_rows(both, ["model", "condition", "input_kind"])
    for r in rows:
        r["slice_type"], r["slice_value"] = "overall", "all"
    for col in ("strategy", "block_group", "coverage_band"):
        sub = metric_rows(both, ["model", "condition", "input_kind", col])
        for r in sub:
            r["slice_type"], r["slice_value"] = col, r.pop(col)
        rows += sub
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "llm_metrics.csv", index=False)
    scored.to_csv(out / "parsed_answers.csv.gz", index=False,
                  compression="gzip")

    lead = metrics[(metrics.slice_type == "overall")].copy()
    lead["label"] = lead.model + " [" + lead.condition + "/" + lead.input_kind + "]"
    lines = ["# LLM leaderboard (rho_k2, identical case sets)", ""]
    for (cond, kind), g in lead.groupby(["condition", "input_kind"]):
        lines += [f"## condition={cond}, input={kind}", "",
                  "| model | n | parse | MAE k2 | group-macro | worst-group | "
                  "bias | cov90 | occ MAE | lifetime MAE | C MAE |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]

        def fmt(v, f="{:.4f}"):
            return f.format(v) if np.isfinite(v) else "-"

        for _, r in g.sort_values("mae_group_macro",
                                  na_position="last").iterrows():
            lines.append(
                f"| {r.model} | {int(r.n)} | {fmt(r.parse_rate, '{:.2f}')} | "
                f"{fmt(r.get('mae', np.nan))} | "
                f"{fmt(r.get('mae_group_macro', np.nan))} | "
                f"{fmt(r.get('mae_worst_group', np.nan))} | "
                f"{fmt(r.get('bias', np.nan), '{:+.3f}')} | "
                f"{fmt(r.get('interval_cover90', np.nan), '{:.2f}')} | "
                f"{fmt(r.get('occ_mae', np.nan))} | "
                f"{fmt(r.get('lifetime_mae', np.nan))} | "
                f"{fmt(r.get('C_mae', np.nan))} |")
        lines.append("")
    (out / "llm_leaderboard.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:40]))
    print(f"\nwrote {out/'llm_metrics.csv'}, {out/'llm_leaderboard.md'}, "
          f"{out/'parsed_answers.csv.gz'}")


if __name__ == "__main__":
    main()
