#!/usr/bin/env python3
"""Evaluate paired target-isolation answers without repairing predictions."""

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def extract(text):
    if not isinstance(text, str):
        return None
    for frag in reversed(re.findall(r"\{[^{}]*\}", text, flags=re.S)):
        try:
            obj = json.loads(frag)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", action="append", required=True)
    ap.add_argument("--cases", default="results/target_diagnostics/prompt_isolation/selected_cases.csv")
    ap.add_argument("--out-dir", default="results/target_diagnostics/prompt_isolation/eval")
    args = ap.parse_args()
    rows = []
    for spec in args.answers:
        for path in glob.glob(spec):
            for line in open(path):
                try:
                    r = json.loads(line)
                    r["_path"] = path
                    rows.append(r)
                except json.JSONDecodeError:
                    pass
    latest = {}
    for r in rows:
        latest[(r.get("model"), r.get("thinking"), r.get("prompt_id"))] = r
    truth = pd.read_csv(args.cases).set_index("case_id")
    parsed = []
    for r in latest.values():
        obj = extract(r.get("answer"))
        keys = r.get("required_keys") or []
        rec = {k: r.get(k) for k in
               ["model", "thinking", "prompt_id", "case_id", "prompt_variant",
                "finish_reason", "_path"]}
        rec["parse_ok"] = bool(obj and all(k in obj for k in keys))
        if obj:
            rec.update({f"pred_{k}": obj.get(k) for k in keys})
        if r.get("case_id") in truth.index:
            t = truth.loc[r["case_id"]]
            for k in range(2, 6):
                rec[f"true_rho_k{k}"] = t[f"rho_W5_k{k}"]
            rec["true_C_one_step"] = t["C_one_step"]
        parsed.append(rec)
    p = pd.DataFrame(parsed)
    metrics = []
    for labels, g in p.groupby(["model", "thinking", "prompt_variant"],
                               dropna=False):
        row = dict(zip(["model", "thinking", "prompt_variant"], labels))
        row.update(n=len(g), parse_rate=g.parse_ok.mean())
        prof = []
        monotone = []
        for _, r in g.iterrows():
            vals = []
            for k in range(2, 6):
                try:
                    pred = float(r[f"pred_rho_k{k}"])
                    prof.append(abs(pred - r[f"true_rho_k{k}"]))
                    vals.append(pred)
                except (KeyError, TypeError, ValueError):
                    vals = []
                    break
            if vals:
                monotone.append(not all(vals[i] >= vals[i+1]
                                        for i in range(3)))
        row["profile_mae"] = np.mean(prof) if prof else np.nan
        ok2 = pd.to_numeric(g.get("pred_rho_k2"), errors="coerce")
        row["rho_k2_mae"] = (ok2 - g.true_rho_k2).abs().mean()
        row["monotonicity_violation_rate"] = (
            np.mean(monotone) if monotone else np.nan)
        metrics.append(row)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    p.to_csv(out / "parsed.csv", index=False)
    m = pd.DataFrame(metrics)
    m.to_csv(out / "metrics.csv", index=False)
    print(m.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
