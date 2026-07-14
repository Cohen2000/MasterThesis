#!/usr/bin/env python3
"""run_deepseek_api.py: Phase-3-Pilotprompts gegen die DeepSeek Chat Completions API.

Ziel: gleicher 180-Prompt-Pilot wie bei den lokalen Modellen/Gemini:
  results/phase3/prompts.jsonl  ->  answers_deepseek-v4-pro.jsonl

Default: deepseek-v4-pro mit Thinking Mode enabled, reasoning_effort=max,
max_tokens=8192 und Kostenbremse bei 3 USD.

Aufruf aus src/:
  export DEEPSEEK_API_KEY=...
  python run_deepseek_api_8192.py --limit 3
  python run_deepseek_api_8192.py

Resume-Logik:
  Nur Zeilen mit parsebarer pred gelten als erledigt. Unparsebare Zeilen werden
  mit raw/reasoning geloggt und beim naechsten Lauf erneut versucht.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests fehlt: python3 -m pip install --user requests")

FINAL_RE = re.compile(r"FINAL\s*[:=]\s*\**\s*([01](?:[\.,]\d+)?|[\.,]\d+)", re.IGNORECASE)


def parse_final(text):
    """Parse committed number, clamped to [0,1]."""
    if not text:
        return None
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[-1]
        hits = FINAL_RE.findall(tail)
        if hits:
            v = float(hits[-1].replace(",", "."))
            return min(max(v, 0.0), 1.0)
    hits = FINAL_RE.findall(text)
    if not hits:
        return None
    v = float(hits[-1].replace(",", "."))
    return min(max(v, 0.0), 1.0)


# Official list-price defaults from DeepSeek docs, USD per 1M tokens.
# Override with CLI args if DeepSeek changes prices.
PRICE_DEFAULTS = {
    "deepseek-v4-pro": {
        "cache_hit_input": 0.003625,
        "cache_miss_input": 0.435,
        "output": 0.87,
    },
    "deepseek-v4-flash": {
        "cache_hit_input": 0.0028,
        "cache_miss_input": 0.14,
        "output": 0.28,
    },
}


def estimate_cost_usd(usage, model, args):
    """Estimate request cost from DeepSeek usage fields."""
    if not usage:
        return 0.0
    if args.price_input_cache_hit is not None:
        p_hit = args.price_input_cache_hit
        p_miss = args.price_input_cache_miss
        p_out = args.price_output
    else:
        prices = PRICE_DEFAULTS.get(model, PRICE_DEFAULTS["deepseek-v4-pro"])
        p_hit = prices["cache_hit_input"]
        p_miss = prices["cache_miss_input"]
        p_out = prices["output"]

    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    prompt = usage.get("prompt_tokens", 0) or 0
    comp = usage.get("completion_tokens", 0) or 0

    # If detailed cache fields are absent, conservatively treat all prompt tokens as cache miss.
    if hit is None or miss is None:
        hit = 0
        miss = prompt

    return (hit * p_hit + miss * p_miss + comp * p_out) / 1_000_000.0


def call_deepseek(sess, args, prompt):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": args.max_tokens,
    }

    if args.thinking in ("enabled", "disabled"):
        body["thinking"] = {"type": args.thinking}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort

    # DeepSeek docs say temperature is not supported / has no effect in reasoning mode.
    # Keep it unset by default; allow explicit use for non-thinking ablations.
    if args.temperature is not None:
        body["temperature"] = args.temperature

    headers = {
        "Authorization": f"Bearer {args.api_key}",
        "Content-Type": "application/json",
    }

    delay = args.initial_retry_delay
    for attempt in range(args.max_retries + 1):
        t0 = time.time()
        try:
            r = sess.post(
                f"{args.base_url}/chat/completions",
                headers=headers,
                json=body,
                timeout=args.timeout,
            )
        except requests.RequestException as e:
            if attempt == args.max_retries:
                raise
            print(f"    Netzfehler ({e}), Retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, args.max_retry_delay)
            continue

        dt = time.time() - t0
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            if attempt == args.max_retries:
                print(f"    Endgueltiger HTTP-Fehler {r.status_code}: {r.text[:500]}", flush=True)
                r.raise_for_status()
            wait = r.headers.get("Retry-After")
            wait = float(wait) if wait else delay
            print(
                f"    HTTP {r.status_code}, Retry in {wait:.0f}s "
                f"(Versuch {attempt + 1}/{args.max_retries + 1})",
                flush=True,
            )
            time.sleep(wait)
            delay = min(delay * 2, args.max_retry_delay)
            continue

        if r.status_code >= 400:
            print(f"    HTTP {r.status_code}: {r.text[:1000]}", flush=True)
            r.raise_for_status()

        return r.json(), dt

    raise RuntimeError("unreachable")


def extract_texts(data):
    choices = data.get("choices") or []
    if not choices:
        return "", "", ""
    ch = choices[0]
    msg = ch.get("message") or {}
    raw = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish = ch.get("finish_reason") or ""
    return raw, reasoning, finish


def load_done(path):
    done = set()
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("pred") is not None:
                    done.add((r["case_id"], r["condition"]))
    return done


def current_cost(path):
    total = 0.0
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                try:
                    total += float(json.loads(line).get("est_cost_usd") or 0.0)
                except Exception:
                    pass
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="../results/phase3/prompts.jsonl")
    ap.add_argument("--pilot-cases", default="../results/phase3/pilot_cases.csv")
    ap.add_argument("--out", default=None,
                    help="Default: ../results/phase3/answers_<model>.jsonl")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="DEEPSEEK_API_KEY")
    ap.add_argument("--thinking", choices=["enabled", "disabled", "omit"], default="enabled")
    ap.add_argument("--reasoning-effort", choices=["high", "max"], default="max")
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="Output cap incl. reasoning tokens. 8192 is the strong/best-effort run cap.")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Usually leave unset for thinking mode; DeepSeek says it has no effect there.")
    ap.add_argument("--sleep", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--max-retries", type=int, default=6)
    ap.add_argument("--initial-retry-delay", type=float, default=15.0)
    ap.add_argument("--max-retry-delay", type=float, default=300.0)
    ap.add_argument("--unparsed-retries", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--max-usd", type=float, default=3.0,
                    help="Stop before next request if accumulated estimated cost exceeds this.")
    ap.add_argument("--price-input-cache-hit", type=float, default=None)
    ap.add_argument("--price-input-cache-miss", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    args = ap.parse_args()

    if (args.price_input_cache_hit is None) != (args.price_input_cache_miss is None) or \
       (args.price_input_cache_hit is None) != (args.price_output is None):
        sys.exit("Wenn du Preise ueberschreibst, setze alle drei: cache-hit, cache-miss, output.")

    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.api_key:
        sys.exit(f"{args.api_key_env} ist nicht gesetzt")

    if args.out is None:
        safe = args.model.replace("/", "-")
        args.out = f"../results/phase3/answers_{safe}_8192.jsonl"

    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.conditions:
        prompts = [p for p in prompts if p["condition"] in args.conditions]

    done = load_done(args.out)
    todo = [p for p in prompts if (p["case_id"], p["condition"]) not in done]
    if args.limit:
        todo = todo[:args.limit]

    spent = current_cost(args.out)
    print(f"{len(prompts)} Prompts, {len(done)} erledigt, {len(todo)} offen -> {args.out}")
    print(f"model={args.model} thinking={args.thinking} reasoning_effort={args.reasoning_effort} "
          f"max_tokens={args.max_tokens} max_usd={args.max_usd} bisher≈${spent:.4f}")

    sess = requests.Session()
    t_start = time.time()

    for i, p in enumerate(todo):
        if spent >= args.max_usd:
            print(f"Kostenbremse erreicht: estimated ${spent:.4f} >= ${args.max_usd:.2f}. Stop.", flush=True)
            break

        label = f"[{i + 1}/{len(todo)}] {p['case_id']} | {p['condition']}"
        pred, raw, reasoning, usage, finish, cid, dt, req_cost = (None, "", "", {}, "", "", 0.0, 0.0)
        status = ""

        for sub in range(args.unparsed_retries + 1):
            data, dt = call_deepseek(sess, args, p["prompt"])
            cid = data.get("id", "")
            usage = data.get("usage") or {}
            raw, reasoning, finish = extract_texts(data)
            status = finish or data.get("object", "")
            # Prefer final answer, but fallback to combined text if model misplaced FINAL.
            pred = parse_final(raw)
            if pred is None:
                pred = parse_final((reasoning or "") + "\n" + (raw or ""))
            req_cost = estimate_cost_usd(usage, args.model, args)
            if pred is not None:
                break
            print(f"  {label}: kein FINAL parsebar (finish={finish}), "
                  f"Wiederholung {sub + 1}/{args.unparsed_retries}", flush=True)
            time.sleep(args.sleep)

        spent += req_cost
        details = usage.get("completion_tokens_details") or {}
        row = {
            "case_id": p["case_id"], "condition": p["condition"],
            "model": args.model, "pred": pred,
            "preds_all": [pred] if pred is not None else [],
            "n_samples": 1,
            "temperature": args.temperature,
            "think_budget": args.max_tokens,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort,
            "forced": False,
            "n_gen_tokens": usage.get("completion_tokens"),
            "thought_tokens": details.get("reasoning_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "est_cost_usd": round(req_cost, 8),
            "est_cost_usd_cum": round(spent, 8),
            "gen_seconds": round(dt, 2),
            "status": status,
            "finish_reason": finish,
            "completion_id": cid,
            "api": f"chat/completions {args.base_url}",
            "raw": raw,
            "reasoning_content": reasoning,
        }
        with open(args.out, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

        el = time.time() - t_start
        print(f"{label}: pred={pred} finish={finish} "
              f"prompt_tok={row['prompt_tokens']} gen_tok={row['n_gen_tokens']} "
              f"reason_tok={row['thought_tokens']} cost≈${req_cost:.5f} cum≈${spent:.4f} "
              f"{dt:.1f}s (gesamt {el / 60:.1f} min)", flush=True)
        if i + 1 < len(todo):
            time.sleep(args.sleep)

    score(args)


def score(args):
    try:
        import pandas as pd
    except ImportError:
        print("pandas fehlt, Scoring uebersprungen")
        return
    if not Path(args.pilot_cases).exists():
        print(f"{args.pilot_cases} nicht gefunden, Scoring uebersprungen")
        return
    if not Path(args.out).exists():
        return
    cases = pd.read_csv(args.pilot_cases)[["case_id", "strategy", "rho_true"]]
    ans = pd.read_json(args.out, lines=True)
    ans = ans.drop_duplicates(subset=["case_id", "condition"], keep="last")
    m = ans.merge(cases, on="case_id", how="left")
    m["ae"] = (m["pred"] - m["rho_true"]).abs()
    print(f"\n=== {args.model}: MAE gegen rho_true "
          f"({int(m['pred'].notna().sum())}/{len(m)} geparst, cost≈${m['est_cost_usd'].sum():.4f}) ===")
    tab = m.pivot_table(index="condition", columns="strategy", values="ae", aggfunc="mean").round(3)
    tab["all"] = m.groupby("condition")["ae"].mean().round(3)
    print(tab.to_string())
    unp = m[m["pred"].isna()][["case_id", "condition", "finish_reason"]]
    if len(unp):
        print(f"\nUnparsebar ({len(unp)}), erneut laufen lassen fuer Retry:")
        print(unp.to_string(index=False))


if __name__ == "__main__":
    main()
