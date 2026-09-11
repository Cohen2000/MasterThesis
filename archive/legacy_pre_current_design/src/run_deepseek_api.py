#!/usr/bin/env python3
"""run_deepseek_api.py (v2): Phase-3-Pilotprompts gegen die DeepSeek API.

Aenderungen gegenueber run_deepseek_api_8192.py, mit Begruendung:
  1. Default jetzt reasoning_effort=high statt max. high ist laut DeepSeek-Doku
     der Default im Thinking-Mode; max ("Think Max") produziert laut Doku und
     Modellkarte sehr lange Denkketten (Kontextfenster >= 384k empfohlen) und
     hat im Smoke-Test 16384 Tokens nur mit Denken gefuellt, ohne je zu
     antworten. max bleibt als expliziter Kurvenpunkt verfuegbar.
  2. max_tokens Default 65536 als Sicherheitsnetz (Output-Maximum der API ist
     384k; der Deckel soll nie binden, nur Ausreisser begrenzen).
  3. Kein Unparsed-Retry mehr bei finish_reason=length: gleicher Prompt +
     gleiches Budget laeuft wieder ins Limit, das hat im Smoke nur 3x Kosten
     erzeugt. Stattdessen Length-Eskalation: bis zu --length-escalations mal
     wird max_tokens verdoppelt und einmal neu versucht.
  4. Kosten werden jetzt ueber ALLE Versuche eines Prompts summiert. v1 hat nur
     den letzten Versuch gezaehlt; der Smoke hat real ca. 3x mehr gekostet als
     geloggt (9 Requests, geloggt wie 3).
  5. Kostenbremse wird vor jedem einzelnen Request geprueft, nicht nur pro
     Prompt, und ist bei parallelen Workern atomar.
  6. --workers (Default 2) wie beim Gemini-Runner; einzelne langsame Antworten
     blockieren den Lauf nicht mehr.
  Hinweis: temperature/top_p werden im Thinking-Mode von DeepSeek ignoriert
  (Doku: "will not trigger an error but will also have no effect").

Aufruf aus src/:
  export DEEPSEEK_API_KEY=...
  python3 run_deepseek_api.py --limit 3     # Smoke
  python3 run_deepseek_api.py               # voller Lauf, resume-faehig
Optionaler zweiter Kurvenpunkt (teurer, erst nach dem high-Lauf entscheiden):
  python3 run_deepseek_api.py --reasoning-effort max --max-tokens 131072 \
      --max-usd 8 --out ../results/phase3/answers_deepseek-v4-pro_max.jsonl
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import requests
except ImportError:
    sys.exit("requests fehlt: python3 -m pip install --user requests")

FINAL_RE = re.compile(r"FINAL\s*[:=]\s*\**\s*([01](?:[\.,]\d+)?|[\.,]\d+)",
                      re.IGNORECASE)


def parse_final(text):
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


# Preise USD pro 1M Tokens, Stand 2026-07-05 (Promo-Niveau, von mehreren
# Preistrackern als aktiv bestaetigt). Listenpreis v4-pro ohne Promo waere
# 1.74 (miss) / 3.48 (out). Bei Abweichungen per CLI ueberschreiben und den
# realen Verbrauch auf platform.deepseek.com gegenpruefen.
PRICE_DEFAULTS = {
    "deepseek-v4-pro": {"cache_hit_input": 0.003625, "cache_miss_input": 0.435,
                        "output": 0.87},
    "deepseek-v4-flash": {"cache_hit_input": 0.0028, "cache_miss_input": 0.14,
                          "output": 0.28},
}


def estimate_cost_usd(usage, model, args):
    if not usage:
        return 0.0
    if args.price_input_cache_hit is not None:
        p_hit, p_miss, p_out = (args.price_input_cache_hit,
                                args.price_input_cache_miss, args.price_output)
    else:
        pr = PRICE_DEFAULTS.get(model, PRICE_DEFAULTS["deepseek-v4-pro"])
        p_hit, p_miss, p_out = pr["cache_hit_input"], pr["cache_miss_input"], pr["output"]
    hit = usage.get("prompt_cache_hit_tokens")
    miss = usage.get("prompt_cache_miss_tokens")
    prompt = usage.get("prompt_tokens", 0) or 0
    comp = usage.get("completion_tokens", 0) or 0
    if hit is None or miss is None:
        hit, miss = 0, prompt
    return (hit * p_hit + miss * p_miss + comp * p_out) / 1_000_000.0


class StartLimiter:
    def __init__(self, interval):
        self.interval = max(interval, 0.0)
        self.lock = threading.Lock()
        self.next_t = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_t)
            self.next_t = start + self.interval
        delay = start - time.monotonic()
        if delay > 0:
            time.sleep(delay)


class FinalFailure(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def call_deepseek(sess, args, prompt, max_tokens, limiter, stop_event):
    body = {"model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False, "max_tokens": max_tokens}
    if args.thinking in ("enabled", "disabled"):
        body["thinking"] = {"type": args.thinking}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    if args.temperature is not None:
        body["temperature"] = args.temperature
    headers = {"Authorization": f"Bearer {args.api_key}",
               "Content-Type": "application/json"}

    delay = args.initial_retry_delay
    for attempt in range(args.max_retries + 1):
        if stop_event.is_set():
            raise FinalFailure("Lauf gestoppt")
        limiter.wait()
        t0 = time.time()
        try:
            r = sess.post(f"{args.base_url}/chat/completions", headers=headers,
                          json=body, timeout=args.timeout)
        except requests.RequestException as e:
            if attempt == args.max_retries:
                raise FinalFailure(f"Netzfehler: {e}")
            print(f"    Netzfehler ({e}), Retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, args.max_retry_delay)
            continue
        dt = time.time() - t0
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            if attempt == args.max_retries:
                raise FinalFailure(f"HTTP {r.status_code}: {r.text[:300]}")
            wait = r.headers.get("Retry-After")
            wait = float(wait) if wait else delay
            print(f"    HTTP {r.status_code}, Retry in {wait:.0f}s "
                  f"(Versuch {attempt + 1}/{args.max_retries + 1})", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, args.max_retry_delay)
            continue
        if r.status_code >= 400:
            raise FinalFailure(f"HTTP {r.status_code}: {r.text[:500]}")
        return r.json(), dt
    raise RuntimeError("unreachable")


def extract_texts(data):
    choices = data.get("choices") or []
    if not choices:
        return "", "", ""
    ch = choices[0]
    msg = ch.get("message") or {}
    return (msg.get("content") or "", msg.get("reasoning_content") or "",
            ch.get("finish_reason") or "")


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
    ap.add_argument("--thinking", choices=["enabled", "disabled", "omit"],
                    default="enabled")
    ap.add_argument("--reasoning-effort", choices=["high", "max"], default="high",
                    help="high = DeepSeek-Default im Thinking-Mode. max nur als "
                         "expliziter Kurvenpunkt, denkt sehr lang und teuer.")
    ap.add_argument("--max-tokens", type=int, default=65536,
                    help="Sicherheitsnetz fuer Denken+Antwort zusammen, soll "
                         "nicht binden (API-Maximum 384k)")
    ap.add_argument("--length-escalations", type=int, default=1,
                    help="bei finish=length: so oft max_tokens verdoppeln und "
                         "einmal neu versuchen")
    ap.add_argument("--temperature", type=float, default=None,
                    help="im Thinking-Mode von DeepSeek wirkungslos (Doku); nur "
                         "fuer --thinking disabled sinnvoll")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="globaler Mindestabstand zwischen Request-Starts in s")
    ap.add_argument("--timeout", type=float, default=1200)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--initial-retry-delay", type=float, default=15.0)
    ap.add_argument("--max-retry-delay", type=float, default=300.0)
    ap.add_argument("--unparsed-retries", type=int, default=1,
                    help="Wiederholungen bei fehlendem FINAL trotz finish=stop")
    ap.add_argument("--max-consecutive-failures", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    ap.add_argument("--max-usd", type=float, default=3.0,
                    help="harte Bremse auf die geschaetzten kumulierten Kosten")
    ap.add_argument("--price-input-cache-hit", type=float, default=None)
    ap.add_argument("--price-input-cache-miss", type=float, default=None)
    ap.add_argument("--price-output", type=float, default=None)
    args = ap.parse_args()

    prices = (args.price_input_cache_hit, args.price_input_cache_miss,
              args.price_output)
    if any(p is not None for p in prices) and any(p is None for p in prices):
        sys.exit("Wenn du Preise ueberschreibst, setze alle drei: "
                 "cache-hit, cache-miss, output.")

    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.api_key:
        sys.exit(f"{args.api_key_env} ist nicht gesetzt")
    if args.temperature is not None and args.thinking == "enabled":
        print("HINWEIS: temperature wird im Thinking-Mode von DeepSeek ignoriert.",
              flush=True)
    if args.out is None:
        safe = args.model.replace("/", "-")
        args.out = f"../results/phase3/answers_{safe}.jsonl"

    prompts = [json.loads(l) for l in open(args.prompts)]
    if args.conditions:
        prompts = [p for p in prompts if p["condition"] in args.conditions]
    done = load_done(args.out)
    todo = [p for p in prompts if (p["case_id"], p["condition"]) not in done]
    if args.limit:
        todo = todo[:args.limit]

    budget_lock = threading.Lock()
    state = {"spent": current_cost(args.out), "done": 0, "failed": 0,
             "consecutive": 0}
    print(f"{len(prompts)} Prompts, {len(done)} erledigt, {len(todo)} offen "
          f"-> {args.out}")
    print(f"model={args.model} thinking={args.thinking} "
          f"reasoning_effort={args.reasoning_effort} max_tokens={args.max_tokens} "
          f"workers={args.workers} max_usd={args.max_usd} "
          f"bisher~${state['spent']:.4f}", flush=True)

    limiter = StartLimiter(args.sleep)
    write_lock = threading.Lock()
    stop_event = threading.Event()
    t_start = time.time()
    tls = threading.local()

    def get_session():
        if not hasattr(tls, "sess"):
            tls.sess = requests.Session()
        return tls.sess

    def check_budget():
        with budget_lock:
            if state["spent"] >= args.max_usd:
                if not stop_event.is_set():
                    print(f"Kostenbremse: ~${state['spent']:.4f} >= "
                          f"${args.max_usd:.2f}. Stop, Resume moeglich.",
                          flush=True)
                    stop_event.set()
                raise BudgetExceeded()

    def process(p):
        label = f"{p['case_id']} | {p['condition']}"
        if stop_event.is_set():
            return
        sess = get_session()
        pred, raw, reasoning, usage, finish, cid, dt = (None, "", "", {}, "", "", 0.0)
        prompt_cost = 0.0
        mt = args.max_tokens
        try:
            esc_left = args.length_escalations
            unparsed_left = args.unparsed_retries
            while True:
                check_budget()
                data, dt = call_deepseek(sess, args, p["prompt"], mt, limiter,
                                         stop_event)
                cid = data.get("id", "")
                usage = data.get("usage") or {}
                raw, reasoning, finish = extract_texts(data)
                req_cost = estimate_cost_usd(usage, args.model, args)
                prompt_cost += req_cost
                with budget_lock:
                    state["spent"] += req_cost
                pred = parse_final(raw)
                if pred is None:
                    pred = parse_final((reasoning or "") + "\n" + (raw or ""))
                if pred is not None:
                    break
                if finish == "length" and esc_left > 0:
                    esc_left -= 1
                    mt *= 2
                    print(f"  {label}: finish=length, eskaliere max_tokens auf "
                          f"{mt}", flush=True)
                    continue
                if finish != "length" and unparsed_left > 0:
                    unparsed_left -= 1
                    print(f"  {label}: kein FINAL parsebar (finish={finish}), "
                          f"noch {unparsed_left + 1} Versuch(e)", flush=True)
                    continue
                break
        except BudgetExceeded:
            return
        except FinalFailure as e:
            # Teilkosten fehlgeschlagener Versuche stecken bereits im
            # Live-Zaehler (Bremse bleibt korrekt), landen aber nicht in der
            # jsonl; realen Verbrauch auf platform.deepseek.com gegenpruefen.
            with budget_lock:
                state["failed"] += 1
                state["consecutive"] += 1
                bad = state["consecutive"]
            print(f"UEBERSPRUNGEN {label}: {e}", flush=True)
            if bad >= args.max_consecutive_failures:
                print(f"{bad} endgueltige Fehlschlaege in Folge, stoppe den Lauf.",
                      flush=True)
                stop_event.set()
            return

        details = usage.get("completion_tokens_details") or {}
        row = {
            "case_id": p["case_id"], "condition": p["condition"],
            "model": args.model, "pred": pred,
            "preds_all": [pred] if pred is not None else [],
            "n_samples": 1,
            "temperature": args.temperature,
            "think_budget": mt,
            "thinking": args.thinking,
            "reasoning_effort": args.reasoning_effort,
            "forced": False,
            "n_gen_tokens": usage.get("completion_tokens"),
            "thought_tokens": details.get("reasoning_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "prompt_cache_hit_tokens": usage.get("prompt_cache_hit_tokens"),
            "prompt_cache_miss_tokens": usage.get("prompt_cache_miss_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "est_cost_usd": round(prompt_cost, 8),
            "gen_seconds": round(dt, 2),
            "status": finish or data.get("object", ""),
            "finish_reason": finish,
            "completion_id": cid,
            "api": f"chat/completions {args.base_url}",
            "raw": raw, "reasoning_content": reasoning,
        }
        with write_lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with budget_lock:
            state["done"] += 1
            state["consecutive"] = 0
            n, spent = state["done"], state["spent"]
        el = time.time() - t_start
        rate = n / el * 60 if el > 0 else 0
        eta = (len(todo) - n - state["failed"]) / rate if rate > 0 else float("inf")
        print(f"[{n}/{len(todo)}] {label}: pred={pred} finish={finish} "
              f"reason_tok={row['thought_tokens']} cost~${prompt_cost:.5f} "
              f"cum~${spent:.4f} {dt:.1f}s ({rate:.1f}/min, ETA {eta:.0f} min)",
              flush=True)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(process, p) for p in todo]
            for f in as_completed(futs):
                f.result()
    except KeyboardInterrupt:
        stop_event.set()
        print("\nAbgebrochen. Neustart setzt automatisch fort (Resume).", flush=True)

    if state["failed"]:
        print(f"{state['failed']} Prompts uebersprungen, erneut laufen lassen.",
              flush=True)
    score(args)


def score(args):
    try:
        import pandas as pd
    except ImportError:
        print("pandas fehlt, Scoring uebersprungen")
        return
    if not Path(args.pilot_cases).exists() or not Path(args.out).exists():
        print("pilot_cases.csv oder Antwortdatei fehlt, Scoring uebersprungen")
        return
    cases = pd.read_csv(args.pilot_cases)[["case_id", "strategy", "rho_true"]]
    ans = pd.read_json(args.out, lines=True)
    ans = ans.drop_duplicates(subset=["case_id", "condition"], keep="last")
    m = ans.merge(cases, on="case_id", how="left")
    m["ae"] = (m["pred"] - m["rho_true"]).abs()
    print(f"\n=== {args.model}: MAE gegen rho_true "
          f"({int(m['pred'].notna().sum())}/{len(m)} geparst, "
          f"cost~${m['est_cost_usd'].sum():.4f}) ===")
    tab = m.pivot_table(index="condition", columns="strategy",
                        values="ae", aggfunc="mean").round(3)
    tab["all"] = m.groupby("condition")["ae"].mean().round(3)
    print(tab.to_string())
    unp = m[m["pred"].isna()][["case_id", "condition", "finish_reason"]]
    if len(unp):
        print(f"\nUnparsebar ({len(unp)}), erneut laufen lassen fuer Retry:")
        print(unp.to_string(index=False))


if __name__ == "__main__":
    main()
