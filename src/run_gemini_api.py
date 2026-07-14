#!/usr/bin/env python3
"""run_gemini_api.py (v2): Phase-3-Pilotprompts gegen die Gemini Interactions API.

Aenderungen gegenueber v1, mit Begruendung:
  1. temperature wird per Default NICHT mehr gesetzt. Google-Doku zu Gemini 3.x:
     temperature/top_p/top_k auf Default lassen; Werte < 1.0 koennen "looping or
     degraded performance, particularly in complex mathematical or reasoning
     tasks" ausloesen. Der v1-Default 0.0 war genau dieses Antipattern.
  2. thinking_level wird per Default NICHT gesetzt (= Modell-Default, laut Doku
     "medium" fuer gemini-3.5-flash). Explizite Level nur fuer die Level-Kurve.
  3. Ein endgueltig fehlgeschlagener Prompt crasht nicht mehr den Lauf, sondern
     wird uebersprungen (Resume holt ihn nach). Nach 3 endgueltigen Fehlschlaegen
     in Folge stoppt der Lauf sauber.
  4. --workers N (Default 3): parallele Requests mit globalem Mindestabstand
     zwischen Request-Starts (--sleep, Default 4.5 s => maximal ~13 Starts/min,
     sicher unter dem Free-Tier-RPM). Einzelne langsame Antworten blockieren so
     nicht mehr den ganzen Lauf.
  5. timeout Default 900 s (v1: 300 s hat laufende Generierungen abgebrochen).
  6. 429: Retry-After wird respektiert, sonst Backoff ab 30 s bis 600 s.

Aufruf aus src/:
  export GEMINI_API_KEY=...
  python3 run_gemini_api.py --limit 3          # Smoke
  python3 run_gemini_api.py --workers 3        # voller Lauf, resume-faehig
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


def extract_from_response(data):
    """Sichtbaren Text und Thought-Summaries einsammeln (REST hat kein output_text)."""
    texts, summaries = [], []
    if isinstance(data.get("output_text"), str) and data["output_text"]:
        texts.append(data["output_text"])
    for out in data.get("outputs") or []:
        if out.get("type") == "text" and out.get("text"):
            texts.append(out["text"])
    for st in data.get("steps") or []:
        t = st.get("type")
        if t == "model_output":
            for c in st.get("content") or []:
                if c.get("type") == "text" and c.get("text"):
                    texts.append(c["text"])
        elif t == "thought":
            summ = st.get("summary")
            if isinstance(summ, list):
                for c in summ:
                    if c.get("type") == "text" and c.get("text"):
                        summaries.append(c["text"])
            elif isinstance(summ, dict) and summ.get("text"):
                summaries.append(summ["text"])
    deduped, seen = [], set()
    for txt in texts:
        if txt not in seen:
            deduped.append(txt)
            seen.add(txt)
    return "\n".join(deduped), "\n".join(summaries)


class StartLimiter:
    """Globaler Mindestabstand zwischen Request-Starts ueber alle Worker."""

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


def call_interactions(sess, args, prompt, limiter, stop_event):
    body = {"model": args.model, "input": prompt, "store": args.store,
            "generation_config": {"thinking_summaries": "auto"}}
    if args.thinking_level and args.thinking_level != "default":
        body["generation_config"]["thinking_level"] = args.thinking_level
    if args.temperature is not None:
        body["generation_config"]["temperature"] = args.temperature
    if args.seed is not None:
        body["generation_config"]["seed"] = args.seed
    headers = {"x-goog-api-key": args.api_key, "Content-Type": "application/json"}
    if args.api_revision:
        headers["Api-Revision"] = args.api_revision

    delay = args.initial_retry_delay
    for attempt in range(args.max_retries + 1):
        if stop_event.is_set():
            raise FinalFailure("Lauf gestoppt")
        limiter.wait()
        t0 = time.time()
        try:
            r = sess.post(f"{args.base_url}/interactions", json=body,
                          headers=headers, timeout=args.timeout)
        except requests.RequestException as e:
            if attempt == args.max_retries:
                raise FinalFailure(f"Netzfehler: {e}")
            print(f"    Netzfehler ({e}), Retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
            continue
        dt = time.time() - t0
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After", delay))
            if attempt == args.max_retries:
                raise FinalFailure("429 auch nach allen Retries")
            print(f"    429 rate limit, warte {wait:.0f}s "
                  f"(Versuch {attempt + 1}/{args.max_retries + 1})", flush=True)
            time.sleep(wait)
            delay = min(delay * 2, 600)
            continue
        if r.status_code >= 500:
            if attempt == args.max_retries:
                raise FinalFailure(f"HTTP {r.status_code} auch nach allen Retries")
            print(f"    HTTP {r.status_code}, Retry in {delay:.0f}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 600)
            continue
        if r.status_code >= 400:
            raise FinalFailure(f"HTTP {r.status_code}: {r.text[:400]}")
        return r.json(), dt
    raise RuntimeError("unreachable")


def load_done(path):
    done = set()
    if Path(path).exists():
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("pred") is not None and r.get("case_id") and r.get("condition"):
                    done.add((r["case_id"], r["condition"]))
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="../results/phase3/prompts.jsonl")
    ap.add_argument("--pilot-cases", default="../results/phase3/pilot_cases.csv")
    ap.add_argument("--out", default=None,
                    help="Default: ../results/phase3/answers_<model>.jsonl")
    ap.add_argument("--model", default="gemini-3.5-flash")
    ap.add_argument("--base-url",
                    default="https://generativelanguage.googleapis.com/v1beta")
    ap.add_argument("--api-key-env", default="GEMINI_API_KEY")
    ap.add_argument("--api-revision", default="2026-05-20",
                    help="Api-Revision-Header; leer = nicht senden")
    ap.add_argument("--thinking-level", default="default",
                    choices=["default", "minimal", "low", "medium", "high"],
                    help="default = Parameter nicht senden (Modell-Default, laut "
                         "Doku medium fuer gemini-3.5-flash)")
    ap.add_argument("--temperature", type=float, default=None,
                    help="NICHT setzen. Google raet fuer Gemini 3.x explizit vom "
                         "Aendern ab (Looping-/Degradationsrisiko bei Reasoning).")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--store", action="store_true",
                    help="Interactions serverseitig speichern (Default: aus)")
    ap.add_argument("--workers", type=int, default=3,
                    help="parallele Requests; Startrate bleibt durch --sleep begrenzt")
    ap.add_argument("--sleep", type=float, default=4.5,
                    help="globaler Mindestabstand zwischen Request-Starts in s")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--max-retries", type=int, default=4)
    ap.add_argument("--initial-retry-delay", type=float, default=30.0)
    ap.add_argument("--unparsed-retries", type=int, default=2)
    ap.add_argument("--max-consecutive-failures", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--conditions", nargs="*", default=None)
    args = ap.parse_args()

    args.api_key = os.environ.get(args.api_key_env, "")
    if not args.api_key:
        sys.exit(f"{args.api_key_env} ist nicht gesetzt")
    if args.temperature is not None:
        print("WARNUNG: temperature explizit gesetzt. Google raet fuer Gemini 3.x "
              "davon ab (Looping-/Degradationsrisiko).", flush=True)
    if args.out is None:
        safe = args.model.replace("/", "-")
        args.out = f"../results/phase3/answers_{safe}.jsonl"

    with open(args.prompts) as f:
        prompts = [json.loads(l) for l in f]
    if args.conditions:
        prompts = [p for p in prompts if p["condition"] in args.conditions]
    done = load_done(args.out)
    todo = [p for p in prompts if (p["case_id"], p["condition"]) not in done]
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(prompts)} Prompts, {len(done)} erledigt, {len(todo)} offen "
          f"-> {args.out}")
    lvl = args.thinking_level if args.thinking_level != "default" \
        else "default (laut Doku medium)"
    print(f"model={args.model} thinking_level={lvl} "
          f"temperature={'default (1.0)' if args.temperature is None else args.temperature} "
          f"workers={args.workers} sleep={args.sleep}s", flush=True)

    limiter = StartLimiter(args.sleep)
    write_lock = threading.Lock()
    fail_lock = threading.Lock()
    stop_event = threading.Event()
    state = {"done": 0, "failed": 0, "consecutive": 0}
    t_start = time.time()
    tls = threading.local()

    def get_session():
        if not hasattr(tls, "sess"):
            tls.sess = requests.Session()
        return tls.sess

    def process(p):
        label = f"{p['case_id']} | {p['condition']}"
        if stop_event.is_set():
            return
        sess = get_session()
        pred, raw, summary, usage, status, iid, dt = (None, "", "", {}, "", "", 0.0)
        try:
            for sub in range(args.unparsed_retries + 1):
                data, dt = call_interactions(sess, args, p["prompt"], limiter,
                                             stop_event)
                status = data.get("status", "")
                iid = data.get("id", "")
                usage = data.get("usage") or {}
                raw, summary = extract_from_response(data)
                pred = parse_final(raw)
                if pred is not None or sub == args.unparsed_retries:
                    break
                print(f"  {label}: kein FINAL parsebar (status={status}), "
                      f"Versuch {sub + 2}/{args.unparsed_retries + 1}", flush=True)
        except FinalFailure as e:
            with fail_lock:
                state["failed"] += 1
                state["consecutive"] += 1
                bad = state["consecutive"]
            print(f"UEBERSPRUNGEN {label}: {e}", flush=True)
            if bad >= args.max_consecutive_failures:
                print(f"{bad} endgueltige Fehlschlaege in Folge, stoppe den Lauf. "
                      f"Neustart setzt automatisch fort (Resume).", flush=True)
                stop_event.set()
            return

        row = {
            "case_id": p["case_id"], "condition": p["condition"],
            "model": args.model, "pred": pred,
            "preds_all": [pred] if pred is not None else [],
            "n_samples": 1,
            "temperature": args.temperature, "seed": args.seed,
            "think_budget": -1,
            "thinking_level": None if args.thinking_level == "default"
            else args.thinking_level,
            "forced": False,
            "n_gen_tokens": usage.get("total_output_tokens"),
            "thought_tokens": usage.get("total_thought_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "gen_seconds": round(dt, 2),
            "status": status, "interaction_id": iid,
            "api": f"interactions {args.base_url}",
            "raw": raw, "thought_summary": summary,
        }
        with write_lock:
            with open(args.out, "a") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with fail_lock:
            state["done"] += 1
            state["consecutive"] = 0
            n = state["done"]
        el = time.time() - t_start
        rate = n / el * 60 if el > 0 else 0
        eta = (len(todo) - n - state["failed"]) / rate if rate > 0 else float("inf")
        print(f"[{n}/{len(todo)}] {label}: pred={pred} status={status} "
              f"thought_tok={row['thought_tokens']} {dt:.1f}s "
              f"({rate:.1f}/min, ETA {eta:.0f} min)", flush=True)

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
          f"({int(m['pred'].notna().sum())}/{len(m)} geparst) ===")
    tab = m.pivot_table(index="condition", columns="strategy",
                        values="ae", aggfunc="mean").round(3)
    tab["all"] = m.groupby("condition")["ae"].mean().round(3)
    print(tab.to_string())
    unp = m[m["pred"].isna()][["case_id", "condition"]]
    if len(unp):
        print(f"\nUnparsebar ({len(unp)}), erneut laufen lassen fuer Retry:")
        print(unp.to_string(index=False))


if __name__ == "__main__":
    main()
