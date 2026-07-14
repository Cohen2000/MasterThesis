#!/usr/bin/env python3
"""Run LLMs over the v2.1 prompts and write eval-ready answer files.

Backends
  api  OpenAI-compatible chat completions (NVIDIA NIM, DeepSeek, ...).
       Endpoint via --base-url, key via the env var named in --api-key-env.
       --thinking on|off sends {"chat_template_kwargs": {"thinking": bool}}
       (DeepSeek hybrid reasoning on NIM); --thinking none omits it.
  hf   local HuggingFace transformers (cluster: Qwen / R1-distill).

Every answer line carries prompt_id (the resume key), the full raw reply in
"answer" (including any <think> block), separate "reasoning" when the server
returns reasoning_content, token usage and latency. Reruns with the same
--out skip prompt_ids already present, so interrupted jobs just continue.
Sharding (--shard-index/--shard-count) splits the prompt list stably for
SLURM array jobs; shards write separate files that eval_llm_v2.py reads
together via a glob.

Examples
  python src/run_llm_v2.py --backend api --prompts results/llm_v2/prompts.jsonl \
      --out results/llm_v2/answers_deepseek-v4-pro_nim.jsonl \
      --model deepseek-ai/deepseek-v4-pro --thinking on --limit 3
  python3 run_llm_v2.py --backend hf --prompts prompts_v2.jsonl \
      --out answers_qwen32b_v2.jsonl --model Qwen/Qwen2.5-32B-Instruct
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


def load_prompts(path, shard_index, shard_count, only_ids):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows.sort(key=lambda r: r["prompt_id"])
    if only_ids:
        keep = set(only_ids)
        rows = [r for r in rows if r["prompt_id"] in keep]
    if shard_count > 1:
        rows = [r for i, r in enumerate(rows) if i % shard_count == shard_index]
    return rows


def done_ids(out_path):
    """prompt_ids with a successful record; error records are retried."""
    ids = set()
    if Path(out_path).exists():
        for line in open(out_path):
            try:
                r = json.loads(line)
                fr = str(r.get("finish_reason") or "")
                if not fr.startswith("error"):
                    ids.add(r["prompt_id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


# ---------------------------------------------------------------- api backend
def api_call(base_url, api_key, model, prompt, temperature, max_tokens,
             thinking, timeout, retries, sleep):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens}
    if thinking in ("on", "off"):
        body["chat_template_kwargs"] = {"thinking": thinking == "on"}
    data = json.dumps(body).encode()
    last_err = None
    for attempt in range(retries):
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                out = json.loads(resp.read().decode())
            msg = out["choices"][0].get("message", {})
            return {
                "answer": msg.get("content") or "",
                "reasoning": msg.get("reasoning_content"),
                "finish_reason": out["choices"][0].get("finish_reason"),
                "usage": out.get("usage"),
                "model_echo": out.get("model"),
                "latency_s": round(time.time() - t0, 2),
            }
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read(500).decode(errors='replace')}"
            wait = sleep * (2 ** attempt)
            ra = e.headers.get("Retry-After") if e.headers else None
            if ra:
                try:
                    wait = max(wait, float(ra))
                except ValueError:
                    pass
            if e.code in (408, 409, 429, 500, 502, 503, 504):
                print(f"  [retry {attempt+1}/{retries}] {last_err[:120]} "
                      f"-> wait {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(last_err)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError,
                KeyError) as e:
            last_err = repr(e)
            print(f"  [retry {attempt+1}/{retries}] {last_err[:120]}",
                  flush=True)
            time.sleep(sleep * (2 ** attempt))
    raise RuntimeError(f"giving up after {retries} attempts: {last_err}")


# ----------------------------------------------------------------- hf backend
class HFModel:
    def __init__(self, model_name, max_new_tokens, temperature):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, device_map="auto")
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    def __call__(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.model.device)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens,
                      "pad_token_id": self.tok.eos_token_id}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        t0 = time.time()
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new = out[0][inputs["input_ids"].shape[1]:]
        return {
            "answer": self.tok.decode(new, skip_special_tokens=True),
            "reasoning": None,
            "finish_reason": ("length" if len(new) >= self.max_new_tokens
                              else "stop"),
            "usage": {"prompt_tokens": int(inputs["input_ids"].shape[1]),
                      "completion_tokens": int(len(new))},
            "model_echo": None,
            "latency_s": round(time.time() - t0, 2),
        }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="results/llm_v2/prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["api", "hf"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url",
                    default="https://integrate.api.nvidia.com/v1")
    ap.add_argument("--api-key-env", default="NVIDIA_API_KEY")
    ap.add_argument("--thinking", choices=["on", "off", "none"],
                    default="none",
                    help="chat_template_kwargs.thinking for hybrid-reasoning "
                         "models on OpenAI-compatible servers (NIM DeepSeek)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8192,
                    help="api backend: max completion tokens")
    ap.add_argument("--max-new-tokens", type=int, default=3072,
                    help="hf backend: generation budget")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ids", default="",
                    help="comma-separated prompt_ids (smoke tests)")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="base seconds between api calls / backoff unit")
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    only_ids = [x for x in args.ids.split(",") if x.strip()]
    rows = load_prompts(args.prompts, args.shard_index, args.shard_count,
                        only_ids)
    done = done_ids(args.out)
    todo = [r for r in rows if r["prompt_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(rows)} prompts in shard {args.shard_index+1}/"
          f"{args.shard_count}; {len(done)} done; {len(todo)} to run "
          f"-> {args.out}", flush=True)
    if not todo:
        return

    if args.backend == "api":
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            sys.exit(f"env var {args.api_key_env} is empty")
        call = lambda prompt: api_call(
            args.base_url, api_key, args.model, prompt, args.temperature,
            args.max_tokens, args.thinking, args.timeout, args.retries,
            args.sleep)
    else:
        hf = HFModel(args.model, args.max_new_tokens, args.temperature)
        call = hf

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    n_err = 0
    with open(args.out, "a") as fh:
        for i, r in enumerate(todo, 1):
            try:
                res = call(r["prompt"])
            except Exception as e:  # keep going, record the failure
                res = {"answer": "", "reasoning": None,
                       "finish_reason": f"error: {e}", "usage": None,
                       "model_echo": None, "latency_s": None}
                n_err += 1
            rec = {"id": r["prompt_id"], "prompt_id": r["prompt_id"],
                   "case_id": r["case_id"], "condition": r["condition"],
                   "input_kind": r["input_kind"], "strategy": r["strategy"],
                   "model": args.model, "backend": args.backend,
                   "thinking": args.thinking, "ts": time.strftime(
                       "%Y-%m-%dT%H:%M:%S"), **res}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            tag = (res.get("finish_reason") or "?")[:24]
            print(f"[{i}/{len(todo)}] {r['prompt_id']} {tag} "
                  f"{res.get('latency_s')}s", flush=True)
            if args.backend == "api" and args.sleep > 0:
                time.sleep(args.sleep)
    print(f"done, {n_err} errors; answers in {args.out}", flush=True)
    if n_err:
        print("rerun the same command: error records do not count as done "
              "and will be retried; the evaluator keeps the latest record "
              "per prompt_id")


if __name__ == "__main__":
    main()
