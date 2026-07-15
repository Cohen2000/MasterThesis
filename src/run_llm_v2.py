#!/usr/bin/env python3
"""Run LLMs over the v2.1 prompts and write eval-ready answer files.

Backends
  api  OpenAI-compatible chat completions (NVIDIA NIM, DeepSeek, Gemini ...).
       Endpoint via --base-url, key via the env var named in --api-key-env.
       --thinking on|off sends {"chat_template_kwargs": {"thinking": bool}}
       (DeepSeek hybrid reasoning on NIM); --thinking none omits it.
       --reasoning-effort none|minimal|low|medium|high sends a top-level
       "reasoning_effort" (Mistral Small 4 on NIM, thinking level on the
       Gemini OpenAI-compat endpoint); omitted by default.
       --include-thoughts asks Gemini to return thought summaries;
       --rate-limit-max-wait N waits patiently on HTTP 429 (free tiers)
       instead of burning --retries.
  hf   local HuggingFace transformers (cluster: Qwen3.6 / R1-distill).
       --thinking on|off passes enable_thinking=True/False to the chat
       template (Qwen3-family hybrid thinking); --thinking none omits the
       kwarg (models such as R1-distill that always think).

Every answer line carries prompt_id (the resume key), the final reply in
"answer" and the reasoning separately in "reasoning" (api: the server's
reasoning_content; hf: the <think>...</think> block, with "answer" holding
the text after </think>), plus token usage and latency. Reruns with the
same --out skip prompt_ids whose stored record is complete (parseable
final JSON with all nine keys, not error/length-truncated); everything
else is retried by appending. Sharding (--shard-index/--shard-count)
splits the prompt list stably for SLURM array jobs; shards write separate
files that eval_llm_v2.py reads together via a glob.

Examples
  python src/run_llm_v2.py --backend api --prompts results/llm_v2/prompts.jsonl \
      --out results/llm_v2/answers_deepseek-v4-pro_nim.jsonl \
      --model deepseek-ai/deepseek-v4-pro --thinking on --limit 3
  python src/run_llm_v2.py --backend api --prompts results/llm_v2/prompts.jsonl \
      --out results/llm_v21/answers_mistral-small-4_high.jsonl \
      --model mistralai/mistral-small-4-119b-2603 --reasoning-effort high \
      --temperature 0.7 --top-p 0.95 --limit 3
  python3 run_llm_v2.py --backend hf --prompts prompts.jsonl \
      --out answers_qwen36-27b_think.jsonl --model Qwen/Qwen3.6-27B \
      --thinking on --temperature 1.0 --top-p 0.95 --top-k 20
"""

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

PRED_KEYS = ["rho_k2", "rho_k3", "rho_k4", "rho_k5", "mean_occupancy",
             "C_one_step", "lifetime_mean_over_T", "lo90", "hi90"]


def extract_last_json(text):
    """Last balanced {...} object that contains rho_k2.

    Kept behavior-identical to eval_llm_v2.extract_last_json (code fences
    tolerated) so that resume and evaluation agree on what parses.
    """
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


def is_complete_record(record):
    """Whether a stored response counts as complete for resume.

    Complete means: not an error, not length-truncated, and the answer
    contains a parseable final JSON object with all nine required keys.
    Values are NOT validated or repaired here; out-of-range or
    non-monotonic predictions are an outcome, not a retry reason.
    """
    finish_reason = str(record.get("finish_reason") or "").lower()
    if finish_reason.startswith("error") or finish_reason == "length":
        return False
    obj = extract_last_json(record.get("answer"))
    return isinstance(obj, dict) and all(k in obj for k in PRED_KEYS)


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
    """prompt_ids with a structurally complete record (is_complete_record).

    Error, length-truncated, empty, unparsable and schema-incomplete
    records stay retryable; retries append, the evaluator keeps the
    latest record per prompt_id.
    """
    ids = set()
    if Path(out_path).exists():
        with open(out_path) as fh:
            for line in fh:
                try:
                    record = json.loads(line)
                    if is_complete_record(record):
                        ids.add(record["prompt_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


# ---------------------------------------------------------------- api backend
def retry_wait_seconds(headers, body_text):
    """Server-suggested wait: Retry-After header or Gemini-style retryDelay.

    Gemini 429 bodies carry RetryInfo like '"retryDelay": "39s"'. Returns
    the largest suggestion found, 0.0 if the server suggests nothing.
    """
    waits = [0.0]
    ra = headers.get("Retry-After") if headers else None
    if ra:
        try:
            waits.append(float(ra))
        except ValueError:
            pass
    m = re.search(r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"',
                  body_text or "")
    if m:
        waits.append(float(m.group(1)))
    return max(waits)


def gemini_thinking_body(include_thoughts, reasoning_effort):
    """Return (extra_body, top-level reasoning_effort) for the request.

    Gemini rejects requests carrying reasoning_effort alongside a custom
    thinking_config ('Expected one of either ...; found both', HTTP 400),
    so with include_thoughts the effort moves into the config as
    thinking_level and nothing is sent top-level.
    """
    if not include_thoughts:
        return None, reasoning_effort
    config = {"include_thoughts": True}
    if reasoning_effort is not None:
        config["thinking_level"] = reasoning_effort
    return {"google": {"thinking_config": config}}, None


def read_stream(resp):
    """Assemble an OpenAI-style completion object from an SSE stream.

    Streaming keeps bytes flowing during long generations, so gateway
    timeouts (NIM HTTP 504 on long thinking runs) do not kill the request.
    """
    content, reasoning = [], []
    finish_reason, usage, model = None, None, None
    for raw in resp:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        for ch in chunk.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            if delta.get("reasoning_content"):
                reasoning.append(delta["reasoning_content"])
            if ch.get("finish_reason"):
                finish_reason = ch["finish_reason"]
    msg = {"content": "".join(content)}
    if reasoning:
        msg["reasoning_content"] = "".join(reasoning)
    return {"choices": [{"message": msg, "finish_reason": finish_reason}],
            "usage": usage, "model": model}


def api_call(base_url, api_key, model, prompt, temperature, max_tokens,
             thinking, timeout, retries, sleep, top_p=None,
             reasoning_effort=None, extra_body=None, rate_limit_max_wait=0.0,
             stream=False):
    url = base_url.rstrip("/") + "/chat/completions"
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    if top_p is not None:
        body["top_p"] = top_p
    if thinking in ("on", "off"):
        body["chat_template_kwargs"] = {"thinking": thinking == "on"}
    if reasoning_effort is not None:
        # Mistral Small 4 hybrid reasoning on NIM and Gemini thinking level;
        # "none" disables reasoning and must still be sent explicitly.
        body["reasoning_effort"] = reasoning_effort
    if extra_body:
        # Gemini OpenAI-compat vendor extension (e.g. thinking_config)
        body["extra_body"] = extra_body
    data = json.dumps(body).encode()
    last_err = None
    attempt = 0
    quota_wait = max(sleep, 1.0)  # doubling backoff for patient 429 handling
    while attempt < retries:
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {api_key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if stream:
                    out = read_stream(resp)
                else:
                    out = json.loads(resp.read().decode())
            msg = out["choices"][0].get("message", {})
            return {
                "answer": msg.get("content") or "",
                "reasoning": (msg.get("reasoning_content")
                              or msg.get("reasoning")),
                "finish_reason": out["choices"][0].get("finish_reason"),
                "usage": out.get("usage"),
                "model_echo": out.get("model"),
                "latency_s": round(time.time() - t0, 2),
            }
        except urllib.error.HTTPError as e:
            detail = e.read(2000).decode(errors="replace")
            last_err = f"HTTP {e.code}: {detail[:500]}"
            suggested = retry_wait_seconds(e.headers, detail)
            if e.code == 429 and rate_limit_max_wait > 0:
                # free-tier quota mode: 429 means "wait", not "fail" -- do
                # not consume an attempt; RPM quotas recover within a
                # minute, daily quotas reset at midnight Pacific
                wait = min(max(quota_wait, suggested), rate_limit_max_wait)
                quota_wait = min(quota_wait * 2, rate_limit_max_wait)
                print(f"  [rate-limit] {last_err[:120]} -> wait {wait:.0f}s",
                      flush=True)
                time.sleep(wait)
                continue
            attempt += 1
            wait = max(sleep * (2 ** (attempt - 1)), suggested)
            if e.code in (408, 409, 429, 500, 502, 503, 504):
                print(f"  [retry {attempt}/{retries}] {last_err[:120]} "
                      f"-> wait {wait:.1f}s", flush=True)
                time.sleep(wait)
                continue
            raise RuntimeError(last_err)
        except (OSError, http.client.HTTPException, json.JSONDecodeError,
                KeyError) as e:
            # OSError covers URLError/timeouts/connection resets; the
            # HTTPException branch catches mid-stream disconnects
            # (IncompleteRead) when streaming
            attempt += 1
            last_err = repr(e)
            print(f"  [retry {attempt}/{retries}] {last_err[:120]}",
                  flush=True)
            time.sleep(sleep * (2 ** (attempt - 1)))
    raise RuntimeError(f"giving up after {retries} attempts: {last_err}")


# ----------------------------------------------------------------- hf backend
def split_think(text):
    """Split '<think>...</think> tail' into (reasoning, answer).

    Lossless: reasoning + answer together cover the full generated text.
    Without a closing </think> everything stays in "answer" (a truncated
    think block then fails the resume completeness check and is retried).
    """
    if not isinstance(text, str) or "</think>" not in text:
        return None, text
    head, tail = text.split("</think>", 1)
    if head.startswith("<think>"):
        head = head[len("<think>"):]
    return head.strip(), tail.strip()


class HFModel:
    def __init__(self, model_name, max_new_tokens, temperature,
                 top_p=None, top_k=None, seed=0, thinking="none"):
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
        self.top_p = top_p
        self.top_k = top_k
        self.seed = seed
        self.thinking = thinking

    def __call__(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        template_kwargs = {}
        if self.thinking in ("on", "off"):
            # Qwen3-family hybrid thinking switch; extra kwargs reach the
            # jinja template, templates without the variable ignore it.
            template_kwargs["enable_thinking"] = self.thinking == "on"
        text = self.tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            **template_kwargs)
        # the templated text already contains all special tokens; adding
        # them again would prepend a second BOS on DeepSeek-style templates
        inputs = self.tok(text, return_tensors="pt",
                          add_special_tokens=False).to(self.model.device)
        gen_kwargs = {"max_new_tokens": self.max_new_tokens,
                      "pad_token_id": self.tok.eos_token_id}
        if self.temperature and self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
            if self.top_p is not None:
                gen_kwargs["top_p"] = self.top_p
            if self.top_k is not None:
                gen_kwargs["top_k"] = self.top_k
        else:
            gen_kwargs.update(do_sample=False)
        t0 = time.time()
        # reseed per generation: results do not depend on resume order
        self.torch.manual_seed(self.seed)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new = out[0][inputs["input_ids"].shape[1]:]
        raw = self.tok.decode(new, skip_special_tokens=True)
        reasoning, answer = split_think(raw)
        return {
            "answer": answer,
            "reasoning": reasoning,
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
                    help="api: chat_template_kwargs.thinking (NIM DeepSeek); "
                         "hf: enable_thinking for the chat template (Qwen3 "
                         "family); none omits the switch entirely")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["none", "minimal", "low", "medium", "high"],
                    help="api backend: top-level reasoning_effort (NIM "
                         "Mistral Small 4: none|low|medium|high; Gemini "
                         "OpenAI-compat maps it to the thinking level, "
                         "minimal is its lowest for gemini-3 models); "
                         "'none' is sent explicitly, omitting the flag "
                         "sends nothing")
    ap.add_argument("--include-thoughts", action="store_true",
                    help="api backend: send extra_body.google.thinking_config"
                         ".include_thoughts=true (Gemini OpenAI-compat) so "
                         "thought summaries come back in the response")
    ap.add_argument("--stream", action="store_true",
                    help="api backend: use SSE streaming and reassemble the "
                         "full response client-side; avoids gateway timeouts "
                         "(HTTP 504) on long thinking generations")
    ap.add_argument("--rate-limit-max-wait", type=float, default=0.0,
                    help="api backend: if > 0, treat HTTP 429 as quota "
                         "exhaustion -- retry the same request indefinitely "
                         "with doubling backoff capped at this many seconds "
                         "(free tiers; daily quotas reset at midnight "
                         "Pacific). 0 keeps 429 within normal --retries.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None,
                    help="nucleus sampling; omitted from the request/"
                         "generation when not given")
    ap.add_argument("--top-k", type=int, default=None,
                    help="hf backend: top-k sampling (e.g. 20 for Qwen3)")
    ap.add_argument("--seed", type=int, default=0,
                    help="hf backend: torch seed, re-applied per generation")
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
        extra_body, effort = gemini_thinking_body(args.include_thoughts,
                                                  args.reasoning_effort)
        call = lambda prompt: api_call(
            args.base_url, api_key, args.model, prompt, args.temperature,
            args.max_tokens, args.thinking, args.timeout, args.retries,
            args.sleep, top_p=args.top_p,
            reasoning_effort=effort, extra_body=extra_body,
            rate_limit_max_wait=args.rate_limit_max_wait, stream=args.stream)
    else:
        hf = HFModel(args.model, args.max_new_tokens, args.temperature,
                     top_p=args.top_p, top_k=args.top_k, seed=args.seed,
                     thinking=args.thinking)
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
                   "thinking": args.thinking,
                   "reasoning_effort": args.reasoning_effort,
                   "temperature": args.temperature, "top_p": args.top_p,
                   "top_k": args.top_k, "seed": args.seed,
                   "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **res}
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
