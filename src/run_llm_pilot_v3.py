#!/usr/bin/env python3
"""Run the pilot prompts through a local open-weights model. Cluster-side, v3.

v3 fixes three problems that surfaced in the R1-Distill smoke run
(all four answers pred=None, every generation ~152s = exactly the
3072-token cap at ~20 tok/s -> the model was cut off mid-<think>):

  1. THINK-BUDGET FORCING (--think-budget N): phase 1 lets the model think
     for at most N new tokens. If no parseable "FINAL:" was produced, the
     partial output is continued ON THE TOKEN LEVEL with "\n</think>\n\nFINAL:"
     and up to --force-tokens more tokens are generated greedily, which forces
     a committed number. Caps the runtime AND guarantees a parseable answer.
     (Budget-forcing in the spirit of s1 / Muennighoff et al. 2025.)
  2. RESUME BUG: v2 counted pred=None rows as done, so a resubmit would skip
     exactly the failures. v3 only skips (case_id, condition) with a parsed pred.
  3. DOUBLE BOS: the DeepSeek chat template already starts with the BOS string;
     v2 then tokenized with add_special_tokens=True, prepending a second BOS.
     v3 tokenizes the templated text with add_special_tokens=False. For Qwen
     this is a no-op (its template has no BOS prefix and its tokenizer adds
     none), so Qwen runs stay bit-identical to v2.

Unchanged: greedy default for instruct models, sampling flags for R1
(temperature 0.6, top_p 0.95, seeded and reproducible), crash-safe
append+flush, no system prompt, conversation retry for non-thinking models.

Usage (cluster):
  python run_llm_pilot_v3.py --prompts prompts.jsonl --out answers_qwen32b.jsonl \
      --model Qwen/Qwen2.5-32B-Instruct
  python run_llm_pilot_v3.py --prompts prompts.jsonl --out answers_r1_32b.jsonl \
      --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
      --temperature 0.6 --top-p 0.95 --seed 0 --think-budget 2048
Smoke:  add --limit 4        Pipeline test without GPU:  add --mock
"""
import argparse
import json
import os
import re
import statistics
import time
import zlib

FINAL_RE = re.compile(r"FINAL\s*[:=]\s*\**\s*([01](?:[\.,]\d+)?|[\.,]\d+)",
                      re.IGNORECASE)


def parse_final(text):
    """Parse the committed number; prefer text after the last </think>."""
    if not text:
        return None
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1]
        hits = FINAL_RE.findall(tail)
        if hits:
            v = float(hits[-1].replace(",", "."))
            return min(max(v, 0.0), 1.0)
    hits = FINAL_RE.findall(text)
    if not hits:
        return None
    v = float(hits[-1].replace(",", "."))
    return min(max(v, 0.0), 1.0)


def load_done(path):
    """Only rows with a parsed prediction count as done (v3 fix #2)."""
    done = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    r = json.loads(line)
                    if r.get("pred") is not None:
                        done.add((r["case_id"], r["condition"]))
                except Exception:
                    pass
    return done


class MockModel:
    name = "mock"

    def answer(self, prompt, cfg, sample_idx=0):
        h = zlib.crc32((prompt + str(sample_idx)).encode()) % 1000
        val = 0.05 + 0.55 * h / 999.0
        raw = (f"<think>mock thinking, decoy FINAL: 0.999 inside think"
               f"</think>\nFINAL: {val:.3f}")
        return raw, parse_final(raw), False, 42


class HFModel:
    def __init__(self, model_id, dtype, temperature, top_p, seed):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.name = model_id
        self.temperature = temperature
        self.top_p = top_p
        self.seed = seed
        print(f"loading {model_id} ...", flush=True)
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=getattr(torch, dtype), device_map="cuda:0")
        self.model.eval()
        print(f"loaded in {time.time()-t0:.0f}s (temperature={temperature}, "
              f"top_p={top_p}, seed={seed})", flush=True)

    # -- low level ----------------------------------------------------------
    def _encode_chat(self, msgs):
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        # v3 fix #3: the templated text already contains all special tokens
        enc = self.tok(text, return_tensors="pt", add_special_tokens=False)
        return {k: v.to(self.model.device) for k, v in enc.items()
                if k in ("input_ids", "attention_mask")}

    def _gen(self, enc, max_new, greedy=False, sample_idx=0):
        kw = dict(max_new_tokens=max_new, pad_token_id=self.tok.eos_token_id)
        if not greedy and self.temperature and self.temperature > 0:
            self.torch.manual_seed(self.seed + 7919 * sample_idx)
            kw.update(do_sample=True, temperature=self.temperature,
                      top_p=self.top_p)
        else:
            kw.update(do_sample=False)
        with self.torch.no_grad():
            out = self.model.generate(**enc, **kw)
        return out[0][enc["input_ids"].shape[1]:]

    def _strip_trailing_eos(self, ids):
        eos = self.tok.eos_token_id
        while len(ids) and int(ids[-1]) == eos:
            ids = ids[:-1]
        return ids

    # -- high level ---------------------------------------------------------
    def answer(self, prompt, cfg, sample_idx=0):
        """Returns (raw_text, pred, forced, n_gen_tokens)."""
        enc = self._encode_chat([{"role": "user", "content": prompt}])
        phase1_budget = cfg.think_budget if cfg.think_budget > 0 \
            else cfg.max_new_tokens
        gen = self._gen(enc, phase1_budget, sample_idx=sample_idx)
        n_tok = int(gen.shape[0])
        text = self.tok.decode(gen, skip_special_tokens=True)
        pred = parse_final(text)
        if pred is not None:
            return text, pred, False, n_tok

        if cfg.think_budget > 0:
            # v3 fix #1: force the answer by continuing on the token level
            force_str = ("\n\nFINAL:" if "</think>" in text
                         else "\n</think>\n\nFINAL:")
            gen_clean = self._strip_trailing_eos(gen)
            fids = self.tok(force_str, return_tensors="pt",
                            add_special_tokens=False
                            ).input_ids.to(self.model.device)
            full = self.torch.cat(
                [enc["input_ids"], gen_clean.unsqueeze(0), fids], dim=1)
            enc2 = {"input_ids": full,
                    "attention_mask": self.torch.ones_like(full)}
            gen2 = self._gen(enc2, cfg.force_tokens, greedy=True)
            tail = self.tok.decode(gen2, skip_special_tokens=True)
            raw = text + force_str + tail
            return raw, parse_final(raw), True, n_tok + int(gen2.shape[0])

        # non-thinking models: conversation retry (unchanged from v2)
        follow = ("Reply with exactly one line and nothing else:\n"
                  "FINAL: <your numeric estimate between 0 and 1>")
        enc2 = self._encode_chat([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": text},
            {"role": "user", "content": follow}])
        gen2 = self._gen(enc2, 64, greedy=True)
        raw = text + "\n---RETRY---\n" + self.tok.decode(
            gen2, skip_special_tokens=True)
        return raw, parse_final(raw), False, n_tok + int(gen2.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.jsonl")
    ap.add_argument("--out", default="answers.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--max-new-tokens", type=int, default=512,
                    help="cap for non-thinking models (no --think-budget)")
    ap.add_argument("--think-budget", type=int, default=0,
                    help=">0: reasoning mode; thinking capped at N tokens, "
                         "then the answer is forced (use for R1-Distill)")
    ap.add_argument("--force-tokens", type=int, default=16)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (instruct models); 0.6 for R1-Distill")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=1,
                    help=">1: median over k sampled answers (needs temperature>0)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.n_samples > 1 and args.temperature <= 0 and not args.mock:
        raise SystemExit("--n-samples > 1 requires --temperature > 0")

    with open(args.prompts) as fh:
        prompts = [json.loads(l) for l in fh if l.strip()]
    if args.limit:
        prompts = prompts[: args.limit]

    done = load_done(args.out)
    todo = [p for p in prompts if (p["case_id"], p["condition"]) not in done]
    print(f"{len(prompts)} prompts, {len(done)} already done, {len(todo)} to run",
          flush=True)
    if not todo:
        return

    model = MockModel() if args.mock else HFModel(
        args.model, args.dtype, args.temperature, args.top_p, args.seed)

    t0 = time.time()
    with open(args.out, "a") as out_fh:
        for k, p in enumerate(todo):
            t1 = time.time()
            preds, raws, forced_any, tok_tot = [], [], False, 0
            for si in range(args.n_samples):
                raw, pred, forced, ntk = model.answer(p["prompt"], args, si)
                preds.append(pred); raws.append(raw)
                forced_any |= forced; tok_tot += ntk
            good = [v for v in preds if v is not None]
            final = statistics.median(good) if good else None
            rec = {"case_id": p["case_id"], "condition": p["condition"],
                   "model": model.name, "pred": final, "preds_all": preds,
                   "n_samples": args.n_samples, "temperature": args.temperature,
                   "seed": args.seed, "think_budget": args.think_budget,
                   "forced": forced_any, "n_gen_tokens": tok_tot,
                   "gen_seconds": round(time.time() - t1, 1),
                   "raw": raws[-1][-6000:]}
            out_fh.write(json.dumps(rec) + "\n")
            out_fh.flush()
            el = time.time() - t0
            eta = el / (k + 1) * (len(todo) - k - 1)
            print(f"[{k+1}/{len(todo)}] {p['case_id']} {p['condition']:16s} "
                  f"pred={final}{' (forced)' if forced_any else ''}  "
                  f"({rec['gen_seconds']}s, {tok_tot} tok, ETA {eta/60:.0f} min)",
                  flush=True)
    print("done.")


if __name__ == "__main__":
    main()
