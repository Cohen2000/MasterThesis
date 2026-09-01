#!/usr/bin/env python3
"""vLLM runner for the G3 cluster models.

The HF backend in `run_llm_v2.py` generates one sequence at a time and reached
17 tok/s on an H100. Qwen3.6-27B is a hybrid Mamba model, so batch-1 decoding
leaves nearly all of the GPU idle. vLLM's continuous batching measured 824
tok/s aggregate on the same hardware and the same weights -- a 48x difference,
which is what makes a generous reasoning budget affordable instead of forcing a
trade against truncation.

What is deliberately unchanged: the frozen prompt text, the per-record
`gen_seed` (vLLM takes a seed per request, so the contract carries over
exactly), the sampling parameters, the append-only answers file, resume by
`prompt_id`, the record schema that `is_complete_record` and the frozen
evaluation read, and -- critically -- the chat templating. The prompt is
wrapped by `apply_chat_template` with the same `enable_thinking` switch the HF
path uses and handed to vLLM as token ids, so the model sees exactly the
conversation it would have seen there. Passing the raw string instead makes
the model continue the text rather than answer it: measured on a first attempt,
that drove 171 of 256 non-thinking generations into the token cap and cut
completeness to 33% against a historical 53%. The inference stack is a different implementation of the same
sampling definition; that difference is recorded in every record via
`backend: "vllm"` and the engine version.

Never repairs a prediction. A truncated or unparseable answer stays incomplete
and therefore retryable, exactly as on the HF path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_prompts(path: str, shard_index: int, shard_count: int,
                 only_ids: list[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in
            Path(path).read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: r["prompt_id"])
    if only_ids:
        keep = set(only_ids)
        rows = [r for r in rows if r["prompt_id"] in keep]
    if shard_count > 1:
        rows = [r for i, r in enumerate(rows) if i % shard_count == shard_index]
    return rows


def completed_ids(path: Path) -> set[str]:
    """prompt_ids already answered completely, so a rerun resumes."""
    from run_llm_v2 import is_complete_record

    done = set()
    if not path.exists():
        return done
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if is_complete_record(record):
            done.add(record.get("prompt_id"))
    return done


def templated_token_ids(tokenizer, prompt: str, thinking: str) -> list[int]:
    """The exact encoding `run_llm_v2.HFModel` would have produced.

    The chat template supplies the special tokens, so the follow-up tokenize
    must not add them again -- a second BOS is a silent quality loss rather
    than an error. `enable_thinking` is the Qwen3-family hybrid switch and is
    the only way `--thinking off` has any effect at all; templates without the
    variable ignore it.
    """
    kwargs = {}
    if thinking in ("on", "off"):
        kwargs["enable_thinking"] = thinking == "on"
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False, add_generation_prompt=True, **kwargs)
    return tokenizer.encode(text, add_special_tokens=False)


def build_record(row: dict, output, args, engine_version: str,
                 elapsed: float) -> dict:
    completion = output.outputs[0]
    text = completion.text
    # vLLM reports "length" the same way the HF path does, so a truncated
    # generation stays incomplete and retryable rather than silently counting.
    finish_reason = completion.finish_reason
    n_out = len(completion.token_ids)
    n_in = len(output.prompt_token_ids or [])
    return {
        "id": row["prompt_id"], "prompt_id": row["prompt_id"],
        "case_id": row.get("case_id"), "condition": row.get("condition"),
        "input_kind": row.get("input_kind"), "strategy": row.get("strategy"),
        "subset": row.get("subset"), "stated_arm": row.get("stated_arm"),
        "seed_slot": row.get("seed_slot"),
        "model": args.model, "backend": "vllm",
        "engine": f"vllm {engine_version}",
        "thinking": args.thinking,
        "temperature": args.temperature, "top_p": args.top_p,
        "top_k": args.top_k,
        "seed": int(row["gen_seed"]),
        "rep": row.get("rep"),
        "prompt_sha256": row.get("prompt_sha256"),
        "required_keys": row.get("required_keys"),
        "max_tokens": args.max_new_tokens,
        "answer": text,
        "reasoning": None,
        "finish_reason": finish_reason,
        "usage": {"prompt_tokens": n_in, "completion_tokens": n_out,
                  "total_tokens": n_in + n_out},
        "total_tokens": n_in + n_out,
        "latency_s": elapsed,
        "ts": time.time(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.6-27B")
    parser.add_argument("--thinking", choices=["on", "off"], default="on")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=65536)
    parser.add_argument("--max-model-len", type=int, default=0,
                        help="0 = prompt headroom + max-new-tokens")
    parser.add_argument("--max-num-seqs", type=int, default=64,
                        help="concurrent sequences; each holds a Mamba cache "
                             "block, so this must stay under the block count")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--ids", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=64,
                        help="prompts handed to one generate() call. Must be "
                             "much larger than --max-num-seqs: generate() "
                             "returns only when its slowest sequence is done, "
                             "so a chunk equal to the concurrency drains to a "
                             "single running sequence and wastes the GPU. A "
                             "large chunk lets vLLM refill the running set as "
                             "sequences finish and confines the drain to one "
                             "tail per chunk. But nothing is written until a "
                             "chunk completes, so a chunk as large as the "
                             "shard means a job killed at its wall clock loses "
                             "everything: 64 against a concurrency of 32 keeps "
                             "the refill while writing several times per job.")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vllm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    only = args.ids.split(",") if args.ids else None
    rows = load_prompts(args.prompts, args.shard_index, args.shard_count, only)
    if args.limit:
        rows = rows[:args.limit]
    out_path = Path(args.out)
    done = completed_ids(out_path)
    todo = [r for r in rows if r["prompt_id"] not in done]
    print(f"{len(rows)} prompts in shard {args.shard_index + 1}/"
          f"{args.shard_count}; {len(done)} already complete; "
          f"{len(todo)} to run -> {out_path}", flush=True)
    if not todo:
        return 0

    longest_prompt = max(len(r["prompt"]) for r in todo) // 2 + 512
    max_model_len = args.max_model_len or (longest_prompt + args.max_new_tokens
                                           + 1024)
    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max_model_len, max_num_seqs=args.max_num_seqs,
              trust_remote_code=True)

    written = 0
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start:start + args.chunk]
        params = [SamplingParams(temperature=args.temperature,
                                 top_p=args.top_p, top_k=args.top_k,
                                 max_tokens=args.max_new_tokens,
                                 seed=int(r["gen_seed"])) for r in chunk]
        prompts = [{"prompt_token_ids":
                    templated_token_ids(tokenizer, r["prompt"], args.thinking)}
                   for r in chunk]
        t0 = time.time()
        outputs = llm.generate(prompts, params)
        elapsed = (time.time() - t0) / max(1, len(chunk))
        # Append-only: a later failure never overwrites an earlier answer.
        with out_path.open("a") as handle:
            for row, output in zip(chunk, outputs):
                handle.write(json.dumps(
                    build_record(row, output, args, vllm.__version__, elapsed),
                    separators=(",", ":")) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        written += len(chunk)
        stopped = sum(1 for o in outputs
                      if o.outputs[0].finish_reason == "stop")
        print(f"[{written}/{len(todo)}] chunk done: {stopped}/{len(chunk)} "
              f"finished naturally, {elapsed:.1f}s/prompt", flush=True)
    print(f"done: {written} records appended to {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
