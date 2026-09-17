#!/usr/bin/env python3
"""Qwen generation with per-request persistence, for the H revision and later passes.

Why not run_qwen_batch.py
-------------------------
That runner hands fixed chunks to LLM.generate, which returns only when the whole
chunk has finished. A chunk still running when the job's wall time ends loses
every answer in it, including the ones that had already finished, and one long
thinking answer (the previous run went up to 79k tokens) keeps its chunk open.
The previous pass lost twelve job attempts that way.

This runner drives the same offline engine step by step: requests are admitted up
to max_num_seqs at a time, every finished request is written at once by atomic
rename, and a freed slot is refilled immediately (continuous batching across all
selected passes, so one model load serves every mode and repeat). New requests
are admitted only until --admit-seconds; after that the job lets the running ones
finish and stops at --stop-seconds at the latest. A killed job therefore loses at
most the requests in flight, never a finished one.

What is unchanged
-----------------
Model, revision, tokenizer and chat template; LLM(...) arguments; per-request
sampling (model-card values per mode, top_k 20, presence penalty 1.5, the output
allowance min(258048, context - input - 8), seed = request seed mod 2**31,
skip_special_tokens False); free generation without a grammar; the reasoning split;
the result-file layout <mode>_r<repeat>/<request id>.json. GENERATION_CONFIG below
is compared against run_qwen_batch.py by scripts/check_qwen_reuse.py.

Requests are submitted with LLM.enqueue, which in vLLM 0.29.0 is the first half
of LLM.generate (_add_completion_requests: the same prompt rendering and
tokenisation, a counter request id, FINAL_ONLY outputs); the loop then calls
LLMEngine.step as LLM._run_engine does. step() reports the id enqueue assigned
before vLLM appended its random suffix, so both forms are mapped back to the
study's request id. If enqueue or step is unavailable the runner stops instead of
silently falling back to chunked generation.
"""
import argparse, hashlib, json, os, sys, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import read_json

MODES = {
    'thinking':    dict(temperature=1.0, top_p=0.95, enable_thinking=True,  config_id='qwen_thinking'),
    'nonthinking': dict(temperature=0.7, top_p=0.80, enable_thinking=False, config_id='qwen_nonthinking'),
}
TOP_K = 20
PRESENCE_PENALTY = 1.5
GENERATION_CONFIG = {
    'modes': MODES, 'top_k': TOP_K, 'presence_penalty': PRESENCE_PENALTY,
    'min_p': 0.0, 'repetition_penalty': 1.0,
    'max_tokens': 258048, 'max_model_len': 262144, 'context_margin': 8,
    'seed_rule': 'request_seed % 2**31', 'skip_special_tokens': False,
    'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.90,
    'limit_mm_per_prompt': {'image': 0, 'video': 0}, 'enforce_eager': False,
    'engine_seed': 20260916, 'structured_output': None,
    'chat_template': 'tokenizer.apply_chat_template(add_generation_prompt=True, enable_thinking=mode)',
}


def split_reasoning(text, thinking):
    """Identical to run_qwen_batch.split_reasoning; see there for the reasoning."""
    if '</think>' in text:
        head, _, tail = text.partition('</think>')
        return head.split('<think>')[-1].strip(), tail.strip(), True
    if thinking:
        return text.strip(), '', False
    return '', text.strip(), True


def load_requests(run, passes, arms, shard_index, shard_count):
    rows = [json.loads(l) for l in (run / 'requests.jsonl').read_text().splitlines()]
    wanted = {(MODES[m]['config_id'], rep): m for m, rep in passes}
    obs = {}
    out = []
    for r in rows:
        key = (r['config_id'], r['repeat_index'])
        if key not in wanted or (arms and r['arm'] not in arms):
            continue
        if r['status'] == 'skipped_empty':
            continue
        oid = r['observation_id']
        if oid not in obs:
            obs[oid] = read_json(run / 'observations' / 'sample' / f'{oid}.json')
        if obs[oid]['prompt_sha256'] != r['prompt_sha256']:
            raise ValueError(f'prompt hash mismatch for {r["id"]}')
        out.append({**r, 'mode': wanted[key], 'messages': obs[oid]['messages']})
    out.sort(key=lambda r: r['id'])
    if shard_count > 1:
        out = [r for i, r in enumerate(out) if i % shard_count == shard_index]
    return out


def result_path(out, r):
    return out / f'{r["mode"]}_r{r["repeat_index"]}' / f'{r["id"]}.json'


def pending(out, requests):
    """Resume rule: a request is done iff its result file exists (atomic rename)."""
    done = {r['id'] for r in requests if result_path(out, r).exists()}
    return done, [r for r in requests if r['id'] not in done]


def write_result(out, r, payload):
    rec = {'id': r['id'], 'observation_id': r['observation_id'],
           'graph_id': r['graph_id'], 'arm': r['arm'],
           'sample_index': r['sample_index'], 'repeat_index': r['repeat_index'],
           'config_id': r['config_id'], 'seed': r['seed'],
           'prompt_sha256': r['prompt_sha256'], 'utc': time.time(), **payload}
    p = result_path(out, r)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(rec, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def parse_passes(text):
    passes = []
    for item in text.split(','):
        mode, rep = item.split(':')
        if mode not in MODES: raise ValueError(mode)
        passes.append((mode, int(rep)))
    return passes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--passes', default='thinking:1,thinking:2,thinking:3,nonthinking:1,nonthinking:2,nonthinking:3')
    ap.add_argument('--arms', default='', help='comma-separated arm filter, e.g. H')
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--max-num-seqs', type=int, default=16)
    ap.add_argument('--admit-seconds', type=float, default=0.0,
                    help='admit no new request after this many seconds (0: no limit)')
    ap.add_argument('--stop-seconds', type=float, default=0.0,
                    help='stop stepping after this many seconds (0: no limit)')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    started = time.time()
    run = Path(a.run); out = Path(a.out)
    passes = parse_passes(a.passes)
    arms = set(filter(None, a.arms.split(',')))
    requests = load_requests(run, passes, arms, a.shard_index, a.shard_count)
    done, todo = pending(out, requests)
    if a.limit:
        todo = todo[:a.limit]
    print(f'shard {a.shard_index}/{a.shard_count} passes={a.passes} arms={sorted(arms) or "all"}: '
          f'{len(requests)} planned, {len(done)} already done, {len(todo)} to run', flush=True)
    if not todo:
        print('NOTHING_TO_DO', flush=True)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    import vllm, inspect
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(a.model)
    g = GENERATION_CONFIG
    llm = LLM(model=a.model, tokenizer=a.model, dtype=g['dtype'],
              tensor_parallel_size=g['tensor_parallel_size'],
              max_model_len=g['max_model_len'],
              gpu_memory_utilization=g['gpu_memory_utilization'],
              limit_mm_per_prompt=g['limit_mm_per_prompt'],
              max_num_seqs=a.max_num_seqs, enforce_eager=g['enforce_eager'], seed=g['engine_seed'])
    engine = llm.llm_engine
    if not hasattr(llm, 'enqueue') or not hasattr(engine, 'step'):
        raise SystemExit(f'ENGINE_API_MISSING in vllm {vllm.__version__}')
    print('MODEL_LOADED', f'{time.time()-started:.1f}s', 'vllm', vllm.__version__,
          'enqueue', str(inspect.signature(llm.enqueue)), flush=True)
    alias = {}

    pending = deque(todo)
    inflight = {}
    written = 0; admitted = 0; out_tokens = 0
    last_report = time.time()
    while pending or inflight:
        now = time.time() - started
        if a.stop_seconds and now > a.stop_seconds:
            print(f'STOP_DEADLINE with {len(inflight)} in flight and {len(pending)} not admitted', flush=True)
            break
        may_admit = not (a.admit_seconds and now > a.admit_seconds)
        while pending and may_admit and len(inflight) < a.max_num_seqs:
            r = pending.popleft()
            cfg = MODES[r['mode']]
            text = tok.apply_chat_template(r['messages'], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=cfg['enable_thinking'])
            n_in = len(tok(text, add_special_tokens=False)['input_ids'])
            budget = g['max_model_len'] - n_in - g['context_margin']
            if budget <= 0:
                write_result(out, r, {'status': 'input_too_long', 'input_tokens': n_in})
                continue
            mt = min(g['max_tokens'], budget)
            params = SamplingParams(temperature=cfg['temperature'], top_p=cfg['top_p'],
                                    top_k=TOP_K, presence_penalty=PRESENCE_PENALTY,
                                    max_tokens=mt, seed=r['seed'] % (2**31),
                                    skip_special_tokens=False)
            (internal,) = llm.enqueue([text], [params], use_tqdm=False)
            alias[internal] = r['id']
            alias[internal.rsplit('-', 1)[0]] = r['id']
            inflight[r['id']] = (r, n_in, mt, time.time())
            admitted += 1
        if not may_admit and pending and not inflight:
            break
        if not inflight:
            continue
        for o in engine.step():
            rid = alias.get(o.request_id)
            if not o.finished or rid not in inflight:
                continue
            r, n_in, mt, t0 = inflight.pop(rid)
            c = o.outputs[0]
            cfg = MODES[r['mode']]
            reasoning, final, closed = split_reasoning(c.text, cfg['enable_thinking'])
            n_out = len(c.token_ids)
            if c.finish_reason == 'length' or n_out >= mt:
                end = 'output_limit'
            elif c.finish_reason == 'stop':
                end = 'model_end'
            else:
                end = f'other:{c.finish_reason}'
            prompt_ids = getattr(o, 'prompt_token_ids', None)
            write_result(out, r, {
                'status': 'completed', 'raw_text': c.text, 'final_text': final,
                'engine_prompt_tokens': len(prompt_ids) if prompt_ids is not None else None,
                'reasoning_text': reasoning, 'reasoning_closed': closed,
                'terminal': True, 'started': True, 'mock': False,
                'finish_reason': c.finish_reason, 'end_state': end,
                'input_tokens': n_in, 'output_tokens': n_out, 'max_tokens': mt,
                'seconds': time.time() - t0, 'model': a.model,
                'mode': r['mode'], 'repeat_index': r['repeat_index'],
                'runner': 'run_qwen_engine.py', 'runner_sha256': runner_sha,
                'vllm_version': vllm.__version__, 'max_num_seqs': a.max_num_seqs})
            written += 1; out_tokens += n_out
        if time.time() - last_report > 120:
            last_report = time.time()
            print(f'PROGRESS t={time.time()-started:.0f}s written={written} inflight={len(inflight)} '
                  f'pending={len(pending)} out_tokens={out_tokens}', flush=True)
    if inflight:
        print(f'ABANDONED_IN_FLIGHT {len(inflight)}: {sorted(inflight)[:5]}', flush=True)
    present = sum(result_path(out, r).exists() for r in requests)
    print(f'SHARD_DONE written={written} admitted={admitted} out_tokens={out_tokens} '
          f'{present} of {len(requests)} results present', flush=True)


if __name__ == '__main__':
    main()
