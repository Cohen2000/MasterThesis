#!/usr/bin/env python3
"""Qwen answers for the 384 samples, on one GPU with the vLLM engine (run on the computing cluster).

Every sample is answered three times with a thinking phase and three times without. A request is
tried once: a finished answer is skipped when the job is restarted, a started one is never asked again.

  python scripts/qwen.py --samples results/samples --out results/qwen --model <folder of Qwen3.6-35B-A3B>
"""
import argparse, hashlib, json, os, sys, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from study.common import LABEL, REPEATS, digest, read_json, seed, sha, write_json

MODEL = 'Qwen/Qwen3.6-35B-A3B'        # revision 995ad96eacd98c81ed38be0c5b274b04031597b0
VERSIONS = {'vllm': '0.29.0', 'transformers': '5.17.0', 'torch': '2.13.0+cu130'}
# Fixed labels that are part of the request IDs and seeds (R, H, B and S were frozen on different days).
VERSION = {'R': 'panel888-access-v9-20260922', 'H': 'panel888-access-v9-20260922', 'B': 'panel888-access-v9-20260922',
           'S': 'panel888-access-v10-20260923'}
# Sampling settings of the model card. The presence penalty discourages loops while thinking
# and is off without thinking, where the answer is one short JSON object.
MODES = {
    'thinking':    dict(temperature=1.0, top_p=0.95, presence_penalty=1.5, enable_thinking=True,  config_id='qwen_thinking'),
    'nonthinking': dict(temperature=0.7, top_p=0.80, presence_penalty=0.0, enable_thinking=False, config_id='qwen_nonthinking'),
}
TOP_K = 20
GENERATION_CONFIG = {
    'max_tokens': 258048, 'max_model_len': 262144, 'context_margin': 8,
    'dtype': 'bfloat16', 'tensor_parallel_size': 1, 'gpu_memory_utilization': 0.90,
    'limit_mm_per_prompt': {'image': 0, 'video': 0}, 'enforce_eager': False,
    'engine_seed': 20260921,
    'structured_output': {'json_object': True, 'reasoning_parser': 'qwen3'},
}


def payload(mode, messages, request_seed, version):
    """What one request asks for; its fingerprint is stored with the answer."""
    thinking = MODES[mode]['enable_thinking']
    return {'model': MODEL, 'messages': messages, 'max_tokens': 258048,
            'temperature': 1. if thinking else .7, 'top_p': .95 if thinking else .80,
            'top_k': 20, 'min_p': 0., 'presence_penalty': 1.5 if thinking else 0., 'repetition_penalty': 1.,
            'chat_template_kwargs': {'enable_thinking': thinking}, 'seed': request_seed,
            'structured_output': {'json_object': True, 'reasoning_parser': 'qwen3', 'applies': 'after reasoning end'},
            'design_version': version,
            'executed_transport': 'vllm offline engine (LLM.enqueue + LLMEngine.step)', 'executed_streaming': False}


def requests(samples, passes=None):
    """One request per sample, mode and repeat, each with its own seed."""
    out = []
    for s in samples:
        version = VERSION[s['arm']]
        for mode, cfg in MODES.items():
            for repeat in range(1, REPEATS+1):
                if passes and (mode, repeat) not in passes: continue
                config = cfg['config_id']
                request_seed = seed('llm', s['graph_id'], LABEL[s['arm']], s['sample_index'], repeat, config+':'+version)
                out.append({'id': f"{s['id']}__{config}__r{repeat}__{version}", 'observation_id': s['id'],
                            'graph_id': s['graph_id'], 'arm': s['arm'], 'sample_index': s['sample_index'],
                            'repeat_index': repeat, 'config_id': config, 'mode': mode, 'seed': request_seed,
                            'messages': s['messages'], 'prompt_sha256': s['prompt_sha256'],
                            'payload_sha256': digest(payload(mode, s['messages'], request_seed, version))})
    return sorted(out, key=lambda r: r['id'])


def split_reasoning(text, thinking):
    """Split raw output at the chat template's </think> marker.

    Only the text after </think> is the final answer. A thinking-mode output that
    never closes its reasoning has no final answer (reasoning_closed=False).
    """
    if '</think>' in text:
        head, _, tail = text.partition('</think>')
        return head.split('<think>')[-1].strip(), tail.strip(), True
    if thinking:
        return text.strip(), '', False
    return '', text.strip(), True


def result_path(out, r):
    return out / f'{r["mode"]}_r{r["repeat_index"]}' / f'{r["id"]}.json'


def pending(out, requests):
    """Never repeat an admitted attempt; require exact identity for completed ones."""
    done=set()
    for r in requests:
        path=result_path(out,r)
        if path.exists():
            d=read_json(path)
            for key in ('id','prompt_sha256','payload_sha256','seed'):
                if d.get(key)!=r.get(key): raise ValueError(f'resume {key} mismatch: {path}')
            if d.get('runner_sha256')!=sha(__file__): raise ValueError('runner changed on resume')
            done.add(r['id'])
        elif path.with_suffix('.attempt').exists():
            attempt=read_json(path.with_suffix('.attempt'))
            if attempt['payload_sha256']!=r['payload_sha256']: raise ValueError('attempt binding mismatch')
            write_result(out,r,{'status':'interrupted','started':True,'terminal':True,
                               'technical_error':True,'final_text':'','end_state':'process_interrupted'})
            done.add(r['id'])
    return done,[r for r in requests if r['id'] not in done]


def write_result(out, r, payload):
    rec = {'id': r['id'], 'observation_id': r['observation_id'],
           'graph_id': r['graph_id'], 'arm': r['arm'],
           'sample_index': r['sample_index'], 'repeat_index': r['repeat_index'],
           'config_id': r['config_id'], 'seed': r['seed'],
           'prompt_sha256': r['prompt_sha256'], 'payload_sha256': r['payload_sha256'],
           'runner_sha256':sha(__file__), 'utc': time.time(), **payload}
    p = result_path(out, r)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(rec, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--passes', default='thinking:1,thinking:2,thinking:3,nonthinking:1,nonthinking:2,nonthinking:3')
    ap.add_argument('--shard-index', type=int, default=0, help='this job of --shard-count parallel jobs')
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--max-num-seqs', type=int, default=16)
    ap.add_argument('--admit-seconds', type=float, default=0.0, help='start no new request after this many seconds (0: no limit)')
    ap.add_argument('--stop-seconds', type=float, default=0.0, help='stop after this many seconds (0: no limit)')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    started = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = open(out/f'shard_{a.shard_index}.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)        # two jobs never write the same answers
    passes = {(mode, int(repeat)) for mode, repeat in (x.split(':') for x in a.passes.split(','))}
    todo = requests([read_json(p) for p in sorted(Path(a.samples).glob('*.json'))], passes)
    todo = [r for i, r in enumerate(todo) if i % a.shard_count == a.shard_index]
    planned = todo
    done, todo = pending(out, todo)
    if a.limit:
        todo = todo[:a.limit]
    print(f'{len(planned)} requests, {len(done)} done, {len(todo)} to run', flush=True)
    if not todo:
        return
    requests_all = planned

    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer
    import vllm, inspect, transformers, torch
    for module, name in ((vllm, 'vllm'), (transformers, 'transformers'), (torch, 'torch')):
        if module.__version__ != VERSIONS[name]: raise ValueError(f'{name} {module.__version__} is not the version of the study')
    runner_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    tok = AutoTokenizer.from_pretrained(a.model)
    g = GENERATION_CONFIG
    extra = {'reasoning_parser': g['structured_output']['reasoning_parser']}
    llm = LLM(model=a.model, tokenizer=a.model, dtype=g['dtype'],
              tensor_parallel_size=g['tensor_parallel_size'],
              max_model_len=g['max_model_len'],
              gpu_memory_utilization=g['gpu_memory_utilization'],
              limit_mm_per_prompt=g['limit_mm_per_prompt'],
              max_num_seqs=a.max_num_seqs, enforce_eager=g['enforce_eager'], seed=g['engine_seed'],
              **extra)
    engine = llm.llm_engine
    if not hasattr(llm, 'enqueue') or not hasattr(engine, 'step'):
        raise SystemExit(f'ENGINE_API_MISSING in vllm {vllm.__version__}')
    print('MODEL_LOADED', f'{time.time()-started:.1f}s', 'vllm', vllm.__version__,
          'enqueue', str(inspect.signature(llm.enqueue)), flush=True)
    alias = {}

    queue = deque(todo)
    inflight = {}
    written = 0; admitted = 0; out_tokens = 0
    last_report = time.time()
    while queue or inflight:
        now = time.time() - started
        if a.stop_seconds and now > a.stop_seconds:
            print(f'STOP_DEADLINE with {len(inflight)} in flight and {len(queue)} not admitted', flush=True)
            break
        may_admit = not (a.admit_seconds and now > a.admit_seconds)
        while queue and may_admit and len(inflight) < a.max_num_seqs:
            r = queue.popleft()
            cfg = MODES[r['mode']]
            text = tok.apply_chat_template(r['messages'], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=cfg['enable_thinking'])
            n_in = len(tok(text, add_special_tokens=False)['input_ids'])
            budget = g['max_model_len'] - n_in - g['context_margin']
            if budget <= 0:
                write_result(out, r, {'status':'input_too_long','input_tokens':n_in,'terminal':True,'started':True,'technical_error':True,'final_text':''})
                continue
            mt = min(g['max_tokens'], budget)
            so = StructuredOutputsParams(json_object=True)
            params = SamplingParams(temperature=cfg['temperature'], top_p=cfg['top_p'],
                                    top_k=TOP_K, presence_penalty=cfg['presence_penalty'],
                                    max_tokens=mt, seed=r['seed'] % (2**31),
                                    skip_special_tokens=False, structured_outputs=so)
            write_json(result_path(out,r).with_suffix('.attempt'),
                       {'id':r['id'],'payload_sha256':r['payload_sha256'],'admitted_utc':time.time(),'input_tokens':n_in})
            (internal,) = llm.enqueue([text], [params], use_tqdm=False)
            alias[internal] = r['id']
            alias[internal.rsplit('-', 1)[0]] = r['id']
            inflight[r['id']] = (r, n_in, mt, time.time())
            admitted += 1
        if not may_admit and queue and not inflight:
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
                'runner': 'qwen.py', 'runner_sha256': runner_sha,
                'structured_output': g['structured_output'],
                'design_version': VERSION['S'],
                'vllm_version': vllm.__version__, 'max_num_seqs': a.max_num_seqs})
            written += 1; out_tokens += n_out
        if time.time() - last_report > 120:
            last_report = time.time()
            print(f'PROGRESS t={time.time()-started:.0f}s written={written} inflight={len(inflight)} '
                  f'queued={len(queue)} out_tokens={out_tokens}', flush=True)
    for rid,(r,n_in,mt,t0) in inflight.items():
        write_result(out,r,{'status':'interrupted','started':True,'terminal':True,
                           'technical_error':True,'final_text':'','input_tokens':n_in,
                           'end_state':'job_deadline','seconds':time.time()-t0})
    if inflight:
        print(f'ABANDONED_IN_FLIGHT {len(inflight)}: {sorted(inflight)[:5]}', flush=True)
    present = sum(result_path(out, r).exists() for r in requests_all)
    print(f'{written} answers written, {present} of {len(requests_all)} present', flush=True)


if __name__ == '__main__':
    main()
