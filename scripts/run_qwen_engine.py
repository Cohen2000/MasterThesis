#!/usr/bin/env python3
"""Qwen offline engine for the revised generic-JSON protocol.

One attempt per request. Admission is persisted before enqueue; interrupted
admissions are never automatically regenerated. Completed records and the
whole runner configuration are bound to immutable hashes.
"""
import argparse, hashlib, json, os, sys, time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import read_json, DESIGN_VERSION, digest, sha, write_json

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
    'engine_seed': 20260916,
    'structured_output': {'json_object': True, 'reasoning_parser': 'qwen3'},
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
    from main_experiment.integrity import validate_request
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
        validate_request(r)
        oid = r['observation_id']
        if oid not in obs:
            obs[oid] = read_json(run / 'observations' / 'sample' / f'{oid}.json')
        if digest(obs[oid]['messages']) != r['prompt_sha256']:
            raise ValueError(f'prompt hash mismatch for {r["id"]}')
        out.append({**r, 'mode': wanted[key], 'messages': obs[oid]['messages']})
    out.sort(key=lambda r: r['id'])
    if shard_count > 1:
        out = [r for i, r in enumerate(out) if i % shard_count == shard_index]
    return out


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
    if a.shard_count<1 or not 0<=a.shard_index<a.shard_count: raise ValueError('invalid shard')
    from main_experiment.integrity import bind
    out.mkdir(parents=True,exist_ok=True)
    import fcntl
    lock=open(out/f'shard_{a.shard_index}.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    model=Path(a.model)
    model_files=sorted(model.glob('*.json'))+sorted(model.glob('*.safetensors'))
    if not model_files: raise ValueError('local pinned model artifacts missing')
    # Content hashes, not a mutable directory name. Computed once per invocation.
    model_hashes={p.name:sha(p) for p in model_files}
    with open(out/'binding.lock','a') as binding_lock:
        fcntl.flock(binding_lock,fcntl.LOCK_EX)
        bind(out/'engine_inputs.json',{'runner_sha256':sha(__file__),
             'requests_sha256':sha(run/'requests.jsonl'),'generation':GENERATION_CONFIG,
             'model_files':model_hashes,
             'max_num_seqs':a.max_num_seqs,'shard_count':a.shard_count})
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
    from vllm.sampling_params import StructuredOutputsParams
    from transformers import AutoTokenizer
    import vllm, inspect, transformers, torch
    from main_experiment.requests import EXECUTION_POLICY
    for module,name in ((vllm,'vllm'),(transformers,'transformers'),(torch,'torch')):
        if module.__version__!=EXECUTION_POLICY['qwen'][name+'_version']:
            raise ValueError(f'unpinned {name} version: {module.__version__}')
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
                                    top_k=TOP_K, presence_penalty=PRESENCE_PENALTY,
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
                'runner': 'run_qwen_engine.py', 'runner_sha256': runner_sha,
                'structured_output': g['structured_output'],
                'design_version': DESIGN_VERSION,
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
    present = sum(result_path(out, r).exists() for r in requests)
    print(f'SHARD_DONE written={written} admitted={admitted} out_tokens={out_tokens} '
          f'{present} of {len(requests)} results present', flush=True)


if __name__ == '__main__':
    main()
