#!/usr/bin/env python3
"""Run one Qwen generation pass over the planned requests with vLLM.

Design choices, and why
-----------------------
Offline batch API instead of an HTTP server. vLLM's LLM.generate does continuous
batching internally, so the server buys nothing here and would add a transport,
streaming and heartbeat failure class that this study does not need. With no
socket there is no "is the connection alive but idle" question to answer: a
request either produced tokens or the process died, and the per-request result
file distinguishes the two.

No guided or structured decoding. Forcing a JSON grammar would constrain the very
first token and truncate the thinking phase, which the design forbids. Generation
is free-form; the final answer is recovered afterwards with the frozen strict
parser, exactly as the offline evaluation expects.

One result file per request id, written by atomic rename. That gives per-request
atomicity, makes resume a directory listing, and makes completeness verifiable by
counting files rather than by trusting a log. A killed job loses at most the
requests in flight.

One generation index per invocation, in its own directory. Resume matches on
request id, so writing a second repeat into the first one's directory would see
the ids as done and silently collapse the repeat-to-repeat variation that the
study measures.
"""
import argparse, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import read_json, CONFIGS

MODES = {
    # Model-card sampling for the two modes. top_k and the penalties are shared.
    'thinking':    dict(temperature=1.0, top_p=0.95, enable_thinking=True,  config_id='qwen_thinking'),
    'nonthinking': dict(temperature=0.7, top_p=0.80, enable_thinking=False, config_id='qwen_nonthinking'),
}
TOP_K = 20
PRESENCE_PENALTY = 1.5
def split_reasoning(text, thinking):
    """Separate the reasoning block from the final answer.

    The pinned chat template opens the block in the *prompt*, not in the output:
    with thinking enabled the generation prompt ends with "<think>\n", so the
    model emits reasoning and then only the closing "</think>". With thinking
    disabled the template writes "<think>\n\n</think>\n\n" into the prompt and
    the output carries no marker at all. The split below mirrors the template's
    own parsing (it splits on "</think>" and takes the last "<think>" before it),
    so a stray opening tag inside the reasoning cannot confuse it.

    Returns (reasoning, final, closed). `closed` is False only in thinking mode
    when no closing tag ever appeared, which means the generation stopped inside
    the reasoning phase and no final answer exists.
    """
    if '</think>' in text:
        head, _, tail = text.partition('</think>')
        return head.split('<think>')[-1].strip(), tail.strip(), True
    if thinking:
        return text.strip(), '', False
    return '', text.strip(), True


def load_requests(run, mode, repeat, shard_index, shard_count):
    cfg = MODES[mode]['config_id']
    rows = [json.loads(l) for l in (run / 'requests.jsonl').read_text().splitlines()]
    rows = [r for r in rows if r['config_id'] == cfg and r['repeat_index'] == repeat]
    obs = {}
    for f in (run / 'observations' / 'sample').glob('*.json'):
        d = read_json(f)
        obs[d['id']] = d
    out = []
    for r in rows:
        if r['status'] == 'skipped_empty':
            continue                      # empty sample: frozen median, never dispatched
        o = obs[r['observation_id']]
        if o['prompt_sha256'] != r['prompt_sha256']:
            raise ValueError(f'prompt hash mismatch for {r["id"]}')
        out.append({**r, 'messages': o['messages']})
    out.sort(key=lambda r: r['id'])
    if shard_count > 1:
        out = [r for i, r in enumerate(out) if i % shard_count == shard_index]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--mode', required=True, choices=sorted(MODES))
    ap.add_argument('--repeat', type=int, required=True)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    ap.add_argument('--tensor-parallel-size', type=int, default=1)
    ap.add_argument('--max-model-len', type=int, default=262144)
    ap.add_argument('--max-tokens', type=int, default=258048)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    ap.add_argument('--max-num-seqs', type=int, default=16)
    ap.add_argument('--chunk', type=int, default=16)
    ap.add_argument('--deadline-seconds', type=float, default=0.0,
                    help='stop starting new chunks this many seconds from now')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    started = time.time()
    run = Path(a.run)
    out = Path(a.out) / f'{a.mode}_r{a.repeat}'
    out.mkdir(parents=True, exist_ok=True)

    requests = load_requests(run, a.mode, a.repeat, a.shard_index, a.shard_count)
    done = {p.stem for p in out.glob('*.json')}
    todo = [r for r in requests if r['id'] not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f'shard {a.shard_index}/{a.shard_count} mode={a.mode} repeat={a.repeat}: '
          f'{len(requests)} planned, {len(done)} already done, {len(todo)} to run', flush=True)
    if not todo:
        print('NOTHING_TO_DO', flush=True)
        return

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    cfg = MODES[a.mode]
    # Text only: the checkpoint is a ConditionalGeneration wrapper; allowing zero
    # image and video items keeps the encoder cache from being allocated at all.
    llm = LLM(model=a.model, tokenizer=a.model, dtype='bfloat16',
              tensor_parallel_size=a.tensor_parallel_size,
              max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization,
              limit_mm_per_prompt={'image': 0, 'video': 0},
              max_num_seqs=a.max_num_seqs, enforce_eager=False, seed=20260916)
    print('MODEL_LOADED', f'{time.time()-started:.1f}s', flush=True)

    for start in range(0, len(todo), a.chunk):
        if a.deadline_seconds and time.time() - started > a.deadline_seconds:
            print(f'DEADLINE_REACHED after {start} of {len(todo)}', flush=True)
            break
        batch = todo[start:start + a.chunk]
        prompts, keep = [], []
        for r in batch:
            text = tok.apply_chat_template(r['messages'], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=cfg['enable_thinking'])
            n_in = len(tok(text, add_special_tokens=False)['input_ids'])
            budget = a.max_model_len - n_in - 8
            if budget <= 0:
                _write(out, r, {'status': 'input_too_long', 'input_tokens': n_in})
                continue
            prompts.append(text)
            keep.append((r, n_in, min(a.max_tokens, budget)))
        if not prompts:
            continue
        params = [SamplingParams(temperature=cfg['temperature'], top_p=cfg['top_p'],
                                 top_k=TOP_K, presence_penalty=PRESENCE_PENALTY,
                                 max_tokens=mt, seed=r['seed'] % (2**31),
                                 skip_special_tokens=False)
                  for (r, _, mt) in keep]
        t0 = time.time()
        outs = llm.generate(prompts, params)
        dt = time.time() - t0
        for (r, n_in, mt), o in zip(keep, outs):
            c = o.outputs[0]
            raw = c.text
            reasoning, final, closed = split_reasoning(raw, cfg['enable_thinking'])
            n_out = len(c.token_ids)
            # A generation that used its whole allowance stopped because of the
            # limit, not because the model was finished. Kept apart from a
            # regular end-of-sequence and from a technical interruption.
            if c.finish_reason == 'length' or n_out >= mt:
                end = 'output_limit'
            elif c.finish_reason == 'stop':
                end = 'model_end'
            else:
                end = f'other:{c.finish_reason}'
            _write(out, r, {
                'status': 'completed', 'raw_text': raw, 'final_text': final,
                'reasoning_text': reasoning, 'reasoning_closed': closed,
                'terminal': True, 'started': True, 'mock': False,
                'finish_reason': c.finish_reason, 'end_state': end,
                'input_tokens': n_in, 'output_tokens': n_out, 'max_tokens': mt,
                'seconds': dt / len(keep), 'model': a.model,
                'mode': a.mode, 'repeat_index': a.repeat})
        print(f'{start+len(batch)}/{len(todo)} chunk {dt:.1f}s '
              f'({sum(len(o.outputs[0].token_ids) for o in outs)} out-tokens)', flush=True)
    n = len(list(out.glob('*.json')))
    print(f'SHARD_DONE {n} of {len(requests)} results present', flush=True)


def _write(out, r, payload):
    rec = {'id': r['id'], 'observation_id': r['observation_id'],
           'graph_id': r['graph_id'], 'arm': r['arm'],
           'sample_index': r['sample_index'], 'repeat_index': r['repeat_index'],
           'config_id': r['config_id'], 'seed': r['seed'],
           'prompt_sha256': r['prompt_sha256'], 'utc': time.time(), **payload}
    p = out / f'{r["id"]}.json'
    tmp = p.with_suffix('.tmp')
    with open(tmp, 'w') as f:
        json.dump(rec, f); f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


if __name__ == '__main__':
    main()
