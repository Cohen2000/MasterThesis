#!/usr/bin/env python3
"""Measure a small number of serving configurations on development inputs.

Reports what actually matters for planning the main pass: how long the model
takes to load, how long a *complete* answer takes (long reasoning included), the
aggregate token rate, and where the generations stop. It never touches the main
test observations.
"""
import argparse, json, os, time
from pathlib import Path




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--prompts', required=True)
    ap.add_argument('--out', default='probe_result.json')
    ap.add_argument('--tp', type=int, default=2)
    ap.add_argument('--max-model-len', type=int, default=262144)
    ap.add_argument('--max-tokens', type=int, default=258048)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.90)
    ap.add_argument('--seqs', default='4,16')
    a = ap.parse_args()

    import torch
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    gpus = [{'name': torch.cuda.get_device_name(i),
             'total_gb': round(torch.cuda.get_device_properties(i).total_memory / 1e9, 1)}
            for i in range(torch.cuda.device_count())]
    rows = [json.loads(l) for l in Path(a.prompts).read_text().splitlines()]
    tok = AutoTokenizer.from_pretrained(a.model)

    t0 = time.time()
    # Text only: the checkpoint is a ConditionalGeneration wrapper, and allowing
    # zero image and video items keeps the encoder cache from being allocated.
    llm = LLM(model=a.model, tokenizer=a.model, dtype='bfloat16',
              tensor_parallel_size=a.tp, max_model_len=a.max_model_len,
              gpu_memory_utilization=a.gpu_memory_utilization,
              limit_mm_per_prompt={'image': 0, 'video': 0},
              max_num_seqs=max(int(x) for x in a.seqs.split(',')), seed=20260916)
    load_s = time.time() - t0
    print(f'LOAD {load_s:.1f}s', flush=True)

    result = {'gpus': gpus, 'tp': a.tp, 'load_seconds': load_s,
              'max_model_len': a.max_model_len, 'max_tokens': a.max_tokens,
              'configs': []}
    for seqs in [int(x) for x in a.seqs.split(',')]:
        batch = rows[:seqs]
        prompts, params = [], []
        for r in batch:
            text = tok.apply_chat_template(r['messages'], tokenize=False,
                                           add_generation_prompt=True,
                                           enable_thinking=True)
            n_in = len(tok(text, add_special_tokens=False)['input_ids'])
            prompts.append(text)
            params.append(SamplingParams(temperature=1.0, top_p=0.95, top_k=20,
                                         presence_penalty=1.5, seed=20260916,
                                         max_tokens=min(a.max_tokens,
                                                        a.max_model_len - n_in - 8),
                                         skip_special_tokens=False))
        t = time.time()
        outs = llm.generate(prompts, params)
        dt = time.time() - t
        lens = [len(o.outputs[0].token_ids) for o in outs]
        fin = {}
        closed = 0
        for o in outs:
            fin[o.outputs[0].finish_reason] = fin.get(o.outputs[0].finish_reason, 0) + 1
            if '</think>' in o.outputs[0].text: closed += 1
        peak = [round(torch.cuda.max_memory_allocated(i) / 1e9, 1)
                for i in range(torch.cuda.device_count())]
        cfg = {'max_num_seqs': seqs, 'n_prompts': len(batch), 'wall_seconds': dt,
               'output_tokens_total': sum(lens), 'output_tokens_max': max(lens),
               'output_tokens_median': sorted(lens)[len(lens) // 2],
               'tokens_per_second': sum(lens) / dt,
               'seconds_per_answer': dt / len(batch),
               'finish_reasons': fin, 'thinking_blocks_closed': closed,
               'peak_gpu_gb': peak}
        result['configs'].append(cfg)
        print('CONFIG ' + json.dumps(cfg), flush=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print('PROBE_DONE ' + json.dumps({k: v for k, v in result.items() if k != 'configs'}),
          flush=True)


if __name__ == '__main__':
    main()
