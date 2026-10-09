#!/usr/bin/env python3
"""DeepSeek and GPT answers for the 384 samples, three per sample (paid APIs).

  python scripts/api.py deepseek --out DIR             one request per answer, eight at a time
  python scripts/api.py openai --out DIR [--tools]     GPT through the Batch API, 96 requests per batch;
                                                       with --tools GPT may run Python code

Every model gets the same prompt, with reasoning set to high and a JSON answer. Without --execute
nothing is sent; the script only says how many answers are missing. A stored answer is never asked
again, and a request that was sent but brought no answer is sent again only with --retry.

Needs DEEPSEEK_API_KEY or OPENAI_API_KEY. The runs of the study additionally kept a spending limit
and, for DeepSeek, waited for the cheaper hours.
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.answers import parse_final  # noqa: E402
from study.common import OUT, REPEATS, digest, read_json  # noqa: E402

MODEL = {'deepseek': 'deepseek-flash', 'openai': 'gpt-6-sol'}
PYTHON_TOOL = {'type': 'code_interpreter', 'container': {'type': 'auto'}}     # the hosted Python tool of GPT


def payload(provider, messages, tools=False):
    """The request: the prompt, reasoning high, a JSON answer and the largest answer length the model allows."""
    if provider == 'deepseek':
        return {'model': MODEL[provider], 'messages': messages, 'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high',
                'response_format': {'type': 'json_object'}, 'max_tokens': 393216, 'stream': False}
    body = {'model': MODEL[provider], 'input': messages, 'reasoning': {'effort': 'high', 'summary': 'auto'},
            'text': {'format': {'type': 'json_object'}}, 'max_output_tokens': 128000}
    if tools: body.update({'tools': [PYTHON_TOOL], 'max_tool_calls': 10})
    return body


def planned(samples, run):
    """One request per sample and repeat; `run` is deepseek, openai or openai_tools.
    First all first answers, then all second ones, so an interrupted run misses the same repeat everywhere."""
    rows = [{'id': f"{s['id']}__{run}__r{repeat}", 'repeat_index': repeat, 'arm': s['arm'], 'graph_id': s['graph_id'],
             'messages': s['messages'], 'prompt_sha256': s['prompt_sha256'], 'order': (repeat, s['sample_index'], s['stratum'], s['graph_id'], s['arm'])}
            for s in samples for repeat in range(1, REPEATS+1)]
    return sorted(rows, key=lambda r: r['order'])


def provider_id(request_id):
    """The name a request has at the provider: a fingerprint, so it tells nothing about network or arm."""
    return 'req_'+hashlib.sha256(request_id.encode()).hexdigest()[:32]


def _reasoning_tokens(usage):
    details = (usage or {}).get('output_tokens_details') or (usage or {}).get('completion_tokens_details') or {}
    return details.get('reasoning_tokens', (usage or {}).get('reasoning_tokens'))


def deepseek_record(row, answer):
    """What is stored of one DeepSeek answer: the final text, the reasoning text and the complete raw answer."""
    choice = answer['choices'][0]
    final, cut = choice['message'].get('content') or '', choice.get('finish_reason') == 'length'
    values, validity = (None, 'max_tokens') if cut else parse_final(final)
    return {'id': row['id'], 'kind': 'main', 'arm': row['arm'], 'graph_id': row['graph_id'], 'prompt_sha256': row['prompt_sha256'],
            'provider': 'deepseek', 'requested_model': MODEL['deepseek'], 'received_utc': datetime.now(timezone.utc).isoformat(),
            'repeat_index': row['repeat_index'], 'returned_model': answer.get('model'), 'usage': answer.get('usage'),
            'reasoning_tokens': _reasoning_tokens(answer.get('usage')), 'reasoning_exposure': 'raw_provider_reasoning',
            'finish_reason': choice.get('finish_reason'), 'limit_hit': cut, 'final_text': final,
            'reasoning_content': choice['message'].get('reasoning_content'), 'validity': validity, 'prediction': values,
            'raw_response': answer}


def openai_record(row, line):
    """What is stored of one GPT answer. `line` is the request's line in the batch output. The final
    answer is the last message; GPT releases only a summary of its reasoning."""
    body = (line.get('response') or {}).get('body') or {}
    output = body.get('output') or []
    messages = [item for item in output if item.get('type') == 'message']
    final = ''.join(part.get('text', '') for part in ((messages[-1].get('content') or []) if messages else [])
                    if part.get('type') == 'output_text')
    code = [item for item in output if item.get('type') == 'code_interpreter_call']
    cut = body.get('status') == 'incomplete' and (body.get('incomplete_details') or {}).get('reason') == 'max_output_tokens'
    values, validity = (None, 'max_output_tokens') if cut else parse_final(final)
    return {'id': row['id'], 'kind': 'main', 'arm': None, 'prompt_sha256': row['prompt_sha256'], 'provider': 'openai',
            'requested_model': MODEL['openai'], 'received_utc': datetime.now(timezone.utc).isoformat(),
            'repeat_index': row['repeat_index'], 'returned_model': body.get('model'), 'usage': body.get('usage'),
            'reasoning_tokens': _reasoning_tokens(body.get('usage')), 'reasoning_exposure': 'provider_reasoning_summary',
            'reasoning_summary': [part for item in output if item.get('type') == 'reasoning' for part in (item.get('summary') or [])],
            'final_text': final, 'limit_hit': cut, 'tool_calls': len(code),
            'containers': sorted({item['container_id'] for item in code if item.get('container_id')}),
            'validity': validity, 'prediction': values, 'raw_response': body, 'provider_id': line['custom_id'],
            'error': line.get('error'), 'raw_batch_response': line}


def batch_line(row, tools):
    return {'custom_id': provider_id(row['id']), 'method': 'POST', 'url': '/v1/responses', 'body': payload('openai', row['messages'], tools)}


def _json(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def call(url, key, body=None, content_type='application/json', raw=False):
    headers = {'Authorization': 'Bearer '+key, **({'Content-Type': content_type} if body is not None else {})}
    with urllib.request.urlopen(urllib.request.Request(url, data=body, headers=headers), timeout=3600) as response:
        return response.read() if raw else json.loads(response.read())


def append(path, value):
    """Add one line and force it to disk, so a crash loses nothing."""
    with path.open('a') as f:
        f.write(json.dumps(value, ensure_ascii=False)+'\n'); f.flush(); os.fsync(f.fileno())


def run_deepseek(todo, out):
    key, lock = os.environ['DEEPSEEK_API_KEY'], Lock()

    def ask(row):
        with lock: append(out/'attempts.jsonl', {'id': row['id']})        # noted before it is sent
        answer = call('https://api.deepseek.com/chat/completions', key, _json(payload('deepseek', row['messages'])))
        with lock: append(out/'responses.jsonl', deepseek_record(row, answer))
    with ThreadPoolExecutor(8) as pool: list(pool.map(ask, todo))


def run_openai(todo, out, tools):
    key = os.environ['OPENAI_API_KEY']
    for first in range(0, len(todo), 96):
        chunk = todo[first:first+96]
        for row in chunk: append(out/'attempts.jsonl', {'id': row['id']})
        lines = b''.join(_json(batch_line(r, tools))+b'\n' for r in chunk)
        mark = 'masterthesis'+hashlib.sha256(lines).hexdigest()[:24]
        form = (f'--{mark}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n--{mark}\r\nContent-Disposition: form-data; '
                f'name="file"; filename="batch.jsonl"\r\nContent-Type: application/jsonl\r\n\r\n').encode()+lines+f'\r\n--{mark}--\r\n'.encode()
        uploaded = call('https://api.openai.com/v1/files', key, form, f'multipart/form-data; boundary={mark}')
        batch = call('https://api.openai.com/v1/batches', key, _json({'input_file_id': uploaded['id'], 'endpoint': '/v1/responses', 'completion_window': '24h'}))
        while batch['status'] not in ('completed', 'expired', 'cancelled', 'failed'):
            time.sleep(120)
            batch = call('https://api.openai.com/v1/batches/'+batch['id'], key)
        rows = {provider_id(r['id']): r for r in chunk}
        if batch.get('output_file_id'):
            for line in call(f"https://api.openai.com/v1/files/{batch['output_file_id']}/content", key, raw=True).decode().splitlines():
                line = json.loads(line)
                if ((line.get('response') or {}).get('body') or {}).get('usage'):      # an answer without usage was not produced
                    append(out/'responses.jsonl', openai_record(rows[line['custom_id']], line))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('provider', choices=tuple(MODEL))
    ap.add_argument('--out', type=Path, required=True, help='folder of this run')
    ap.add_argument('--tools', action='store_true', help='GPT may run Python code')
    ap.add_argument('--execute', action='store_true', help='send the requests')
    ap.add_argument('--retry', action='store_true', help='also send requests again that were sent but brought no answer')
    a = ap.parse_args()
    samples = [read_json(p) for p in sorted((OUT/'samples').glob('*.json'))]
    todo = planned(samples, a.provider+('_tools' if a.tools else ''))
    lines = lambda name: [json.loads(x) for x in (a.out/name).read_text().splitlines()] if (a.out/name).exists() else []
    done = {r['id'] for r in lines('responses.jsonl')}
    sent = {r['id'] for r in lines('attempts.jsonl')}-done
    todo = [r for r in todo if r['id'] not in done and (a.retry or r['id'] not in sent)]
    print(f'{len(done)} answers stored, {len(sent)} sent without an answer, {len(todo)} to ask')
    if a.execute and todo:
        if any(digest(r['messages']) != r['prompt_sha256'] for r in todo): raise SystemExit('a prompt does not match its sample')
        a.out.mkdir(parents=True, exist_ok=True)
        run_deepseek(todo, a.out) if a.provider == 'deepseek' else run_openai(todo, a.out, a.tools)


if __name__ == '__main__':
    main()
