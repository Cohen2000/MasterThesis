#!/usr/bin/env python3
"""Offline planning and explicit execution for the frozen v10 API comparison.

No command contacts a provider unless --execute is supplied. Production reads
the sealed 360 observation JSON files from the verified Qwen archive.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from main_experiment.common import MAIN_KEYS, digest  # noqa: E402
from main_experiment.evaluation import parse_final  # noqa: E402
from main_experiment.observation import messages  # noqa: E402

API_MAIN_ARMS = ('R', 'S', 'H', 'B')
MODELS = {'deepseek': 'deepseek-flash', 'openai': 'gpt-6-sol'}
REPEATS = {'deepseek': 1, 'openai': 3}
OUTPUT_TOKENS = 8192
MARGIN = 1.25
# Conservative planning ceilings in USD per million tokens. DeepSeek rates are
# deliberately above the expected Flash rates; confirm live prices before a run.
RATES = {'deepseek': (1.0, 3.0), 'openai': (1.0, 5.0)}  # OpenAI Batch rates
EXPECTED = {'deepseek': 288, 'openai': 864}


def load_keys():
    path = Path.home() / '.config/masterthesis/api_keys.env'
    if not path.exists():
        return
    if path.stat().st_mode & 0o077:
        raise ValueError('api_keys.env must have mode 600')
    if path.parent.stat().st_mode & 0o077:
        raise ValueError('api_keys.env directory must have mode 700')
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        name, sep, value = line.partition('=')
        if sep and name in ('OPENAI_API_KEY', 'DEEPSEEK_API_KEY'):
            if not os.environ.get(name):
                os.environ[name] = value


def observations(directory):
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))]
    if len(rows) != 360 or len({r['id'] for r in rows}) != 360:
        raise ValueError('frozen observation set must contain 360 distinct rows')
    for row in rows:
        if row['graph_id'] not in MAIN_KEYS or row['arm'] not in (*API_MAIN_ARMS, 'S_obs'):
            raise ValueError('unexpected graph or arm')
        if row['block_sha256'] != digest(row['block']):
            raise ValueError('observation block hash mismatch')
        if row['prompt_sha256'] != digest(row['messages']):
            raise ValueError('frozen prompt hash mismatch')
        if row['messages'] != messages(row['block']):
            raise ValueError('canonical prompt content mismatch')
    cells = {(r['graph_id'], r['arm'], r['sample_index']) for r in rows}
    expected = {(g, arm, i) for g in MAIN_KEYS for arm in (*API_MAIN_ARMS, 'S_obs') for i in (1, 2, 3)}
    if cells != expected:
        raise ValueError('incomplete main observation grid')
    return rows


def manifest(rows, provider, one=False):
    result = []
    for row in sorted(rows, key=lambda r: (r['stratum'], r['graph_id'], r['arm'], r['sample_index'])):
        if row['arm'] not in API_MAIN_ARMS:
            continue
        for repeat in range(1, REPEATS[provider] + 1):
            rid = f"{row['id']}__{provider}__r{repeat}"
            result.append({'id': rid, 'observation_id': row['id'], 'graph_id': row['graph_id'],
                           'stratum': row['stratum'], 'arm': row['arm'],
                           'sample_index': row['sample_index'], 'repeat_index': repeat,
                           'prompt_sha256': row['prompt_sha256'], 'messages': row['messages']})
    if len(result) != EXPECTED[provider] or len({r['id'] for r in result}) != len(result):
        raise ValueError('incomplete or duplicate request manifest')
    return result[:1] if one else result


def payload(provider, row):
    if provider == 'openai':
        return {'model': MODELS[provider], 'input': row['messages'],
                'reasoning': {'effort': 'high'}, 'text': {'format': {'type': 'json_object'}},
                'max_output_tokens': OUTPUT_TOKENS}
    return {'model': MODELS[provider], 'messages': row['messages'],
            'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high',
            'response_format': {'type': 'json_object'}, 'max_tokens': OUTPUT_TOKENS,
            'stream': False}


def batch_line(row):
    return {'custom_id': row['id'], 'method': 'POST', 'url': '/v1/responses',
            'body': payload('openai', row)}


def estimate(rows, provider):
    # Two characters/token is above the checked DeepSeek text-token count for
    # every frozen prompt; 256 tokens cover provider message framing. Full
    # output allowance includes reasoning tokens.
    input_tokens = sum(math.ceil(sum(len(m['content']) for m in r['messages']) / 2) + 256 for r in rows)
    output_tokens = len(rows) * OUTPUT_TOKENS
    input_rate, output_rate = RATES[provider]
    base = (input_tokens * input_rate + output_tokens * output_rate) / 1_000_000
    return {'input_tokens': input_tokens, 'output_tokens': output_tokens,
            'estimated_usd': round(base * MARGIN, 4), 'safety_margin': MARGIN}


def guard(rows, provider, budget, execute):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate request IDs')
    expected = EXPECTED[provider]
    if len(rows) != expected:
        raise ValueError(f'incomplete manifest: {len(rows)} of {expected}')
    cost = estimate(rows, provider)
    if budget is None or budget <= 0:
        raise ValueError('explicit positive --budget-usd required')
    if provider == 'deepseek' and budget != 10:
        raise ValueError('DeepSeek production budget is fixed at USD 10')
    if cost['estimated_usd'] > budget:
        raise ValueError(f"projected USD {cost['estimated_usd']} exceeds budget USD {budget}")
    if execute:
        load_keys()
        key = 'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'OPENAI_API_KEY'
        if not os.environ.get(key):
            raise ValueError(f'{key} absent')
    return cost


def summary(rows, provider, budget=None):
    cost = estimate(rows, provider)
    report = {'provider': provider, 'model': MODELS[provider], 'reasoning': 'high',
              'requests': len(rows), 'arms': API_MAIN_ARMS,
              'graph_strata': {'real': 8, 'surrogate': 8, 'synthetic': 8},
              'sampler_draws': 3, 'model_repeats': REPEATS[provider], **cost,
              'budget_usd': budget}
    print(json.dumps(report, indent=2))


def request(url, key, body=None, method='GET', content_type='application/json'):
    headers = {'Authorization': 'Bearer ' + key}
    if body is not None:
        headers['Content-Type'] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=120) as response:
        return json.loads(response.read())


def json_body(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def execute_deepseek(rows, run_dir):
    key = os.environ['DEEPSEEK_API_KEY']
    run_dir.mkdir(parents=True, exist_ok=False)
    with (run_dir / 'responses.jsonl').open('w') as out:
        for row in rows:
            answer = request('https://api.deepseek.com/chat/completions', key,
                             json_body(payload('deepseek', row)), 'POST')
            choice = answer['choices'][0]
            final = choice['message'].get('content') or ''
            values, validity = parse_final(final)
            record = {'id': row['id'], 'prompt_sha256': row['prompt_sha256'],
                      'returned_model': answer.get('model'), 'usage': answer.get('usage'),
                      'finish_reason': choice.get('finish_reason'), 'final_text': final,
                      'reasoning': choice['message'].get('reasoning_content'),
                      'validity': validity, 'prediction': values, 'raw_response': answer}
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            os.fsync(out.fileno())


def execute_openai(rows, run_dir):
    key = os.environ['OPENAI_API_KEY']
    batch = ''.join(json.dumps(batch_line(r), ensure_ascii=False, separators=(',', ':')) + '\n' for r in rows).encode()
    boundary = 'masterthesis' + hashlib.sha256(batch).hexdigest()[:24]
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
            f'Content-Type: application/jsonl\r\n\r\n').encode() + batch + f'\r\n--{boundary}--\r\n'.encode()
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / 'batch.jsonl').write_bytes(batch)
    uploaded = request('https://api.openai.com/v1/files', key, body, 'POST',
                       f'multipart/form-data; boundary={boundary}')
    atomic_json(run_dir / 'upload.json', uploaded)
    created = request('https://api.openai.com/v1/batches', key,
                      json_body({'input_file_id': uploaded['id'], 'endpoint': '/v1/responses',
                                 'completion_window': '24h'}), 'POST')
    atomic_json(run_dir / 'batch.json', created)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('check', 'cost', 'prepare', 'smoke', 'submit', 'status', 'collect'))
    ap.add_argument('--provider', choices=tuple(MODELS), required=True)
    ap.add_argument('--observations', type=Path)
    ap.add_argument('--budget-usd', type=float)
    ap.add_argument('--output', type=Path, help='prepared JSONL path, or run directory')
    ap.add_argument('--execute', action='store_true', help='explicitly allow provider network requests')
    a = ap.parse_args()
    if a.command in ('status', 'collect'):
        if not a.output or not (a.output / 'batch.json').exists() or a.provider != 'openai':
            ap.error('status/collect require an existing OpenAI Batch run directory')
        if not a.execute:
            print('dry run: no provider request; add --execute to contact OpenAI')
            return
        load_keys()
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            ap.error('OPENAI_API_KEY absent')
        batch = json.loads((a.output / 'batch.json').read_text())
        state = request('https://api.openai.com/v1/batches/' + batch['id'], key)
        print(json.dumps({'id': state['id'], 'status': state['status'], 'request_counts': state.get('request_counts')}, indent=2))
        if a.command == 'collect':
            file_id = state.get('output_file_id')
            if not file_id:
                raise ValueError('batch has no output file yet')
            req = urllib.request.Request('https://api.openai.com/v1/files/' + file_id + '/content',
                                         headers={'Authorization': 'Bearer ' + key})
            with urllib.request.urlopen(req, timeout=120) as response:
                raw = response.read()
            (a.output / 'batch_output.jsonl').write_bytes(raw)
            lines = [json.loads(line) for line in raw.splitlines()]
            expected = [json.loads(line)['custom_id'] for line in (a.output / 'batch.jsonl').read_text().splitlines()]
            if len(lines) != len(expected) or {line['custom_id'] for line in lines} != set(expected):
                raise ValueError('Batch result IDs incomplete or duplicated')
            parsed = []
            for line in lines:
                body = (line.get('response') or {}).get('body') or {}
                content = [part.get('text', '') for item in body.get('output', [])
                           for part in item.get('content', []) if part.get('type') == 'output_text']
                final = ''.join(content)
                values, validity = parse_final(final)
                parsed.append({'id': line['custom_id'], 'final_text': final,
                               'usage': body.get('usage'), 'returned_model': body.get('model'),
                               'validity': validity, 'prediction': values, 'error': line.get('error')})
            (a.output / 'responses.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in parsed))
        return
    if not a.observations:
        ap.error('--observations is required')
    rows = manifest(observations(a.observations), a.provider)
    summary(rows, a.provider, a.budget_usd)
    if a.command == 'check':
        print('frozen observations and prompts: valid')
    elif a.command == 'cost':
        return
    elif a.command == 'prepare':
        if not a.output:
            ap.error('--output is required')
        lines = (batch_line(r) if a.provider == 'openai' else
                 {'id': r['id'], 'prompt_sha256': r['prompt_sha256'], 'payload': payload(a.provider, r)}
                 for r in rows)
        a.output.write_text(''.join(json.dumps(line, ensure_ascii=False) + '\n' for line in lines))
    else:
        if not a.output:
            ap.error('--output run directory is required')
        guard(rows, a.provider, a.budget_usd, a.execute)
        if not a.execute:
            print('dry run: budget and manifest valid; no provider request')
            return
        if a.command == 'smoke':
            rows = rows[:1]
        if a.provider == 'deepseek':
            execute_deepseek(rows, a.output)
        else:
            execute_openai(rows, a.output)


if __name__ == '__main__':
    main()
