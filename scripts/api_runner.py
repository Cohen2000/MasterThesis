#!/usr/bin/env python3
"""Offline planning and explicit execution for the frozen v10 API comparison.

No command contacts a provider unless --execute is supplied. Production reads
the local copy of the 360 sealed observation JSON files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from threading import Lock
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
# USD per million tokens: conservative DeepSeek safety ceilings and GPT Batch.
RATES = {'deepseek': (1.0, 3.0), 'openai': (1.0, 5.0)}  # OpenAI Batch rates
DEEPSEEK_OFFPEAK_RATES = (0.15, 0.60)
DEEPSEEK_PEAK_BUFFER_MINUTES = 10
DEEPSEEK_DEFAULT_CONCURRENCY = 8
DEEPSEEK_BUDGET_USD = 10
OPENAI_BUDGET_USD = 200
EXPECTED = {'deepseek': 288, 'openai': 864}
REASONING_EXPOSURE = {'deepseek': 'raw_provider_reasoning',
                      'openai': 'provider_reasoning_summary',
                      'qwen': 'generated_reasoning_block'}


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


def smoke_observation(path, provider):
    row = json.loads(path.read_text())
    if row.get('domain') not in ('training', 'pool_train', 'pool_dev'):
        raise ValueError('smoke requires a training/dev/pool observation')
    if row.get('graph_id') in MAIN_KEYS or row.get('empty'):
        raise ValueError('smoke cannot use a main graph or empty observation')
    if row.get('arm') not in API_MAIN_ARMS:
        raise ValueError('smoke arm must be R/S/H/B')
    if row.get('block_sha256') != digest(row['block']) or row.get('prompt_sha256') != digest(row['messages']):
        raise ValueError('smoke block or prompt hash mismatch')
    if row['messages'] != messages(row['block']):
        raise ValueError('smoke canonical prompt mismatch')
    return {'id': row['id'] + '__' + provider + '__smoke',
            'prompt_sha256': row['prompt_sha256'], 'messages': row['messages'],
            'kind': 'smoke'}


def manifest(rows, provider):
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
    return result


def payload(provider, row):
    if provider == 'openai':
        return {'model': MODELS[provider], 'input': row['messages'],
                'reasoning': {'effort': 'high', 'summary': 'auto'},
                'text': {'format': {'type': 'json_object'}},
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
    result = {'input_tokens': input_tokens, 'output_tokens': output_tokens,
              'estimated_usd': round(base * MARGIN, 4), 'safety_margin': MARGIN}
    if provider == 'deepseek':
        result['offpeak_at_token_allowance_usd'] = round(
            (input_tokens * DEEPSEEK_OFFPEAK_RATES[0]
             + output_tokens * DEEPSEEK_OFFPEAK_RATES[1]) / 1_000_000, 4)
    return result


def reserve_usd(row):
    """Conservative DeepSeek charge for one unsent request, including margin."""
    chars = sum(len(m['content']) for m in row['messages'])
    return MARGIN * ((math.ceil(chars / 2) + 256) * RATES['deepseek'][0]
                     + OUTPUT_TOKENS * RATES['deepseek'][1]) / 1_000_000


def actual_usd(record):
    """Charge completed token usage at conservative cache-miss ceilings."""
    usage = record.get('usage') or {}
    prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
    if not all(type(x) is int and x >= 0 for x in (prompt, completion)):
        raise ValueError(f"{record.get('id')}: completed DeepSeek usage is missing or invalid")
    return MARGIN * (prompt * RATES['deepseek'][0]
                     + completion * RATES['deepseek'][1]) / 1_000_000


def next_deepseek_peak(now):
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('DeepSeek scheduling requires timezone-aware UTC')
    now = now.astimezone(timezone.utc)
    for day in range(8):
        date = (now + timedelta(days=day)).date()
        if date.weekday() >= 5:
            continue
        for hour in (1, 6):
            start = datetime(date.year, date.month, date.day, hour, tzinfo=timezone.utc)
            if start > now:
                return start
    raise AssertionError('next DeepSeek peak not found')


def deepseek_window(now):
    """Return off-peak allowance and time left before the next UTC peak."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('DeepSeek scheduling requires timezone-aware UTC')
    now = now.astimezone(timezone.utc)
    peak = now.weekday() < 5 and (1 <= now.hour < 4 or 6 <= now.hour < 10)
    next_peak = next_deepseek_peak(now)
    remaining = next_peak - now
    allowed = not peak and remaining > timedelta(minutes=DEEPSEEK_PEAK_BUFFER_MINUTES)
    return allowed, peak, next_peak, remaining


def require_deepseek_offpeak(now, announce=False):
    allowed, peak, next_peak, remaining = deepseek_window(now)
    if announce:
        print('DeepSeek pricing window: ' + ('PEAK' if peak else 'OFF-PEAK'))
        print('current UTC time:', now.astimezone(timezone.utc).isoformat())
        print('next peak start UTC:', next_peak.isoformat())
        safe_remaining = max(timedelta(0), remaining - timedelta(minutes=DEEPSEEK_PEAK_BUFFER_MINUTES)) if not peak else timedelta(0)
        print('safe off-peak time remaining:', str(safe_remaining))
    if peak:
        raise ValueError('DeepSeek execution blocked: current UTC time is inside the provider peak-price window. Retry during off-peak.')
    if not allowed:
        raise ValueError('DeepSeek execution paused: within the 10-minute pre-peak buffer. Resume in the next off-peak window.')


def guard(rows, provider, budget, execute):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate request IDs')
    expected = EXPECTED[provider]
    if len(rows) != expected:
        raise ValueError(f'incomplete manifest: {len(rows)} of {expected}')
    cost = estimate(rows, provider)
    if budget is None or budget <= 0:
        raise ValueError('explicit positive --budget-usd required')
    if provider == 'deepseek' and budget != DEEPSEEK_BUDGET_USD:
        raise ValueError('DeepSeek production budget is fixed at USD 10')
    if provider == 'openai' and budget > OPENAI_BUDGET_USD:
        raise ValueError('GPT experiment budget cannot exceed the approved USD 200 cap')
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
              'reasoning_exposure': REASONING_EXPOSURE[provider],
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
    with urllib.request.urlopen(req, timeout=1800) as response:
        return json.loads(response.read())


def json_body(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def reasoning_tokens(usage):
    usage = usage or {}
    details = usage.get('output_tokens_details') or usage.get('completion_tokens_details') or {}
    return details.get('reasoning_tokens', usage.get('reasoning_tokens'))


def openai_record(body, rid, prompt_sha256=None):
    """Keep the provider summary as returned; hidden reasoning is unavailable."""
    output = body.get('output') or []
    content = [part.get('text', '') for item in output for part in item.get('content', [])
               if part.get('type') == 'output_text']
    summaries = [part for item in output if item.get('type') == 'reasoning'
                 for part in (item.get('summary') or [])]
    final = ''.join(content)
    values, validity = parse_final(final)
    return {'id': rid, 'prompt_sha256': prompt_sha256,
            'returned_model': body.get('model'), 'usage': body.get('usage'),
            'reasoning_tokens': reasoning_tokens(body.get('usage')),
            'reasoning_exposure': REASONING_EXPOSURE['openai'],
            'reasoning_summary': summaries, 'final_text': final,
            'validity': validity, 'prediction': values, 'raw_response': body}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def deepseek_progress(rows, run_dir, smoke=None):
    """Read durable completions; never retry a launched request with no answer."""
    planned = {r['id']: r for r in rows}
    completed = {}
    for record in read_jsonl(run_dir / 'responses.jsonl'):
        rid = record['id']
        if rid in completed:
            raise ValueError(f'duplicate completed DeepSeek ID: {rid}')
        if rid in planned:
            if record.get('prompt_sha256') != planned[rid]['prompt_sha256']:
                raise ValueError(f'completed prompt mismatch: {rid}')
        elif not (rid.endswith('__deepseek__smoke') and record.get('kind') == 'smoke'):
            raise ValueError(f'unexpected completed DeepSeek ID: {rid}')
        actual_usd(record)  # Incomplete usage cannot silently lower the budget.
        completed[rid] = record
    attempts = [r['id'] for r in read_jsonl(run_dir / 'attempts.jsonl')]
    if len(attempts) != len(set(attempts)):
        raise ValueError('duplicate DeepSeek launch IDs')
    uncertain = set(attempts) - set(completed)
    if uncertain:
        raise ValueError(f'{len(uncertain)} launched DeepSeek request(s) have uncertain outcomes; reconcile before resuming')
    if smoke and smoke['id'] in completed and completed[smoke['id']]['prompt_sha256'] != smoke['prompt_sha256']:
        raise ValueError('completed smoke prompt mismatch')
    spent = sum(actual_usd(r) for r in completed.values())
    remaining_main = [r for r in rows if r['id'] not in completed]
    remaining_smoke = [smoke] if smoke and smoke['id'] not in completed else []
    reserve = sum(reserve_usd(r) for r in remaining_main + remaining_smoke)
    if spent + reserve > DEEPSEEK_BUDGET_USD:
        raise ValueError(f'DeepSeek USD 10 guard: completed conservative spend {spent:.4f} + remaining reserve {reserve:.4f} exceeds budget')
    return completed, spent, remaining_main, remaining_smoke


def execute_deepseek(rows, run_dir, max_concurrency, smoke=None):
    key = os.environ['DEEPSEEK_API_KEY']
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'batch.json').exists():
        raise ValueError('DeepSeek output directory contains an OpenAI Batch')
    lock = Lock()
    completed, spent, remaining_main, remaining_smoke = deepseek_progress(rows, run_dir, smoke)
    if smoke is None and not any(record.get('kind') == 'smoke' for record in completed.values()):
        raise ValueError('DeepSeek production requires a completed training smoke in the same output directory')
    pending = remaining_smoke if smoke else remaining_main
    print(f'DeepSeek completed: {len(completed)}; conservative charged spend: USD {spent:.4f}; remaining in this command: {len(pending)}')
    if not pending:
        return
    with (run_dir / 'responses.jsonl').open('a') as answers, (run_dir / 'attempts.jsonl').open('a') as attempts:
        def send(row):
            require_deepseek_offpeak(datetime.now(timezone.utc))
            with lock:
                attempts.write(json.dumps({'id': row['id']}) + '\n')
                attempts.flush()
                os.fsync(attempts.fileno())
            answer = request('https://api.deepseek.com/chat/completions', key,
                             json_body(payload('deepseek', row)), 'POST')
            choice = answer['choices'][0]
            final = choice['message'].get('content') or ''
            values, validity = parse_final(final)
            return {'id': row['id'], 'kind': row.get('kind', 'main'),
                    'prompt_sha256': row['prompt_sha256'],
                    'returned_model': answer.get('model'), 'usage': answer.get('usage'),
                    'reasoning_tokens': reasoning_tokens(answer.get('usage')),
                    'reasoning_exposure': REASONING_EXPOSURE['deepseek'],
                    'finish_reason': choice.get('finish_reason'), 'final_text': final,
                    'reasoning_content': choice['message'].get('reasoning_content'),
                    'validity': validity, 'prediction': values, 'raw_response': answer}

        for start in range(0, len(pending), max_concurrency):
            try:
                require_deepseek_offpeak(datetime.now(timezone.utc), announce=True)
            except ValueError as error:
                raise SystemExit(f'{error} Completed responses are durable; rerun the same command during the next off-peak window.') from None
            _, spent, remaining_main, remaining_smoke = deepseek_progress(rows, run_dir, smoke)
            wave = [r for r in pending[start:start + max_concurrency] if r['id'] not in completed]
            if not wave:
                continue
            errors = []
            with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
                futures = [pool.submit(send, row) for row in wave]
                for future in as_completed(futures):
                    try:
                        record = future.result()
                        answers.write(json.dumps(record, ensure_ascii=False) + '\n')
                        answers.flush()
                        os.fsync(answers.fileno())
                        completed[record['id']] = record
                    except Exception as error:
                        errors.append(error)
            if errors:
                raise RuntimeError(f'{len(errors)} DeepSeek request(s) failed or were blocked; completed responses were saved. Check uncertain attempts before resuming.') from errors[0]
    main_remaining = sum(row['id'] not in completed for row in rows)
    print(f'DeepSeek command complete: {len(completed)} recorded responses; {main_remaining} main requests remain.')


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


def execute_openai_smoke(row, run_dir):
    """One synchronous technical check; production remains Batch."""
    run_dir.mkdir(parents=True, exist_ok=False)
    answer = request('https://api.openai.com/v1/responses', os.environ['OPENAI_API_KEY'],
                     json_body(payload('openai', row)), 'POST')
    atomic_json(run_dir / 'smoke.json', openai_record(answer, row['id'], row['prompt_sha256']))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('check', 'cost', 'prepare', 'smoke', 'submit', 'status', 'collect'))
    ap.add_argument('--provider', choices=tuple(MODELS), required=True)
    ap.add_argument('--observations', type=Path)
    ap.add_argument('--smoke-observation', type=Path, help='one frozen training/dev/pool JSON file')
    ap.add_argument('--budget-usd', type=float)
    ap.add_argument('--max-concurrency', type=int, default=DEEPSEEK_DEFAULT_CONCURRENCY)
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
            submitted_rows = [json.loads(line) for line in (a.output / 'batch.jsonl').read_text().splitlines()]
            submitted = {item['custom_id']: digest(item['body']['input']) for item in submitted_rows}
            if len(submitted) != len(submitted_rows) or len(lines) != len(submitted) or {line['custom_id'] for line in lines} != set(submitted):
                raise ValueError('Batch result IDs incomplete or duplicated')
            parsed = []
            for line in lines:
                body = (line.get('response') or {}).get('body') or {}
                record = openai_record(body, line['custom_id'], submitted[line['custom_id']])
                record['error'] = line.get('error')
                record['raw_batch_response'] = line
                parsed.append(record)
            (a.output / 'responses.jsonl').write_text(''.join(json.dumps(x) + '\n' for x in parsed))
        return
    if not a.observations:
        ap.error('--observations is required')
    if not 1 <= a.max_concurrency <= 16:
        ap.error('--max-concurrency must be between 1 and 16')
    rows = manifest(observations(a.observations), a.provider)
    smoke = None
    if a.command == 'smoke':
        if not a.smoke_observation:
            ap.error('--smoke-observation is required for smoke')
        smoke = smoke_observation(a.smoke_observation, a.provider)
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
        if a.provider == 'deepseek':
            execute_deepseek(rows, a.output, 1 if smoke else a.max_concurrency, smoke)
        elif smoke:
            execute_openai_smoke(smoke, a.output)
        else:
            execute_openai(rows, a.output)


if __name__ == '__main__':
    main()
