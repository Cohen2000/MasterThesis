#!/usr/bin/env python3
"""Offline planning and explicit execution for the frozen v11 API comparison.

No command contacts a provider unless --execute is supplied. Production reads
the local copy of the 288 sealed observation JSON files.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from threading import Lock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from main_experiment.common import MAIN_KEYS, digest  # noqa: E402
from main_experiment.evaluation import parse_final  # noqa: E402
from main_experiment.observation import messages, parse  # noqa: E402

API_MAIN_ARMS = ('R', 'S', 'H', 'B')
MODELS = {'deepseek': 'deepseek-flash', 'openai': 'gpt-6-sol'}
INITIAL_REPEATS = 1
MAX_REPEATS = 3
GENERATION_CAP = 128000
MARGIN = 1.25
# USD per million tokens: DeepSeek off-peak and GPT Batch.
RATES = {'deepseek': (0.15, 0.60), 'openai': (1.0, 5.0)}
OPENAI_STANDARD_RATES = (2.0, 10.0)
DEEPSEEK_PEAK_BUFFER_MINUTES = 10
DEEPSEEK_DEFAULT_CONCURRENCY = 8
BATCH_SIZE_DEFAULT = 96
DEEPSEEK_BUDGET_USD = 10
OPENAI_BUDGET_USD = 200
OBSERVATION_COUNT = 288
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
    if len(rows) != OBSERVATION_COUNT or len({r['id'] for r in rows}) != OBSERVATION_COUNT:
        raise ValueError('v11 API observation set must contain 288 distinct rows')
    for row in rows:
        if set(row) != {'id', 'graph_id', 'stratum', 'arm', 'sample_index',
                        'block', 'block_sha256', 'messages', 'prompt_sha256'}:
            raise ValueError('API observation contains hidden or unexpected metadata')
        if row['graph_id'] not in MAIN_KEYS or row['arm'] not in API_MAIN_ARMS:
            raise ValueError('unexpected graph or arm')
        if row['block_sha256'] != digest(row['block']):
            raise ValueError('observation block hash mismatch')
        if row['prompt_sha256'] != digest(row['messages']):
            raise ValueError('frozen prompt hash mismatch')
        if row['messages'] != messages(row['block']):
            raise ValueError('canonical prompt content mismatch')
        block = parse(row['block'])
        if (row['arm'] in ('R', 'H')) != ('n_panel' in block):
            raise ValueError('R/H released n_panel missing or unexpected')
        if row['arm'] == 'S' and 'traversals' not in row['block']:
            raise ValueError('S crawl information missing')
        if row['arm'] == 'B' and '\np=' not in row['block']:
            raise ValueError('B sampling probability missing')
    cells = {(r['graph_id'], r['arm'], r['sample_index']) for r in rows}
    expected = {(g, arm, i) for g in MAIN_KEYS for arm in API_MAIN_ARMS for i in (1, 2, 3)}
    if cells != expected:
        raise ValueError('incomplete main observation grid')
    freeze = json.loads((ROOT / 'docs/results/panel888_v11_main_20260923/API_FREEZE.json').read_text())
    frozen = [(r['id'], r['block_sha256'], r['prompt_sha256']) for r in sorted(rows, key=lambda r: r['id'])]
    fingerprint = hashlib.sha256(json.dumps(frozen, separators=(',', ':')).encode()).hexdigest()
    if fingerprint != freeze['observation_hash_manifest_sha256']:
        raise ValueError('local v11 API freeze differs from committed hashes')
    return rows


def technical_observation(path, provider, kind):
    row = json.loads(path.read_text())
    if row.get('domain') not in ('training', 'pool_train', 'pool_dev'):
        raise ValueError(f'{kind} requires a training/dev/pool observation')
    if row.get('graph_id') in MAIN_KEYS or row.get('empty'):
        raise ValueError(f'{kind} cannot use a main graph or empty observation')
    if row.get('arm') not in API_MAIN_ARMS:
        raise ValueError(f'{kind} arm must be R/S/H/B')
    if row.get('block_sha256') != digest(row['block']) or row.get('prompt_sha256') != digest(row['messages']):
        raise ValueError(f'{kind} block or prompt hash mismatch')
    if row['messages'] != messages(row['block']):
        raise ValueError(f'{kind} canonical prompt mismatch')
    return {'id': row['id'] + '__' + provider + '__' + kind,
            'prompt_sha256': row['prompt_sha256'], 'messages': row['messages'],
            'graph_id': row['graph_id'], 'arm': row['arm'], 'kind': kind}


def smoke_observation(path, provider):
    return technical_observation(path, provider, 'smoke')


def pilot_manifest(directory, provider):
    rows = [technical_observation(path, provider, 'pilot') for path in sorted(directory.glob('*.json'))]
    if not 8 <= len(rows) <= 12 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('pilot needs 8-12 distinct training/dev/pool observations')
    if any(sum(r['arm'] == arm for r in rows) < 2 for arm in API_MAIN_ARMS):
        raise ValueError('pilot needs at least two observations per R/S/H/B arm')
    if len({r['graph_id'] for r in rows}) < 2:
        raise ValueError('pilot needs at least two training/dev/pool graphs')
    return rows


def provider_id(internal_id):
    return 'req_' + hashlib.sha256(internal_id.encode()).hexdigest()[:32]


def manifest(rows, provider, repeats=INITIAL_REPEATS):
    if not 1 <= repeats <= MAX_REPEATS:
        raise ValueError('repeats must be 1..3')
    result = []
    for row in sorted(rows, key=lambda r: (r['stratum'], r['graph_id'], r['arm'], r['sample_index'])):
        if row['arm'] not in API_MAIN_ARMS:
            continue
        for repeat in range(1, repeats + 1):
            rid = f"{row['id']}__{provider}__r{repeat}"
            result.append({'id': rid, 'provider_id': provider_id(rid),
                           'observation_id': row['id'], 'graph_id': row['graph_id'],
                           'stratum': row['stratum'], 'arm': row['arm'],
                           'sample_index': row['sample_index'], 'repeat_index': repeat,
                           'prompt_sha256': row['prompt_sha256'], 'messages': row['messages']})
    expected = OBSERVATION_COUNT * repeats
    if len(result) != expected or len({r['id'] for r in result}) != expected or \
            len({r['provider_id'] for r in result}) != expected:
        raise ValueError('incomplete or duplicate request manifest')
    return result


def payload(provider, row):
    if provider == 'openai':
        return {'model': MODELS[provider], 'input': row['messages'],
                'reasoning': {'effort': 'high', 'summary': 'auto'},
                'text': {'format': {'type': 'json_object'}},
                'max_output_tokens': GENERATION_CAP}
    return {'model': MODELS[provider], 'messages': row['messages'],
            'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high',
            'response_format': {'type': 'json_object'}, 'max_tokens': GENERATION_CAP,
            'stream': False}


def batch_line(row):
    return {'custom_id': row.get('provider_id', provider_id(row['id'])), 'method': 'POST', 'url': '/v1/responses',
            'body': payload('openai', row)}


def input_allowance(row):
    # Conservative proxy checked against frozen local tokenizer counts.
    return math.ceil(sum(len(m['content']) for m in row['messages']) / 2) + 256


def usage_tokens(record, provider):
    usage = record.get('usage') or {}
    names = ('prompt_tokens', 'completion_tokens') if provider == 'deepseek' else ('input_tokens', 'output_tokens')
    counts = tuple(usage.get(name) for name in names)
    if not all(type(x) is int and x >= 0 for x in counts):
        raise ValueError(f"{record.get('id')}: completed {provider} usage is missing or invalid")
    return counts


def actual_usd(record, provider):
    input_count, output_count = usage_tokens(record, provider)
    rates = OPENAI_STANDARD_RATES if provider == 'openai' and record.get('kind') in ('smoke', 'pilot') else RATES[provider]
    return (input_count * rates[0] + output_count * rates[1]) / 1_000_000


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    left = math.floor(position)
    return values[left] + (values[math.ceil(position)] - values[left]) * (position - left)


def token_stats(records, provider):
    measured = [r for r in records if r.get('kind') in ('pilot', 'main')]
    output = [usage_tokens(r, provider)[1] for r in measured]
    if not output:
        return None
    reasoning = [r['reasoning_tokens'] for r in measured
                 if type(r.get('reasoning_tokens')) is int and r['reasoning_tokens'] >= 0]
    def describe(values):
        return {'n': len(values), 'mean': round(statistics.mean(values), 1),
                'median': round(statistics.median(values), 1),
                'p90': round(percentile(values, .90), 1),
                'p95': round(percentile(values, .95), 1), 'maximum': max(values)}
    stats = {'generated_tokens': describe(output),
             'reasoning_tokens': describe(reasoning) if reasoning else None}
    stats['conservative_tokens_per_request'] = math.ceil(max(max(output), 1.25 * percentile(output, .95)))
    return stats


def estimate(rows, provider, records=()):
    """Pilot-based run projection; generation cap appears only in worst-case figures."""
    records = list(records)
    completed = {r['id'] for r in records if r.get('kind') == 'main'}
    remaining = [r for r in rows if r['id'] not in completed]
    input_tokens = sum(map(input_allowance, remaining))
    actual = sum(actual_usd(r, provider) for r in records)
    rates = RATES[provider]
    theoretical = (sum(map(input_allowance, rows)) * rates[0]
                   + len(rows) * GENERATION_CAP * rates[1]) / 1_000_000
    pilot_stats = token_stats([r for r in records if r.get('kind') == 'pilot'], provider)
    stats = token_stats(records, provider)
    result = {'generation_cap_per_request': GENERATION_CAP,
              'remaining_requests': len(remaining), 'remaining_input_token_allowance': input_tokens,
              'completed_actual_spend_usd': round(actual, 4), 'safety_margin': MARGIN,
              'theoretical_full_main_worst_case_usd': round(theoretical, 4),
              'pilot_usage': pilot_stats, 'observed_usage': stats,
              'projected_total_usd': None,
              'conservative_projected_total_usd': None}
    if stats:
        mean = stats['generated_tokens']['mean']
        conservative = stats['conservative_tokens_per_request']
        remaining_input_cost = input_tokens * rates[0]
        result['projected_total_usd'] = round(actual + (remaining_input_cost + len(remaining) * mean * rates[1]) / 1_000_000, 4)
        result['conservative_projected_total_usd'] = round(
            actual + MARGIN * (remaining_input_cost + len(remaining) * conservative * rates[1]) / 1_000_000, 4)
    return result


def in_flight_worst_usd(rows, provider, standard=False):
    rates = OPENAI_STANDARD_RATES if standard else RATES[provider]
    return MARGIN * (sum(map(input_allowance, rows)) * rates[0]
                     + len(rows) * GENERATION_CAP * rates[1]) / 1_000_000


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


def guard(rows, provider, budget, execute, repeats=INITIAL_REPEATS):
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('duplicate request IDs')
    expected = OBSERVATION_COUNT * repeats
    if len(rows) != expected:
        raise ValueError(f'incomplete manifest: {len(rows)} of {expected}')
    if budget is None or budget <= 0:
        raise ValueError('explicit positive --budget-usd required')
    if provider == 'deepseek' and budget != DEEPSEEK_BUDGET_USD:
        raise ValueError('DeepSeek production budget is fixed at USD 10')
    if provider == 'openai' and budget > OPENAI_BUDGET_USD:
        raise ValueError('GPT experiment budget cannot exceed the approved USD 200 cap')
    if execute:
        load_keys()
        key = 'DEEPSEEK_API_KEY' if provider == 'deepseek' else 'OPENAI_API_KEY'
        if not os.environ.get(key):
            raise ValueError(f'{key} absent')
    return estimate(rows, provider)


def summary(rows, provider, budget=None, records=(), repeats=INITIAL_REPEATS):
    cost = estimate(rows, provider, records)
    report = {'provider': provider, 'model': MODELS[provider], 'reasoning': 'high',
              'reasoning_exposure': REASONING_EXPOSURE[provider],
              'requests': len(rows), 'arms': API_MAIN_ARMS,
              'graph_strata': {'real': 8, 'surrogate': 8, 'synthetic': 8},
              'sampler_draws': 3, 'model_repeats': repeats, **cost,
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


def openai_record(body, rid, prompt_sha256=None, kind='main', arm=None, repeat_index=None):
    """Keep the provider summary as returned; hidden reasoning is unavailable."""
    output = body.get('output') or []
    content = [part.get('text', '') for item in output for part in item.get('content', [])
               if part.get('type') == 'output_text']
    summaries = [part for item in output if item.get('type') == 'reasoning'
                 for part in (item.get('summary') or [])]
    final = ''.join(content)
    limit_hit = (body.get('status') == 'incomplete'
                 and (body.get('incomplete_details') or {}).get('reason') == 'max_output_tokens')
    values, validity = (None, 'max_output_tokens') if limit_hit else parse_final(final)
    return {'id': rid, 'kind': kind, 'arm': arm, 'prompt_sha256': prompt_sha256,
            'provider': 'openai', 'requested_model': MODELS['openai'],
            'received_utc': datetime.now(timezone.utc).isoformat(), 'repeat_index': repeat_index,
            'returned_model': body.get('model'), 'usage': body.get('usage'),
            'reasoning_tokens': reasoning_tokens(body.get('usage')),
            'reasoning_exposure': REASONING_EXPOSURE['openai'],
            'reasoning_summary': summaries, 'final_text': final, 'limit_hit': limit_hit,
            'validity': validity, 'prediction': values, 'raw_response': body}


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def deepseek_progress(rows, run_dir, extra=()):
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
        elif not (record.get('kind') in ('smoke', 'pilot') and
                  rid.endswith('__deepseek__' + record['kind'])):
            raise ValueError(f'unexpected completed DeepSeek ID: {rid}')
        actual_usd(record, 'deepseek')  # Missing usage cannot silently lower the budget.
        completed[rid] = record
    attempts = [r['id'] for r in read_jsonl(run_dir / 'attempts.jsonl')]
    if len(attempts) != len(set(attempts)):
        raise ValueError('duplicate DeepSeek launch IDs')
    uncertain = set(attempts) - set(completed)
    if uncertain:
        raise ValueError(f'{len(uncertain)} launched DeepSeek request(s) have uncertain outcomes; reconcile before resuming')
    for row in extra:
        if row['id'] in completed and completed[row['id']]['prompt_sha256'] != row['prompt_sha256']:
            raise ValueError('completed technical prompt mismatch')
    spent = sum(actual_usd(r, 'deepseek') for r in completed.values())
    remaining_main = [r for r in rows if r['id'] not in completed]
    remaining_extra = [r for r in extra if r['id'] not in completed]
    return completed, spent, remaining_main, remaining_extra


def execute_deepseek(rows, run_dir, max_concurrency, smoke=None, pilot=None):
    key = os.environ['DEEPSEEK_API_KEY']
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'batch.json').exists():
        raise ValueError('DeepSeek output directory contains an OpenAI Batch')
    lock = Lock()
    extra = [smoke] if smoke else pilot or []
    completed, spent, remaining_main, remaining_extra = deepseek_progress(rows, run_dir, extra)
    if smoke is None and not any(record.get('kind') == 'smoke' for record in completed.values()):
        raise ValueError('DeepSeek production requires a completed training smoke in the same output directory')
    if smoke is None and pilot is None:
        pilot_records = [r for r in completed.values() if r.get('kind') == 'pilot']
        if len(pilot_records) < 8 or {r.get('arm') for r in pilot_records} != set(API_MAIN_ARMS):
            raise ValueError('DeepSeek production requires a completed stratified token pilot')
    pending = remaining_extra if extra else remaining_main
    print(f'DeepSeek completed: {len(completed)}; actual off-peak spend: USD {spent:.4f}; remaining in this command: {len(pending)}')
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
            limit_hit = choice.get('finish_reason') == 'length'
            values, validity = (None, 'max_tokens') if limit_hit else parse_final(final)
            return {'id': row['id'], 'kind': row.get('kind', 'main'),
                    'arm': row.get('arm'), 'graph_id': row.get('graph_id'),
                    'prompt_sha256': row['prompt_sha256'],
                    'provider': 'deepseek', 'requested_model': MODELS['deepseek'],
                    'received_utc': datetime.now(timezone.utc).isoformat(),
                    'repeat_index': row.get('repeat_index'),
                    'returned_model': answer.get('model'), 'usage': answer.get('usage'),
                    'reasoning_tokens': reasoning_tokens(answer.get('usage')),
                    'reasoning_exposure': REASONING_EXPOSURE['deepseek'],
                    'finish_reason': choice.get('finish_reason'), 'limit_hit': limit_hit,
                    'final_text': final,
                    'reasoning_content': choice['message'].get('reasoning_content'),
                    'validity': validity, 'prediction': values, 'raw_response': answer}

        for start in range(0, len(pending), max_concurrency):
            try:
                require_deepseek_offpeak(datetime.now(timezone.utc), announce=True)
            except ValueError as error:
                raise SystemExit(f'{error} Completed responses are durable; rerun the same command during the next off-peak window.') from None
            _, spent, remaining_main, remaining_extra = deepseek_progress(rows, run_dir, extra)
            wave = [r for r in pending[start:start + max_concurrency] if r['id'] not in completed]
            if not wave:
                continue
            if smoke is None and pilot is None:
                projected = estimate(rows, 'deepseek', completed.values())['conservative_projected_total_usd']
                if projected is None or projected > DEEPSEEK_BUDGET_USD:
                    raise ValueError(f'DeepSeek projected total USD {projected} exceeds USD 10 budget')
            if spent + in_flight_worst_usd(wave, 'deepseek') > DEEPSEEK_BUDGET_USD:
                raise ValueError('DeepSeek actual spend plus 128k worst case for the next wave exceeds USD 10')
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


def openai_progress(rows, run_dir):
    planned = {r['id']: r for r in rows}
    technical = read_jsonl(run_dir / 'technical_responses.jsonl')
    main_records = read_jsonl(run_dir / 'responses.jsonl')
    all_records = technical + main_records
    ids = [r['id'] for r in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate completed OpenAI IDs')
    for record in main_records:
        if record['id'] not in planned or record.get('prompt_sha256') != planned[record['id']]['prompt_sha256']:
            raise ValueError('unexpected OpenAI Batch ID or prompt hash')
    for record in all_records:
        actual_usd(record, 'openai')
    attempts = [r['id'] for r in read_jsonl(run_dir / 'technical_attempts.jsonl')]
    if len(attempts) != len(set(attempts)) or set(attempts) - set(ids):
        raise ValueError('OpenAI technical request has an uncertain outcome; reconcile before resuming')
    return technical, main_records


def execute_openai_technical(rows, run_dir, budget=OPENAI_BUDGET_USD):
    """Synchronous DEV/POOL smoke or token pilot, saved one response at a time."""
    run_dir.mkdir(parents=True, exist_ok=True)
    technical, main_records = openai_progress([], run_dir)
    if main_records or list(run_dir.glob('chunk_*.batch.json')):
        raise ValueError('technical pilot must precede Batch production')
    if rows[0]['kind'] == 'pilot' and not any(r.get('kind') == 'smoke' for r in technical):
        raise ValueError('run the one-request technical smoke before the token pilot')
    completed = {r['id'] for r in technical}
    with (run_dir / 'technical_responses.jsonl').open('a') as answers, (run_dir / 'technical_attempts.jsonl').open('a') as attempts:
        for row in rows:
            if row['id'] in completed:
                continue
            spent = sum(actual_usd(r, 'openai') for r in technical)
            if spent + in_flight_worst_usd([row], 'openai', standard=True) > budget:
                raise ValueError('GPT actual spend plus 128k technical request worst case exceeds user budget')
            attempts.write(json.dumps({'id': row['id']}) + '\n')
            attempts.flush(); os.fsync(attempts.fileno())
            body = request('https://api.openai.com/v1/responses', os.environ['OPENAI_API_KEY'],
                           json_body(payload('openai', row)), 'POST')
            record = openai_record(body, row['id'], row['prompt_sha256'], row['kind'], row['arm'])
            answers.write(json.dumps(record, ensure_ascii=False) + '\n')
            answers.flush(); os.fsync(answers.fileno())
            technical.append(record)


def chunk_numbers(run_dir):
    return sorted(int(p.name[6:9]) for p in run_dir.glob('chunk_[0-9][0-9][0-9].batch.json'))


def execute_openai(rows, run_dir, batch_size, budget=OPENAI_BUDGET_USD):
    """Submit only the next Batch chunk after collecting every previous chunk."""
    run_dir.mkdir(parents=True, exist_ok=True)
    technical, main_records = openai_progress(rows, run_dir)
    pilots = [r for r in technical if r.get('kind') == 'pilot']
    if len(pilots) < 8 or {r.get('arm') for r in pilots} != set(API_MAIN_ARMS):
        raise ValueError('GPT Batch requires a completed stratified token pilot')
    numbers = chunk_numbers(run_dir)
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError('Batch chunk numbering is incomplete')
    if numbers and not (run_dir / f'chunk_{numbers[-1]:03d}.collected.json').exists():
        raise ValueError('collect the previous Batch chunk before submitting another')
    next_number = len(numbers) + 1
    prefix = run_dir / f'chunk_{next_number:03d}'
    if any(prefix.with_suffix(suffix).exists() for suffix in ('.input.jsonl', '.upload.json', '.batch.json')):
        raise ValueError('unreconciled Batch upload or creation attempt')
    completed = {r['id'] for r in main_records}
    pending = [r for r in rows if r['id'] not in completed]
    if not pending:
        print(f'All {len(rows)} target GPT requests are collected.')
        return
    projection = estimate(rows, 'openai', technical + main_records)
    if projection['conservative_projected_total_usd'] is None or projection['conservative_projected_total_usd'] > budget:
        raise ValueError(f"GPT conservative projected total USD {projection['conservative_projected_total_usd']} exceeds user budget USD {budget}")
    chunk = pending[:batch_size]
    mapping = {r.get('provider_id', provider_id(r['id'])): r['id'] for r in chunk}
    if len(mapping) != len(chunk):
        raise ValueError('opaque provider ID collision')
    spent = sum(actual_usd(r, 'openai') for r in technical + main_records)
    if spent + in_flight_worst_usd(chunk, 'openai') > budget:
        raise ValueError('GPT actual spend plus 128k worst case for next Batch chunk exceeds user budget')
    print(f"GPT next chunk: {len(chunk)} requests; conservative complete-run projection USD {projection['conservative_projected_total_usd']}")
    batch = ''.join(json.dumps(batch_line(r), ensure_ascii=False, separators=(',', ':')) + '\n' for r in chunk).encode()
    boundary = 'masterthesis' + hashlib.sha256(batch).hexdigest()[:24]
    body = (f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
            f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="batch.jsonl"\r\n'
            f'Content-Type: application/jsonl\r\n\r\n').encode() + batch + f'\r\n--{boundary}--\r\n'.encode()
    prefix.with_suffix('.input.jsonl').write_bytes(batch)
    atomic_json(prefix.with_suffix('.mapping.json'), mapping)
    key = os.environ['OPENAI_API_KEY']
    uploaded = request('https://api.openai.com/v1/files', key, body, 'POST',
                       f'multipart/form-data; boundary={boundary}')
    atomic_json(prefix.with_suffix('.upload.json'), uploaded)
    created = request('https://api.openai.com/v1/batches', key,
                      json_body({'input_file_id': uploaded['id'], 'endpoint': '/v1/responses',
                                 'completion_window': '24h'}), 'POST')
    atomic_json(prefix.with_suffix('.batch.json'), created)


def latest_chunk(run_dir):
    numbers = chunk_numbers(run_dir)
    if not numbers:
        raise ValueError('no submitted GPT Batch chunk')
    number = numbers[-1]
    prefix = run_dir / f'chunk_{number:03d}'
    return prefix, json.loads(prefix.with_suffix('.batch.json').read_text())


def download_openai_file(file_id, key):
    req = urllib.request.Request('https://api.openai.com/v1/files/' + file_id + '/content',
                                 headers={'Authorization': 'Bearer ' + key})
    with urllib.request.urlopen(req, timeout=120) as response:
        return response.read()


def collect_openai(run_dir, state, key):
    prefix, _ = latest_chunk(run_dir)
    marker = prefix.with_suffix('.collected.json')
    if marker.exists():
        print('Latest Batch chunk is already collected.')
        return
    if state.get('status') not in ('completed', 'expired', 'cancelled', 'failed'):
        raise ValueError('Batch chunk is not terminal yet')
    raw_lines = []
    for field, suffix in (('output_file_id', '.output.jsonl'), ('error_file_id', '.errors.jsonl')):
        file_id = state.get(field)
        if file_id:
            path = prefix.with_suffix(suffix)
            if not path.exists():
                path.write_bytes(download_openai_file(file_id, key))
            raw_lines.extend(json.loads(line) for line in path.read_text().splitlines())
    submitted_rows = read_jsonl(prefix.with_suffix('.input.jsonl'))
    mapping = json.loads(prefix.with_suffix('.mapping.json').read_text())
    submitted = {item['custom_id']: digest(item['body']['input']) for item in submitted_rows}
    ids = [line['custom_id'] for line in raw_lines]
    if len(submitted) != len(submitted_rows) or set(mapping) != set(submitted) or \
            len(set(mapping.values())) != len(mapping) or len(ids) != len(set(ids)) or set(ids) != set(submitted):
        raise ValueError('Batch chunk outputs/errors incomplete or duplicate; reconcile before next submission')
    prior = {r['id'] for r in read_jsonl(run_dir / 'responses.jsonl')}
    if {mapping[line['custom_id']] for line in raw_lines} & prior:
        raise ValueError('duplicate previously collected OpenAI request ID')
    with (run_dir / 'responses.jsonl').open('a') as out:
        for line in raw_lines:
            internal_id = mapping[line['custom_id']]
            body = (line.get('response') or {}).get('body') or {}
            repeat_index = int(internal_id.rsplit('__r', 1)[1])
            record = openai_record(body, internal_id, submitted[line['custom_id']],
                                   repeat_index=repeat_index)
            record['provider_id'] = line['custom_id']
            record['error'] = line.get('error')
            record['raw_batch_response'] = line
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush(); os.fsync(out.fileno())
    atomic_json(marker, {'batch_id': state['id'], 'requests': len(submitted)})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('check', 'cost', 'prepare', 'smoke', 'pilot', 'submit', 'status', 'collect'))
    ap.add_argument('--provider', choices=tuple(MODELS), required=True)
    ap.add_argument('--observations', type=Path)
    ap.add_argument('--smoke-observation', type=Path, help='one frozen training/dev/pool JSON file')
    ap.add_argument('--pilot-observations', type=Path, help='8-12 frozen training/dev/pool JSON files')
    ap.add_argument('--budget-usd', type=float)
    ap.add_argument('--max-concurrency', type=int, default=DEEPSEEK_DEFAULT_CONCURRENCY)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument('--repeats', type=int, default=INITIAL_REPEATS,
                    help='target total model repeats, 1..3; completed IDs are skipped')
    ap.add_argument('--output', type=Path, help='prepared JSONL path, or run directory')
    ap.add_argument('--execute', action='store_true', help='explicitly allow provider network requests')
    a = ap.parse_args()
    if a.command in ('status', 'collect'):
        if not a.output or a.provider != 'openai':
            ap.error('status/collect require an existing OpenAI Batch run directory')
        if not a.execute:
            print('dry run: no provider request; add --execute to contact OpenAI')
            return
        load_keys()
        key = os.environ.get('OPENAI_API_KEY')
        if not key:
            ap.error('OPENAI_API_KEY absent')
        _, batch = latest_chunk(a.output)
        state = request('https://api.openai.com/v1/batches/' + batch['id'], key)
        print(json.dumps({'id': state['id'], 'status': state['status'], 'request_counts': state.get('request_counts')}, indent=2))
        if a.command == 'collect':
            collect_openai(a.output, state, key)
        return
    if not a.observations:
        ap.error('--observations is required')
    if not 1 <= a.max_concurrency <= 16:
        ap.error('--max-concurrency must be between 1 and 16')
    if not 1 <= a.batch_size <= 96:
        ap.error('--batch-size must be between 1 and 96')
    rows = manifest(observations(a.observations), a.provider, a.repeats)
    smoke = None
    if a.command == 'smoke':
        if not a.smoke_observation:
            ap.error('--smoke-observation is required for smoke')
        smoke = smoke_observation(a.smoke_observation, a.provider)
    pilot = None
    if a.command == 'pilot':
        if not a.pilot_observations:
            ap.error('--pilot-observations is required for pilot')
        pilot = pilot_manifest(a.pilot_observations, a.provider)
    records = []
    if a.output and a.output.is_dir():
        if a.provider == 'deepseek':
            records = list(deepseek_progress(rows, a.output)[0].values())
        else:
            technical, completed = openai_progress(rows, a.output)
            records = technical + completed
    summary(rows, a.provider, a.budget_usd, records, a.repeats)
    if a.command == 'check':
        print('frozen observations and prompts: valid')
    elif a.command == 'cost':
        return
    elif a.command == 'prepare':
        if not a.output:
            ap.error('--output is required')
        selected = rows[:a.batch_size] if a.provider == 'openai' else rows
        lines = (batch_line(r) if a.provider == 'openai' else
                 {'id': r['id'], 'prompt_sha256': r['prompt_sha256'], 'payload': payload(a.provider, r)}
                 for r in selected)
        a.output.write_text(''.join(json.dumps(line, ensure_ascii=False) + '\n' for line in lines))
        if a.provider == 'openai':
            atomic_json(a.output.with_suffix(a.output.suffix + '.mapping.json'),
                        {r['provider_id']: r['id'] for r in selected})
    else:
        if not a.output:
            ap.error('--output run directory is required')
        guard(rows, a.provider, a.budget_usd, a.execute, a.repeats)
        if not a.execute:
            print('dry run: manifest and budget cap valid; production needs completed token pilot; no provider request')
            return
        if a.provider == 'deepseek':
            execute_deepseek(rows, a.output, 1 if smoke else a.max_concurrency, smoke, pilot)
        elif smoke or pilot:
            execute_openai_technical([smoke] if smoke else pilot, a.output, a.budget_usd)
        else:
            execute_openai(rows, a.output, a.batch_size, a.budget_usd)


if __name__ == '__main__':
    main()
