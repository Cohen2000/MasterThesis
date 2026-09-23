#!/usr/bin/env python3
"""Offline planning and explicit execution for the frozen v11 API comparison.

No command contacts a provider unless --execute is supplied. Production reads
the local copy of the 288 sealed observation JSON files.
"""
import argparse
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from threading import Lock
import time
import urllib.error
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
GENERATION_CAP = 128000                      # GPT-6 Sol maximum
GENERATION_CAPS = {'deepseek': 393216, 'openai': GENERATION_CAP}   # provider maxima
MARGIN = 1.25
# USD per million tokens: DeepSeek off-peak (cache miss) and GPT Batch. GPT input
# is charged at the cache-write price (1.25x input), an upper bound for input.
RATES = {'deepseek': (0.15, 0.60), 'openai': (1.25, 5.0)}
OPENAI_STANDARD_RATES = (2.5, 10.0)
# DeepSeek does not document whether start or completion time sets the price, so
# no wave starts within an hour of a peak window.
DEEPSEEK_PEAK_BUFFER_MINUTES = 60
# HTTP statuses returned before any generation; such requests are not billed.
UNBILLED_REJECTIONS = (400, 401, 402, 403, 404, 422, 429)
# Optional GPT variant with the hosted Python tool (separate run directory and IDs).
CODE_INTERPRETER = {'type': 'code_interpreter', 'container': {'type': 'auto'}}
TOOL_CALL_LIMIT = 10
# 1 GB container: USD 0.03 per 20-minute session; charged twice per container as a bound.
CONTAINER_USD = 0.06
OPENAI_POLL_SECONDS = 15
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
        body = {'model': MODELS[provider], 'input': row['messages'],
                'reasoning': {'effort': 'high', 'summary': 'auto'},
                'text': {'format': {'type': 'json_object'}},
                'max_output_tokens': GENERATION_CAPS['openai']}
        if row.get('tools'):
            body.update({'tools': [CODE_INTERPRETER], 'max_tool_calls': TOOL_CALL_LIMIT})
        return body
    return {'model': MODELS[provider], 'messages': row['messages'],
            'thinking': {'type': 'enabled'}, 'reasoning_effort': 'high',
            'response_format': {'type': 'json_object'}, 'max_tokens': GENERATION_CAPS['deepseek'],
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
    containers = len(record.get('containers') or []) if provider == 'openai' else 0
    return (input_count * rates[0] + output_count * rates[1]) / 1_000_000 + containers * CONTAINER_USD


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
                   + len(rows) * GENERATION_CAPS[provider] * rates[1]) / 1_000_000
    pilot_stats = token_stats([r for r in records if r.get('kind') == 'pilot'], provider)
    stats = token_stats(records, provider)
    result = {'generation_cap_per_request': GENERATION_CAPS[provider],
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
    """Hard upper bound: generation cannot exceed the cap and the input allowance
    exceeds provider input counts (checked on every DeepSeek answer)."""
    rates = OPENAI_STANDARD_RATES if standard else RATES[provider]
    total = 0.
    for row in rows:
        input_tokens = input_allowance(row)
        if row.get('tools'):
            # Every internal tool turn can re-read the prompt plus all generated output.
            input_tokens = (TOOL_CALL_LIMIT + 1) * (input_tokens + GENERATION_CAPS[provider])
            total += CONTAINER_USD * 1_000_000
        total += input_tokens * rates[0] + GENERATION_CAPS[provider] * rates[1]
    return total / 1_000_000


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
        raise ValueError(f'DeepSeek execution paused: within the {DEEPSEEK_PEAK_BUFFER_MINUTES}-minute pre-peak buffer. Resume in the next off-peak window.')


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


class ProviderRejected(RuntimeError):
    """HTTP rejection before generation; the request was not billed."""

    def __init__(self, status, detail):
        super().__init__(f'HTTP {status}: {detail}')
        self.status = status
        self.detail = detail


def request(url, key, body=None, method='GET', content_type='application/json', timeout=1800):
    headers = {'Authorization': 'Bearer ' + key}
    if body is not None:
        headers['Content-Type'] = content_type
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors='replace')[:4000]
        if error.code in UNBILLED_REJECTIONS:
            raise ProviderRejected(error.code, detail) from None
        raise RuntimeError(f'HTTP {error.code}: {detail}') from None


def append_durable(path, value):
    with path.open('a') as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + '\n')
        handle.flush()
        os.fsync(handle.fileno())


def rejection(rid, error):
    return {'id': rid, 'status': error.status, 'detail': error.detail,
            'received_utc': datetime.now(timezone.utc).isoformat()}


def live_launches(attempt_ids, rejected_ids):
    """Launches per ID that may have generated output (rejections excluded)."""
    rejected = Counter(rejected_ids)
    return {rid: count - rejected[rid] for rid, count in Counter(attempt_ids).items()}


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
    messages = [item for item in output if item.get('type') == 'message']
    # The final answer is the last assistant message; earlier ones precede tool calls.
    content = [part.get('text', '') for part in ((messages[-1].get('content') or []) if messages else [])
               if part.get('type') == 'output_text']
    summaries = [part for item in output if item.get('type') == 'reasoning'
                 for part in (item.get('summary') or [])]
    tool_calls = [item for item in output if item.get('type') == 'code_interpreter_call']
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
            'tool_calls': len(tool_calls),
            'containers': sorted({item['container_id'] for item in tool_calls if item.get('container_id')}),
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
    launches = live_launches([r['id'] for r in read_jsonl(run_dir / 'attempts.jsonl')],
                            [r['id'] for r in read_jsonl(run_dir / 'rejected.jsonl')])
    if any(count > 1 or count < 0 for count in launches.values()):
        raise ValueError('duplicate DeepSeek launch IDs')
    uncertain = {rid for rid, count in launches.items() if count == 1 and rid not in completed}
    if uncertain:
        raise ValueError(f'{len(uncertain)} launched DeepSeek request(s) have uncertain outcomes; reconcile before resuming')
    for row in extra:
        if row['id'] in completed and completed[row['id']]['prompt_sha256'] != row['prompt_sha256']:
            raise ValueError('completed technical prompt mismatch')
    spent = sum(actual_usd(r, 'deepseek') for r in completed.values())
    remaining_main = [r for r in rows if r['id'] not in completed]
    remaining_extra = [r for r in extra if r['id'] not in completed]
    return completed, spent, remaining_main, remaining_extra


def dispatch_order(rows):
    """Balanced order: if the budget ends early, missing requests fall in the last
    repeat/sample index across all graphs and arms, not in one stratum."""
    return sorted(rows, key=lambda r: (r.get('repeat_index') or 0, r.get('sample_index') or 0,
                                       r.get('stratum', ''), r.get('graph_id', ''), r.get('arm', '')))


def execute_deepseek(rows, run_dir, max_concurrency, smoke=None, pilot=None, budget=DEEPSEEK_BUDGET_USD):
    """Rolling pool. A request starts only if recorded spend plus the hard worst
    case of every open request, including the new one, stays within budget, so a
    request is never started that the balance might not cover."""
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
        projection = estimate(rows, 'deepseek', completed.values())
        print(f"DeepSeek projection: expected USD {projection['projected_total_usd']}, "
              f"conservative USD {projection['conservative_projected_total_usd']} (budget USD {budget})")
    pending = dispatch_order(remaining_extra if extra else remaining_main)
    print(f'DeepSeek completed: {len(completed)}; actual off-peak spend: USD {spent:.4f}; remaining in this command: {len(pending)}')
    if not pending:
        return
    require_deepseek_offpeak(datetime.now(timezone.utc), announce=True)
    with (run_dir / 'responses.jsonl').open('a') as answers, (run_dir / 'attempts.jsonl').open('a') as attempts:
        def send(row):
            with lock:
                attempts.write(json.dumps({'id': row['id']}) + '\n')
                attempts.flush()
                os.fsync(attempts.fileno())
            try:
                # Non-streaming responses carry keep-alive blank lines while the model works.
                answer = request('https://api.deepseek.com/chat/completions', key,
                                 json_body(payload('deepseek', row)), 'POST', timeout=3600)
            except ProviderRejected as error:
                with lock:
                    append_durable(run_dir / 'rejected.jsonl', rejection(row['id'], error))
                raise
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

        open_requests, errors, stop = {}, [], None
        with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
            while pending or open_requests:
                while pending and not stop and not errors and len(open_requests) < max_concurrency:
                    try:
                        require_deepseek_offpeak(datetime.now(timezone.utc))
                    except ValueError as error:
                        stop = f'{error} Rerun the same command then.'
                        break
                    reserve = in_flight_worst_usd(list(open_requests.values()) + [pending[0]], 'deepseek')
                    if spent + reserve > budget:
                        if not open_requests:
                            stop = (f'budget reached: spend USD {spent:.4f} plus worst case of one more request '
                                    f'exceeds USD {budget}')
                        break
                    row = pending.pop(0)
                    open_requests[pool.submit(send, row)] = row
                if not open_requests:
                    break
                done, _ = wait(open_requests, return_when=FIRST_COMPLETED)
                for future in done:
                    row = open_requests.pop(future)
                    try:
                        record = future.result()
                    except Exception as error:
                        errors.append(error)
                        print(f"DeepSeek request failed: {row['id']}: {error}", flush=True)
                        continue
                    answers.write(json.dumps(record, ensure_ascii=False) + '\n')
                    answers.flush()
                    os.fsync(answers.fileno())
                    completed[record['id']] = record
                    spent += actual_usd(record, 'deepseek')
                    prompt_tokens = usage_tokens(record, 'deepseek')[0]
                    if prompt_tokens > input_allowance(row):
                        errors.append(ValueError(f"{row['id']}: {prompt_tokens} prompt tokens exceed the input allowance"))
                    print(f"{datetime.now(timezone.utc):%H:%M:%S} done {len(completed)} | open {len(open_requests)} | "
                          f"pending {len(pending)} | out {usage_tokens(record, 'deepseek')[1]} | "
                          f"{record['validity']} | spend USD {spent:.4f}", flush=True)
    main_remaining = sum(row['id'] not in completed for row in rows)
    print(f'DeepSeek command complete: {len(completed)} recorded responses; {main_remaining} main requests remain; '
          f'actual off-peak spend USD {spent:.4f}.')
    if errors:
        raise RuntimeError(f'{len(errors)} DeepSeek request(s) failed; completed responses were saved. Check uncertain attempts before resuming.') from errors[0]
    if stop:
        print('stopped: ' + stop)


def shared_spend(directories):
    """Recorded spend of other GPT run directories drawing on the same budget;
    they must have no open Batch chunk or open technical response."""
    spent = 0.
    for directory in directories:
        numbers = chunk_numbers(directory)
        if numbers and not (directory / f'chunk_{numbers[-1]:03d}.collected.json').exists():
            raise ValueError(f'{directory} has an uncollected Batch chunk; collect it first')
        technical, main_records, resumable = openai_progress([], directory, technical_only=True, check_main=False)
        if resumable:
            raise ValueError(f'{directory} has open technical responses')
        spent += sum(actual_usd(r, 'openai') for r in technical + main_records)
    return spent


def openai_progress(rows, run_dir, technical_only=False, check_main=True):
    """Collected records; launched background responses without a record are
    returned as resumable {id: response_id}, never relaunched."""
    planned = {r['id']: r for r in rows}
    technical = read_jsonl(run_dir / 'technical_responses.jsonl')
    main_records = read_jsonl(run_dir / 'responses.jsonl')
    all_records = technical + main_records
    ids = [r['id'] for r in all_records]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate completed OpenAI IDs')
    for record in main_records if check_main else ():
        if record['id'] not in planned or record.get('prompt_sha256') != planned[record['id']]['prompt_sha256']:
            raise ValueError('unexpected OpenAI Batch ID or prompt hash')
    for record in all_records:
        actual_usd(record, 'openai')
    attempts = read_jsonl(run_dir / 'technical_attempts.jsonl')
    launches = live_launches([r['id'] for r in attempts if 'response_id' not in r],
                             [r['id'] for r in read_jsonl(run_dir / 'technical_rejected.jsonl')])
    response_ids = {r['id']: r['response_id'] for r in attempts if 'response_id' in r}
    if any(count > 1 or count < 0 for count in launches.values()):
        raise ValueError('duplicate OpenAI technical launch IDs')
    open_ids = {rid for rid, count in launches.items() if count == 1 and rid not in ids}
    if open_ids - set(response_ids):
        raise ValueError('OpenAI technical request has an uncertain outcome; reconcile before resuming')
    resumable = {rid: response_ids[rid] for rid in open_ids}
    if resumable and not technical_only:
        raise ValueError(f'{len(resumable)} OpenAI technical response(s) still open; rerun the technical command to collect them')
    return (technical, main_records, resumable) if technical_only else (technical, main_records)


def execute_openai_technical(rows, run_dir, budget=OPENAI_BUDGET_USD, shared=()):
    """DEV/POOL smoke or token pilot as background responses: all are launched
    (within the budget), then polled and saved one at a time. A launched response
    is resumed, never relaunched."""
    run_dir.mkdir(parents=True, exist_ok=True)
    technical, main_records, resumable = openai_progress([], run_dir, technical_only=True)
    if main_records or list(run_dir.glob('chunk_*.batch.json')):
        raise ValueError('technical pilot must precede Batch production')
    if rows[0]['kind'] == 'pilot' and not any(r.get('kind') == 'smoke' for r in technical):
        raise ValueError('run the one-request technical smoke before the token pilot')
    if set(resumable) - {r['id'] for r in rows}:
        raise ValueError('an open technical response belongs to another command; rerun that command first')
    key = os.environ['OPENAI_API_KEY']
    completed = {r['id'] for r in technical}
    by_id = {r['id']: r for r in rows}
    open_ids = dict(resumable)
    spent = sum(actual_usd(r, 'openai') for r in technical) + shared_spend(shared)
    for row in rows:
        if row['id'] in completed or row['id'] in open_ids:
            continue
        reserve = in_flight_worst_usd([by_id[rid] for rid in open_ids] + [row], 'openai', standard=True)
        if spent + reserve > budget:
            raise ValueError('GPT actual spend plus 128k technical request worst case exceeds user budget')
        append_durable(run_dir / 'technical_attempts.jsonl', {'id': row['id']})
        try:
            launched = request('https://api.openai.com/v1/responses', key,
                               json_body({**payload('openai', row), 'background': True}), 'POST')
        except ProviderRejected as error:
            append_durable(run_dir / 'technical_rejected.jsonl', rejection(row['id'], error))
            raise
        open_ids[row['id']] = launched['id']
        append_durable(run_dir / 'technical_attempts.jsonl', {'id': row['id'], 'response_id': launched['id']})
    print(f'GPT technical: {len(open_ids)} background response(s) open', flush=True)
    while open_ids:
        for rid, response_id in list(open_ids.items()):
            body = request('https://api.openai.com/v1/responses/' + response_id, key)
            if body.get('status') in ('queued', 'in_progress'):
                continue
            row = by_id[rid]
            record = openai_record(body, rid, row['prompt_sha256'], row['kind'], row['arm'])
            record['response_id'] = response_id
            try:
                actual_usd(record, 'openai')
            except ValueError:
                append_durable(run_dir / 'technical_failed.jsonl', {'id': rid, 'raw_response': body})
                raise ValueError(f"GPT technical response {response_id} ended with status {body.get('status')} and no usage; reconcile manually") from None
            append_durable(run_dir / 'technical_responses.jsonl', record)
            technical.append(record)
            del open_ids[rid]
            print(f"{datetime.now(timezone.utc):%H:%M:%S} {row['kind']} {rid}: status {body.get('status')}, "
                  f"output tokens {usage_tokens(record, 'openai')[1]}, {record['validity']}, "
                  f"technical spend USD {sum(actual_usd(r, 'openai') for r in technical):.4f}; open {len(open_ids)}", flush=True)
        if open_ids:
            time.sleep(OPENAI_POLL_SECONDS)


def chunk_numbers(run_dir):
    return sorted(int(p.name[6:9]) for p in run_dir.glob('chunk_[0-9][0-9][0-9].batch.json'))


def execute_openai(rows, run_dir, batch_size, budget=OPENAI_BUDGET_USD, shared=()):
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
    pending = dispatch_order([r for r in rows if r['id'] not in completed])
    if not pending:
        print(f'All {len(rows)} target GPT requests are collected.')
        return
    other = shared_spend(shared)
    projection = estimate(rows, 'openai', technical + main_records)
    if projection['conservative_projected_total_usd'] is None or projection['conservative_projected_total_usd'] + other > budget:
        raise ValueError(f"GPT conservative projected total USD {projection['conservative_projected_total_usd']} plus shared spend USD {other:.4f} exceeds user budget USD {budget}")
    chunk = pending[:batch_size]
    mapping = {r.get('provider_id', provider_id(r['id'])): r['id'] for r in chunk}
    if len(mapping) != len(chunk):
        raise ValueError('opaque provider ID collision')
    records = technical + main_records
    spent = sum(actual_usd(r, 'openai') for r in records) + other
    if chunk[0].get('tools'):
        # Tool turns have no hard token bound; reserve twice the costliest observed
        # tool request at Batch prices for every request in the chunk.
        observed = [actual_usd({**r, 'kind': 'main'}, 'openai') for r in records if r.get('kind') in ('pilot', 'main')]
        reserve = 2 * max(observed) * len(chunk)
    else:
        reserve = in_flight_worst_usd(chunk, 'openai')
    if spent + reserve > budget:
        raise ValueError(f'GPT spend USD {spent:.4f} plus reserve USD {reserve:.2f} for next Batch chunk ({len(chunk)} requests) exceeds user budget; use a smaller --batch-size')
    print(f'GPT spend so far USD {spent:.4f} (incl. shared); reserve for this chunk USD {reserve:.2f}')
    failed_before = {r['id'] for r in read_jsonl(run_dir / 'batch_failures.jsonl')} & set(mapping.values())
    print(f"GPT next chunk: {len(chunk)} requests ({len(failed_before)} resubmitted after unbilled Batch failure); "
          f"conservative complete-run projection USD {projection['conservative_projected_total_usd']}")
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
    records, failures = [], []
    for line in raw_lines:
        internal_id = mapping[line['custom_id']]
        body = (line.get('response') or {}).get('body') or {}
        repeat_index = int(internal_id.rsplit('__r', 1)[1])
        record = openai_record(body, internal_id, submitted[line['custom_id']],
                               repeat_index=repeat_index)
        record['provider_id'] = line['custom_id']
        record['error'] = line.get('error')
        record['raw_batch_response'] = line
        try:
            actual_usd(record, 'openai')
        except ValueError:
            # No usage: nothing was generated or billed; the ID stays pending.
            failures.append({'id': internal_id, 'provider_id': line['custom_id'],
                             'batch_id': state['id'], 'raw_batch_response': line})
            continue
        records.append(record)
    for failure in failures:
        append_durable(run_dir / 'batch_failures.jsonl', failure)
    for record in records:
        append_durable(run_dir / 'responses.jsonl', record)
    atomic_json(marker, {'batch_id': state['id'], 'requests': len(submitted),
                         'collected': len(records), 'failed_unbilled': len(failures)})
    spent = sum(actual_usd(r, 'openai') for r in read_jsonl(run_dir / 'technical_responses.jsonl')
                + read_jsonl(run_dir / 'responses.jsonl'))
    print(f'Collected {len(records)} responses; {len(failures)} failed without usage (stay pending); '
          f'conservative total GPT spend USD {spent:.4f}')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=('check', 'cost', 'smoke', 'pilot', 'submit', 'status', 'collect'))
    ap.add_argument('--provider', choices=tuple(MODELS), required=True)
    ap.add_argument('--observations', type=Path)
    ap.add_argument('--smoke-observation', type=Path, help='one frozen training/dev/pool JSON file')
    ap.add_argument('--pilot-observations', type=Path, help='8-12 frozen training/dev/pool JSON files')
    ap.add_argument('--budget-usd', type=float)
    ap.add_argument('--max-concurrency', type=int, default=DEEPSEEK_DEFAULT_CONCURRENCY)
    ap.add_argument('--batch-size', type=int, default=BATCH_SIZE_DEFAULT)
    ap.add_argument('--repeats', type=int, default=INITIAL_REPEATS,
                    help='target total model repeats, 1..3; completed IDs are skipped')
    ap.add_argument('--output', type=Path, help='run directory')
    ap.add_argument('--execute', action='store_true', help='explicitly allow provider network requests')
    ap.add_argument('--tools', action='store_true', help='GPT variant with the hosted Python tool (own run directory)')
    ap.add_argument('--shared-budget-dir', type=Path, action='append', default=[],
                    help='other GPT run directory whose spend counts against the same budget')
    a = ap.parse_args()
    if a.tools and a.provider != 'openai':
        ap.error('--tools is a GPT variant')
    label = 'openai_tools' if a.tools else a.provider
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
    if not 1 <= a.max_concurrency <= 64:
        ap.error('--max-concurrency must be between 1 and 64')
    if not 1 <= a.batch_size <= OBSERVATION_COUNT:
        ap.error(f'--batch-size must be between 1 and {OBSERVATION_COUNT}')
    rows = manifest(observations(a.observations), label, a.repeats)
    smoke = None
    if a.command == 'smoke':
        if not a.smoke_observation:
            ap.error('--smoke-observation is required for smoke')
        smoke = smoke_observation(a.smoke_observation, label)
    pilot = None
    if a.command == 'pilot':
        if not a.pilot_observations:
            ap.error('--pilot-observations is required for pilot')
        pilot = pilot_manifest(a.pilot_observations, label)
    for row in [*rows, *([smoke] if smoke else []), *(pilot or [])]:
        row['tools'] = a.tools
    if a.output and a.provider == 'openai' and a.command in ('smoke', 'pilot', 'submit') and a.execute:
        a.output.mkdir(parents=True, exist_ok=True)
        marker = a.output / 'variant.json'
        variant = {'label': label, 'tools': [CODE_INTERPRETER] if a.tools else [],
                   'max_tool_calls': TOOL_CALL_LIMIT if a.tools else None}
        if marker.exists() and json.loads(marker.read_text()) != variant:
            ap.error(f'{a.output} belongs to another GPT variant')
        if a.output.resolve() in {d.resolve() for d in a.shared_budget_dir}:
            ap.error('--shared-budget-dir must name other run directories')
        atomic_json(marker, variant)
    records = []
    if a.output and a.output.is_dir():
        if a.provider == 'deepseek':
            records = list(deepseek_progress(rows, a.output)[0].values())
        else:
            technical, completed, _ = openai_progress(rows, a.output, technical_only=True)
            records = technical + completed
    summary(rows, a.provider, a.budget_usd, records, a.repeats)
    if a.command == 'check':
        print('frozen observations and prompts: valid')
    elif a.command == 'cost':
        return
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
            execute_openai_technical([smoke] if smoke else pilot, a.output, a.budget_usd, a.shared_budget_dir)
        else:
            execute_openai(rows, a.output, a.batch_size, a.budget_usd, a.shared_budget_dir)


if __name__ == '__main__':
    main()
