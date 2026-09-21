#!/usr/bin/env python3
"""Independent audit of the evaluated Qwen answers.

Re-parses every collected final text with a separately written strict parser
(not evaluation.parse_final) and requires the evaluation to agree on validity
and on the absolute errors of every valid answer. Also requires one response
per non-empty Qwen request with matching prompt/payload hashes, and no Sol or
DeepSeek response anywhere.
"""
import csv
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import AUDIT, PREPARED, QWEN, QWEN_CONFIGS, RESULTS, read_json, read_jsonl, sha, write_json

KEYS = [f'rho_{k}' for k in range(2, 6)]


def strict(raw):
    """Profile from an answer, or None. One whole-answer code fence may wrap the JSON."""
    text = raw.strip()
    if text.startswith('```'):
        match = re.fullmatch(r'```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*', text, re.S)
        if not match: return None
        text = match[1]

    def unique(items):
        keys = [k for k, _ in items]
        if len(keys) != len(set(keys)): raise ValueError('duplicate key')
        return dict(items)
    try:
        value = json.loads(text, object_pairs_hook=unique)
    except (ValueError, TypeError, OverflowError):
        return None
    if type(value) is not dict or sorted(value) != KEYS: return None
    profile = [value[k] for k in KEYS]
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in profile): return None
    return profile if all(a >= b for a, b in zip(profile, profile[1:])) else None


def main():
    requests = {r['id']: r for r in read_jsonl(PREPARED/'requests.jsonl')}
    expected = {i for i, r in requests.items() if r['config_id'] in QWEN_CONFIGS and r['status'] != 'skipped_empty'}
    responses = read_jsonl(QWEN/'responses.jsonl')
    ids = Counter(r['id'] for r in responses)
    assert set(ids) == expected and max(ids.values()) == 1
    for r in responses:
        assert r['prompt_sha256'] == requests[r['id']]['prompt_sha256'] and r['payload_sha256'] == requests[r['id']]['payload_sha256']
    evaluated = {r['id']: r for r in csv.DictReader(open(QWEN/'evaluation/answer_errors.csv'))}
    truth = {p.stem: read_json(p)['truth'] for p in (PREPARED/'observations/sample').glob('*.json')}
    counts = Counter()
    for r in responses:
        profile = None if r.get('technical_error') else strict(r.get('final_text', ''))
        row = evaluated[r['id']]
        assert (profile is not None) == (row['valid'] == 'True'), r['id']
        if profile is not None:
            error = np.abs(np.array(profile)-truth[requests[r['id']]['observation_id']])
            assert abs(float(row['AE2'])-error[0]) < 1e-14 and abs(float(row['ProfileAE'])-error.mean()) < 1e-14
        mode = requests[r['id']]['config_id']
        counts[mode, 'valid' if profile is not None else 'technical_error' if r.get('technical_error') else 'invalid'] += 1
        counts[mode, 'output_limit'] += bool(r.get('limit_hit'))
    assert not (RESULTS/'api/responses.jsonl').exists() or not read_jsonl(RESULTS/'api/responses.jsonl')
    assert read_json(RESULTS/'api/ledger.json')['requests'] == {}
    result = {'verified': True, 'responses': len(responses), 'expected': len(expected),
              'counts': {f'{m}:{k}': v for (m, k), v in sorted(counts.items())},
              'valid': sum(v for (m, k), v in counts.items() if k == 'valid'),
              'responses_sha256': sha(QWEN/'responses.jsonl'), 'sol_deepseek_started': 0}
    write_json(AUDIT/'qwen_audit.json', result)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
