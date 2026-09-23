#!/usr/bin/env python3
"""Verify completed raw Qwen answers against the frozen v10 request manifest."""
import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import (DESIGN_VERSION, QWEN_CONFIGS, ROOT, read_json,
                                    read_jsonl, sha, write_json)
from main_experiment.requests import EXECUTION_POLICY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, required=True)
    ap.add_argument('--answers', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    requests = {r['id']: r for r in read_jsonl(a.prepared / 'requests.jsonl')
                if r['config_id'] in QWEN_CONFIGS and r['status'] != 'skipped_empty'}
    if len(requests) != 2160:
        raise ValueError(f'unexpected Qwen request count: {len(requests)}')
    files = sorted(a.answers.glob('*_r*/*.json'))
    answers = [read_json(p) for p in files]
    ids = [x['id'] for x in answers]
    if len(ids) != len(set(ids)) or set(ids) != set(requests):
        raise ValueError(f'missing/duplicate/unexpected answers: {len(ids)} of {len(requests)}')
    runner = sha(ROOT / 'scripts/run_qwen_engine.py')
    end_states = collections.Counter()
    for x in answers:
        r = requests[x['id']]
        for key in ('prompt_sha256', 'payload_sha256', 'seed', 'config_id', 'repeat_index',
                    'observation_id'):
            if x.get(key) != r.get(key):
                raise ValueError(f'{x["id"]}: {key} mismatch')
        if x.get('runner_sha256') != runner:
            raise ValueError(f'{x["id"]}: runner mismatch')
        if x.get('status') != 'completed' or x.get('technical_error'):
            raise ValueError(f'{x["id"]}: incomplete/technical failure')
        if x.get('design_version') != DESIGN_VERSION:
            raise ValueError(f'{x["id"]}: design version mismatch')
        if x.get('vllm_version') != EXECUTION_POLICY['qwen']['vllm_version']:
            raise ValueError(f'{x["id"]}: engine version mismatch')
        end_states[x.get('end_state', x['status'])] += 1
    result = {'verified': True, 'answers': len(answers), 'requests': len(requests),
              'runner_sha256': runner, 'end_states': dict(end_states),
              'request_manifest_sha256': sha(a.prepared / 'requests.jsonl')}
    write_json(a.output, result)
    print(result)


if __name__ == '__main__':
    main()
