#!/usr/bin/env python3
"""Freeze and verify v10 Qwen inputs before production."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import code_hashes, digest, read_json, read_jsonl, sha, write_json
from main_experiment.observation import messages, parse, serialize
from main_experiment.requests import validate_request

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--old', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    prepared = a.prepared
    report = read_json(prepared / 'report.json')
    assert report['offline_ready'] and report['design_version'] == 'panel888-access-v10-20260923'
    assert read_json(prepared / 'inputs.json')['code'] == code_hashes(), 'source changed after preparation'
    obs = {r['id']: r for f in (prepared / 'observations/sample').glob('*.json')
           for r in [read_json(f)]}
    assert len(obs) == 360
    for r in obs.values():
        assert serialize(parse(r['block'])) == r['block']
        assert messages(r['block']) == r['messages']
        assert digest(r['block']) == r['block_sha256']
    requests = read_jsonl(prepared / 'requests.jsonl')
    assert len(requests) == 4320
    for r in requests:
        validate_request(r)
        assert r['observation_id'] in obs
        assert r['prompt_sha256'] == obs[r['observation_id']]['prompt_sha256']
    old = {r['id']: r for r in read_jsonl(a.old / 'requests.jsonl')}
    identity = ('id', 'config_id', 'repeat_index', 'prompt_sha256', 'payload_sha256', 'seed')
    rhb = [r for r in requests if r['arm'] in ('R', 'H', 'B')]
    assert len(rhb) == 2592
    assert all(r['id'] in old and all(r[k] == old[r['id']][k] for k in identity) for r in rhb)
    new_qwen = [r for r in requests if r['arm'] in ('S', 'S_obs') and r['config_id'].startswith('qwen')]
    assert len(new_qwen) == 864
    out = {'design_version': report['design_version'], 'verified': True,
           'observations': len(obs), 'requests': len(requests),
           'requests_by_arm': dict(Counter(r['arm'] for r in requests)),
           'rhb_byte_identical_requests': len(rhb), 'new_qwen_requests': len(new_qwen),
           'prepared_report_sha256': sha(prepared / 'report.json'),
           'prepared_inputs_sha256': sha(prepared / 'inputs.json'),
           'requests_sha256': sha(prepared / 'requests.jsonl'),
           'observation_file_sha256': {p.name: sha(p) for p in sorted((prepared / 'observations/sample').glob('*.json'))}}
    write_json(a.output, out)
    print('V10_REQUESTS_SEALED', len(obs), len(requests), len(new_qwen))

if __name__ == '__main__': main()
