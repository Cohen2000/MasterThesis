#!/usr/bin/env python3
"""Verify a downloaded Qwen production archive before anything is collected from it.

Checks every archive checksum, that requests and observations are byte-identical
to the local prepared study, that exactly one answer exists per non-empty Qwen
request (no duplicates, no unknown IDs) with matching prompt/payload hashes and
seed, that every answer came from the committed runner and the pinned engine,
and that the model identity and source commit are the recorded ones.
"""
import argparse
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import PREPARED, QWEN_CONFIGS, RESULTS, ROOT, read_json, read_jsonl, sha, write_json
from main_experiment.requests import EXECUTION_POLICY, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', required=True)
    parser.add_argument('--commit', required=True, help='frozen source commit the cluster ran')
    args = parser.parse_args()
    folder = Path(args.archive)
    sums = read_json(folder/'CHECKSUMS.json')
    for name, h in sums.items():
        assert not Path(name).is_absolute() and '..' not in Path(name).parts
        assert sha(folder/name) == h, name
    assert sha(folder/'requests.jsonl') == sha(PREPARED/'requests.jsonl')
    local = sorted(p.name for p in (PREPARED/'observations/sample').glob('*.json'))
    assert sorted(p.name for p in (folder/'observations/sample').glob('*.json')) == local
    for name in local: assert sha(folder/'observations/sample'/name) == sha(PREPARED/'observations/sample'/name)
    requests = {r['id']: r for r in read_jsonl(PREPARED/'requests.jsonl')
                if r['config_id'] in QWEN_CONFIGS and r['status'] != 'skipped_empty'}
    answers = [read_json(p) for p in (folder/'answers').glob('*_r*/*.json')]
    ids = Counter(a['id'] for a in answers)
    assert max(ids.values()) == 1 and set(ids) == set(requests), 'missing, duplicate or unknown answers'
    runner = sha(ROOT/'scripts/run_qwen_engine.py')
    for a in answers:
        for key in ('prompt_sha256', 'payload_sha256', 'seed', 'config_id', 'repeat_index'):
            assert a[key] == requests[a['id']][key], (a['id'], key)
        assert a['runner_sha256'] == runner
        if a.get('status') == 'completed':
            assert a['vllm_version'] == EXECUTION_POLICY['qwen']['vllm_version']
            assert a['design_version'] == requests[a['id']]['design_version']
    identity = read_json(folder/'model_identity.json')
    assert identity['revision_pinned'] == REVISION
    assert (folder/'SPEC_COMMIT').read_text().strip() == args.commit
    assert read_json(folder/'ARCHIVE_REPORT.json')['readback_mismatches'] == []
    result = {'verified': True, 'checksums': len(sums), 'answers': len(answers),
              'end_states': dict(Counter(a.get('end_state', a.get('status')) for a in answers)),
              'requests_sha256': sha(folder/'requests.jsonl'), 'model_identity_sha256': sha(folder/'model_identity.json'),
              'model_revision': identity['revision_pinned'], 'source_commit': args.commit, 'runner_sha256': runner}
    write_json(RESULTS/'qwen_archive_verification.json', result)
    print(result)


if __name__ == '__main__':
    main()
