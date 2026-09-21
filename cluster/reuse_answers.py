#!/usr/bin/env python3
"""Seed a new production workspace with existing answers to byte-identical requests.

usage: python cluster/reuse_answers.py <old mainexp dir> <new mainexp dir>   (on uc3, before submission)

An answer file is copied only if the request has the same ID, configuration, repeat,
prompt hash, payload hash and seed in the new and the old run/requests.jsonl, and
the file itself carries the same ID, prompt hash, payload hash and seed. The runner
then admits only the remaining requests. answers/REUSED_ANSWERS.json lists every copied
file with its SHA256.
"""
import hashlib, json, shutil, sys
from pathlib import Path

old, new = Path(sys.argv[1]), Path(sys.argv[2])
IDENTITY = ('id', 'config_id', 'repeat_index', 'prompt_sha256', 'payload_sha256', 'seed')
old_requests = {r['id']: r for r in map(json.loads, (old/'run/requests.jsonl').read_text().splitlines())}
new_requests = {r['id']: r for r in map(json.loads, (new/'run/requests.jsonl').read_text().splitlines())}
assert not (new/'answers').exists(), 'new workspace already has answers'
copied = {}
for rid, r in new_requests.items():
    if not r['config_id'].startswith('qwen') or r['status'] == 'skipped_empty': continue
    previous = old_requests.get(rid)
    if previous is None or any(previous[k] != r[k] for k in IDENTITY): continue   # new or changed request
    mode = 'thinking' if r['config_id'] == 'qwen_thinking' else 'nonthinking'
    source = old/'answers'/f'{mode}_r{r["repeat_index"]}'/f'{rid}.json'
    if not source.exists(): continue                                  # never completed there
    answer = json.loads(source.read_text())
    for key in ('id', 'prompt_sha256', 'payload_sha256', 'seed'):
        assert answer[key] == r[key], (rid, key)
    target = new/'answers'/source.parent.name/source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied[str(target.relative_to(new))] = hashlib.sha256(target.read_bytes()).hexdigest()
(new/'answers/REUSED_ANSWERS.json').write_text(json.dumps(
    {'source_workspace': str(old), 'rule': 'identical id, config, repeat, prompt, payload and seed',
     'files': copied}, indent=1, sort_keys=True)+'\n')
qwen = sum(r['config_id'].startswith('qwen') and r['status'] != 'skipped_empty' for r in new_requests.values())
print(f'reused {len(copied)} of {qwen} Qwen answers; {qwen-len(copied)} to generate')
