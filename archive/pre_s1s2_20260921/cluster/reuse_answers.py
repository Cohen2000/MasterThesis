#!/usr/bin/env python3
"""Seed a new production workspace with existing answers to byte-identical requests.

usage: python cluster/reuse_answers.py <old mainexp dir> <new mainexp dir>   (on uc3, before submission)

An answer file is copied only if its request line in the new run/requests.jsonl is
byte-identical to the line in the old one (same ID, prompt, payload, seed) and the
file itself carries the same ID, prompt hash, payload hash and seed. The runner then
admits only the remaining requests. answers/REUSED_ANSWERS.json lists every copied
file with its SHA256.
"""
import hashlib, json, shutil, sys
from pathlib import Path

old, new = Path(sys.argv[1]), Path(sys.argv[2])
old_lines = {json.loads(l)['id']: l for l in (old/'run/requests.jsonl').read_text().splitlines()}
new_lines = {json.loads(l)['id']: l for l in (new/'run/requests.jsonl').read_text().splitlines()}
assert not (new/'answers').exists(), 'new workspace already has answers'
copied = {}
for rid, line in new_lines.items():
    r = json.loads(line)
    if not r['config_id'].startswith('qwen') or r['status'] == 'skipped_empty': continue
    if old_lines.get(rid) != line: continue                          # new or changed request
    mode = 'thinking' if r['config_id'] == 'qwen_thinking' else 'nonthinking'
    source = old/'answers'/f'{mode}_r{r["repeat_index"]}'/f'{rid}.json'
    answer = json.loads(source.read_text())
    for key in ('id', 'prompt_sha256', 'payload_sha256', 'seed'):
        assert answer[key] == r[key], (rid, key)
    target = new/'answers'/source.parent.name/source.name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied[str(target.relative_to(new))] = hashlib.sha256(target.read_bytes()).hexdigest()
(new/'answers/REUSED_ANSWERS.json').write_text(json.dumps(
    {'source_workspace': str(old), 'rule': 'request line byte-identical; id/prompt/payload/seed checked',
     'files': copied}, indent=1, sort_keys=True)+'\n')
qwen = sum(json.loads(l)['config_id'].startswith('qwen') and json.loads(l)['status'] != 'skipped_empty'
           for l in new_lines.values())
print(f'reused {len(copied)} of {qwen} Qwen answers; {qwen-len(copied)} to generate')
