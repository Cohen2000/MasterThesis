#!/usr/bin/env python3
"""Progress and output-format monitor for a Qwen answer directory (all arms).

usage: python status.py <mainexp dir> [answers subdir]
"""
import json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.evaluation import parse_final

base = Path(sys.argv[1] if len(sys.argv) > 1 else '.')
answers = sys.argv[2] if len(sys.argv) > 2 else 'answers'
planned = [json.loads(l) for l in (base / 'run' / 'requests.jsonl').read_text().splitlines()]
by_id = {r['id']: r for r in planned}
want = {r['id'] for r in planned if r['config_id'].startswith('qwen') and r['status'] != 'skipped_empty'}
rows = [json.loads(f.read_text()) for f in (base / answers).glob('*_r*/*.json')]
ids = [d['id'] for d in rows]
per = Counter(f"{d['mode']}_r{d['repeat_index']}" for d in rows if d.get('status') == 'completed')
arm = Counter(d['arm'] for d in rows)
end = Counter(d.get('end_state', d.get('status')) for d in rows)
valid = Counter()
for d in rows:
    if d.get('status') != 'completed': continue
    v, reason = parse_final(d.get('final_text', ''))
    valid[(d['mode'], 'valid' if v is not None else reason.split(':')[0])] += 1
toks = sorted(d.get('output_tokens', 0) for d in rows)
print(json.dumps({
    'planned_qwen': len(want), 'present': len(rows), 'missing': len(want - set(ids)),
    'unexpected': len(set(ids) - want), 'duplicates': len(ids) - len(set(ids)),
    'complete': not (want - set(ids)),
    'per_pass': dict(sorted(per.items())), 'per_arm': dict(sorted(arm.items())), 'end_states': dict(end),
    'unclosed_reasoning': sum(1 for d in rows if d.get('status') == 'completed' and not d.get('reasoning_closed', True)),
    'empty_final_text': sum(1 for d in rows if d.get('status') == 'completed' and not (d.get('final_text') or '').strip()),
    'strict_parser': {f'{m}:{k}': n for (m, k), n in sorted(valid.items())},
    'output_tokens': {'n': len(toks), 'max': toks[-1] if toks else None,
                      'median': toks[len(toks) // 2] if toks else None, 'sum': sum(toks)},
    'payload_hash_mismatch':sum(1 for d in rows if d['id'] in by_id and d.get('payload_sha256')!=by_id[d['id']]['payload_sha256']),
    'prompt_hash_mismatch': sum(1 for d in rows if d['id'] in by_id and d['prompt_sha256'] != by_id[d['id']]['prompt_sha256']),
    'engine_vs_own_input_tokens_mismatch': sum(1 for d in rows if d.get('engine_prompt_tokens') is not None
                                               and d['engine_prompt_tokens'] != d.get('input_tokens')),
}, indent=1))
