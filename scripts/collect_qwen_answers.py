#!/usr/bin/env python3
"""Merge the per-request answer files of a Qwen run into one evaluation input.

The runner writes one file per request id so that a killed job loses at most the
requests in flight and completeness is a directory count rather than a claim.
This collects them, maps the runner's end states onto the fields the frozen
evaluator expects, and refuses to emit a file that is silently incomplete.
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import read_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='offline run directory with requests.jsonl')
    ap.add_argument('--answers', required=True, help='directory holding <mode>_r<n>/ result files')
    ap.add_argument('--out', required=True)
    ap.add_argument('--configs', default='qwen_thinking,qwen_nonthinking')
    ap.add_argument('--allow-incomplete', action='store_true')
    a = ap.parse_args()

    run = Path(a.run)
    configs = set(a.configs.split(','))
    planned = [json.loads(l) for l in (run / 'requests.jsonl').read_text().splitlines()]
    expected = {r['id'] for r in planned
                if r['config_id'] in configs and r['status'] != 'skipped_empty'}

    found = {}
    for f in sorted(Path(a.answers).rglob('*.json')):
        if f.name.endswith('.tmp.json'):
            continue
        d = json.loads(f.read_text())
        if d['id'] in found:
            raise ValueError(f'duplicate result for {d["id"]}')
        found[d['id']] = d

    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    if extra:
        raise ValueError(f'{len(extra)} results outside the planned set, first {extra[:3]}')

    states = Counter(d.get('end_state', d.get('status')) for d in found.values())
    report = {'expected': len(expected), 'found': len(found), 'missing': len(missing),
              'end_states': dict(states),
              'output_limit_hits': states.get('output_limit', 0),
              'unclosed_reasoning': sum(1 for d in found.values()
                                        if d.get('status') == 'completed'
                                        and not d.get('reasoning_closed', True)),
              'empty_final_text': sum(1 for d in found.values()
                                      if d.get('status') == 'completed'
                                      and not (d.get('final_text') or '').strip())}
    print(json.dumps(report, indent=1))
    if missing and not a.allow_incomplete:
        Path(a.out + '.missing.txt').write_text('\n'.join(missing) + '\n')
        sys.exit(f'{len(missing)} of {len(expected)} answers missing; '
                 f'an incomplete set is never a finished main run. '
                 f'ids written to {a.out}.missing.txt')

    lines = []
    for rid in sorted(found):
        d = found[rid]
        completed = d.get('status') == 'completed'
        rec = {'id': rid, 'started': True, 'mock': False,
               'terminal': bool(completed),
               'technical_error': not completed,
               'limit_hit': d.get('end_state') == 'output_limit',
               'finish_reason': d.get('finish_reason'),
               'end_state': d.get('end_state', d.get('status')),
               'input_tokens': d.get('input_tokens'),
               'output_tokens': d.get('output_tokens'),
               'max_tokens': d.get('max_tokens'),
               'model': d.get('model'), 'mode': d.get('mode'),
               'reasoning_closed': d.get('reasoning_closed'),
               'raw_text': d.get('raw_text', ''),
               'reasoning_text': d.get('reasoning_text', '')}
        if completed:
            # Only the final answer is ever parsed; the reasoning is carried
            # alongside for the record and never fed to the parser.
            rec['final_text'] = d.get('final_text', '')
        lines.append(json.dumps(rec, sort_keys=True))
    Path(a.out).write_text('\n'.join(lines) + '\n')
    print(f'wrote {len(lines)} records to {a.out}')


if __name__ == '__main__':
    main()
