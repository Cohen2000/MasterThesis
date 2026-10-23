#!/usr/bin/env python3
"""Arm S without the walk log: GPT-6 Sol, high reasoning, three answers per sample.

Dry run by default; --execute sends requests, --retry resends unanswered requests,
and --evaluate scores rho_2 separately by stratum in results/ablation_s/summary.csv.
"""
import argparse
import csv
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import api  # noqa: E402
from study.answers import parse_final  # noqa: E402
from study.common import FINAL, OUT, ROOT, digest, read_json  # noqa: E402
from study.sample import HEAD, PROMPTS, WALK_HEAD, parse  # noqa: E402


def samples():
    """Keep the sampling rule and released L, removing only the log from the input."""
    if not (OUT/'samples').is_dir():
        subprocess.run([sys.executable, '-B', str(ROOT/'scripts/samples.py')], check=True)
    read = lambda name: (PROMPTS/name).read_text().rstrip('\n')
    rule = read('S.txt').split('The table also gives', 1)[0].rstrip()+' The walk length L is released.'
    selected = []
    for path in sorted((OUT/'samples').glob('*.json')):
        s = read_json(path)
        if s['arm'] != 'S': continue
        lines = s['block'].splitlines()
        head = lines.index(WALK_HEAD)
        rows = [line.split(',') for line in lines[head+1:]]
        block = '\n'.join(lines[:head]+[f'L={sum(int(r[4]) for r in rows)}', HEAD]
                          +[','.join(r[:3]) for r in rows])
        user = [read('task.txt'), 'Sampling rule: '+rule, block,
                'Return the four full-archive estimates in the specified JSON format.']
        messages = [{'role': 'system', 'content': read('system.txt')}, {'role': 'user', 'content': '\n'.join(user)}]
        assert all(word not in m['content'] for m in messages for word in ('inv_events', 'traversals', 'c_e'))
        selected.append({**s, 'original_block': s['block'], 'block': block, 'block_sha256': digest(block),
                         'messages': messages, 'prompt_sha256': digest(messages)})
    assert Counter(s['stratum'] for s in selected) == {'real': 36, 'surrogate': 36, 'synthetic': 24}
    assert len({s['id'] for s in selected}) == 96
    return selected


def evaluate(selected, responses, out):
    """Average errors within each network first, then equally over its stratum."""
    truth = read_json(FINAL/'TRUTH.json')
    by_id = {s['id']: s for s in selected}
    estimates, baseline = defaultdict(list), {}
    for s in selected:
        o = parse(s['original_block'])
        rows = o['table']
        recurrent = [r[0].count('1') >= 2 for r in rows]
        weights = [r[1]**2/r[2] if r[1] else 0. for r in rows]
        values = {'plugin': sum(r[1] for r, keep in zip(rows, recurrent) if keep)/o['D_obs'],
                  'proxy': sum(w for w, keep in zip(weights, recurrent) if keep)/sum(weights),
                  'hh': sum(r[5] for r, keep in zip(rows, recurrent) if keep)/sum(r[5] for r in rows)}
        baseline[s['id']] = values
        for method, value in values.items(): estimates[s['id'], method].append(value)
    with (FINAL/'PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            s = by_id.get(r['observation_id'])
            if s and r['method'] == 'gpt_6_sol' and r['arm'] == 'S' and r['group'] == s['stratum'] and r['valid'] == 'True':
                assert r['source'] == s['graph_id']
                estimates[s['id'], 'gpt_main'].append(json.loads(r['prediction'])[0])
    planned = {r['id']: r for r in api.planned(selected, 'openai_noweights')}
    for request_id, r in {r['id']: r for r in responses}.items():
        assert request_id in planned, f'unknown answer: {request_id}'
        assert r['prompt_sha256'] == planned[request_id]['prompt_sha256']
        values, _ = parse_final(r.get('final_text') or '')
        if values is not None and not r.get('limit_hit'):
            sample_id = request_id.rsplit('__openai_noweights__r', 1)[0]
            estimates[sample_id, 'gpt_noweights'].append(values[0])
    summary = []
    for stratum in ('real', 'surrogate', 'synthetic'):
        for method in ('plugin', 'proxy', 'hh', 'gpt_main', 'gpt_noweights'):
            networks, close_plugin, close_hh = defaultdict(list), [], []
            for s in selected:
                if s['stratum'] != stratum: continue
                for value in estimates[s['id'], method]:
                    networks[s['graph_id']].append(100*(value-truth[s['graph_id']][0]))
                    close_plugin.append(abs(value-baseline[s['id']]['plugin']) <= 0.005+1e-12)
                    close_hh.append(abs(value-baseline[s['id']]['hh']) <= 0.005+1e-12)
            summary.append({'stratum': stratum, 'method': method, 'networks': len(networks),
                            'n_valid': sum(map(len, networks.values())),
                            'mae_pp': mean(mean(abs(e) for e in v) for v in networks.values()) if networks else None,
                            'signed_error_pp': mean(mean(v) for v in networks.values()) if networks else None,
                            'within_plugin_0.5pp': mean(close_plugin) if close_plugin and method.startswith('gpt_') else None,
                            'within_hh_0.5pp': mean(close_hh) if close_hh and method.startswith('gpt_') else None})
    expected = {'plugin': {'real': 30.1, 'surrogate': 24.6, 'synthetic': 12.7},
                'gpt_main': {'real': 8.4, 'surrogate': 9.2, 'synthetic': 2.9}, 'hh': {'real': 8.8}}
    for r in summary:
        target = expected.get(r['method'], {}).get(r['stratum'])
        if target is not None:
            assert round(r['mae_pp'], 1) == target, f'sanity check failed: {r}'
    out.mkdir(parents=True, exist_ok=True)
    with (out/'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(summary)
    writer = csv.DictWriter(sys.stdout, fieldnames=list(summary[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows({k: f'{v:.4f}' if type(v) is float else v for k, v in r.items()} for r in summary)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--execute', action='store_true', help='send the requests')
    ap.add_argument('--retry', action='store_true', help='resend requests sent without an answer')
    ap.add_argument('--evaluate', action='store_true', help='score rho_2 and write summary.csv')
    a = ap.parse_args()
    selected, out = samples(), OUT/'ablation_s'
    lines = lambda name: [json.loads(x) for x in (out/name).read_text().splitlines()] if (out/name).exists() else []
    if a.evaluate:
        evaluate(selected, lines('responses.jsonl'), out)
        return
    todo = api.planned(selected, 'openai_noweights')
    assert len(todo) == 288
    done = {r['id'] for r in lines('responses.jsonl')}
    sent = {r['id'] for r in lines('attempts.jsonl')}-done
    todo = [r for r in todo if r['id'] not in done and (a.retry or r['id'] not in sent)]
    print(f'{len(done)} answers stored, {len(sent)} sent without an answer, {len(todo)} to ask')
    if a.execute and todo:
        if any(digest(r['messages']) != r['prompt_sha256'] for r in todo): raise SystemExit('a prompt does not match its sample')
        out.mkdir(parents=True, exist_ok=True)
        api.run_openai(todo, out, tools=False)


if __name__ == '__main__':
    main()
