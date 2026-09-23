#!/usr/bin/env python3
"""Paired R/H panel-size release on the frozen v10 draws and ET cache."""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import MAIN_KEYS, TRAIN, REAL_TEST, digest, read_json, write_json
from main_experiment.observation import parse, serialize, messages, features, FEATURE_NAMES
from main_experiment.requests import planned, validate_request
import build_v10_et as et

ARMS = ('R', 'H')
DEFAULT_INPUT = Path.home() / '.local/share/masterthesis/rh_panel_sensitivity/inputs'
DEFAULT_OLD = Path.home() / '.local/share/masterthesis/v10_observations'
DEFAULT_OUT = Path.home() / '.local/share/masterthesis/rh_panel_sensitivity/run'


def released_id(old_id):
    stem, sample = old_id.rsplit('__', 1)
    if not sample.startswith('s'):
        raise ValueError('unexpected sampler ID')
    return stem + '-panel-release__' + sample


def budgets(directory):
    result = {}
    with (directory / 'budget_summary.csv').open() as f:
        for row in csv.DictReader(f):
            result[row['graph_id']] = {arm: int(row['n_panel' if arm == 'R' else 'n_panel_history'])
                                       for arm in ARMS}
    for path in (directory / 'pool_observations').glob('*.json'):
        row = read_json(path)
        result[row['key']] = {arm: int(row['budget']['n_panel' if arm == 'R' else 'n_panel_history'])
                              for arm in ARMS}
    return result


def release(row, panel):
    if row['arm'] not in ARMS or type(panel) is not int or panel < 1:
        raise ValueError('bad panel release')
    old = parse(row['block'])
    assert 'n_panel' not in old
    new = {**old, 'n_panel': panel}
    block = serialize(new)
    assert parse(block) == new
    assert {k: v for k, v in parse(block).items() if k != 'n_panel'} == old
    assert block.replace(f'n_panel={panel}\n', '') == row['block']
    changed = {**row, 'id': released_id(row['id']), 'block': block,
               'block_sha256': digest(block), 'messages': messages(block),
               'n_panel': panel, 'paired_hidden_id': row['id']}
    changed['prompt_sha256'] = digest(changed['messages'])
    assert changed['messages'][0] == row['messages'][0]
    assert changed['prompt_sha256'] != row['prompt_sha256']
    return changed


def prepare(old_dir, input_dir, out):
    out.mkdir(parents=True, exist_ok=True)
    budget = budgets(input_dir)
    old_rows = [read_json(p) for p in old_dir.glob('*.json')]
    if len(old_rows) != 360 or len({r['id'] for r in old_rows}) != 360:
        raise ValueError('expected 360 unique frozen observations')
    rows = []
    for row in old_rows:
        if row['arm'] not in ARMS:
            continue
        if row['graph_id'] not in MAIN_KEYS or row['block_sha256'] != digest(row['block']) or \
           row['prompt_sha256'] != digest(row['messages']) or row['messages'] != messages(row['block']):
            raise ValueError('old observation hash/content mismatch')
        fresh = release(row, budget[row['graph_id']][row['arm']])
        rows.append(fresh)
        write_json(out / 'observations/sample' / (fresh['id'] + '.json'), fresh)
    cells = {(r['graph_id'], r['arm'], r['sample_index']) for r in rows}
    expected = {(graph, arm, draw) for graph in MAIN_KEYS for arm in ARMS for draw in (1, 2, 3)}
    if len(rows) != 144 or cells != expected:
        raise ValueError('paired 144-observation grid incomplete')
    requests = [r for r in planned(rows) if r['config_id'].startswith('qwen')]
    if len(requests) != 864 or len({r['id'] for r in requests}) != 864:
        raise ValueError('Qwen 864-request grid incomplete')
    if any(sum(r['config_id'] == mode for r in requests) != 432
           for mode in ('qwen_thinking', 'qwen_nonthinking')):
        raise ValueError('Qwen mode count')
    for r in requests:
        validate_request(r)
    (out / 'requests.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in requests))
    write_json(out / 'pairing.json', {'paired_observations': len(rows), 'qwen_requests': len(requests),
                                     'qwen_thinking': 432, 'qwen_nonthinking': 432,
                                     'old_observation_hashes': {r['paired_hidden_id']: next(
                                         x['block_sha256'] for x in old_rows if x['id'] == r['paired_hidden_id'])
                                         for r in rows}})
    print('PAIRED', len(rows), 'QWEN', len(requests))


def et_cache(input_dir, out):
    old_rows = read_json(input_dir / 'et_rows.json')
    old_x = np.load(input_dir / 'et_features.npy', mmap_mode='r')
    if len(old_rows) != len(old_x) or old_x.shape[1] != len(FEATURE_NAMES):
        raise ValueError('old ET cache shape mismatch')
    budget = budgets(input_dir)
    indices = [i for i, r in enumerate(old_rows) if r['arm'] in ARMS]
    rows = []
    panel_feature = []
    for i in indices:
        row = dict(old_rows[i]); n = budget[row['source']][row['arm']]
        panel_feature.append(math.log1p(n))
        if row['domain'] == 'main':
            row['id'] = released_id(row['id'])
        rows.append(row)
    et_dir = out / 'et'
    et_dir.mkdir(parents=True, exist_ok=True)
    np.save(et_dir / 'features.npy', np.column_stack((old_x[indices], panel_feature)))
    write_json(et_dir / 'rows.json', rows)
    print('ET_CACHE', len(rows), 'features', old_x.shape[1] + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'et-cache', 'et-select', 'et-train'))
    ap.add_argument('--old', type=Path, default=DEFAULT_OLD)
    ap.add_argument('--inputs', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--index', type=int)
    a = ap.parse_args()
    if a.stage == 'prepare': prepare(a.old, a.inputs, a.out)
    elif a.stage == 'et-cache': et_cache(a.inputs, a.out)
    else:
        if a.index is None or not 0 <= a.index < 2 * len(et.FOLDS):
            raise ValueError('ET index must be 0..17')
        et.OUT = a.out / 'et'
        et.ARMS = ARMS
        if a.stage == 'et-select': et.select_one(a.index)
        else: et.train_one(a.index)


if __name__ == '__main__':
    main()
