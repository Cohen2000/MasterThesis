#!/usr/bin/env python3
"""Paired R/H panel-size release on the frozen stage-1 draws and ET cache."""
# Plain-words overview: in the node-panel arms R and H, the observation block originally
# did not say how many panel nodes were sampled. This script adds that single number
# (n_panel) to the otherwise identical R/H observations ("released panel"), then reruns
# only what depends on it: new Qwen requests, ExtraTrees with one extra feature, and the
# comparison with the old version. Plugin, median and MLE cannot change and are checked
# to stay identical. It also wrote the frozen observations that the paid API models saw
# (freeze-api) and the matching training draws for the token pilot (freeze-technical).
# Stages are run one at a time: python scripts/add_panel_size.py <stage>.
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from study.common import STAGE1_GRAPHS, TRAINING_SOURCES, digest, fold_for, read_json, seed, write_json
from study.observation import parse, serialize, messages, FEATURE_NAMES
from study.estimators import plugin
from study.mle import fit as mle_fit
from study.model_requests import planned, payload, protocol_version, validate_request
import extratrees as et

# Only the two node-panel arms are affected.
ARMS = ('R', 'H')
DEFAULT_INPUT = Path.home() / '.local/share/masterthesis/rh_panel_sensitivity/inputs'
DEFAULT_OLD = Path.home() / '.local/share/masterthesis/v10_observations'
DEFAULT_OUT = Path.home() / '.local/share/masterthesis/rh_panel_sensitivity/run'
DEFAULT_API = Path.home() / '.local/share/masterthesis/v11_api_observations'
DEFAULT_TECHNICAL = Path.home() / '.local/share/masterthesis/v11_token_pilot'


# New observation ID = old ID with '-panel-release' added, so old and new stay paired.
def released_id(old_id):
    stem, sample = old_id.rsplit('__', 1)
    if not sample.startswith('s'):
        raise ValueError('unexpected sampler ID')
    return stem + '-panel-release__' + sample


# Panel size (number of sampled nodes) per graph and arm, from the preparation inputs.
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


# Add 'n_panel=<size>' to one observation block. The asserts prove that nothing else in
# the block changed, and new hashes are computed for the changed block and prompt.
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


# Stage 'prepare': release all 144 R/H test observations (24 graphs of that panel: 8 real,
# 8 surrogates, 8 synthetic; x 2 arms x 3 draws) and write the 864 matching Qwen requests
# (2 modes x 3 repeats each). Each request keeps the random seed of its old twin, so only
# the prompt differs.
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
        if row['graph_id'] not in STAGE1_GRAPHS or row['block_sha256'] != digest(row['block']) or \
           row['prompt_sha256'] != digest(row['messages']) or row['messages'] != messages(row['block']):
            raise ValueError('old observation hash/content mismatch')
        fresh = release(row, budget[row['graph_id']][row['arm']])
        rows.append(fresh)
        write_json(out / 'observations/sample' / (fresh['id'] + '.json'), fresh)
    cells = {(r['graph_id'], r['arm'], r['sample_index']) for r in rows}
    expected = {(graph, arm, draw) for graph in STAGE1_GRAPHS for arm in ARMS for draw in (1, 2, 3)}
    if len(rows) != 144 or cells != expected:
        raise ValueError('paired 144-observation grid incomplete')
    requests = [r for r in planned(rows) if r['config_id'].startswith('qwen')]
    if len(requests) != 864 or len({r['id'] for r in requests}) != 864:
        raise ValueError('Qwen 864-request grid incomplete')
    if any(sum(r['config_id'] == mode for r in requests) != 432
           for mode in ('qwen_thinking', 'qwen_nonthinking')):
        raise ValueError('Qwen mode count')
    by_new_id = {r['id']: r for r in rows}
    for r in requests:
        # Keep the sealed Qwen generation stream paired. Only the prompt and
        # released block change; the variant observation ID must not reseed it.
        old_sampler = by_new_id[r['observation_id']]['paired_hidden_id'].rsplit('__', 2)[1]
        version = protocol_version(r['arm'])
        r['seed'] = seed('llm', r['graph_id'], old_sampler, r['sample_index'],
                         r['repeat_index'], r['config_id'] + ':' + version)
        r['payload'] = payload(r['config_id'], by_new_id[r['observation_id']]['messages'],
                               r['seed'], version)
        r['payload_sha256'] = digest(r['payload'])
        validate_request(r)
    (out / 'requests.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in requests))
    write_json(out / 'pairing.json', {'paired_observations': len(rows), 'qwen_requests': len(requests),
                                     'qwen_thinking': 432, 'qwen_nonthinking': 432,
                                     'old_observation_hashes': {r['paired_hidden_id']: next(
                                         x['block_sha256'] for x in old_rows if x['id'] == r['paired_hidden_id'])
                                         for r in rows}})
    print('PAIRED', len(rows), 'QWEN', len(requests))


# Stage 'freeze-api': the frozen observation set sent to the paid API models.
def freeze_api(old_dir, released_dir, destination):
    """Copy only the completed, released evidence to a local API freeze."""
    old = [read_json(p) for p in old_dir.glob('*.json')]
    released = [read_json(p) for p in released_dir.glob('*.json')]
    selected = [r for r in old if r['arm'] in ('S', 'B')] + released
    cells = {(r['graph_id'], r['arm'], r['sample_index']) for r in selected}
    expected = {(g, arm, i) for g in STAGE1_GRAPHS for arm in ('R', 'S', 'H', 'B') for i in (1, 2, 3)}
    if len(selected) != 288 or len(cells) != 288 or cells != expected:
        raise ValueError('API observation grid incomplete')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('API freeze already exists; do not overwrite frozen inputs')
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for row in selected:
        arm = row['arm']
        if row['block_sha256'] != digest(row['block']) or row['prompt_sha256'] != digest(row['messages']):
            raise ValueError('source observation hash mismatch')
        if row['messages'] != messages(row['block']):
            raise ValueError('source prompt mismatch')
        parsed = parse(row['block'])
        if (arm in ('R', 'H')) != ('n_panel' in parsed):
            raise ValueError('R/H panel release missing or unexpected')
        if arm == 'S' and 'traversals' not in row['block'] or arm == 'B' and '\np=' not in row['block']:
            raise ValueError('S/B design evidence missing')
        frozen = {key: row[key] for key in ('id', 'graph_id', 'stratum', 'arm', 'sample_index',
                                           'block', 'block_sha256', 'messages', 'prompt_sha256')}
        write_json(destination / (row['id'] + '.json'), frozen)
        hashes[row['id']] = {'block_sha256': row['block_sha256'],
                             'prompt_sha256': row['prompt_sha256']}
    print('API_FREEZE', len(hashes), 'R/S/H/B', *(sum(r['arm'] == a for r in selected)
                                                        for a in ('R', 'S', 'H', 'B')))
    return hashes


# Stage 'freeze-technical': the same release for the 12 training draws of the API token pilot.
def freeze_technical(source, input_dir, destination):
    """Apply the same R/H release to completed training pilot draws."""
    budget = budgets(input_dir)
    rows = [read_json(p) for p in source.glob('*.json')]
    if len(rows) != 12 or destination.exists() and any(destination.iterdir()):
        raise ValueError('expected 12 technical draws and an empty destination')
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row['domain'] not in ('training', 'pool_train', 'pool_dev'):
            raise ValueError('technical draw is not training/dev/pool')
        changed = release(row, budget[row['graph_id']][row['arm']]) if row['arm'] in ARMS else row
        frozen = {key: changed[key] for key in ('id', 'domain', 'empty', 'graph_id', 'arm',
                                               'block', 'block_sha256', 'messages', 'prompt_sha256')}
        write_json(destination / (changed['id'] + '.json'), frozen)
    print('TECHNICAL_FREEZE', len(rows))


# Stage 'et-cache': reuse the old ExtraTrees feature table and append log(1 + n_panel).
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


# Sealed predictions of the earlier version, used to prove what did not change.
def old_predictions():
    path = Path(__file__).resolve().parents[1] / 'results/stage1_scores/PREDICTIONS.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {(r['observation_id'], r['method'], r['repeat_index'] or None): r for r in rows}


# True profiles of the real training sources (needed for the median baseline).
def training_truths(input_dir):
    truth = {}
    for path in (input_dir / 'training_observations').glob('*.json'):
        row = read_json(path)
        source = row['source_family']
        if source in truth and truth[source] != row['truth']:
            raise ValueError('training truth mismatch')
        truth[source] = row['truth']
    if set(truth) != set(TRAINING_SOURCES):
        raise ValueError('incomplete training sources')
    return truth


# Stage 'offline': recompute plugin, median and MLE on the released blocks and stop if
# any of them differs from the sealed value (they do not use n_panel).
def offline(input_dir, out):
    old = old_predictions(); truths = training_truths(input_dir)
    rows = []
    for path in sorted((out / 'observations/sample').glob('*.json')):
        row = read_json(path); o = parse(row['block']); fold = fold_for(row['graph_id'])
        median = np.median([truths[source] for source in TRAINING_SOURCES if source != fold], axis=0).tolist()
        hidden_block = dict(o); hidden_block.pop('n_panel')
        mle_new = list(map(float, mle_fit(o).rho))
        mle_hidden = list(map(float, mle_fit(hidden_block).rho))
        if not np.allclose(mle_new, mle_hidden, atol=1e-12, rtol=0):
            raise ValueError(f'MLE changed from panel release: {row["id"]}')
        profiles = {'plugin': plugin(o), 'median': median, 'mle': mle_new}
        for method, profile in profiles.items():
            previous = json.loads(old[row['paired_hidden_id'], method, None]['prediction'])
            if method != 'mle' and not np.allclose(profile, previous, atol=1e-12, rtol=0):
                raise ValueError(f'{method} unexpectedly changed for {row["id"]}')
            # The same serialized counts give exactly the same local MLE fit;
            # retain the stored stage-1 numerical value for an exact zero contrast.
            if method == 'mle': profile = previous
            rows.append({'id': row['id'], 'old_id': row['paired_hidden_id'], 'graph_id': row['graph_id'],
                         'stratum': row['stratum'], 'arm': row['arm'], 'sample_index': row['sample_index'],
                         'method': method, 'prediction': profile, 'truth': row['truth']})
    if len(rows) != 432:
        raise ValueError('incomplete offline methods')
    write_json(out / 'offline.json', rows)
    print('OFFLINE', len(rows), 'plugin/median/MLE unchanged by construction and re-evaluation')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'et-cache', 'et-select', 'et-train',
                                      'offline', 'freeze-api', 'freeze-technical'))
    ap.add_argument('--old', type=Path, default=DEFAULT_OLD)
    ap.add_argument('--inputs', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--api-out', type=Path, default=DEFAULT_API)
    ap.add_argument('--technical-old', type=Path,
                    default=Path.home() / '.local/share/masterthesis/v10_token_pilot')
    ap.add_argument('--technical-out', type=Path, default=DEFAULT_TECHNICAL)
    ap.add_argument('--index', type=int)
    a = ap.parse_args()
    if a.stage == 'prepare': prepare(a.old, a.inputs, a.out)
    elif a.stage == 'et-cache': et_cache(a.inputs, a.out)
    elif a.stage == 'offline': offline(a.inputs, a.out)
    elif a.stage == 'freeze-api': freeze_api(a.old, a.out / 'observations/sample', a.api_out)
    elif a.stage == 'freeze-technical': freeze_technical(a.technical_old, a.inputs,
                                                         a.technical_out)
    else:
        if a.index is None or not 0 <= a.index < 2 * len(et.FOLDS):
            raise ValueError('ET index must be 0..17')
        et.OUT = a.out / 'et'
        et.ARMS = ARMS
        if a.stage == 'et-select': et.select_one(a.index)
        else: et.train_one(a.index)


if __name__ == '__main__':
    main()
