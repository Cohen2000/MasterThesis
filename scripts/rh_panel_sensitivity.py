#!/usr/bin/env python3
"""Paired R/H panel-size release on the frozen v10 draws and ET cache."""
import argparse
import csv
import json
import math
from collections import defaultdict
from statistics import fmean
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import MAIN_KEYS, TRAIN, REAL_TEST, digest, fold_for, read_json, seed, write_csv, write_json
from main_experiment.observation import parse, serialize, messages, features, FEATURE_NAMES
from main_experiment.baselines import plugin
from main_experiment.shared_mle import fit as mle_fit
from main_experiment.evaluation import parse_final
from main_experiment.requests import planned, payload, protocol_version, validate_request
import build_v10_et as et
from build_v10_results import et_predictions

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


def old_predictions():
    path = Path(__file__).resolve().parents[1] / 'docs/results/panel888_v10_main_20260923/PREDICTIONS.csv'
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return {(r['observation_id'], r['method'], r['repeat_index'] or None): r for r in rows}


def training_truths(input_dir):
    truth = {}
    for path in (input_dir / 'training_observations').glob('*.json'):
        row = read_json(path)
        source = row['source_family']
        if source in truth and truth[source] != row['truth']:
            raise ValueError('training truth mismatch')
        truth[source] = row['truth']
    if set(truth) != set(TRAIN):
        raise ValueError('incomplete training sources')
    return truth


def offline(input_dir, out):
    old = old_predictions(); truths = training_truths(input_dir)
    rows = []
    for path in sorted((out / 'observations/sample').glob('*.json')):
        row = read_json(path); o = parse(row['block']); fold = fold_for(row['graph_id'])
        median = np.median([truths[source] for source in TRAIN if source != fold], axis=0).tolist()
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
            # retain the sealed v10 numerical value for an exact zero contrast.
            if method == 'mle': profile = previous
            rows.append({'id': row['id'], 'old_id': row['paired_hidden_id'], 'graph_id': row['graph_id'],
                         'stratum': row['stratum'], 'arm': row['arm'], 'sample_index': row['sample_index'],
                         'method': method, 'prediction': profile, 'truth': row['truth']})
    if len(rows) != 432:
        raise ValueError('incomplete offline methods')
    write_json(out / 'offline.json', rows)
    print('OFFLINE', len(rows), 'plugin/median/MLE unchanged by construction and re-evaluation')


def side_channel(input_dir, out):
    truth = training_truths(input_dir); budget = budgets(input_dir)
    rows = []
    for arm in ARMS:
        for source in REAL_TEST:
            train = [s for s in TRAIN if s != source]
            x = np.array([math.log1p(budget[s][arm]) for s in train])
            y = np.array([truth[s][0] for s in train])
            coefficient = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
            actual = truth[source][0]
            rows.append({'arm': arm, 'source': source, 'n_panel': budget[source][arm],
                         'rho_2': actual, 'median_prediction': float(np.median(y)),
                         'n_panel_prediction': float(coefficient @ [1, math.log1p(budget[source][arm])]),
                         'median_AE2': abs(float(np.median(y)) - actual),
                         'n_panel_AE2': abs(float(coefficient @ [1, math.log1p(budget[source][arm])]) - actual)})
    write_csv(out / 'SIDE_CHANNEL.csv', rows)
    print('SIDE_CHANNEL', {arm: {'median_MAE2': fmean(r['median_AE2'] for r in rows if r['arm'] == arm),
                                 'n_panel_MAE2': fmean(r['n_panel_AE2'] for r in rows if r['arm'] == arm)}
                           for arm in ARMS})


def comparison(out):
    old = old_predictions()
    new = {(r['id'], r['method'], None): r['prediction'] for r in read_json(out / 'offline.json')}
    et_results = et_predictions(out / 'et')
    if len(et_results) != 144:
        raise ValueError('R/H ET predictions incomplete')
    for oid, prediction in et_results.items():
        new[oid, 'et', None] = prediction
    requests = {r['id']: r for r in (json.loads(line) for line in (out / 'requests.jsonl').read_text().splitlines())}
    answer_files = list((out / 'answers').glob('*_r*/*.json'))
    if len(answer_files) != 864:
        raise ValueError(f'Qwen answers incomplete: {len(answer_files)}/864')
    seen = set(); invalid = defaultdict(int); completed = defaultdict(int)
    for path in answer_files:
        answer = read_json(path); rid = answer['id']
        if rid not in requests or rid in seen:
            raise ValueError('unknown/duplicate Qwen answer')
        seen.add(rid); request = requests[rid]
        if any(answer.get(key) != request[key] for key in ('seed', 'prompt_sha256', 'payload_sha256')):
            raise ValueError('Qwen answer identity mismatch')
        if answer.get('status') == 'completed':
            if any(key not in answer for key in ('raw_text', 'reasoning_text', 'final_text')):
                raise ValueError('Qwen raw/reasoning/final output missing')
            completed[request['config_id']] += 1
        prediction, reason = parse_final(answer.get('final_text', ''))
        if answer.get('status') != 'completed' or answer.get('technical_error'):
            prediction = None; reason = answer.get('end_state', 'technical_error')
        if prediction is None: invalid[request['config_id']] += 1
        new[request['observation_id'], request['config_id'], str(request['repeat_index'])] = prediction
    if seen != set(requests):
        raise ValueError('missing Qwen IDs')
    graph_rows = []; summary_rows = []
    observations = [read_json(path) for path in sorted((out / 'observations/sample').glob('*.json'))]
    for row in observations:
        for method in ('plugin', 'median', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking'):
            repeats = (None,) if not method.startswith('qwen') else ('1', '2', '3')
            for repeat in repeats:
                old_row = old[row['paired_hidden_id'], method, repeat]
                old_prediction = json.loads(old_row['prediction']) if old_row['prediction'] else None
                new_prediction = new[row['id'], method, repeat]
                for access, prediction in (('hidden', old_prediction), ('released', new_prediction)):
                    error = np.abs(np.array(prediction) - np.array(row['truth'])) if prediction is not None else None
                    graph_rows.append({'graph_id': row['graph_id'], 'stratum': row['stratum'],
                                       'arm': row['arm'], 'sample_index': row['sample_index'],
                                       'method': method, 'repeat_index': repeat, 'access': access,
                                       'valid': prediction is not None,
                                       'MAE_2': float(error[0]) if error is not None else None,
                                       'ProfileMAE': float(np.mean(error)) if error is not None else None})
    grouped = defaultdict(list)
    for r in graph_rows:
        grouped[r['graph_id'], r['stratum'], r['arm'], r['method'], r['access']].append(r)
    graph_comparison = []
    for graph, stratum, arm, method in sorted({k[:4] for k in grouped}):
        entry = {'graph_id': graph, 'stratum': stratum, 'arm': arm, 'method': method}
        for access in ('hidden', 'released'):
            group = grouped[graph, stratum, arm, method, access]
            for metric in ('MAE_2', 'ProfileMAE'):
                valid = [r[metric] for r in group if r[metric] is not None]
                entry[f'{access}_{metric}'] = fmean(valid) if valid else None
            entry[f'{access}_validity'] = fmean(int(r['valid']) for r in group)
        for metric in ('MAE_2', 'ProfileMAE'):
            a, b = entry[f'hidden_{metric}'], entry[f'released_{metric}']
            entry[f'delta_{metric}'] = b-a if a is not None and b is not None else None
        graph_comparison.append(entry)
    for stratum in ('real', 'surrogate', 'synthetic'):
        for arm in ARMS:
            for method in ('plugin', 'median', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking'):
                group = [r for r in graph_comparison if (r['stratum'], r['arm'], r['method']) ==
                         (stratum, arm, method)]
                if len(group) != 8: raise ValueError('eight source/graph rows required')
                def eight_source_mean(field):
                    values = [r[field] for r in group if r[field] is not None]
                    return fmean(values) if len(values) == 8 else None
                summary_rows.append({'stratum': stratum, 'arm': arm, 'method': method,
                                     **{field: eight_source_mean(field)
                                        for field in ('hidden_MAE_2', 'released_MAE_2', 'delta_MAE_2',
                                                      'hidden_ProfileMAE', 'released_ProfileMAE',
                                                      'delta_ProfileMAE')},
                                     **{field: fmean(r[field] for r in group)
                                        for field in ('hidden_validity', 'released_validity')}})
    write_csv(out / 'GRAPH_COMPARISON.csv', graph_comparison)
    write_csv(out / 'COMPARISON.csv', summary_rows)
    choices = []
    for path in sorted((out / 'et/choices').glob('*/*.json')):
        item = read_json(path); selected = item['selected']
        choices.append({'arm': item['arm'], 'outer_fold': item['outer_fold'],
                        'anchor': selected['anchor'],
                        'min_samples_leaf': selected['min_samples_leaf'],
                        'max_features': selected['max_features'],
                        'inner_MAE_2': selected['mean_inner_source_MAE2']})
    if len(choices) != 18:
        raise ValueError('18 R/H ET choices required')
    write_csv(out / 'ET_CHOICES.csv', choices)
    write_json(out / 'VERIFICATION.json', {'observations': 144, 'requests': 864,
                                           'present': len(seen), 'complete': sum(completed.values()),
                                           'missing': 0, 'duplicates': 0, 'hash_mismatch': 0,
                                           'modes': {mode: {'requested': 432, 'complete': completed[mode],
                                                            'invalid': invalid[mode], 'missing': 0,
                                                            'duplicates': 0, 'hash_mismatch': 0}
                                                     for mode in ('qwen_thinking', 'qwen_nonthinking')}})
    fmt = lambda n: '' if n is None else f'{n:.4f}'
    lines = ['# R/H panel-size release sensitivity', '',
             'Separate from the frozen v10 main results. The same 144 R/H draws release only',
             '`n_panel` and the corresponding sampling-rule text. R+N retains complete retrieved',
             'dyad histories; H+N retains the original truncated history and Temporal_access.',
             'Neither releases full-archive sizes, a sampling fraction, calibration target or truth.',
             'Qwen uses the sealed v10 model settings and the original paired generation seeds.',
             'Negative delta means lower error.',
             'Real-source equal-source MAE_2 is primary; ProfileMAE is secondary. Qwen accuracy',
             'is conditional on strict valid final JSON answers. No outputs are repaired.', '',
             '| Stratum | Arm | Method | Hidden MAE_2 | Released MAE_2 | Delta | Hidden ProfileMAE | Released ProfileMAE | Delta | Validity hidden → released |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in summary_rows:
        lines.append('| ' + ' | '.join([r['stratum'], r['arm'], r['method'],
            *[fmt(r[k]) for k in ('hidden_MAE_2', 'released_MAE_2', 'delta_MAE_2',
                                  'hidden_ProfileMAE', 'released_ProfileMAE', 'delta_ProfileMAE')],
            fmt(r['hidden_validity']) + ' → ' + fmt(r['released_validity'])]) + ' |')
    side = list(csv.DictReader((out / 'SIDE_CHANNEL.csv').open()))
    lines += ['', '## Panel-size-only diagnostic', '',
              'Leave-one-real-source-out linear regression on `log1p(n_panel)` versus the',
              'training-median-only baseline; this is diagnostic, not a new main estimator.', '',
              '| Arm | Median-only MAE_2 | Panel-size-only MAE_2 |', '|---|---:|---:|']
    for arm in ARMS:
        group = [r for r in side if r['arm'] == arm]
        lines.append(f"| {arm} | {fmt(fmean(float(r['median_AE2']) for r in group))} | "
                     f"{fmt(fmean(float(r['n_panel_AE2']) for r in group))} |")
    lines += ['', '## Real graph-level deltas', '',
              'Counts use the eight real graph means; the full paired dataset is in',
              '`GRAPH_COMPARISON.csv`.', '',
              '| Arm | Method | Improved / 8 | Largest improvement | Largest worsening |',
              '|---|---|---:|---|---|']
    for arm in ARMS:
        for method in ('et', 'qwen_thinking', 'qwen_nonthinking'):
            group = sorted((r for r in graph_comparison if r['stratum'] == 'real' and
                            r['arm'] == arm and r['method'] == method),
                           key=lambda r: r['delta_MAE_2'])
            lines.append(f"| {arm} | {method} | {sum(r['delta_MAE_2'] < 0 for r in group)} | "
                         f"{group[0]['graph_id']} ({fmt(group[0]['delta_MAE_2'])}) | "
                         f"{group[-1]['graph_id']} (+{fmt(group[-1]['delta_MAE_2'])}) |")
    lines += ['', 'The ExtraTrees comparison uses one additional `log1p_n_panel` feature in',
              'the same training rows, nested LOSO folds, grid, anchors, weights and seeds.',
              'Plugin and median were re-evaluated and are identical. MLE was fitted to both',
              'paired blocks locally and is identical by construction; the sealed v10',
              'numerical predictions are retained to avoid platform-level optimizer drift.',
              'See `GRAPH_COMPARISON.csv` for all 288 graph/arm/method paired deltas and',
              '`ET_CHOICES.csv` for the 18 selected ET configurations.', '']
    (out / 'REPORT.md').write_text('\n'.join(lines))
    print('COMPARISON', len(summary_rows), 'graph rows', len(graph_comparison), 'invalid', dict(invalid))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=('prepare', 'et-cache', 'et-select', 'et-train',
                                      'offline', 'side-channel', 'comparison'))
    ap.add_argument('--old', type=Path, default=DEFAULT_OLD)
    ap.add_argument('--inputs', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--out', type=Path, default=DEFAULT_OUT)
    ap.add_argument('--index', type=int)
    a = ap.parse_args()
    if a.stage == 'prepare': prepare(a.old, a.inputs, a.out)
    elif a.stage == 'et-cache': et_cache(a.inputs, a.out)
    elif a.stage == 'offline': offline(a.inputs, a.out)
    elif a.stage == 'side-channel': side_channel(a.inputs, a.out)
    elif a.stage == 'comparison': comparison(a.out)
    else:
        if a.index is None or not 0 <= a.index < 2 * len(et.FOLDS):
            raise ValueError('ET index must be 0..17')
        et.OUT = a.out / 'et'
        et.ARMS = ARMS
        if a.stage == 'et-select': et.select_one(a.index)
        else: et.train_one(a.index)


if __name__ == '__main__':
    main()
