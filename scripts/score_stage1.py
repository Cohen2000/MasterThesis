#!/usr/bin/env python3
"""Score the stage-1 panel, or compose its tables with the R/H panel release."""
# Plain-words overview: this script scores the stage-1 panel (eight real sources, their
# surrogates and the synthetic graphs). It has two modes:
# - default: score plugin, median, design estimator, MLE, ExtraTrees and Qwen on the
#   prepared observations and write PREDICTIONS.csv and ET_CHOICES.csv;
# - --panel-release: take those predictions and swap in the R/H rows with the panel
#   size shown (scripts/add_panel_size.py).
# The final tables are built by src/pipeline/report.py, which starts from this script's
# output (results/stage1_tables). evaluate_api.py reuses errors(), mean_by_source() and draw_mcse().
import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from study.estimators import design_estimate, plugin
from study.common import (ARMS, PREPARED, STAGE1_REAL, RESULTS, TRAINING_SOURCES, fold_for,
                                    read_json, read_jsonl, write_csv)
from study.answer_format import parse_final
from study.observation import parse
from study.mle import fit as mle_fit

# 'design' is the inverse-probability estimator for the walk arms (S, S_obs); it was the
# S reference at this stage.
METHODS = ('plugin', 'median', 'design', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
QWEN = ('qwen_thinking', 'qwen_nonthinking')
REFERENCE = {'R': 'plugin', 'S': 'design', 'S_obs': 'design', 'H': 'mle', 'B': 'mle'}


# A prediction profile is valid if it has 4 finite values in [0, 1] that never increase
# (rho_2 >= rho_3 >= rho_4 >= rho_5).
def profile_valid(x):
    return bool(x is not None and len(x) == 4 and all(math.isfinite(float(a)) and 0 <= a <= 1 for a in x)
                and all(a >= b for a, b in zip(x, x[1:])))


# Error measures of one prediction against the truth:
# AE2 = |rho_2 error| (the main metric), ProfileAE = mean |error| over rho_2..rho_5,
# signed_rho2 = rho_2 error with sign (positive = overestimate).
def errors(x, truth):
    if x is None: return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    d = np.asarray(x, float) - np.asarray(truth, float)
    if not np.isfinite(d).all(): return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    return {'AE2': float(abs(d[0])), 'ProfileAE': float(np.abs(d).mean()),
            'signed_rho2': float(d[0])}


# The 'median' baseline: ignore the observation and predict the median true profile of the
# real training sources (leaving out the source being tested).
def median_profiles(prepared):
    truth = {}
    for p in (prepared / 'observations/training').glob('*.json'):
        r = read_json(p)
        truth[r['source_family']] = r['truth']
    if set(truth) != set(TRAINING_SOURCES): raise ValueError('incomplete real training sources')
    return {fold: np.median([truth[s] for s in TRAINING_SOURCES if s != fold], axis=0).tolist()
            for fold in (*STAGE1_REAL, 'synthetic')}


# Collect the ExtraTrees prediction of every observation from the fitted models' output files.
def et_predictions(folder):
    out = {}
    for path in sorted((folder / 'models').glob('*/*/predictions.json')):
        d = read_json(path)
        for row in d['observations']:
            if row['id'] in out: raise ValueError('duplicate ET prediction')
            out[row['id']] = row['prediction']
    return out


# Hyper-parameters chosen for ExtraTrees by nested cross-validation (one row per arm and fold).
def et_choices(folder):
    rows = []
    for path in sorted((folder / 'choices').glob('*/*.json')):
        d = read_json(path)
        selected = d['selected']
        rows.append({'arm': d['arm'], 'outer_fold': d['outer_fold'],
                     'anchor': selected['anchor'],
                     'min_samples_leaf': selected['min_samples_leaf'],
                     'max_features': selected['max_features'],
                     'inner_MAE_2': selected['mean_inner_source_MAE2']})
    if len(rows) != len(ARMS) * 9: raise ValueError('incomplete ET nested-CV choices')
    return rows


# Walk-audit result: graphs where the walk arm cannot be corrected at this sampling budget.
def gate_flags(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 24: raise ValueError('incomplete walk audit')
    return {r['graph_id']: r['not_correctable_at_this_budget'] == 'True' for r in rows}


# Score every method that needs no language model on every observation.
def offline_rows(prepared, et_dir, gate):
    medians = median_profiles(prepared)
    et = et_predictions(et_dir)
    observations = {r['id']: r for p in (prepared / 'observations/sample').glob('*.json')
                    for r in [read_json(p)]}
    if len(observations) != 360 or len(et) != 360: raise ValueError('incomplete main observations/ET')
    results = []; lookup = {}
    for oid, r in sorted(observations.items()):
        o = parse(r['block'])
        fold = fold_for(r['graph_id'])
        pred = {'plugin': plugin(o), 'median': medians[fold], 'et': et[oid]}
        if r['arm'] in ('S', 'S_obs'): pred['design'] = design_estimate(o)
        fit = mle_fit(o)
        pred['mle'] = fit.rho
        ref = pred[REFERENCE[r['arm']]]
        inv_hajek = None
        if r['arm'] in ('S', 'S_obs'):
            inv_hajek = [sum(row[3] for row in o['table'] if row[0].count('1') >= k) /
                         sum(row[3] for row in o['table']) for k in range(2, 6)]
        for method, p in pred.items():
            record = {'id': oid, 'observation_id': oid, 'source': r['graph_id'],
                      'stratum': r['stratum'], 'arm': r['arm'], 'sample_index': r['sample_index'],
                      'repeat_index': None, 'method': method, 'prediction': p,
                      'valid': all(math.isfinite(float(v)) for v in p),
                      'profile_valid': profile_valid(p),
                      'fallback': bool(fit.fallback_used) if method == 'mle' else False,
                      'fit_status': fit.fit_status if method == 'mle' else '',
                      'mle_flags': json.dumps(fit.flags, sort_keys=True) if method == 'mle' else '{}',
                      'inv_events_hajek_rho2': inv_hajek[0] if inv_hajek else None,
                      'reference_method': REFERENCE[r['arm']], 'reference_rho2': ref[0],
                      'plugin_rho2': pred['plugin'][0],
                      'not_correctable_at_this_budget': gate.get(r['graph_id'], False)
                      if r['arm'] in ('S', 'S_obs') else False,
                      **errors(p, r['truth'])}
            results.append(record)
            lookup[oid, method] = record
    return results, lookup, observations


# Read the Qwen answers, check that each belongs to exactly one planned request (same
# prompt, payload and seed), parse it with the shared strict parser and score it.
def qwen_rows(answers, prepared, observations, lookup):
    requests = {r['id']: r for r in read_jsonl(prepared / 'requests.jsonl')
                if r['config_id'] in QWEN and r['status'] != 'skipped_empty'}
    if len(requests) != 2160: raise ValueError('unexpected Qwen manifest size')
    found = {}
    for path in Path(answers).glob('*_r*/*.json'):
        a = read_json(path)
        rid = a['id']
        if rid not in requests or rid in found: raise ValueError('unknown/duplicate Qwen answer')
        for key in ('prompt_sha256', 'payload_sha256', 'seed'):
            if a.get(key) != requests[rid][key]: raise ValueError('answer identity mismatch')
        found[rid] = a
    if set(found) != set(requests): raise ValueError(f'incomplete Qwen answers: {len(found)}/{len(requests)}')
    rows = []
    for rid, req in sorted(requests.items()):
        answer = found[rid]
        obs = observations[req['observation_id']]
        values, reason = parse_final(answer.get('final_text', ''))
        if answer.get('status') != 'completed' or answer.get('technical_error'):
            values, reason = None, answer.get('end_state', 'technical_error')
        offline = lookup[obs['id'], 'plugin']
        rows.append({'id': rid, 'observation_id': obs['id'], 'source': obs['graph_id'],
                     'stratum': obs['stratum'], 'arm': obs['arm'],
                     'sample_index': obs['sample_index'], 'repeat_index': req['repeat_index'],
                     'method': req['config_id'], 'prediction': values,
                     'valid': values is not None, 'profile_valid': values is not None,
                     'validation_reason': reason, 'fallback': False, 'fit_status': '',
                     'reference_method': offline['reference_method'],
                     'reference_rho2': offline['reference_rho2'],
                     'plugin_rho2': offline['plugin_rho2'],
                     'inv_events_hajek_rho2': offline['inv_events_hajek_rho2'],
                     'not_correctable_at_this_budget': offline['not_correctable_at_this_budget'],
                     **errors(values, obs['truth'])})
    return rows


# Average a field per source (graph). Rows without a value (invalid answers) are skipped.
def mean_by_source(rows, field):
    by = defaultdict(list)
    for r in rows:
        if r[field] is not None: by[r['source']].append(float(r[field]))
    return {s: float(np.mean(v)) for s, v in by.items()}


# Monte-Carlo standard error of the equal-source MAE, treating the three observation
# draws of a source as independent clusters (how much the MAE could move with new draws).
def draw_mcse(rows, field):
    sources = sorted({r['source'] for r in rows})
    variances = []
    for source in sources:
        chosen = [r for r in rows if r['source'] == source and r[field] is not None]
        clusters = defaultdict(list)
        for r in chosen: clusters[r['sample_index']].append(float(r[field]))
        if len(clusters) < 2:
            variances.append(0.)
            continue
        y = np.array([sum(x) for x in clusters.values()])
        n = np.array([len(x) for x in clusters.values()])
        mu = y.sum() / n.sum()
        variances.append(float(len(y) / (len(y)-1) * np.square(y-mu*n).sum() / n.sum()**2))
    return float(np.sqrt(sum(variances)) / len(sources)) if sources else None


# Read a committed PREDICTIONS.csv back into typed Python values.
def _sealed_prediction_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    numeric = ('reference_rho2', 'plugin_rho2', 'inv_events_hajek_rho2', 'AE2', 'ProfileAE', 'signed_rho2')
    for row in rows:
        row['prediction'] = json.loads(row['prediction']) if row['prediction'] else None
        for key in numeric:
            row[key] = float(row[key]) if row[key] else None
        for key in ('valid', 'profile_valid', 'fallback', 'not_correctable_at_this_budget'):
            row[key] = (row[key] == 'True') if row[key] else None
        row['sample_index'] = int(row['sample_index'])
        row['repeat_index'] = int(row['repeat_index']) if row['repeat_index'] else None
    return rows


# Replace the R/H rows by the rows with the panel size shown; every other row is
# copied unchanged and checked to be identical.
def compose_panel_release(old_results, released, out):
    """Substitute the R/H answers and ExtraTrees fits with the panel size shown."""
    old = _sealed_prediction_rows(old_results / 'PREDICTIONS.csv')
    old_by_slot = {(r['observation_id'], r['method'], r['repeat_index']): r for r in old}
    if len(old) != len(old_by_slot):
        raise ValueError('duplicate sealed prediction slot')
    observations = {r['id']: r for p in (released / 'observations/sample').glob('*.json')
                    for r in [read_json(p)]}
    offline = {(r['id'], r['method']): r for r in read_json(released / 'offline.json')}
    et = et_predictions(released / 'et')
    requests = {r['id']: r for r in read_jsonl(released / 'requests.jsonl')}
    answers = {}
    for path in (released / 'answers').glob('*_r*/*.json'):
        answer = read_json(path)
        if answer['id'] in answers or answer['id'] not in requests:
            raise ValueError('duplicate or unexpected released Qwen answer')
        req = requests[answer['id']]
        if any(answer.get(k) != req[k] for k in ('seed', 'prompt_sha256', 'payload_sha256')):
            raise ValueError('released Qwen answer identity mismatch')
        answers[answer['id']] = answer
    if len(observations) != 144 or len(et) != 144 or len(requests) != 864 or set(answers) != set(requests):
        raise ValueError('released R/H evidence incomplete')
    rows = [r for r in old if r['arm'] not in ('R', 'H')]
    for obs in observations.values():
        hidden = obs['paired_hidden_id']
        if obs['arm'] not in ('R', 'H') or 'n_panel' not in parse(obs['block']):
            raise ValueError('released panel observation invalid')
        for method in ('plugin', 'median', 'mle', 'et'):
            original = old_by_slot[hidden, method, None]
            prediction = et[obs['id']] if method == 'et' else offline[obs['id'], method]['prediction']
            record = {**original, 'id': obs['id'], 'observation_id': obs['id'],
                      'prediction': prediction, 'valid': all(math.isfinite(float(x)) for x in prediction),
                      'profile_valid': profile_valid(prediction), **errors(prediction, obs['truth'])}
            if method in ('plugin', 'median', 'mle') and prediction != original['prediction']:
                raise ValueError(f'{method} changed despite identical observed table')
            rows.append(record)
    for rid, req in requests.items():
        obs = observations[req['observation_id']]
        original = old_by_slot[obs['paired_hidden_id'], req['config_id'], req['repeat_index']]
        answer = answers[rid]
        prediction, reason = parse_final(answer.get('final_text', ''))
        if answer.get('status') != 'completed' or answer.get('technical_error'):
            prediction, reason = None, answer.get('end_state', 'technical_error')
        rows.append({**original, 'id': rid, 'observation_id': obs['id'],
                     'prediction': prediction, 'valid': prediction is not None,
                     'profile_valid': prediction is not None, 'validation_reason': reason,
                     **errors(prediction, obs['truth'])})
    if len(rows) != len(old) or len({(r['id'], r['method']) for r in rows}) != len(rows):
        raise ValueError('composed prediction count or identity mismatch')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'PREDICTIONS.csv', rows)
    old_choices = list(csv.DictReader((old_results / 'ET_CHOICES.csv').open()))
    new_choices = list(csv.DictReader((released / 'ET_CHOICES.csv').open()))
    choices = [r for r in old_choices if r['arm'] not in ('R', 'H')] + new_choices
    if len(choices) != 45: raise ValueError('composed ET choices incomplete')
    write_csv(out / 'ET_CHOICES.csv', choices)
    print('PANEL_RELEASE_RESULTS', len(rows), 'Qwen', 2160)


# Command-line entry: default scoring mode, or --panel-release composition mode.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, default=PREPARED)
    ap.add_argument('--et', type=Path, default=RESULTS / 'et')
    ap.add_argument('--gate', type=Path, required=True)
    ap.add_argument('--answers', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--panel-release', type=Path,
                    help='compose the tables from the stage-1 scoring and the completed R/H panel release')
    ap.add_argument('--stage1-results', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'results/stage1_scores')
    a = ap.parse_args()
    if a.panel_release:
        compose_panel_release(a.stage1_results, a.panel_release, a.out)
        return
    gate = gate_flags(a.gate)
    offline, lookup, observations = offline_rows(a.prepared, a.et, gate)
    rows = offline + (qwen_rows(a.answers, a.prepared, observations, lookup) if a.answers else [])
    a.out.mkdir(parents=True, exist_ok=True)
    write_csv(a.out / 'PREDICTIONS.csv', rows)
    write_csv(a.out / 'ET_CHOICES.csv', et_choices(a.et))
    print('STAGE1_RESULTS', len(rows), 'Qwen', bool(a.answers), flush=True)


if __name__ == '__main__': main()
