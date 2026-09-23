#!/usr/bin/env python3
"""Generate v10 results or compose v11 from verified completed predictions."""
import argparse
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.baselines import design_estimate, plugin
from main_experiment.common import (ARMS, PREPARED, REAL_TEST, REFERENCES, RESULTS,
                                    TRAIN, fold_for, graph_stratum, read_json, read_jsonl,
                                    write_csv, write_json)
from main_experiment.evaluation import parse_final
from main_experiment.observation import parse
from main_experiment.shared_mle import fit as mle_fit

METHODS = ('plugin', 'median', 'design', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
QWEN = ('qwen_thinking', 'qwen_nonthinking')
REF_METHOD = {'R': 'plugin', 'S': 'design', 'S_obs': 'design', 'H': 'mle', 'B': 'mle'}


def profile_valid(x):
    return bool(x is not None and len(x) == 4 and all(math.isfinite(float(a)) and 0 <= a <= 1 for a in x)
                and all(a >= b for a, b in zip(x, x[1:])))


def errors(x, truth):
    if x is None: return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    d = np.asarray(x, float) - np.asarray(truth, float)
    if not np.isfinite(d).all(): return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    return {'AE2': float(abs(d[0])), 'ProfileAE': float(np.abs(d).mean()),
            'signed_rho2': float(d[0])}


def median_profiles(prepared):
    truth = {}
    for p in (prepared / 'observations/training').glob('*.json'):
        r = read_json(p)
        truth[r['source_family']] = r['truth']
    if set(truth) != set(TRAIN): raise ValueError('incomplete real training sources')
    return {fold: np.median([truth[s] for s in TRAIN if s != fold], axis=0).tolist()
            for fold in (*REAL_TEST, 'synthetic')}


def et_predictions(folder):
    out = {}
    for path in sorted((folder / 'models').glob('*/*/predictions.json')):
        d = read_json(path)
        for row in d['observations']:
            if row['id'] in out: raise ValueError('duplicate ET prediction')
            out[row['id']] = row['prediction']
    return out


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


def gate_flags(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 24: raise ValueError('incomplete walk audit')
    return {r['graph_id']: r['not_correctable_at_this_budget'] == 'True' for r in rows}


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
        ref = pred[REF_METHOD[r['arm']]]
        mle_old_errors = errors(fit.old_rule_rho, r['truth'])
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
                      'mle_old_rule_AE2': mle_old_errors['AE2'] if method == 'mle' else None,
                      'mle_old_rule_ProfileAE': mle_old_errors['ProfileAE'] if method == 'mle' else None,
                      'mle_old_rule_signed_rho2': mle_old_errors['signed_rho2'] if method == 'mle' else None,
                      'mle_old_rule_fallback': fit.old_rule_fallback_used if method == 'mle' else False,
                      'inv_events_hajek_rho2': inv_hajek[0] if inv_hajek else None,
                      'reference_method': REF_METHOD[r['arm']], 'reference_rho2': ref[0],
                      'plugin_rho2': pred['plugin'][0],
                      'not_correctable_at_this_budget': gate.get(r['graph_id'], False)
                      if r['arm'] in ('S', 'S_obs') else False,
                      **errors(p, r['truth'])}
            results.append(record)
            lookup[oid, method] = record
    return results, lookup, observations


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


def mean_by_source(rows, field):
    by = defaultdict(list)
    for r in rows:
        if r[field] is not None: by[r['source']].append(float(r[field]))
    return {s: float(np.mean(v)) for s, v in by.items()}


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


def summary(rows, stratum):
    selected = [r for r in rows if r['stratum'] == stratum]
    table = []
    for arm in ARMS:
        plugin_rows = [r for r in selected if r['arm'] == arm and r['method'] == 'plugin']
        plugin_mae = float(np.mean(list(mean_by_source(plugin_rows, 'AE2').values())))
        for method in METHODS:
            group = [r for r in selected if r['arm'] == arm and r['method'] == method]
            if not group: continue
            ae = mean_by_source(group, 'AE2')
            profile = mean_by_source(group, 'ProfileAE')
            signed = mean_by_source(group, 'signed_rho2')
            if stratum == 'real' and len(ae) != 8:
                raise ValueError(f'{arm}/{method}: equal-source result needs eight valid sources')
            mae = float(np.mean(list(ae.values()))) if ae else None
            row = {'stratum': stratum, 'arm': arm, 'method': method,
                   'sources': len({r['source'] for r in group}), 'sources_with_valid': len(ae),
                   'MAE_2': mae, 'ProfileMAE': float(np.mean(list(profile.values()))) if profile else None,
                   'signed_rho_2': float(np.mean(list(signed.values()))) if signed else None,
                   'validity': float(np.mean([r['valid'] for r in group])),
                   'fallback_rate': float(np.mean([r['fallback'] for r in group])) if method == 'mle' else None,
                   'old_rule_MAE_2': (float(np.mean(list(mean_by_source(group, 'mle_old_rule_AE2').values())))
                                      if method == 'mle' else None),
                   'old_rule_ProfileMAE': (float(np.mean(list(mean_by_source(group, 'mle_old_rule_ProfileAE').values())))
                                           if method == 'mle' else None),
                   'old_rule_signed_rho_2': (float(np.mean(list(mean_by_source(group, 'mle_old_rule_signed_rho2').values())))
                                             if method == 'mle' else None),
                   'old_rule_fallback_rate': (float(np.mean([r['mle_old_rule_fallback'] for r in group]))
                                              if method == 'mle' else None),
                   'ET_profile_validity': float(np.mean([r['profile_valid'] for r in group])) if method == 'et' else None,
                   'draw_clustered_MCSE_2': draw_mcse(group, 'AE2') if mae is not None else None,
                   'skill_vs_plugin': 1 - mae / plugin_mae if mae is not None and plugin_mae > 0 else None,
                   'not_correctable_at_this_budget_sources': ','.join(sorted({r['source'] for r in group
                       if r['not_correctable_at_this_budget']})) if arm in ('S', 'S_obs') else ''}
            if stratum == 'synthetic': row['ET_in_distribution'] = method == 'et'
            table.append(row)
    return table


def by_source(rows, stratum):
    out = []
    for (source, arm, method), group in itertools.groupby(
            sorted((r for r in rows if r['stratum'] == stratum),
                   key=lambda r: (r['source'], r['arm'], r['method'])),
            key=lambda r: (r['source'], r['arm'], r['method'])):
        group = list(group)
        valid = [r for r in group if r['AE2'] is not None]
        out.append({'source': source, 'stratum': stratum, 'arm': arm, 'method': method,
                    'MAE_2': float(np.mean([r['AE2'] for r in valid])) if valid else None,
                    'ProfileMAE': float(np.mean([r['ProfileAE'] for r in valid])) if valid else None,
                    'signed_rho_2': float(np.mean([r['signed_rho2'] for r in valid])) if valid else None,
                    'validity': len(valid) / len(group),
                    'not_correctable_at_this_budget': any(r['not_correctable_at_this_budget'] for r in group)})
    return out


def signflip(values):
    a = np.asarray(values, float)
    observed = abs(a.mean())
    flipped = (abs(np.mean(a*np.asarray(signs))) for signs in itertools.product((-1, 1), repeat=len(a)))
    return sum(x >= observed - 1e-15 for x in flipped) / 2**len(a)


def paired(rows, arm, first, second, stratum='real'):
    # Qwen comparisons use only slots with a valid answer; offline estimates repeat by draw.
    a = [r for r in rows if r['stratum'] == stratum and r['arm'] == arm and r['method'] == first]
    b = {(r['source'], r['sample_index'], r['repeat_index']): r for r in rows
         if r['stratum'] == stratum and r['arm'] == arm and r['method'] == second}
    by = defaultdict(list)
    for r in a:
        key = (r['source'], r['sample_index'], r['repeat_index'] if second in QWEN else None)
        other = b.get(key)
        if other and r['AE2'] is not None and other['AE2'] is not None:
            by[r['source']].append(r['AE2'] - other['AE2'])
    means = {s: float(np.mean(v)) for s, v in by.items()}
    if len(means) != 8: return {'mean_difference': None, 'wins_out_of_8': None,
                                'exact_signflip_p': None, 'LOSO_min': None, 'LOSO_max': None,
                                'sources_with_pairs': len(means)}
    v = np.array(list(means.values()))
    loo = [float(np.mean(np.delete(v, i))) for i in range(8)]
    return {'mean_difference': float(v.mean()), 'wins_out_of_8': int((v < 0).sum()),
            'exact_signflip_p': signflip(v), 'LOSO_min': min(loo), 'LOSO_max': max(loo),
            'sources_with_pairs': 8}


def inference(rows):
    out = []
    for arm in ARMS:
        ref = REF_METHOD[arm]
        comparisons = [('qwen_thinking', 'plugin'), ('qwen_thinking', ref),
                       (ref, 'plugin'), ('et', ref)]
        if ref == 'plugin':
            comparisons = [('qwen_thinking', 'plugin'), ('et', ref)]
        for first, second in comparisons:
            out.append({'arm': arm, 'comparison': f'{first} vs {second}',
                        **paired(rows, arm, first, second)})
    return out


def anchoring(rows):
    out = []
    for stratum in ('real', 'surrogate', 'synthetic'):
        for arm in ARMS:
            for method in QWEN:
                group = [r for r in rows if r['stratum'] == stratum and r['arm'] == arm
                         and r['method'] == method and abs(r['plugin_rho2'] - r['reference_rho2']) >= .01]
                valid = [r for r in group if r['valid']]
                plugin_near = [r for r in valid if abs(r['prediction'][0] - r['plugin_rho2']) <= .005]
                ref_near = [r for r in valid if abs(r['prediction'][0] - r['reference_rho2']) <= .005]
                inv_near = [r for r in valid if r['inv_events_hajek_rho2'] is not None and
                            abs(r['prediction'][0] - r['inv_events_hajek_rho2']) <= .005]
                both = {r['id'] for r in plugin_near} & {r['id'] for r in ref_near}
                union = ({r['id'] for r in plugin_near} | {r['id'] for r in ref_near} |
                         {r['id'] for r in inv_near})
                denom = len(valid)
                out.append({'stratum': stratum, 'arm': arm, 'method': method,
                            'observations_separated': len({r['observation_id'] for r in group}),
                            'valid_answers': denom, 'validity': denom / len(group) if group else None,
                            'near_plugin': len(plugin_near)/denom if denom else None,
                            'near_reference': len(ref_near)/denom if denom else None,
                            'near_inv_events_hajek': len(inv_near)/denom if denom else None,
                            'near_both': len(both)/denom if denom else None,
                            'neither': (denom-len(union))/denom if denom else None})
    return out


def qwen_repeat_ranges(rows):
    grouped = defaultdict(list)
    for r in rows:
        if r['method'] in QWEN and r['valid'] and r['prediction'] is not None:
            grouped[(r['stratum'], r['arm'], r['method'], r['observation_id'])].append(r['prediction'])
    by_group = defaultdict(list)
    for (stratum, arm, method, oid), profiles in grouped.items():
        if len(profiles) != 3: continue
        ranges = np.ptp(np.asarray(profiles, float), axis=0)
        by_group[(stratum, arm, method)].append(ranges)
    return [{'stratum': stratum, 'arm': arm, 'method': method,
             'complete_observations': len(v),
             **{f'median_range_rho{k}': float(np.median(np.asarray(v)[:, k-2])) for k in range(2, 6)}}
            for (stratum, arm, method), v in sorted(by_group.items())]


def s_contrast(rows):
    out = []
    for stratum in ('real', 'surrogate', 'synthetic'):
        for method in METHODS:
            s = [r for r in rows if r['stratum'] == stratum and r['arm'] == 'S' and r['method'] == method]
            obs = {(r['source'], r['sample_index'], r['repeat_index']): r for r in rows
                   if r['stratum'] == stratum and r['arm'] == 'S_obs' and r['method'] == method}
            by = defaultdict(list)
            for r in s:
                other = obs.get((r['source'], r['sample_index'], r['repeat_index']))
                if other and r['AE2'] is not None and other['AE2'] is not None:
                    by[r['source']].append(r['AE2'] - other['AE2'])
            if not by: continue
            v = np.array([np.mean(x) for x in by.values()])
            loo = [float(np.mean(np.delete(v, i))) for i in range(len(v))] if len(v) > 1 else []
            out.append({'stratum': stratum, 'method': method, 'sources': len(v),
                        'S_minus_S_obs_MAE_2': float(v.mean()), 'S_wins': int((v < 0).sum()),
                        'exact_signflip_p': signflip(v),
                        'LOSO_min': min(loo) if loo else None, 'LOSO_max': max(loo) if loo else None,
                        'not_correctable_at_this_budget_sources': ','.join(sorted(
                            {r['source'] for r in s if r['not_correctable_at_this_budget']}))})
    return out


def markdown(main, infer, out, qwen_complete, design='v10'):
    lines = [f'# {design} main results', '',
             'Primary metric: equal-source MAE_2 across eight real sources. ProfileMAE is secondary.',
             'LLM accuracy uses valid final answers only; no estimate is clipped or repaired.',
             'ET hyperparameters were selected by nested leave-one-real-training-source-out CV.',
             'sp_hospital__pwt remains flagged as not correctable at the 10% S budget.',
             '', f'Qwen complete: {qwen_complete}.', '',
             '| Arm | Method | MAE_2 | ProfileMAE | Signed rho_2 | Validity | Fallback | Old-rule MLE MAE_2 | Old-rule ProfileMAE | Old-rule signed rho_2 | Old-rule fallback | ET profile validity | Skill vs plugin | S flag |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    if design == 'v11':
        lines[7:7] = ['R/H use completed released-panel Qwen answers and panel-aware ET; S/S_obs/B',
                      'use unchanged sealed v10 evidence. Hidden-panel R/H is sensitivity only.',
                      'Workplace retains the original empty-calendar-window limitation.', '']
    form = lambda x: '' if x is None or x == '' else f'{float(x):.4f}'
    for r in main:
        lines.append('| ' + ' | '.join([r['arm'], r['method'], form(r['MAE_2']), form(r['ProfileMAE']),
                                    form(r['signed_rho_2']), form(r['validity']),
                                    form(r['fallback_rate']), form(r['old_rule_MAE_2']),
                                    form(r['old_rule_ProfileMAE']), form(r['old_rule_signed_rho_2']),
                                    form(r['old_rule_fallback_rate']), form(r['ET_profile_validity']),
                                    form(r['skill_vs_plugin']), r['not_correctable_at_this_budget_sources']]) + ' |')
    if infer:
        lines += ['', '## Source-level paired inference', '',
                  'Difference is first method minus second method in source-level MAE_2.', '',
                  '| Arm | Comparison | Mean difference | Wins / 8 | Exact sign-flip p | LOSO range |',
                  '|---|---|---:|---:|---:|---:|']
        for r in infer:
            rng = '' if r['LOSO_min'] is None else f"{r['LOSO_min']:.4f} to {r['LOSO_max']:.4f}"
            lines.append(f"| {r['arm']} | {r['comparison']} | {form(r['mean_difference'])} | "
                         f"{r['wins_out_of_8'] if r['wins_out_of_8'] is not None else ''} | "
                         f"{form(r['exact_signflip_p'])} | {rng} |")
    (out / 'MAIN_RESULTS.md').write_text('\n'.join(lines) + '\n')


def _sealed_prediction_rows(path):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    numeric = ('reference_rho2', 'plugin_rho2', 'inv_events_hajek_rho2',
               'mle_old_rule_AE2', 'mle_old_rule_ProfileAE', 'mle_old_rule_signed_rho2',
               'AE2', 'ProfileAE', 'signed_rho2')
    for row in rows:
        row['prediction'] = json.loads(row['prediction']) if row['prediction'] else None
        for key in numeric:
            row[key] = float(row[key]) if row[key] else None
        for key in ('valid', 'profile_valid', 'fallback', 'mle_old_rule_fallback',
                    'not_correctable_at_this_budget'):
            row[key] = (row[key] == 'True') if row[key] else None
        row['sample_index'] = int(row['sample_index'])
        row['repeat_index'] = int(row['repeat_index']) if row['repeat_index'] else None
    return rows


def composite_v11(old_results, released, out):
    """Substitute completed R/H answers and ET fits; rerun only table arithmetic."""
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
        raise ValueError('v11 composite prediction count or identity mismatch')
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'PREDICTIONS.csv', rows)
    def unchanged_arms(filename, current):
        with (old_results / filename).open() as stream:
            sealed = list(csv.DictReader(stream))
        # S/S_obs/B have identical prediction rows; carry their published
        # decimal representations through without platform-level rounding drift.
        key = lambda r: (r.get('stratum'), r.get('condition'), r['arm'], r.get('method'))
        lookup = {key(r): r for r in sealed}
        if len(lookup) != len(sealed): raise ValueError('duplicate sealed summary row')
        if any(key(x) not in lookup for x in current if x['arm'] in ('S', 'S_obs', 'B')):
            raise ValueError('unchanged arm missing from sealed summary')
        return [lookup[key(x)] if x['arm'] in ('S', 'S_obs', 'B') else x for x in current]
    main_table = unchanged_arms('MAIN_REAL.csv', summary(rows, 'real'))
    write_csv(out / 'SUMMARY.csv', main_table +
              unchanged_arms('SURROGATE.csv', summary(rows, 'surrogate')) +
              unchanged_arms('SYNTHETIC.csv', summary(rows, 'synthetic')))
    write_csv(out / 'PER_SOURCE.csv', [r for stratum in ('real', 'surrogate', 'synthetic')
                                      for r in by_source(rows, stratum)])
    conditions = ('dar_a0', 'dar_a08', 'ad_memoryless', 'ad_memory')
    write_csv(out / 'SYNTHETIC_CONDITIONS.csv', unchanged_arms('SYNTHETIC_CONDITIONS.csv',
        [dict(condition=c, **r) for c in conditions
         for r in summary([x for x in rows if x['source'].startswith(c + '_r')], 'synthetic')]))
    old_choices = list(csv.DictReader((old_results / 'ET_CHOICES.csv').open()))
    new_choices = list(csv.DictReader((released / 'ET_CHOICES.csv').open()))
    choices = [r for r in old_choices if r['arm'] not in ('R', 'H')] + new_choices
    if len(choices) != 45: raise ValueError('v11 ET choices incomplete')
    write_csv(out / 'ET_CHOICES.csv', choices)
    write_csv(out / 'SOURCE_INFERENCE.csv', inference(rows))
    write_csv(out / 'ANCHORING.csv', anchoring(rows))
    write_csv(out / 'QWEN_REPEAT_RANGE.csv', qwen_repeat_ranges(rows))
    # S/S_obs is byte-identical to the sealed v10 table; refer to it there.
    markdown(main_table, inference(rows), out, True, design='v11')
    qwen = [r for r in rows if r['method'] in QWEN]
    write_json(out / 'REPORT.json', {'design_version': 'panel888-access-v11-20260923',
              'observations': 360, 'qwen_answers': len(qwen),
              'qwen_valid': sum(r['valid'] for r in qwen),
              'qwen_invalid': sum(not r['valid'] for r in qwen),
              'qwen_by_arm_mode': {f'{arm}/{mode}': {'answers': sum(r['arm'] == arm and r['method'] == mode for r in qwen),
                                                    'valid': sum(r['arm'] == arm and r['method'] == mode and r['valid'] for r in qwen)}
                                   for arm in ARMS for mode in QWEN},
              'qwen_source_by_arm': {'R': 'released_RH_archive', 'H': 'released_RH_archive',
                                     'S': 'v10_archive', 'S_obs': 'v10_archive', 'B': 'v10_archive'},
              'ET_released_panel_arms': ['R', 'H'],
              'ET_RH_source': 'completed panel-release nested ET fits; docs/results/panel888_v10_RH_panel_release/ET_CHOICES.csv',
              'ET_S_S_obs_B_source': 'sealed v10 main predictions and choices',
              'model_or_sampling_recomputed': False})
    print('V11_RESULTS', len(rows), 'Qwen', 2160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--prepared', type=Path, default=PREPARED)
    ap.add_argument('--et', type=Path, default=RESULTS / 'et')
    ap.add_argument('--gate', type=Path, required=True)
    ap.add_argument('--answers', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--v11-released', type=Path,
                    help='compose v11 from sealed v10 predictions and completed released R/H results')
    ap.add_argument('--v10-results', type=Path,
                    default=Path(__file__).resolve().parents[1] / 'docs/results/panel888_v10_main_20260923')
    a = ap.parse_args()
    if a.v11_released:
        composite_v11(a.v10_results, a.v11_released, a.out)
        return
    gate = gate_flags(a.gate)
    offline, lookup, observations = offline_rows(a.prepared, a.et, gate)
    rows = offline + (qwen_rows(a.answers, a.prepared, observations, lookup) if a.answers else [])
    a.out.mkdir(parents=True, exist_ok=True)
    write_csv(a.out / 'PREDICTIONS.csv', rows)
    main_table = summary(rows, 'real')
    write_csv(a.out / 'MAIN_REAL.csv', main_table)
    write_csv(a.out / 'PER_SOURCE_REAL.csv', by_source(rows, 'real'))
    write_csv(a.out / 'PER_SOURCE_SURROGATE.csv', by_source(rows, 'surrogate'))
    write_csv(a.out / 'PER_SOURCE_SYNTHETIC.csv', by_source(rows, 'synthetic'))
    choices = et_choices(a.et)
    write_csv(a.out / 'ET_CHOICES.csv', choices)
    write_json(a.out / 'ET_CHOICES.json', choices)
    write_csv(a.out / 'SURROGATE.csv', summary(rows, 'surrogate'))
    write_csv(a.out / 'SYNTHETIC.csv', summary(rows, 'synthetic'))
    conditions = ('dar_a0', 'dar_a08', 'ad_memoryless', 'ad_memory')
    write_csv(a.out / 'SYNTHETIC_CONDITIONS.csv', [dict(condition=c, **r) for c in conditions
        for r in summary([x for x in rows if x['source'].startswith(c + '_r')], 'synthetic')])
    infer = inference(rows)
    write_csv(a.out / 'SOURCE_INFERENCE.csv', infer)
    if a.answers: write_csv(a.out / 'ANCHORING.csv', anchoring(rows))
    write_csv(a.out / 'S_VS_S_OBS.csv', s_contrast(rows))
    if a.answers: write_csv(a.out / 'QWEN_REPEAT_RANGE.csv', qwen_repeat_ranges(rows))
    markdown(main_table, infer, a.out, bool(a.answers))
    write_json(a.out / 'REPORT.json', {'observations': len(observations),
                                      'offline_prediction_rows': len(offline),
                                      'qwen_prediction_rows': len(rows)-len(offline),
                                      'qwen_complete': bool(a.answers),
                                      'gate_failures_retained': sorted(k for k, v in gate.items() if v),
                                      'ET_selection': 'nested_leave_one_real_training_source_out',
                                      'primary_metric': 'equal_source_MAE_2'})
    print('V10_RESULTS', len(rows), 'Qwen', bool(a.answers), flush=True)


if __name__ == '__main__': main()
