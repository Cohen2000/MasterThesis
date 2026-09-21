#!/usr/bin/env python3
"""Matched temporal-control analysis: parent vs. P[w,t] surrogate differences.

The eight surrogates are not further independent sources. For each parent, arm,
sampler index and model repeat:
    Delta rho_k    = rho_k(surrogate)    - rho_k(parent)
    Delta rhohat_k = rhohat_k(surrogate) - rhohat_k(parent)
    AE_Delta_2     = |Delta rhohat_2 - Delta rho_2|          (primary)
    Delta_ProfileAE = mean_k |Delta rhohat_k - Delta rho_k|  (secondary)
A pair is valid only if both answers are valid (no imputation). Sources are the
eight parents, weighted equally, with draw-clustered MCSE. If exactly one side
is a deterministic (saturated) H draw, its single observation is reused against
each draw of the other side and no MCSE is reported. Signed error, sign
agreement and correlation are descriptive only.

Also reported: within-replicate synthetic mode contrasts (a08 - a0,
memory - memoryless) for each outer replicate r1/r2.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARMS, DESIGN_VERSION, LLM_REPEATS, PREPARED, REAL_TEST, REFERENCES,
                                    fresh_directory, read_json, read_jsonl, sha, write_csv, write_json)
from main_experiment.evaluation import conditional_summary, resolve
from main_experiment.observation import parse

REFERENCE_METHODS = ('plugin', 'median', 'extratrees_pooled', 'extratrees_real_only', 'primary_corrector')
METRICS = ('AE_Delta_2', 'Delta_ProfileAE', 'signed_delta_error_2', 'sign_agreement_2')


def delta_metrics(parent_truth, surrogate_truth, parent_prediction, surrogate_prediction):
    truth = np.asarray(surrogate_truth)-parent_truth
    if parent_prediction is None or surrogate_prediction is None:
        return {'delta_rho': truth.tolist(), 'delta_rhohat': None, **{m: None for m in METRICS}}
    predicted = np.asarray(surrogate_prediction)-parent_prediction
    error = predicted-truth
    return {'delta_rho': truth.tolist(), 'delta_rhohat': predicted.tolist(),
            'AE_Delta_2': float(abs(error[0])), 'Delta_ProfileAE': float(np.abs(error).mean()),
            'signed_delta_error_2': float(error[0]), 'sign_agreement_2': bool(np.sign(predicted[0]) == np.sign(truth[0]))}


class Predictions:
    """Reference predictions and (optionally) parsed model answers per observation."""

    def __init__(self, responses):
        self.references = read_json(REFERENCES/'primary_baselines.json')['observations']
        planned = read_jsonl(PREPARED/'requests.jsonl')
        self.request = {(r['observation_id'], r['config_id'], r['repeat_index']): r['id'] for r in planned}
        self.records = {r['id']: r for r in read_jsonl(responses)} if responses else {}
        self.configs = sorted({r['config_id'] for r in planned if r['id'] in self.records})

    def methods(self):
        return [*REFERENCE_METHODS, *self.configs]

    def repeats(self, method):
        return range(1, 2 if method in REFERENCE_METHODS else LLM_REPEATS+1)

    def __call__(self, observation, method, repeat):
        if method in REFERENCE_METHODS: return self.references[observation['id']][method]['prediction']
        record = self.records.get(self.request[observation['id'], method, repeat])
        return resolve(parse(observation['block']), record=record)['prediction']


def pair_rows(observations, predict):
    by_key = {(o['graph_id'], o['arm'], o['sample_index']): o for o in observations}
    rows = []
    for parent in REAL_TEST:
        surrogate = parent+'__pwt'
        for arm in ARMS:
            draws = {g: sum(o['graph_id'] == g and o['arm'] == arm for o in observations) for g in (parent, surrogate)}
            one_side_deterministic = min(draws.values()) == 1 < max(draws.values())
            for index in range(1, max(draws.values())+1):
                a = by_key[parent, arm, 1 if draws[parent] == 1 else index]
                b = by_key[surrogate, arm, 1 if draws[surrogate] == 1 else index]
                for method in predict.methods():
                    for repeat in predict.repeats(method):
                        pa, pb = predict(a, method, repeat), predict(b, method, repeat)
                        rows.append({'parent': parent, 'arm': arm, 'method': method, 'sample_index': index,
                                     'repeat_index': repeat, 'parent_observation': a['id'],
                                     'surrogate_observation': b['id'], 'one_side_deterministic': one_side_deterministic,
                                     'valid_pair': pa is not None and pb is not None,
                                     **delta_metrics(a['truth'], b['truth'], pa, pb)})
    return rows


def summarise(rows, predict):
    summary = []; sources = []
    for arm in ARMS:
        for method in predict.methods():
            group = [r for r in rows if r['arm'] == arm and r['method'] == method]
            result = {'arm': arm, 'method': method, 'parents': len(REAL_TEST), 'planned_pairs': len(group),
                      'valid_pairs': sum(r['valid_pair'] for r in group)}
            for metric in METRICS:
                cells = {}; draws = {}
                for parent in REAL_TEST:
                    chosen = [r for r in group if r['parent'] == parent]
                    n = max(r['sample_index'] for r in chosen)
                    values = np.full((n, LLM_REPEATS), np.nan)
                    for r in chosen:
                        if r[metric] is not None:
                            # A deterministic reference fills all repeat columns with its single value.
                            columns = [r['repeat_index']-1] if method not in REFERENCE_METHODS else slice(None)
                            values[r['sample_index']-1, columns] = float(r[metric])
                    cells[parent] = values; draws[parent] = n
                    sources.append({'parent': parent, 'arm': arm, 'method': method, 'metric': metric,
                                    'value': float(np.nanmean(values)) if np.isfinite(values).any() else None,
                                    'valid_pairs': sum(r['valid_pair'] for r in chosen), 'planned_pairs': len(chosen)})
                estimate = conditional_summary(cells, draws)
                result[metric] = estimate['mean']
                result[metric+'_MCSE'] = None if any(r['one_side_deterministic'] for r in group) else estimate['mcse']
            truth = []; predicted = []
            for parent in REAL_TEST:
                valid = [r for r in group if r['parent'] == parent and r['valid_pair']]
                if valid:
                    truth.append(valid[0]['delta_rho'][0])
                    predicted.append(np.mean([r['delta_rhohat'][0] for r in valid]))
            result['descriptive_parent_delta_correlation'] = (
                float(np.corrcoef(truth, predicted)[0, 1]) if len(truth) > 1 and np.ptp(truth) > 0 and np.ptp(predicted) > 0 else None)
            summary.append(result)
    return summary, sources


def mode_mae2(observations, predict, key, arm, method):
    """MAE2 over all valid predictions of one synthetic instance and arm (None if none valid)."""
    errors = []
    for o in observations:
        if o['graph_id'] != key or o['arm'] != arm: continue
        for repeat in predict.repeats(method):
            prediction = predict(o, method, repeat)
            if prediction is not None: errors.append(abs(prediction[0]-o['truth'][0]))
    return float(np.mean(errors)) if errors else None


def synthetic_contrasts(observations, predict):
    rows = []
    for family, (mode_a, mode_b) in (('dar', ('a0', 'a08')), ('ad', ('memoryless', 'memory'))):
        for replicate in (1, 2):
            for arm in ARMS:
                for method in predict.methods():
                    mae = [mode_mae2(observations, predict, f'{family}_{mode}_r{replicate}', arm, method)
                           for mode in (mode_a, mode_b)]
                    rows.append({'family': family, 'outer_replicate': replicate, 'arm': arm, 'method': method,
                                 'mode_a': mode_a, 'mode_b': mode_b, 'MAE2_a': mae[0], 'MAE2_b': mae[1],
                                 'paired_MAE2_b_minus_a': mae[1]-mae[0] if None not in mae else None})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--responses', help='collected responses; references only if omitted')
    args = parser.parse_args()
    out = fresh_directory(args.out)
    write_json(out/'inputs.json', {'requests_sha256': sha(PREPARED/'requests.jsonl'),
                                   'baselines_sha256': sha(REFERENCES/'primary_baselines.json'),
                                   'responses_sha256': sha(args.responses) if args.responses else None})
    observations = [read_json(p) for p in sorted((PREPARED/'observations/sample').glob('*.json'))]
    predict = Predictions(args.responses)
    rows = pair_rows(observations, predict)
    summary, sources = summarise(rows, predict)
    contrasts = synthetic_contrasts(observations, predict)
    write_csv(out/'paired_observations.csv', rows)
    write_csv(out/'paired_sources.csv', sources)
    write_csv(out/'paired_summary.csv', summary)
    write_csv(out/'synthetic_within_replicate_contrasts.csv', contrasts)
    write_json(out/'report.json', {'design_version': DESIGN_VERSION, 'pair_rows': len(rows), 'summary': summary,
                                   'primary': 'AE_Delta_2', 'secondary': 'Delta_ProfileAE',
                                   'source_weighting': 'eight parents equally', 'validity': 'both answers required; no imputation',
                                   'synthetic_contrast_rows': len(contrasts)})
    print('paired control:', len(rows), 'pair rows,', len(predict.configs), 'model configurations')


if __name__ == '__main__':
    main()
