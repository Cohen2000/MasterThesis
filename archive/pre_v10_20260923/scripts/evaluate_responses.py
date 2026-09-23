#!/usr/bin/env python3
"""Validity and conditional accuracy of model answers; no imputation, no model calls.

Every planned request is matched to at most one response (strict ID, prompt and
payload hash binding). Per evidence block x arm x configuration: counts of
valid / invalid / technical failures / output-limit hits, conditional MAE2 and
ProfileMAE over valid answers with equal source weights and hierarchical MCSE
(clustered on the sampler draws), and the same for every reference.
"""
from pathlib import Path
import argparse
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARMS, CONFIGS, DESIGN_VERSION, LLM_REPEATS, PREPARED, REAL_TEST, REFERENCES,
                                    STRATA, digest, fresh_directory, in_stratum, read_json, read_jsonl, sha,
                                    write_csv, write_json)
from main_experiment.evaluation import EVALUATION_VERSION, complete_summary, conditional_summary, errors, resolve
from main_experiment.observation import parse

# Reference name in the output -> method in primary_baselines.json.
REFERENCE_METHODS = {'baseline': 'primary_corrector', 'plugin': 'plugin', 'extratrees': 'extratrees_pooled',
                     'median': 'median'}


def planned_study(run, primary):
    """Planned requests and their observations, with every hash re-checked."""
    planned = read_jsonl(run/'requests.jsonl')
    if len({r['id'] for r in planned}) != len(planned): raise ValueError('duplicate planned ID')
    observations = {}
    for r in planned:
        if r['payload_sha256'] != digest(r['payload']): raise ValueError('payload hash mismatch')
        oid = r['observation_id']
        if oid not in observations: observations[oid] = read_json(run/'observations/sample'/f'{oid}.json')
        o = observations[oid]
        if digest(o['messages']) != r['prompt_sha256']: raise ValueError('prompt hash mismatch')
        if o['block_sha256'] != digest(o['block']): raise ValueError('block hash mismatch')
        reference = primary['observations'][oid]
        if reference['block_sha256'] != o['block_sha256'] or reference['truth'] != o['truth']:
            raise ValueError('baseline observation mismatch')
    return planned, observations


def checked_responses(response_file, planned, mock):
    """One response per planned request at most, bound by prompt and payload hash."""
    known = {r['id']: r for r in planned}
    records = {}
    for r in read_jsonl(response_file) if response_file else []:
        if r['id'] not in known or r['id'] in records: raise ValueError('unknown or duplicate response ID')
        if bool(r.get('mock', False)) != mock: raise ValueError('mock / production mismatch')
        if r.get('extraction'): raise ValueError('post-hoc extraction is not a main response')
        for key in ('prompt_sha256', 'payload_sha256'):
            if r.get(key) != known[r['id']][key]: raise ValueError(f'{key} missing or different: {r["id"]}')
        if r.get('terminal') and 'final_text' not in r and not r.get('technical_error'):
            raise ValueError('terminal response lacks final_text')
        records[r['id']] = r
    return records


def answer_rows(planned, observations, primary, records, mock):
    """One row per planned request: state, errors, and the references' errors."""
    rows = []
    for r in planned:
        obs = observations[r['observation_id']]; reference = primary['observations'][obs['id']]
        outcome = resolve(parse(obs['block']), record=records.get(r['id']))
        row = {k: r[k] for k in ('id', 'graph_id', 'arm', 'sample_index', 'repeat_index', 'config_id', 'stratum')}
        row.update(outcome); row.update(errors(outcome['prediction'], obs['truth']))
        row.update(observation_id=obs['id'], empty=obs['empty'], mock=mock, prediction_json=outcome['prediction'],
                   budget_matched=obs['budget_matched'], reference_name=reference['primary_corrector_name'],
                   reference_status=reference['primary_corrector']['status'],
                   reference_fallback=reference['primary_corrector'].get('fallback', ''))
        for prefix, method in REFERENCE_METHODS.items():
            error = errors(reference[method]['prediction'], obs['truth'])
            for metric in ('AE2', 'ProfileAE'):
                row[f'{prefix}_{metric}_all'] = error[metric]
                row[f'{prefix}_{metric}_matched'] = error[metric] if row['valid'] else None
                row[f'delta_{metric}_vs_{prefix}'] = (row[metric]-error[metric]
                                                      if row[metric] is not None and error[metric] is not None else None)
        rows.append(row)
    return rows


def summary_rows(rows, mock):
    """Per block x arm x configuration: counts, then equal-source estimates with MCSE.

    Model accuracy is reported only for complete cells (every answer terminal);
    reference values are available as soon as the cell's shape is complete.
    """
    summary = []; sources_out = []
    for block in STRATA:
        for arm in ARMS:
            for config in CONFIGS:
                group = [r for r in rows if r['arm'] == arm and r['config_id'] == config and in_stratum(r['graph_id'], block)]
                if not group: continue
                sources = sorted({r['graph_id'] for r in group})
                draws = {s: len({r['sample_index'] for r in group if r['graph_id'] == s}) for s in sources}
                slots = {(r['graph_id'], r['sample_index'], r['repeat_index']) for r in group}
                if len(slots) != len(group): raise ValueError('duplicate sampling slot')
                shape_complete = len(slots) == sum(draws.values())*LLM_REPEATS
                if block == 'real': shape_complete &= set(sources) == set(REAL_TEST)
                complete = bool(shape_complete and all(r['terminal'] for r in group))
                base = {'stratum': block, 'arm': arm, 'config_id': config, 'mock': mock}
                result = {**base, 'complete': complete, 'planned': len(group),
                          'started': sum(r['started'] for r in group), 'terminal': sum(r['terminal'] for r in group),
                          'valid_answers': sum(bool(r['valid']) for r in group),
                          'invalid_terminal': sum(r['terminal'] and not r['valid'] for r in group),
                          'not_started': sum(r['status'] == 'not_started' for r in group),
                          'in_progress': sum(r['status'] == 'in_progress' for r in group),
                          'empty_skips': sum(r['empty'] for r in group),
                          'technical_failures': sum(bool(r.get('technical_error')) for r in group),
                          'limit_hits': sum(bool(r.get('limit_hit')) for r in group),
                          'fenced_valid': sum(r.get('validation_reason') == 'valid_after_fence' for r in group),
                          'accuracy_condition': 'valid answers only; no imputation; equal source weights'}
                metrics = ['valid_fraction', 'AE2', 'ProfileAE', 'signed_rho2']
                metrics += [k for k in group[0] if k.endswith(('_all', '_matched')) or k.startswith('delta_')]
                for metric in metrics:
                    cells = {s: np.full((draws[s], LLM_REPEATS), np.nan) for s in sources}
                    for r in group:
                        value = float(bool(r['valid'])) if metric == 'valid_fraction' and r['terminal'] else r.get(metric)
                        if value is not None: cells[r['graph_id']][r['sample_index']-1, r['repeat_index']-1] = value
                    everything = metric == 'valid_fraction' or metric.endswith('_all')
                    reportable = complete or (metric.endswith('_all') and shape_complete)
                    estimate = None
                    if reportable:
                        estimate = (complete_summary if everything else conditional_summary)(cells, draws)
                    result[metric] = estimate['mean'] if estimate else None
                    result[metric+'_MCSE'] = estimate['mcse'] if estimate else None
                    if metric == 'AE2': result['sources_with_valid_answers'] = estimate['sources_with_valid_answers'] if estimate else 0
                    for source, v in (estimate['sources'].items() if estimate else []):
                        sources_out.append({**base, 'source': source, 'metric': metric, 'value': v['mean'], 'MCSE': v['mcse'],
                                            **{k: v[k] for k in ('draws', 'valid_answers', 'planned_answers')}})
                result['numeric_model_estimate'] = result['AE2'] is not None
                summary.append(result)
    return summary, sources_out


def evaluate(response_file, out, mock=False, run=PREPARED, baselines=REFERENCES/'primary_baselines.json'):
    run = Path(run); out = Path(out)
    if mock and 'mock' not in str(out).lower(): raise ValueError('mock directory must contain mock')
    if read_json(run/'report.json')['design_version'] != DESIGN_VERSION: raise ValueError('design version mismatch')
    primary = read_json(baselines)
    if primary['design_version'] != DESIGN_VERSION: raise ValueError('baseline design mismatch')
    planned, observations = planned_study(run, primary)
    records = checked_responses(response_file, planned, mock)
    fresh_directory(out)
    write_json(out/'evaluation_inputs.json', {
        'evaluation_version': EVALUATION_VERSION, 'mock': mock, 'requests_sha256': sha(run/'requests.jsonl'),
        'baselines_sha256': sha(baselines), 'responses_sha256': sha(response_file) if response_file else None,
        'observations_sha256': digest(observations), 'evaluator_sha256': sha(__file__),
        'parser_sha256': sha(Path(__file__).resolve().parents[1]/'src/main_experiment/evaluation.py')})
    if response_file: (out/'raw_responses.jsonl').write_bytes(Path(response_file).read_bytes())
    rows = answer_rows(planned, observations, primary, records, mock)
    write_csv(out/'answer_errors.csv', rows)
    summary, sources = summary_rows(rows, mock)
    write_csv(out/'summary.csv', summary); write_csv(out/'source_results.csv', sources)
    write_json(out/'report.json', {
        'mock': mock, 'design_version': DESIGN_VERSION, 'evaluation_version': EVALUATION_VERSION,
        'logical_requests': len(rows), 'primary_baselines_sha256': sha(baselines),
        'complete_main_result': bool(summary) and all(r['complete'] for r in summary),
        'metric_label': 'validity and conditional MAE; no replacement',
        'accuracy_undefined_cells': [f"{r['stratum']}/{r['arm']}/{r['config_id']}" for r in summary
                                     if r['complete'] and not r['numeric_model_estimate']]})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--responses'); parser.add_argument('--out', required=True)
    parser.add_argument('--mock', action='store_true')
    args = parser.parse_args()
    evaluate(args.responses, args.out, args.mock)
