#!/usr/bin/env python3
"""Budget sensitivity of the panel888 study (ancillary; the main study stays at 0.10).

The grid BUDGET_GRID = (.025, .05, .10, .20, .30, .40, .50) was fixed before any
10% Qwen accuracy was seen. Only the budget changes: T_g(b) = b * sum_e K_e.
Graphs, arms and their calibration rules, prompts, draws, repeats, references,
LOSO folds, the training pool and the evaluation are those of the main study.
The 0.10 point is the main study itself (results/panel888), never recomputed.

  prepare <b>   one budget: calibrate, draw test/training observations, training
                pool, refit ExtraTrees, reference predictions, Qwen requests
  feasibility   per budget x graph x arm: target, expected coverage, matched?
  audit         counts, seeds, request identities; builds the cluster run/ folder
  evaluate      after Qwen: one tidy table, equal-source summaries and plots

Outputs: results/panel888_budget_sensitivity/ (b025, b050, b200, ... , run, qwen).
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARMS, AUDIT, BUDGET_GRID, BUDGET_SENSITIVITY, BUILD, COVERAGE_FRACTION,
                                    DESIGN_VERSION, LLM_REPEATS, MAIN_KEYS, PREPARED, QWEN, QWEN_CONFIGS,
                                    REFERENCES, ROOT, SEEDS, STRATA, SURROGATES, SYNTH, TRAIN, code_hashes, draws_for,
                                    fresh_directory, in_stratum, read_json, read_jsonl, sha, write_csv, write_json)
from main_experiment.baselines import PRIMARY_REFERENCE_NAME
from main_experiment.data import load_graph
from main_experiment.evaluation import complete_summary, conditional_summary, errors, resolve
from main_experiment.observation import parse
from main_experiment.pool import build_pool, pool_definition
from main_experiment.prepare import domains_of, observation_row
from main_experiment.references import fold_references, main_references, stage_train
from main_experiment.requests import planned, validate_request
from main_experiment.sampling import calibrate, draw
from main_experiment.token_sizes import TokenCounters

EXTRA_BUDGETS = tuple(b for b in BUDGET_GRID if b != COVERAGE_FRACTION)
REFERENCE_METHODS = ('plugin', 'primary_corrector', 'median', 'extratrees_pooled', 'extratrees_real_only')
# Expected observed active dyad-windows of each arm, from the calibrated budget.
EXPECTED_CELLS = {'R': 'node_expected_cells', 'S': 'validation_mean', 'H': 'h_expected_cells',
                  'B': 'bernoulli_expected_cells'}


def folder(fraction):
    """Where one budget's prepared study lives; 0.10 is the main study."""
    if fraction == COVERAGE_FRACTION: return PREPARED
    return BUDGET_SENSITIVITY/f'b{round(fraction*1000):03d}'


def budget_of(fraction, key):
    return read_json(folder(fraction)/'calibration'/f'{key}.json')


# ---------------------------------------------------------------- prepare one budget
def prepare(fraction):
    if fraction not in EXTRA_BUDGETS: raise ValueError('the 0.10 point is the main study; not recomputed')
    out = fresh_directory(folder(fraction))
    write_json(out/'inputs.json', {'coverage_fraction': fraction, 'design_version': DESIGN_VERSION,
                                   'code': code_hashes(), 'main_requests_sha256': sha(PREPARED/'requests.jsonl')})
    main_rows = []; training_rows = []
    for key in (*TRAIN, *SYNTH, *SURROGATES):
        g = load_graph(PREPARED/'graphs'/key)                     # the main study's graphs, unchanged
        budget, walk, _ = calibrate(g, BUILD, fraction)
        write_json(out/'calibration'/f'{key}.json', budget)
        for domain in domains_of(key):
            for arm in ARMS:
                for index in range(1, draws_for(arm, budget, domain)+1):
                    counts, traversals = draw(g, arm, index, domain, budget, walk)
                    row = observation_row(g, arm, index, domain, budget, counts, traversals)
                    write_json(out/'observations'/domain/f'{row["id"]}.json', row)
                    (training_rows if domain == 'training' else main_rows).append(row)
        print(f'b={fraction} {key}: matched={budget["budget_matched_by_arm"]}', flush=True)
    counters = TokenCounters(ROOT/'data/tokenizers')
    sizes = [{'id': r['id'], **counters.count(r['messages'])} for r in main_rows]
    write_csv(out/'prompt_sizes.csv', sizes)
    train_specs = [s for s in pool_definition()['graphs'] if s['partition'] == 'train']
    build_pool(out/'pool', train_specs, fraction)
    stage_train(out, prepared=out)
    models, medians = fold_references(out)
    main_references(out, models, medians, prepared=out)
    requests = [r for r in planned(main_rows) if r['config_id'] in QWEN_CONFIGS]
    with open(out/'requests.jsonl', 'w') as f:
        for r in requests: f.write(json.dumps(r, separators=(',', ':'), allow_nan=False, sort_keys=True)+'\n')
    write_json(out/'seed_manifest.json', [{'seed': s, 'fields': json.loads(v)} for s, v in sorted(SEEDS.items())])
    write_json(out/'report.json', {'coverage_fraction': fraction, 'main_observations': len(main_rows),
                                   'training_observations': len(training_rows), 'qwen_requests': len(requests),
                                   'max_qwen_prompt_tokens': max(max(s['qwen_thinking'], s['qwen_nonthinking']) for s in sizes)})
    print('prepared', read_json(out/'report.json'), flush=True)


# ---------------------------------------------------------------- feasibility
def feasibility():
    rows = []
    for fraction in BUDGET_GRID:
        for key in MAIN_KEYS:
            b = budget_of(fraction, key)
            block = next(s for s in STRATA if in_stratum(key, s))
            for arm in ARMS:
                rows.append({'budget': fraction, 'graph_id': key, 'evidence_block': block, 'arm': arm,
                             'T': b['T'], 'active_dyad_windows': b['active_dyad_windows'],
                             'expected_cells': b[EXPECTED_CELLS[arm]],
                             'expected_coverage': b[EXPECTED_CELLS[arm]]/b['active_dyad_windows'],
                             'matched': b['budget_matched_by_arm'][arm],
                             'reasons': ';'.join(r for r in b['unmatched_reasons'] if r.startswith(arm+':')),
                             'h_saturated': b['h_saturated'] if arm == 'H' else None,
                             'draws': draws_for(arm, b)})
    write_csv(BUDGET_SENSITIVITY/'feasibility.csv', rows)
    summary = []
    for fraction in BUDGET_GRID:
        for arm in ARMS:
            chosen = [r for r in rows if r['budget'] == fraction and r['arm'] == arm]
            unmatched = [f'{r["graph_id"]} ({r["reasons"]}, coverage {r["expected_coverage"]:.3f})'
                         for r in chosen if not r['matched']]
            summary.append({'budget': fraction, 'arm': arm, 'graphs': len(chosen), 'matched': len(chosen)-len(unmatched),
                            'min_expected_coverage': min(r['expected_coverage'] for r in chosen),
                            'max_expected_coverage': max(r['expected_coverage'] for r in chosen),
                            'saturated_H': sum(bool(r['h_saturated']) for r in chosen), 'unmatched': unmatched})
    write_json(BUDGET_SENSITIVITY/'feasibility.json', summary)
    for s in summary:
        print(f"b={s['budget']:.3f} {s['arm']}: matched {s['matched']}/24, coverage "
              f"{s['min_expected_coverage']:.3f}-{s['max_expected_coverage']:.3f}, saturated H {s['saturated_H']}")
        for u in s['unmatched']: print('    unmatched:', u)


# ---------------------------------------------------------------- audit and cluster run folder
def audit():
    """Counts, seed identities and request identities across all budgets and the main study."""
    main_seeds = {r['seed']: r['fields'] for r in read_jsonl(AUDIT/'complete_seed_manifest.jsonl')}
    seeds = dict(main_seeds); request_ids = {r['id'] for r in read_jsonl(PREPARED/'requests.jsonl')}
    main_request_ids = set(request_ids); all_requests = []; counts = {}
    for fraction in EXTRA_BUDGETS:
        out = folder(fraction)
        for r in read_json(out/'seed_manifest.json'):          # same seed value must mean the same stream
            assert seeds.setdefault(r['seed'], r['fields']) == r['fields'], ('seed collision', r)
        observations = {p.stem: read_json(p) for p in (out/'observations/sample').glob('*.json')}
        budgets = {k: budget_of(fraction, k) for k in MAIN_KEYS}
        expected = sum(draws_for(a, budgets[k]) for k in MAIN_KEYS for a in ARMS)
        tag = f'-b{round(fraction*1000):03d}__'
        assert len(observations) == expected and all(tag in oid for oid in observations)
        requests = read_jsonl(out/'requests.jsonl')
        assert len(requests) == expected*len(QWEN_CONFIGS)*LLM_REPEATS
        for r in requests:
            validate_request(r)
            assert r['config_id'] in QWEN_CONFIGS and r['observation_id'] in observations
            assert r['prompt_sha256'] == observations[r['observation_id']]['prompt_sha256']
            assert r['id'] not in request_ids, ('duplicate request id', r['id'])
            request_ids.add(r['id'])
        primary = read_json(out/'primary_baselines.json')['observations']
        assert set(primary) == set(observations)
        for fold in ('synthetic', 'sp_hospital'):
            m = read_json(out/'models_pooled'/fold/'manifest.json')
            assert all(tag in oid for oid in m['observations']) and fold not in m['real_sources']
        counts[fraction] = {'observations': len(observations), 'qwen_requests': len(requests),
                            'training_observations': len(list((out/'observations/training').glob('*.json')))}
        all_requests += requests
    assert not main_request_ids & {r['id'] for r in all_requests}
    run = fresh_directory(BUDGET_SENSITIVITY/'run')
    with open(run/'requests.jsonl', 'w') as f:
        for r in all_requests: f.write(json.dumps(r, separators=(',', ':'), allow_nan=False, sort_keys=True)+'\n')
    for fraction in EXTRA_BUDGETS:
        for p in (folder(fraction)/'observations/sample').glob('*.json'):
            (run/'observations/sample').mkdir(parents=True, exist_ok=True)
            (run/'observations/sample'/p.name).write_bytes(p.read_bytes())
    result = {'verified': True, 'design_version': DESIGN_VERSION, 'budgets': list(EXTRA_BUDGETS),
              'counts': {str(k): v for k, v in counts.items()}, 'qwen_requests': len(all_requests),
              'per_config': dict(Counter(r['config_id'] for r in all_requests)), 'seed_values_checked': len(seeds),
              'requests_sha256': sha(run/'requests.jsonl'), 'main_request_ids_reused': 0}
    write_json(run/'report.json', result)
    write_json(BUDGET_SENSITIVITY/'audit.json', result)
    print(json.dumps(result, indent=1))


# ---------------------------------------------------------------- evaluation after Qwen
def tidy_rows(fraction, references, requests, responses):
    """One row per observation x estimator (x repeat for Qwen) at one budget."""
    rows = []
    for path in sorted((folder(fraction)/'observations/sample').glob('*.json')):
        o = read_json(path); b = budget_of(fraction, o['graph_id'])
        base = {'budget': fraction, 'main_study_point': fraction == COVERAGE_FRACTION, 'graph_id': o['graph_id'],
                'evidence_block': next(s for s in STRATA if in_stratum(o['graph_id'], s)), 'stratum': o['stratum'],
                'arm': o['arm'], 'sample_index': o['sample_index'], 'observation_id': o['id'],
                'T': b['T'], 'expected_coverage': b[EXPECTED_CELLS[o['arm']]]/b['active_dyad_windows'],
                'observed_coverage': o['internal_evaluation']['observed_cell_fraction'],
                'arm_budget_matched': b['budget_matched_by_arm'][o['arm']], 'deterministic_draw': o['deterministic_draw']}
        for method in REFERENCE_METHODS:
            prediction = references[o['id']][method]['prediction']
            name = PRIMARY_REFERENCE_NAME[o['arm']] if method == 'primary_corrector' else ''
            rows.append({**base, 'method': method, 'method_detail': name, 'repeat_index': 0, 'request_id': '',
                         'status': 'reference', 'valid': True, **errors(prediction, o['truth'])})
        for config in QWEN_CONFIGS:
            for repeat in range(1, LLM_REPEATS+1):
                rid = requests[o['id'], config, repeat]
                outcome = resolve(parse(o['block']), responses.get(rid))
                rows.append({**base, 'method': config, 'method_detail': '', 'repeat_index': repeat, 'request_id': rid,
                             'status': outcome['status'], 'valid': bool(outcome['valid']),
                             **errors(outcome['prediction'], o['truth'])})
    return rows


def summarise(rows):
    """Equal-source conditional MAE per budget x block x arm x method, draw-clustered MCSE."""
    summary = []
    for fraction in BUDGET_GRID:
        for block in STRATA:
            for arm in ARMS:
                for method in (*REFERENCE_METHODS, *QWEN_CONFIGS):
                    group = [r for r in rows if r['budget'] == fraction and r['evidence_block'] == block
                             and r['arm'] == arm and r['method'] == method]
                    sources = sorted({r['graph_id'] for r in group})
                    draws = {s: max(r['sample_index'] for r in group if r['graph_id'] == s) for s in sources}
                    result = {'budget': fraction, 'evidence_block': block, 'arm': arm, 'method': method,
                              'sources': len(sources), 'planned': len(group),
                              'valid': sum(r['valid'] for r in group),
                              'missing': sum(r['status'] in ('not_started', 'in_progress') for r in group),
                              'empty_observations': sum(r['status'] == 'empty' for r in group),
                              'expected_coverage': float(np.mean([np.mean([r['expected_coverage'] for r in group if r['graph_id'] == s])
                                                                  for s in sources])),
                              'all_arm_budgets_matched': all(r['arm_budget_matched'] for r in group)}
                    for metric in ('AE2', 'ProfileAE', 'valid'):
                        cells = {s: np.full((draws[s], LLM_REPEATS), np.nan) for s in sources}
                        for r in group:
                            value = float(r['valid']) if metric == 'valid' else r[metric]
                            columns = slice(None) if r['repeat_index'] == 0 else r['repeat_index']-1
                            if value is not None: cells[r['graph_id']][r['sample_index']-1, columns] = value
                        if metric == 'valid' and result['missing']:
                            estimate = {'mean': None, 'mcse': None}
                        else:
                            estimate = (complete_summary if metric == 'valid' else conditional_summary)(cells, draws)
                        label = {'AE2': 'MAE2', 'ProfileAE': 'ProfileMAE', 'valid': 'valid_fraction'}[metric]
                        result[label] = estimate['mean']; result[label+'_MCSE'] = estimate['mcse']
                    summary.append(result)
    return summary


def plot(summary, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lines = {'plugin': 'plug-in', 'primary_corrector': 'primary reference', 'extratrees_pooled': 'ExtraTrees (pooled)',
             'qwen_thinking': 'Qwen thinking', 'qwen_nonthinking': 'Qwen non-thinking'}
    figures = [(block, metric) for block in STRATA for metric in ('MAE2',)]+[('real', 'ProfileMAE'), ('real', 'valid_fraction')]
    for block, metric in figures:
        fig, axes = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
        for ax, arm in zip(axes, ARMS):
            for method, label in lines.items():
                if metric == 'valid_fraction' and not method.startswith('qwen'): continue
                points = sorted((100*r['expected_coverage'], r[metric]) for r in summary
                                if r['evidence_block'] == block and r['arm'] == arm and r['method'] == method
                                and r[metric] is not None)
                if points: ax.plot(*zip(*points), marker='o', label=label)
            ax.axvline(10, color='grey', linestyle=':', linewidth=1)
            ax.set_title(f'{arm}'); ax.set_xlabel('expected dyad-window coverage (%)')
        axes[0].set_ylabel({'MAE2': 'MAE$_2$', 'ProfileMAE': 'ProfileMAE', 'valid_fraction': 'valid answers'}[metric])
        axes[0].text(10.5, axes[0].get_ylim()[1], 'main study (fixed 10%)', fontsize=7, va='top', color='grey')
        axes[-1].legend(fontsize=7)
        fig.suptitle(f'{block} sources: {metric} by budget (equal-source means)')
        fig.tight_layout(); fig.savefig(out/f'{metric}_{block}.png', dpi=150); plt.close(fig)


def evaluate():
    out = fresh_directory(BUDGET_SENSITIVITY/'evaluation')
    rows = []
    for fraction in BUDGET_GRID:
        main = fraction == COVERAGE_FRACTION
        references = read_json((REFERENCES if main else folder(fraction))/'primary_baselines.json')['observations']
        request_file = PREPARED/'requests.jsonl' if main else folder(fraction)/'requests.jsonl'
        requests = {(r['observation_id'], r['config_id'], r['repeat_index']): r['id']
                    for r in read_jsonl(request_file) if r['config_id'] in QWEN_CONFIGS}
        response_file = QWEN/'responses.jsonl' if main else BUDGET_SENSITIVITY/'qwen/responses.jsonl'
        responses = {r['id']: r for r in read_jsonl(response_file)}   # verified by verify_qwen_archive / collect
        rows += tidy_rows(fraction, references, requests, responses)
    ids = [r['request_id'] for r in rows if r['request_id']]
    assert len(ids) == len(set(ids)), 'a request is counted twice'
    write_csv(out/'tidy_results.csv', rows)
    summary = summarise(rows)
    write_csv(out/'summary.csv', summary)
    plot(summary, out)
    print('evaluation:', len(rows), 'rows;', len(summary), 'summary rows')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['prepare', 'feasibility', 'audit', 'evaluate'])
    parser.add_argument('budget', nargs='?', type=float)
    args = parser.parse_args()
    if args.stage == 'prepare': prepare(args.budget)
    else: {'feasibility': feasibility, 'audit': audit, 'evaluate': evaluate}[args.stage]()
