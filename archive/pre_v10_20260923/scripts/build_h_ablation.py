#!/usr/bin/env python3
"""History-fraction ablation of arm H over the full coverage grid (cluster, CPU).

For one h (the main study is h=.60): recalibrate only H's panel size with
sampling.h_parameters on the sealed graphs, redraw H with the unchanged stream
(sampler identity keyed by coverage only, so node panels are common random
numbers with the main H arm), and write
  - test observations per level and the Qwen H request manifest (qwen_run/),
  - the feasibility table,
  - mixed-budget H ExtraTrees (build_mixed_budget_et.fit_one, i.e. the final v9
    protocol, trained on the H training draws of all levels) scored on the tests.
Observation ids carry '-h040' etc. in the sampler part, so seeds and request ids
never collide with the main study. With --h .6 every test block must equal the
sealed observation byte for byte (a check of the redraw logic).

usage: build_h_ablation.py --h 0.4 [--workers N] [--smoke]
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_mixed_budget_et as et                                                  # noqa: E402
from main_experiment.common import (BUDGET_GRID, BUDGET_TOLERANCE, H_FRACTION, PREPARED, QWEN_CONFIGS, REFERENCES, ROOT, SURROGATES,
                                    SYNTH, TRAIN, DESIGN_VERSION, draws_for, fold_for, read_json, write_csv, write_json)
from main_experiment.data import load_graph
from main_experiment.pool import pool_definition, regenerate
from main_experiment.prepare import domains_of, observation_row
from main_experiment.requests import planned
from main_experiment.sampling import draw, h_parameters
from main_experiment.training import FOLDS

H = None   # set in main; read by forked workers


def h_budget(g, base, h):
    b = {**base, **h_parameters(g, base['T'], h)}
    matched = dict(base['budget_matched_by_arm'], H=bool(b['h_within_tolerance']))
    return {**b, 'budget_matched_by_arm': matched, 'budget_matched': all(matched.values())}


def relabel(oid):
    graph, sampler, index = oid.rsplit('__', 2)
    return oid if H == H_FRACTION else f'{graph}__{sampler}-h{round(H*100):03d}__{index}'


def graph_rows(key):
    """Test and real-training H draws of one sealed graph, all levels."""
    g = load_graph(PREPARED/'graphs'/key); tests = []; train = []; feas = []
    for fraction in BUDGET_GRID:
        base = read_json(et.level_dir(fraction)/'calibration'/f'{key}.json')
        b = h_budget(g, base, H)
        feas.append({'h': H, 'budget': fraction, 'graph_id': key, 'T': b['T'], 'J_total': b['J_total'], 'N': g.N,
                     'n_panel_history': b['n_panel_history'], 'h_relative_budget_error': b['h_relative_budget_error'],
                     'h_within_tolerance': b['h_within_tolerance'], 'h_target_unreachable': b['h_target_unreachable'],
                     'h_saturated': b['h_saturated']})
        for domain in domains_of(key):
            for index in range(1, draws_for('H', b, domain)+1):
                row = observation_row(g, 'H', index, domain, b, draw(g, 'H', index, domain, b)[0], None)
                if H == H_FRACTION and domain == 'sample':
                    sealed = read_json(et.level_dir(fraction)/'observations/sample'/f"{row['id']}.json")
                    if sealed['block'] != row['block']: raise AssertionError(f"redraw differs: {row['id']}")
                row['id'] = relabel(row['id']); row['budget'] = fraction; row['h'] = H
                (tests if domain == 'sample' else train).append(row)
    return tests, [{'arm': 'H', 'block': r['block'], 'source_family': key, 'block_group': 'real', 'empty': r['empty'],
                    'budget': r['budget'], 'truth': r['truth']} for r in train], feas


def pool_rows(spec):
    g = regenerate(spec); rows = []
    for fraction in BUDGET_GRID:
        pool = REFERENCES/'pool' if fraction == .10 else et.level_dir(fraction)/'pool'
        b = h_budget(g, read_json(pool/'observations'/f"{spec['key']}.json")['budget'], H)
        for index in range(1, draws_for('H', b, 'pool_train')+1):
            row = observation_row(g, 'H', index, 'pool_train', b, draw(g, 'H', index, 'pool_train', b)[0], None)
            rows.append({'arm': 'H', 'block': row['block'], 'source_family': spec['key'], 'block_group': spec['family'],
                         'empty': row['empty'], 'budget': fraction, 'truth': g.truth})
    return rows


def main():
    global H
    ap = argparse.ArgumentParser()
    ap.add_argument('--h', type=float, required=True)
    ap.add_argument('--workers', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args(); H = a.h
    out = ROOT/'results/panel888_h_ablation'/f'h{round(H*100):03d}'
    keys = list(dict.fromkeys((*TRAIN, *SYNTH, *SURROGATES)))
    specs = [s for s in pool_definition()['graphs'] if s['partition'] == 'train']
    if a.smoke: specs = specs[:8]
    ctx = mp.get_context('fork')
    with ctx.Pool(a.workers) as pool:
        per_graph = pool.map(graph_rows, keys, chunksize=1)
        pool_train = sum(pool.map(pool_rows, specs, chunksize=1), [])
    tests = [r for t, _, _ in per_graph for r in t]
    train = [r for _, tr, _ in per_graph for r in tr]+pool_train
    feas = [f for _, _, fs in per_graph for f in fs]
    write_csv(out/'feasibility.csv', feas)
    for r in tests: write_json(out/'observations'/f"b{round(r['budget']*1000):03d}"/f"{r['id']}.json", r)
    print(f'h={H}: {len(tests)} test observations, {len(train)} training rows', flush=True)

    requests = [r for r in planned(tests) if r['config_id'] in QWEN_CONFIGS]
    run = out/'qwen_run'
    for r in tests: write_json(run/'observations/sample'/f"{r['id']}.json", r)
    (run/'requests.jsonl').write_text(''.join(json.dumps(r, separators=(',', ':'), sort_keys=True)+'\n' for r in requests))
    write_json(run/'report.json', {'design_version': DESIGN_VERSION, 'h': H, 'requests': len(requests), 'verified': True})

    train = [r for r in train if not r['empty']]
    evaluable = [r for r in tests if not r['empty']]
    truth = {r['source_family']: np.asarray(r['truth'], float) for r in train}
    params = dict(et.PARAMETERS)
    if a.smoke: params['n_estimators'] = 3
    with ctx.Pool(a.workers) as pool:
        X = np.array(pool.map(et.featurize, [r['block'] for r in train], chunksize=64))
        Xt = np.array(pool.map(et.featurize, [r['block'] for r in evaluable], chunksize=16))
    et.G.update(train=train, truth=truth, X=X, params=params, out=out/'et', Xt=Xt,
                tests=[{**r, 'fold': fold_for(r['graph_id'])} for r in evaluable])
    results = []
    with ctx.Pool(min(a.workers, len(FOLDS))) as pool:
        for _, fold, n, rows in pool.imap_unordered(et.fit_one, [('H', f) for f in FOLDS]):
            results += rows; print(f'fitted H/{fold}: {n} rows', flush=True)
    write_csv(out/'et/evaluation.csv', sorted(results, key=lambda r: (r['budget'], r['graph_id'])))
    if len(results) != len(evaluable): raise AssertionError('missing ET predictions')
    write_json(out/'report.json', {'h': H, 'test_observations': len(tests), 'evaluable': len(evaluable),
                                   'training_rows': len(train), 'qwen_requests': len(requests), 'smoke': a.smoke,
                                   'budget_tolerance': BUDGET_TOLERANCE})
    print('done', flush=True)


if __name__ == '__main__':
    main()
