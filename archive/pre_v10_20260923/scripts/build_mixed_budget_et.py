#!/usr/bin/env python3
"""Fit arm-specific released-feature residual forests over the full budget grid,
and score them on Main + every budget level in the same run.

Per (arm, fold): training rows = released training draws of the real sources
(held-out source excluded) and the synthetic training pool, pooled over all
seven coverage levels; block/source weights as below; target = truth - anchor;
forest = training.PARAMETERS. Budget is not a feature.

Features and anchors do not depend on the fold, so they are computed once per
row (in parallel) instead of once per fold; the 36 (arm, fold) forests are
fitted in parallel, and each forest predicts its own test observations right
after fitting, so the ~18 GB of forests never have to be reloaded.

usage: build_mixed_budget_et.py [--out DIR] [--workers N] [--smoke]
"""
import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARMS, BUDGET_GRID, BUDGET_SENSITIVITY, COVERAGE_FRACTION, PREPARED, REAL_TEST,
                                    REFERENCES, ROOT, TRAIN, fold_for, read_json, sha, write_csv, write_json)
from main_experiment.observation import FEATURE_NAMES, features, parse
from main_experiment.training import FOLDS, PARAMETERS

BLOCK_WEIGHTS = {'real': .50, 'dar': .25, 'ad': .25}
ANCHOR = [FEATURE_NAMES.index(f'anchor_rho_{k}') for k in range(2, 6)]
G = {}   # shared with forked workers (copy-on-write)


def level_dir(fraction):
    return PREPARED if fraction == COVERAGE_FRACTION else BUDGET_SENSITIVITY/f'b{round(fraction*1000):03d}'


def pool_dir(fraction):
    # The main study's pool lives with the references, the budget levels' pools with their level.
    return REFERENCES/'pool' if fraction == COVERAGE_FRACTION else level_dir(fraction)/'pool'


def load_training():
    rows = []; truth = {}
    for fraction in BUDGET_GRID:
        for p in sorted((level_dir(fraction)/'observations/training').glob('*.json')):
            r = read_json(p); truth[r['source_family']] = r['truth']
            rows.append({'arm': r['arm'], 'block': r['block'], 'source_family': r['source_family'],
                         'block_group': 'real', 'empty': r['empty'], 'budget': fraction})
        pool_files = sorted((pool_dir(fraction)/'observations').glob('*.json'))
        if not pool_files: raise FileNotFoundError(f'no training pool for {fraction}: {pool_dir(fraction)}')
        for p in pool_files:
            g = read_json(p)
            if g['partition'] != 'train': continue
            truth[g['key']] = g['truth']
            rows += [{'arm': r['arm'], 'block': r['block'], 'source_family': g['key'], 'block_group': g['family'],
                      'empty': r['empty'], 'budget': fraction} for r in g['observations']]
    return [r for r in rows if not r['empty']], truth


def load_tests():
    rows = []
    for fraction in BUDGET_GRID:
        for p in sorted((level_dir(fraction)/'observations/sample').glob('*.json')):
            r = read_json(p)
            if r['empty']: continue
            rows.append({'budget': fraction, 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                         'source_family': r['source_family'], 'fold': fold_for(r['graph_id']),
                         'block': r['block'], 'truth': r['truth']})
    return rows


def featurize(block):
    return features(parse(block))


def fit_one(task):
    arm, fold = task
    allowed = set(TRAIN)-({fold} if fold in REAL_TEST else set())
    idx = [i for i, r in enumerate(G['train']) if r['arm'] == arm and
           (r['block_group'] != 'real' or r['source_family'] in allowed)]
    if not idx: raise ValueError(f'no rows for {arm}/{fold}')
    selected = [G['train'][i] for i in idx]
    blocks = {}
    for r in selected: blocks.setdefault(r['block_group'], set()).add(r['source_family'])
    shares = {b: BLOCK_WEIGHTS[b]/len(gs) for b, gs in blocks.items()}
    counts = {}
    for r in selected: counts[r['source_family']] = counts.get(r['source_family'], 0)+1
    w = np.array([shares[r['block_group']]/counts[r['source_family']] for r in selected]); w /= w.sum()
    X = G['X'][idx]
    y = np.array([G['truth'][r['source_family']] for r in selected])-X[:, ANCHOR]
    model = ExtraTreesRegressor(**G['params']).fit(X, y, sample_weight=w)
    d = G['out']/'models'/arm/fold; d.mkdir(parents=True, exist_ok=True)
    with open(d/'model.pkl', 'wb') as f: pickle.dump(model, f, protocol=5)
    write_json(d/'manifest.json', {'arm': arm, 'fold': fold, 'coverage_grid': list(BUDGET_GRID),
               'feature_names': FEATURE_NAMES, 'n_features': len(FEATURE_NAMES), 'training_rows': len(selected),
               'rows_per_budget': {str(b): sum(r['budget'] == b for r in selected) for b in BUDGET_GRID},
               'sources': sorted({r['source_family'] for r in selected}), 'test_source': fold,
               'weights_sum': float(w.sum()), 'model_sha256': sha(d/'model.pkl'), 'parameters': G['params']})
    tests = [i for i, t in enumerate(G['tests']) if t['arm'] == arm and t['fold'] == fold]
    out = []
    if tests:
        Xt = G['Xt'][tests]
        pred = Xt[:, ANCHOR]+model.predict(Xt)
        for i, p in zip(tests, pred):
            t = G['tests'][i]; e = p-np.asarray(t['truth'])
            out.append({'budget': t['budget'], 'graph_id': t['graph_id'], 'arm': arm, 'stratum': t['stratum'],
                        'source_family': t['source_family'], 'fold': fold,
                        'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0]),
                        'prediction': [float(x) for x in p],
                        'in_unit_interval': bool(np.all((p >= 0) & (p <= 1))),
                        'monotone': bool(np.all(np.diff(p) <= 0))})
    return arm, fold, len(selected), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=ROOT/'results/panel888_et_mixed')
    ap.add_argument('--workers', type=int, default=len(os.sched_getaffinity(0)))
    ap.add_argument('--smoke', action='store_true', help='tiny forests on a row subsample; code-path check only')
    a = ap.parse_args()
    train, truth = load_training(); tests = load_tests()
    params = dict(PARAMETERS)
    if a.smoke:
        import inspect
        accepted = inspect.signature(ExtraTreesRegressor).parameters   # older local sklearn lacks e.g. monotonic_cst
        train = train[::25]; params = {k: v for k, v in params.items() if k in accepted}; params['n_estimators'] = 3
    print(f'training rows {len(train)}, test rows {len(tests)}, workers {a.workers}', flush=True)
    ctx = mp.get_context('fork')
    with ctx.Pool(a.workers) as pool:
        X = np.array(pool.map(featurize, [r['block'] for r in train], chunksize=64))
        Xt = np.array(pool.map(featurize, [t['block'] for t in tests], chunksize=16))
    print('features done', flush=True)
    G.update(train=train, truth={k: np.asarray(v, float) for k, v in truth.items()}, X=X, tests=tests, Xt=Xt,
             params=params, out=a.out)
    tasks = [(arm, fold) for arm in ARMS for fold in FOLDS]
    results = []
    with ctx.Pool(min(a.workers, len(tasks))) as pool:
        for arm, fold, n, rows in pool.imap_unordered(fit_one, tasks):
            results += rows
            print(f'fitted {arm}/{fold}: {n} training rows, {len(rows)} test predictions', flush=True)
    results.sort(key=lambda r: (r['budget'], r['arm'], r['graph_id']))
    write_csv(a.out/'evaluation.csv', results)
    expected = len(tests)
    write_json(a.out/'report.json', {'coverage_grid': list(BUDGET_GRID), 'arms': list(ARMS), 'folds': list(FOLDS),
               'training_rows': len(train), 'test_rows': expected, 'predictions': len(results),
               'budget_feature_excluded': True, 'smoke': a.smoke, 'n_estimators': params['n_estimators']})
    if len(results) != expected: raise AssertionError(f'{len(results)} predictions for {expected} test rows')
    print(f'done: {len(results)} predictions', flush=True)


if __name__ == '__main__':
    main()
