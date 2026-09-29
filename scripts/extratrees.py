#!/usr/bin/env python3
"""Arm-specific pooled ExtraTrees at 10%, with nested real-source selection."""
# In plain words: the learned baseline. For every sampling arm one ExtraTrees forest learns
# to correct a simple starting estimate ('anchor': the plugin value or the arm's reference
# estimate) using features of the observation block. Training data: observations of the
# 16 real training sources plus the synthetic pool, whose true answers are known.
# Three steps, each run as a cluster job:
#   cache   compute the feature table of every observation once
#   select  pick the forest settings by nested leave-one-source-out validation
#   train   fit the chosen setting and predict the test observations of the held-out source
# The seed labels 'v10_et_nested' / 'v10_et_final' are fixed inputs of the random streams.
import argparse
import json
import os
import pickle
import sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.baselines import plugin
from main_experiment.common import (ARMS, PREPARED, REAL_TEST, REFERENCES, RESULTS,
                                    TRAIN, fold_for, read_json, seed, write_json)
from main_experiment.observation import FEATURE_NAMES, features, parse

# Output folder, the cross-validation folds (one per real test source + all synthetic graphs) and the settings grid.
OUT = RESULTS / 'et'
FOLDS = (*REAL_TEST, 'synthetic')
GRID = [(anchor, leaf, maxfeat) for anchor in ('plugin', 'reference')
        for leaf in (1, 5, 20) for maxfeat in (.5, 1.)]
REF_START = FEATURE_NAMES.index('anchor_rho_2')


# Every observation that can enter training or testing, with its truth and block (real, pool or test).
def source_rows():
    for path in sorted((PREPARED / 'observations/training').glob('*.json')):
        r = read_json(path)
        yield r, 'real_train', r['truth'], 'real'
    for path in sorted((REFERENCES / 'pool/observations').glob('*.json')):
        graph = read_json(path)
        domain = 'pool_' + graph['partition']
        for r in graph['observations']:
            yield r, domain, graph['truth'], graph['family']
    for path in sorted((PREPARED / 'observations/sample').glob('*.json')):
        r = read_json(path)
        yield r, 'main', r['truth'], r['stratum']


# Step 'cache': features of every non-empty observation, stored as one matrix plus a row index.
def cache():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []; X = []
    for i, (r, domain, truth, family) in enumerate(source_rows(), 1):
        o = parse(r['block'])
        if o['D_obs'] == 0: continue
        x = features(o)
        ref = x[REF_START:REF_START + 4].tolist()
        rows.append({'id': r['id'], 'source': r['source_family'], 'arm': r['arm'],
                     'domain': domain, 'family': family, 'fold': fold_for(r['source_family']),
                     'truth': truth, 'plugin': plugin(o), 'reference': ref})
        X.append(x)
        if i % 1000 == 0: print('features', i, flush=True)
    np.save(OUT / 'features.npy', np.asarray(X, dtype=np.float64))
    write_json(OUT / 'rows.json', rows)
    print('ET_CACHE', len(rows), len(FEATURE_NAMES), flush=True)


def load():
    return read_json(OUT / 'rows.json'), np.load(OUT / 'features.npy', mmap_mode='r')


# Row numbers of one arm and one kind of data (real_train, pool_train, main = test).
def indices(rows, arm, domain):
    return np.array([i for i, r in enumerate(rows) if r['arm'] == arm and r['domain'] == domain], dtype=int)


# Features, true profiles and the anchor estimate for the chosen rows.
def arrays(rows, X, ids, anchor):
    sub = [rows[i] for i in ids]
    truth = np.array([r['truth'] for r in sub], float)
    baseline = np.array([r[anchor] for r in sub], float)
    return np.asarray(X[ids]), truth, baseline


# 300 fully random trees; the forest predicts the correction truth - anchor.
def model(leaf, maxfeat, random_state, jobs):
    return ExtraTreesRegressor(n_estimators=300, criterion='squared_error',
                               min_samples_leaf=leaf, max_features=maxfeat,
                               bootstrap=False, random_state=random_state, n_jobs=jobs)


# Error per graph first, then averaged over graphs (every graph counts equally).
def graph_mae(rows, ids, pred):
    graph = {}
    for i, p in zip(ids, pred):
        r = rows[i]
        e = np.abs(p - np.asarray(r['truth']))
        graph.setdefault(r['source'], []).append(e)
    errors = np.array([np.mean(x, axis=0) for x in graph.values()])
    return float(errors[:, 0].mean()), float(errors.mean())


# Step 'select' for one (arm, outer fold). For every setting of the grid, and for every
# remaining real training source in turn: train on the rest, predict that source, score it.
# The setting with the lowest mean score wins (ties broken by a fixed order).
def select_one(index):
    rows, X = load()
    arm = ARMS[index // len(FOLDS)]
    outer_fold = FOLDS[index % len(FOLDS)]
    inner_sources = [s for s in TRAIN if s != outer_fold]
    pool_ids = indices(rows, arm, 'pool_train')
    real_ids = indices(rows, arm, 'real_train')
    candidates = []
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    for anchor, leaf, maxfeat in GRID:
        source_scores = []
        for inner_source in inner_sources:
            train_ids = np.concatenate([pool_ids, np.array([
                i for i in real_ids if rows[i]['source'] not in (outer_fold, inner_source)], dtype=int)])
            valid_ids = np.array([i for i in real_ids if rows[i]['source'] == inner_source], dtype=int)
            if not len(train_ids) or not len(valid_ids): raise ValueError('incomplete nested real-source CV')
            x_train, truth, base = arrays(rows, X, train_ids, anchor)
            x_valid, _, valid_base = arrays(rows, X, valid_ids, anchor)
            config = f'{anchor}:{leaf}:{int(maxfeat * 100)}'
            m = model(leaf, maxfeat, seed('v10_et_nested', arm, outer_fold, inner_source,
                                          config) % 2**32, cpus)
            m.fit(x_train, truth - base, sample_weight=block_weights(rows, train_ids))
            pred = valid_base + m.predict(x_valid)
            score, _ = graph_mae(rows, valid_ids, pred)
            source_scores.append(score)
        candidates.append({'anchor': anchor, 'min_samples_leaf': leaf,
                           'max_features': maxfeat, 'mean_inner_source_MAE2': float(np.mean(source_scores)),
                           'inner_source_scores': dict(zip(inner_sources, source_scores))})
        print('ET_CV', arm, outer_fold, anchor, leaf, maxfeat, np.mean(source_scores), flush=True)
    best = min(candidates, key=lambda r: (r['mean_inner_source_MAE2'], r['anchor'],
                                          r['min_samples_leaf'], r['max_features']))
    result = {'arm': arm, 'outer_fold': outer_fold, 'selected': best, 'candidates': candidates,
              'selection': 'leave-one-real-training-source-out; pooled composition and block weights'}
    folder = OUT / 'choices' / arm
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder / f'{outer_fold}.json', result)


# Sample weights so that real sources, DAR graphs and AD graphs weigh 50/25/25 percent.
def block_weights(rows, ids):
    # Equal graph weight within each block; real/DAR/AD contribute .50/.25/.25.
    block_weight = {'real': .5, 'dar': .25, 'ad': .25}
    sources = {b: {rows[i]['source'] for i in ids if rows[i]['family'] == b}
               for b in block_weight}
    counts = {}
    for i in ids:
        r = rows[i]
        key = (r['family'], r['source'])
        counts[key] = counts.get(key, 0) + 1
    w = np.array([block_weight[rows[i]['family']] /
                  len(sources[rows[i]['family']]) /
                  counts[(rows[i]['family'], rows[i]['source'])] for i in ids])
    return w / w.sum()


# Step 'train' for one (arm, fold): fit on all training rows that do not belong to the
# held-out source, predict its test observations and save the model and the predictions.
def train_one(index):
    rows, X = load()
    arm = ARMS[index // len(FOLDS)]
    fold = FOLDS[index % len(FOLDS)]
    choice = read_json(OUT / 'choices' / arm / f'{fold}.json')['selected']
    anchor = choice['anchor']
    train_ids = np.array([i for i, r in enumerate(rows)
                          if r['arm'] == arm and (r['domain'] == 'pool_train' or
                           (r['domain'] == 'real_train' and r['source'] != fold))], dtype=int)
    test_ids = np.array([i for i, r in enumerate(rows)
                         if r['arm'] == arm and r['domain'] == 'main' and r['fold'] == fold], dtype=int)
    if not len(train_ids) or not len(test_ids): raise ValueError('missing fold rows')
    if fold in REAL_TEST and any(rows[i]['source'] == fold for i in train_ids):
        raise AssertionError('test source entered training')
    x_train, truth, base = arrays(rows, X, train_ids, anchor)
    x_test, _, test_base = arrays(rows, X, test_ids, anchor)
    m = model(choice['min_samples_leaf'], choice['max_features'],
              seed('v10_et_final', arm, fold) % 2**32, int(os.environ.get('SLURM_CPUS_PER_TASK', 4)))
    m.fit(x_train, truth - base, sample_weight=block_weights(rows, train_ids))
    pred = test_base + m.predict(x_test)
    folder = OUT / 'models' / arm / fold
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'model.pkl').open('wb') as f: pickle.dump(m, f, protocol=5)
    write_json(folder / 'predictions.json', {'arm': arm, 'fold': fold, 'choice': choice,
                                             'train_sources': sorted({rows[i]['source'] for i in train_ids}),
                                             'observations': [{'id': rows[i]['id'], 'prediction': p.tolist()}
                                                              for i, p in zip(test_ids, pred)]})
    print('ET_FOLD', arm, fold, len(train_ids), len(test_ids), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=('cache', 'select', 'train'))
    ap.add_argument('--index', type=int)
    a = ap.parse_args()
    if a.stage == 'cache': cache()
    elif a.stage == 'select':
        if a.index is None or not 0 <= a.index < len(ARMS) * len(FOLDS): raise ValueError('index')
        select_one(a.index)
    else:
        if a.index is None or not 0 <= a.index < len(ARMS) * len(FOLDS): raise ValueError('index')
        train_one(a.index)


if __name__ == '__main__': main()
