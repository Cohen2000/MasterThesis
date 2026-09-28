"""Nested ET selection and final fits per replicate, arm and outer fold.

The model, grid, anchors, block weights and selection score are those of
scripts/build_v10_et.py. Replicate 0 is production: the nine v11 folds reuse the
sealed choices, are refitted to verify reproduction, and predict new sources with
the pickled production model; the new nr_radoslaw_email fold is selected and
fitted on the frozen v11 caches with the v10 seed domains. Replicates k >= 1 use
their own training draws and the seed domains v10_et_nested_r<k>/v10_et_final_r<k>.
"""
import os
import pickle
import sys
import numpy as np
from main_experiment.common import REAL_TEST, ROOT, TRAIN, read_json, seed, write_json
from .core import RADOSLAW, RH, RH_ARMS, V10, stream_domain
from .replicates import v11_cache

sys.path.insert(0, str(ROOT/'scripts'))
import build_v10_et as v10et  # noqa: E402

V11_FOLDS = (*REAL_TEST, 'synthetic')
FOLDS = (*V11_FOLDS, RADOSLAW)


def cpus():
    return int(os.environ.get('SLURM_CPUS_PER_TASK', 4))


def training(k, arm, inputs):
    """(rows, X) of the ET training observations of one arm and replicate."""
    if k == 0:
        rows, X = v11_cache(arm)
        ids = [i for i, r in enumerate(rows) if r['arm'] == arm and r['domain'] in ('real_train', 'pool_train')]
        return [rows[i] for i in ids], np.asarray(X[ids], float)
    rows, Xs = [], []
    for name, folder in sorted(inputs.items()):
        if name.startswith(('draw_real:', 'draw_pool:')) and (folder/f'rows_{arm}.json').exists():
            rows += read_json(folder/f'rows_{arm}.json'); Xs.append(np.load(folder/f'X_{arm}.npy'))
    X = np.vstack(Xs)
    if len(rows) != len(X) or {r['source'] for r in rows if r['domain'] == 'real_train'} != set(TRAIN):
        raise ValueError(f'replicate {k}/{arm}: incomplete training draws')
    return rows, X


def select(task, out, inputs):
    k, arm, outer = task.params['k'], task.params['arm'], task.params['fold']
    rows, X = training(k, arm, inputs)
    pool_ids = np.array([i for i, r in enumerate(rows) if r['domain'] == 'pool_train'], dtype=int)
    real_ids = np.array([i for i, r in enumerate(rows) if r['domain'] == 'real_train'], dtype=int)
    inner_sources = [s for s in TRAIN if s != outer]
    candidates = []
    for anchor, leaf, maxfeat in v10et.GRID:
        scores = []
        for inner in inner_sources:
            train_ids = np.concatenate([pool_ids, np.array(
                [i for i in real_ids if rows[i]['source'] not in (outer, inner)], dtype=int)])
            valid_ids = np.array([i for i in real_ids if rows[i]['source'] == inner], dtype=int)
            if not len(train_ids) or not len(valid_ids): raise ValueError('incomplete nested real-source CV')
            x_train, truth, base = v10et.arrays(rows, X, train_ids, anchor)
            x_valid, _, valid_base = v10et.arrays(rows, X, valid_ids, anchor)
            config = f'{anchor}:{leaf}:{int(maxfeat*100)}'
            m = v10et.model(leaf, maxfeat, seed(stream_domain('et_nested', k), arm, outer, inner, config) % 2**32, cpus())
            m.fit(x_train, truth-base, sample_weight=v10et.block_weights(rows, train_ids))
            scores.append(v10et.graph_mae(rows, valid_ids, valid_base+m.predict(x_valid))[0])
        candidates.append({'anchor': anchor, 'min_samples_leaf': leaf, 'max_features': maxfeat,
                           'mean_inner_source_MAE2': float(np.mean(scores)),
                           'inner_source_scores': dict(zip(inner_sources, map(float, scores)))})
    best = min(candidates, key=lambda r: (r['mean_inner_source_MAE2'], r['anchor'],
                                          r['min_samples_leaf'], r['max_features']))
    write_json(out/'choice.json', {'replicate': k, 'arm': arm, 'outer_fold': outer, 'selected': best,
                                   'candidates': candidates,
                                   'selection': 'leave-one-real-training-source-out; pooled composition and block weights'})
    print('ET_SELECT', k, arm, outer, best['anchor'], best['min_samples_leaf'], best['max_features'], flush=True)


def train(task, out, inputs):
    k, arm, fold = task.params['k'], task.params['arm'], task.params['fold']
    production = k == 0 and fold in V11_FOLDS
    folder = RH/'et_run/et' if arm in RH_ARMS else V10/'et'
    if production:
        choice = read_json(folder/'choices'/arm/f'{fold}.json')['selected']
    else:
        choice = read_json(inputs[f'select:{k}:{arm}:{fold}']/'choice.json')['selected']
    rows, X = training(k, arm, inputs)
    train_ids = np.array([i for i, r in enumerate(rows) if r['domain'] == 'pool_train' or
                          (r['domain'] == 'real_train' and r['source'] != fold)], dtype=int)
    if fold in (*REAL_TEST, RADOSLAW) and any(rows[i]['source'] == fold for i in train_ids):
        raise AssertionError('test source entered training')
    test_rows = read_json(inputs['testset']/f'rows_{arm}.json')
    test_X = np.load(inputs['testset']/f'X_{arm}.npy')
    test_ids = np.array([i for i, r in enumerate(test_rows) if r['fold'] == fold], dtype=int)
    if not len(train_ids) or not len(test_ids): raise ValueError('missing fold rows')
    anchor = choice['anchor']
    x_train, truth, base = v10et.arrays(rows, X, train_ids, anchor)
    x_test, _, test_base = v10et.arrays(test_rows, test_X, test_ids, anchor)
    m = v10et.model(choice['min_samples_leaf'], choice['max_features'],
                    seed(stream_domain('et_final', k), arm, fold) % 2**32, cpus())
    m.fit(x_train, truth-base, sample_weight=v10et.block_weights(rows, train_ids))
    refit = test_base+m.predict(x_test)
    ids = [test_rows[i]['id'] for i in test_ids]
    group = [test_rows[i]['group'] for i in test_ids]
    meta = {'replicate': k, 'arm': arm, 'fold': fold, 'choice': choice, 'n_train': int(len(train_ids)),
            'train_sources': sorted({rows[i]['source'] for i in train_ids if rows[i]['domain'] == 'real_train'})}
    if production:
        stored = {r['id']: r['prediction'] for r in read_json(folder/'models'/arm/fold/'predictions.json')['observations']}
        with open(folder/'models'/arm/fold/'model.pkl', 'rb') as f: sealed_model = pickle.load(f)
        sealed = test_base+sealed_model.predict(x_test)
        old = [i for i, g in enumerate(group) if g == 'v11']
        if set(stored) != {ids[i] for i in old}: raise ValueError(f'{arm}/{fold}: v11 ET prediction set differs')
        stored_arr = np.array([stored[ids[i]] for i in old])
        meta['verification'] = {
            'v11_rows': len(old),
            'refit_vs_v11_max_abs': float(np.max(np.abs(refit[old]-stored_arr))),
            'sealed_model_vs_v11_max_abs': float(np.max(np.abs(sealed[old]-stored_arr))),
            'refit_vs_sealed_model_new_rows_max_abs': float(np.max(np.abs(refit-sealed))) if len(ids) > len(old) else None}
        # v11 rows keep their sealed numbers; new rows use the pickled production model.
        prediction = [stored[ids[i]] if group[i] == 'v11' else sealed[i].tolist() for i in range(len(ids))]
        print('ET_REP0', arm, fold, meta['verification'], flush=True)
    else:
        prediction = refit.tolist()
    write_json(out/'predictions.json', {**meta, 'observations': [
        {'id': i, 'group': g, 'prediction': list(map(float, p))} for i, g, p in zip(ids, group, prediction)]})
