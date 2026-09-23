#!/usr/bin/env python3
"""Recompute every ET prediction from its serialized block and saved forest."""
import pickle
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.baselines import plugin
from main_experiment.common import PREPARED, RESULTS, REAL_TEST, fold_for, read_json, sha, write_json
from main_experiment.observation import FEATURE_NAMES, features, parse

out = RESULTS / 'et'
observations = {r['id']: r for p in (PREPARED / 'observations/sample').glob('*.json')
                for r in [read_json(p)]}
ref_start = FEATURE_NAMES.index('anchor_rho_2')
checked = set(); models = {}
for path in sorted((out / 'models').glob('*/*/predictions.json')):
    saved = read_json(path)
    arm, fold = saved['arm'], saved['fold']
    if fold in REAL_TEST and fold in saved['train_sources']:
        raise AssertionError('held-out source entered ET training')
    with (path.parent / 'model.pkl').open('rb') as f: forest = pickle.load(f)
    models[f'{arm}/{fold}'] = sha(path.parent / 'model.pkl')
    anchor = saved['choice']['anchor']
    for row in saved['observations']:
        oid = row['id']
        if oid in checked: raise AssertionError('duplicate ET prediction')
        obs = observations[oid]
        if obs['arm'] != arm or fold_for(obs['graph_id']) != fold:
            raise AssertionError('wrong ET fold')
        parsed = parse(obs['block'])
        x = features(parsed)
        base = plugin(parsed) if anchor == 'plugin' else x[ref_start:ref_start+4]
        prediction = np.asarray(base) + forest.predict(x[None, :])[0]
        np.testing.assert_allclose(prediction, row['prediction'], atol=1e-12, rtol=1e-12)
        checked.add(oid)
if set(observations) != checked or len(models) != 45:
    raise AssertionError('missing ET prediction or forest')
write_json(out / 'ET_REPRODUCIBILITY.json', {'verified': True, 'observations': len(checked),
                                            'models': models,
                                            'input': 'serialized block and saved forest only'})
print('ET_REPRODUCED', len(checked), len(models))
