#!/usr/bin/env python3
"""Fit arm-specific released-feature residual forests over the full budget grid."""
import json, pickle
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import ARMS, REAL_TEST, TRAIN, BUDGET_GRID, PREPARED, BUDGET_SENSITIVITY, read_json, sha, write_json
from main_experiment.observation import FEATURE_NAMES, features, parse
from main_experiment.baselines import anchor_profile
from main_experiment.training import FOLDS, PARAMETERS

def rows(path, group):
    out=[]
    for p in sorted((Path(path)/'observations/training').glob('*.json')):
        r=read_json(p); r['block_group']=group; out.append(r)
    return out

def pool_rows(path):
    out=[]; truth={}
    for p in sorted((Path(path)/'pool/observations').glob('*.json')):
        g=read_json(p); truth[g['key']]=g['truth']
        out.extend([{**r,'block_group':g['family'],'source_family':g['key']} for r in g['observations']])
    return out,truth

def fit(selected, truth, fold, arm, out):
    allowed=set(TRAIN)-({fold} if fold in REAL_TEST else set())
    selected=[r for r in selected if r['arm']==arm and
              (r['block_group'] != 'real' or r['source_family'] in allowed)]
    if not selected: raise ValueError(f'no rows for {arm}/{fold}')
    # Equalize source families, blocks, and observations within this arm.
    blocks={}
    for r in selected: blocks.setdefault(r['block_group'], set()).add(r['source_family'])
    bw={'real':.50,'dar':.25,'ad':.25}
    shares={b:bw[b]/len(gs) for b,gs in blocks.items()}
    counts={}
    for r in selected: counts[r['source_family']]=counts.get(r['source_family'],0)+1
    w=np.array([shares[r['block_group']]/counts[r['source_family']] for r in selected]); w/=w.sum()
    X=np.array([features(parse(r['block'])) for r in selected])
    y=np.array([np.asarray(truth[r['source_family']])-anchor_profile(parse(r['block'])) for r in selected])
    model=ExtraTreesRegressor(**PARAMETERS).fit(X,y,sample_weight=w)
    d=Path(out)/arm/fold; d.mkdir(parents=True,exist_ok=True)
    with open(d/'model.pkl','wb') as f: pickle.dump(model,f,protocol=5)
    write_json(d/'manifest.json',{'arm':arm,'fold':fold,'coverage_grid':list(BUDGET_GRID),
      'feature_names':FEATURE_NAMES,'n_features':len(FEATURE_NAMES),'training_rows':len(selected),
      'sources':sorted({r['source_family'] for r in selected}),'test_source':fold,
      'weights':w.tolist(),'model_sha256':sha(d/'model.pkl'),'parameters':PARAMETERS})

def main():
    root=Path(BUDGET_SENSITIVITY).parent/'panel888_et_mixed'
    root.mkdir(parents=True,exist_ok=True)
    allrows=[]; truth={}
    sources=[PREPARED]+[BUDGET_SENSITIVITY/f'b{round(b*1000):03d}' for b in BUDGET_GRID if b != .10]
    for s in sources:
        rs=rows(s,'real'); allrows.extend(rs)
        for r in rs: truth[r['source_family']]=r['truth']
        ps,pt=pool_rows(s); allrows.extend(ps); truth.update(pt)
    if len({r['block_group'] for r in allrows}) < 1: raise ValueError('missing training rows')
    for arm in ARMS:
        for fold in FOLDS: fit(allrows,truth,fold,arm,root/'models')
    write_json(root/'report.json',{'coverage_grid':list(BUDGET_GRID),'arms':list(ARMS),'folds':list(FOLDS),
      'training_rows':len(allrows),'features':FEATURE_NAMES,'budget_feature_excluded':True})
    print(json.dumps(read_json(root/'report.json'),indent=2))
if __name__=='__main__': main()
