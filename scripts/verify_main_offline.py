#!/usr/bin/env python3
"""Independent artifact audit; reads but does not run inference or retrain."""
from pathlib import Path
import argparse
import csv
import json
import sys
import pickle
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import REAL_TEST,TRAIN,SYNTH,CONFIGS,sha,digest,read_json,seed, MAIN_OBSERVATIONS, TRAINING_OBSERVATIONS, PLANNED_CALLS
from main_experiment.observation import parse,features,messages,FEATURE_NAMES
from main_experiment.baselines import plugin,corrector
from main_experiment.data import load_graph


def verify(root):
    root=Path(root)
    checksums=read_json(root/'checksums.json')
    for f,h in checksums.items():
        if sha(root/f)!=h: raise AssertionError(f'checksum {f}')
    obs=list((root/'observations/sample').glob('*.json'))
    train=list((root/'observations/training').glob('*.json'))
    assert len(obs)==MAIN_OBSERVATIONS and len(train)==TRAINING_OBSERVATIONS
    keys=set(REAL_TEST)|set(SYNTH); counts={k:0 for k in keys}
    source_truth={}
    for key in list(TRAIN)+list(SYNTH):
        g=load_graph(root/'graphs'/key); m=read_json(root/'graphs'/key/'manifest.json')
        # Independent reference from unique (dyad,window) event occurrences.
        seen=np.unique(g.pair*5+g.w)
        k=np.bincount(seen//5,minlength=g.D)
        truth=[float(np.mean(k>=j)) for j in range(2,6)]
        assert truth==m['truth']; source_truth[key]=truth
        assert g.counts.sum()==g.M and int(g.counts[:,2:].sum())==m['B']>0
        b=read_json(root/'calibration'/key/'budget.json')
        values=[]
        for f in sorted((root/'calibration'/key).glob('validation_*.json')): values+=read_json(f)['volumes']
        assert len(values)==b['validation_n']
        assert float(np.mean(values))==b['validation_mean']
        assert float(np.std(values,ddof=1)/np.sqrt(len(values)))==b['validation_mcse']
        if b['search_limit_reached_without_budget']: assert b['L']==b['C'] and not b['budget_matched']
    for path in obs+train:
        row=read_json(path); o=parse(row['block'])
        assert messages(row['block'])==row['messages']
        assert digest(row['messages'])==row['prompt_sha256']
        assert len(features(o))==88
        assert row['truth']==source_truth[row['graph_id']]
        if path.parent.name=='sample': counts[row['graph_id']]+=1
    assert all(n==16 for n in counts.values())
    models={}
    for fold in [*REAL_TEST,'synthetic']:
        m=read_json(root/'models'/fold/'manifest.json')
        sources=set(TRAIN)-({fold} if fold in REAL_TEST else set())
        assert set(m['sources'])==sources
        np.testing.assert_array_equal(m['median'],np.median([source_truth[s] for s in sorted(sources)],axis=0))
        assert m['training_rows']==len(sources)*16
        assert abs(sum(m['weights'])-1)<1e-12
        assert m['feature_names']==FEATURE_NAMES and m['parameters']['n_estimators']==500
        assert sha(root/'models'/fold/'model.pkl')==m['model_sha256']
        with open(root/'models'/fold/'model.pkl','rb') as f: models[fold]=pickle.load(f)
    for path in obs:
        row=read_json(path); o=parse(row['block'])
        fold=row['graph_id'] if row['stratum']=='real' else 'synthetic'
        base=read_json(root/'baselines'/(row['id']+'.json'))
        median=read_json(root/'models'/fold/'manifest.json')['median']
        expected=models[fold].predict([features(o)])[0] if o['D_obs'] else median
        np.testing.assert_array_equal(base['extratrees']['prediction'],expected)
        if o['D_obs']:
            np.testing.assert_array_equal(base['plugin']['prediction'],plugin(o))
            np.testing.assert_array_equal(base['corrector']['prediction'],corrector(o))
        for result in base.values():
            values=result['prediction']
            assert all(0<=v<=1 for v in values)
            assert all(a>=b for a,b in zip(values,values[1:]))
    requests=[json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
    assert len(requests)==PLANNED_CALLS and len({r['id'] for r in requests})==PLANNED_CALLS
    for config in CONFIGS: assert sum(r['config_id']==config for r in requests)==672
    assert all(not r['started'] and r['status'] in ['not_started','skipped_empty'] for r in requests)
    for row in requests:
        o=read_json(root/'observations/sample'/(row['observation_id']+'.json'))
        assert row['payload'].get('messages',row['payload'].get('input'))==o['messages']
    for oid in {r['observation_id'] for r in requests}:
        assert len({r['prompt_sha256'] for r in requests if r['observation_id']==oid})==1
    seeds=read_json(root/'seed_manifest.json')
    assert len({r['seed'] for r in seeds})==len(seeds)
    for r in seeds: assert seed(*r['fields'][1:])==r['seed']
    sizes=list(csv.DictReader(open(root/'prompt_sizes.csv')))
    assert len(sizes)==MAIN_OBSERVATIONS
    assert all(max(int(r[k]) for k in ['qwen_thinking','qwen_nonthinking','deepseek_message_texts','sol_o200k_message_proxy'])<=4096 for r in sizes)
    cns=read_json(root/'graphs/copenhagen_bluetooth/manifest.json')
    assert cns['copenhagen']['legacy_export_matches'] and cns['copenhagen']['release_md5_verified']=='98892459f73e774cf79e7977edfeee3e'
    import pandas as pd
    export=pd.read_csv(root/'graphs/copenhagen_bluetooth/canonical.csv')
    assert (export.u<export.v).all()
    assert export.equals(export.sort_values(['u','v','t']).reset_index(drop=True))
    result={'verified':True,'checksums':len(checksums),'main_observations':len(obs),'training_observations':len(train),
            'models':7,'planned_requests':len(requests),'started_requests':0,'seed_count':len(seeds)}
    print(json.dumps(result,indent=2)); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); a=p.parse_args(); verify(a.run)
