#!/usr/bin/env python3
"""Audit unchanged R/S/B inputs, regenerated pool and refitted LOSO references."""
import sys,json,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,PREVIOUS_RUN,PREVIOUS_REVISION,REAL_TEST,TRAIN,sha,read_json,write_json
from main_experiment.observation import parse,features
from main_experiment.integrity import validate_request


def main():
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION
    for name,h in read_json(run/'preparation_inputs.json')['sources'].items(): assert sha(ROOT/name)==h,name
    same=0; new_h=0
    for domain in ('sample','training'):
        for p in (run/'observations'/domain).glob('*.json'):
            r=read_json(p)
            if r['arm']=='H':
                assert not (PREVIOUS_RUN/'observations'/domain/p.name).exists()
                new_h+=1; continue
            old=read_json(PREVIOUS_RUN/'observations'/domain/p.name)
            for k in ('block','block_sha256','messages','prompt_sha256','truth'): assert r[k]==old[k],(p.name,k)
            same+=1
    pool_same=0
    for p in (rev/'pool/observations').glob('*.json'):
        assert p.with_suffix('.sha256').read_text().strip()==sha(p)
        r=read_json(p); old=read_json(PREVIOUS_REVISION/'pool/observations'/p.name)
        for k in ('seed','parameters','truth','latents_sha256'): assert r[k]==old[k],(p.name,k)
        byid={x['id']:x for x in old['observations']}
        for o in r['observations']:
            if o['arm']=='H': assert o['id'] not in byid; continue
            assert o['block']==byid[o['id']]['block']
            pool_same+=1
    primary=read_json(rev/'primary_baselines.json'); checked=0
    for folder,method in (('models_pooled','extratrees_pooled'),('models_real_only','extratrees_real_only')):
        for fold in (*REAL_TEST,'synthetic'):
            path=rev/folder/fold; m=read_json(path/'manifest.json')
            assert sha(path/'model.pkl')==m['model_sha256']
            assert fold not in m['sources']
            assert not any('_dev_' in s for s in m['sources'])
            assert set(m['real_sources'])==set(TRAIN)-({fold} if fold in REAL_TEST else set())
            assert m['n_features']==134 and m['parameters']['n_estimators']==500
            with (path/'model.pkl').open('rb') as f: model=pickle.load(f)
            for p in (run/'observations/sample').glob('*.json'):
                o=read_json(p); wanted=o['graph_id'] if o['stratum']=='real' else 'synthetic'
                if wanted!=fold: continue
                pred=model.predict([features(parse(o['block']))])[0]
                np.testing.assert_allclose(pred,primary['observations'][o['id']][method]['prediction'],rtol=0,atol=1e-14)
                checked+=1
    requests=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()]
    for r in requests: validate_request(r)
    ledger=read_json(run.with_name(run.name+'_api')/'ledger.json')
    assert ledger['requests']=={} and ledger['batches']=={}
    result={'verified':True,'unchanged_main_training_RSB':same,'new_H_observations':new_h,
        'unchanged_pool_RSB':pool_same,'pool_graphs':500,'reference_models':14,
        'predictions_recomputed':checked,'requests':len(requests),'api_calls':0,
        'primary_baselines_sha256':sha(rev/'primary_baselines.json')}
    write_json(rev/'revision_audit.json',result); print(json.dumps(result,indent=2))


if __name__=='__main__': main()
