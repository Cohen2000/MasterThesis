#!/usr/bin/env python3
"""Read-only audit of the revised pool, folds, request identities and old/new blocks."""
import argparse,json,pickle,sys
from collections import Counter
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ROOT,CURRENT_RUN,CURRENT_REVISION,REAL_TEST,TRAIN,sha,read_json,write_json)
from main_experiment.pool import pool_definition
from main_experiment.observation import features,parse
from main_experiment.integrity import validate_request


def audit(out):
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION
    old=ROOT/'results/main_experiment/cells10_20260917'
    for name,value in read_json(run/'preparation_inputs.json')['sources'].items():
        assert sha(ROOT/name)==value,f'preparation source changed: {name}'
    matching=0
    for domain in ('sample','training'):
        for f in (run/'observations'/domain).glob('*.json'):
            before=read_json(old/'observations'/domain/f.name); after=read_json(f)
            for field in ('id','block','block_sha256','messages','prompt_sha256','truth'):
                assert before[field]==after[field],(f.name,field)
            matching+=1
    definition=pool_definition(); actual=[]
    assert read_json(rev/'pool_definition.json')==definition
    for spec in definition['graphs']:
        p=rev/'pool/observations'/f"{spec['key']}.json"
        assert p.with_suffix('.sha256').read_text().strip()==sha(p)
        row=read_json(p)
        for k in ('key','family','partition','parameters','seed','stratum'): assert row[k]==spec[k]
        actual.append(row)
    distributions={}
    for family in ('dar','ad'):
        for part,n in (('train',200),('dev',50)):
            group=[r for r in actual if r['family']==family and r['partition']==part]
            counts=Counter(r['parameters']['N'] for r in group)
            assert len(counts)==5 and set(counts.values())=={n//5}
            entry={'N':dict(counts)}
            if family=='ad':
                pairs=Counter((r['parameters']['N'],r['parameters']['rounds']) for r in group)
                assert len(pairs)==25 and set(pairs.values())=={n//25}
                entry['N_rounds']={str(k):v for k,v in sorted(pairs.items())}
            distributions[f'{family}/{part}']=entry
    training_ids={r['key'] for r in actual if r['partition']=='train'}
    dev_ids={r['key'] for r in actual if r['partition']=='dev'}
    primary=read_json(rev/'primary_baselines.json')
    predictions=0; model_count=0
    for folder,method in (('models_pooled','extratrees_pooled'),('models_real_only','extratrees_real_only')):
        for fold in (*REAL_TEST,'synthetic'):
            p=rev/folder/fold; manifest=read_json(p/'manifest.json')
            assert sha(p/'model.pkl')==manifest['model_sha256']
            sources=set(manifest['sources'])
            assert not sources&dev_ids
            if fold in REAL_TEST: assert fold not in sources
            expected=(set(TRAIN)-({fold} if fold in REAL_TEST else set()))
            if folder=='models_pooled': expected|=training_ids
            assert sources==expected,(folder,fold,len(sources),len(expected))
            with open(p/'model.pkl','rb') as f: model=pickle.load(f)
            for path in (run/'observations/sample').glob('*.json'):
                obs=read_json(path); wanted=obs['graph_id'] if obs['stratum']=='real' else 'synthetic'
                if wanted!=fold: continue
                o=parse(obs['block'])
                pred=model.predict([features(o)])[0] if o['D_obs'] else manifest['median']
                np.testing.assert_allclose(pred,primary['observations'][obs['id']][method]['prediction'],rtol=0,atol=1e-14)
                predictions+=1
            model_count+=1
    requests=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()]
    oldids={json.loads(x)['id'] for x in (old/'requests.jsonl').read_text().splitlines()}
    assert len(requests)==3360 and not {r['id'] for r in requests}&oldids
    assert all(not r['started'] for r in requests)
    for r in requests: validate_request(r)
    effective=[r['seed']%(2**31) for r in requests if r['config_id'].startswith('qwen')]
    assert len(effective)==len(set(effective))
    for r in requests:
        if r['config_id'].startswith('qwen'):
            assert r['payload']['structured_output']['json_object'] is True
            assert 'regex' not in r['payload']['structured_output']
    api=run.with_name(run.name+'_api')
    ledger=read_json(api/'ledger.json')
    assert ledger['requests']=={} and ledger['batches']=={}
    assert read_json(api/'release.template.json')['authorized'] is False
    result={'verified':True,'unchanged_observation_blocks':matching,'pool_graphs':len(actual),
            'pool_distributions':distributions,'models_checked':model_count,
            'predictions_recomputed':predictions,'planned_requests':len(requests),
            'request_ids_disjoint_from_previous':True,'qwen_effective_seeds_unique':True,'paid_requests_admitted':0,
            'release_authorized':False,'primary_baselines_sha256':sha(rev/'primary_baselines.json')}
    write_json(out,result); print(json.dumps({k:v for k,v in result.items() if k!='pool_distributions'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',required=True); a=p.parse_args(); audit(a.out)
