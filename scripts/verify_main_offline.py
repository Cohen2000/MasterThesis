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
from main_experiment.common import (REAL_TEST,TRAIN,SYNTH,ARMS,CONFIGS,H_CAP,BUDGET_TOLERANCE,sha,digest,read_json,seed,
                                    planned_sizes,observations_per_graph,draws_for,observation_id)
from main_experiment.observation import parse,features,messages,make,serialize,FEATURE_NAMES
from main_experiment.baselines import plugin,corrector,h_midpoint
from main_experiment.sampling import Walk,draw,recent_counts
from main_experiment.data import load_graph


def verify(root):
    root=Path(root)
    checksums=read_json(root/'checksums.json')
    for f,h in checksums.items():
        if sha(root/f)!=h: raise AssertionError(f'checksum {f}')
    obs=list((root/'observations/sample').glob('*.json'))
    train=list((root/'observations/training').glob('*.json'))
    budgets={k:read_json(root/'calibration'/k/'budget.json') for k in list(TRAIN)+list(SYNTH)}
    design=planned_sizes(budgets,[*REAL_TEST,*SYNTH],TRAIN)
    assert len(obs)==design['main_observations'] and len(train)==design['training_observations']
    keys=set(REAL_TEST)|set(SYNTH); counts={k:0 for k in keys}
    source_truth={}; graphs={}
    # No feature of the superseded suffix arm may survive in the current design.
    assert 'n_panel_suffix' not in FEATURE_NAMES and not any('??' in f for f in FEATURE_NAMES)
    for key in list(TRAIN)+list(SYNTH):
        g=load_graph(root/'graphs'/key); m=read_json(root/'graphs'/key/'manifest.json')
        # Independent reference from unique (dyad,window) event occurrences.
        seen=np.unique(g.pair*5+g.w)
        k=np.bincount(seen//5,minlength=g.D)
        truth=[float(np.mean(k>=j)) for j in range(2,6)]
        assert truth==m['truth']; source_truth[key]=truth
        # B is a share of the whole archive since the design revision; the suffix
        # volume is separate and is what must be positive for arm H to exist.
        assert g.counts.sum()==g.M and g.B==m['B']>0 and g.M_suffix>0
        b=read_json(root/'calibration'/key/'budget.json')
        values=[]
        for f in sorted((root/'calibration'/key).glob('validation_*.json')): values+=read_json(f)['volumes']
        assert len(values)==b['validation_n']
        assert float(np.mean(values))==b['validation_mean']
        assert float(np.std(values,ddof=1)/np.sqrt(len(values)))==b['validation_mcse']
        if b['search_limit_reached_without_budget']: assert b['L']==b['C'] and not b['walk_budget_matched']
        # Arm H, recomputed independently: C = sum min(cap, m_e); d minimises
        # |d*C/D - B| over 1..D with ties to the smaller d.
        C=int(np.minimum(g.m,H_CAP).sum()); dd=np.arange(1,g.D+1)
        d=int(dd[np.argmin(np.abs(dd*C/g.D-g.B))])
        assert (b['C_cap'],b['n_dyads'])==(C,d)
        assert b['h_saturated']==(d==g.D) and b['h_target_unreachable']==(C<g.B)
        rel=(d*C/g.D-g.B)/g.B
        assert abs(rel-b['h_relative_budget_error'])<1e-12 and b['h_within_tolerance']==(abs(rel)<=BUDGET_TOLERANCE)
        assert b['budget_matched_by_arm']['H']==b['h_within_tolerance']
        assert b['budget_matched_by_arm']['S']==b['walk_budget_matched']
        assert b['budget_matched']==all(b['budget_matched_by_arm'].values())
        graphs[key]=g
    # Every stored block is re-derived from its seed and compared byte for byte.
    walks={}
    for path in obs+train:
        row=read_json(path); o=parse(row['block'])
        g=graphs[row['graph_id']]; b=budgets[row['graph_id']]
        assert path.stem==observation_id(row['graph_id'],row['arm'],row['sample_index'])
        assert 1<=row['sample_index']<=draws_for(row['arm'],b)
        if row['arm']=='S' and row['graph_id'] not in walks: walks[row['graph_id']]=Walk(g,root/'build')
        c,re=draw(g,row['arm'],row['sample_index'],path.parent.name,b,walks.get(row['graph_id']))
        assert serialize(make(g,row['arm'],b,c,re))==row['block']
        if row['arm']=='H':
            seen=c.sum(1)>0
            assert o['D_obs']==b['n_dyads']==int(seen.sum())
            assert (c[seen].sum(1)==np.minimum(g.m[seen],H_CAP)).all()
            np.testing.assert_array_equal(c[seen],recent_counts(g.counts[seen]))
            # The at-cap column is the observed count, never the true truncation flag.
            assert sum(r[3] for r in o['table'])==int((c[seen].sum(1)==H_CAP).sum())
            np.testing.assert_array_equal(features(o)[-4:],h_midpoint(o))
        assert messages(row['block'])==row['messages']
        assert digest(row['messages'])==row['prompt_sha256']
        assert len(features(o))==len(FEATURE_NAMES)
        assert row['truth']==source_truth[row['graph_id']]
        if path.parent.name=='sample': counts[row['graph_id']]+=1
    assert all(n==observations_per_graph(budgets[k]) for k,n in counts.items())
    models={}
    for fold in [*REAL_TEST,'synthetic']:
        m=read_json(root/'models'/fold/'manifest.json')
        sources=set(TRAIN)-({fold} if fold in REAL_TEST else set())
        assert set(m['sources'])==sources
        np.testing.assert_array_equal(m['median'],np.median([source_truth[s] for s in sorted(sources)],axis=0))
        assert m['training_rows']==sum(observations_per_graph(budgets[s]) for s in sources)
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
    calls=design['planned_calls']
    assert len(requests)==calls and len({r['id'] for r in requests})==calls
    for config in CONFIGS: assert sum(r['config_id']==config for r in requests)==calls//len(CONFIGS)
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
    assert len(sizes)==design['main_observations']
    assert all(max(int(r[k]) for k in ['qwen_thinking','qwen_nonthinking','deepseek_message_texts','sol_o200k_message_proxy'])<=4096 for r in sizes)
    cns=read_json(root/'graphs/copenhagen_bluetooth/manifest.json')
    assert cns['copenhagen']['legacy_export_matches'] and cns['copenhagen']['release_md5_verified']=='98892459f73e774cf79e7977edfeee3e'
    import pandas as pd
    export=pd.read_csv(root/'graphs/copenhagen_bluetooth/canonical.csv')
    assert (export.u<export.v).all()
    assert export.equals(export.sort_values(['u','v','t']).reset_index(drop=True))
    result={'verified':True,'checksums':len(checksums),'main_observations':len(obs),'training_observations':len(train),
            'blocks_rederived':len(obs)+len(train),
            'deterministic_h_graphs':sorted(k for k,b in budgets.items() if b['h_saturated']),
            'models':7,'planned_requests':len(requests),'started_requests':0,'seed_count':len(seeds)}
    print(json.dumps(result,indent=2)); return result

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); a=p.parse_args(); verify(a.run)
