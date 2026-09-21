#!/usr/bin/env python3
"""Independent panel888 invariant, LOSO, prompt and raw result audit."""
import csv,json,math,re,sys,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,DESIGN_VERSION,sha,read_json,write_json

def strict(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError('duplicate')
            d[k]=v
        return d
    raw=raw.strip()
    if raw.startswith('```'):
        m=re.fullmatch(r'```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*',raw,re.S)
        if not m: return None
        raw=m[1]
    try:
        d=json.loads(raw,object_pairs_hook=pairs)
        if type(d)!=dict or set(d)!={f'rho_{k}' for k in range(2,6)}: return None
        v=[d[f'rho_{k}'] for k in range(2,6)]
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in v): return None
        return v if all(a>=b for a,b in zip(v,v[1:])) else None
    except (ValueError,TypeError,OverflowError): return None

def main():
    from main_experiment.common import MAIN_KEYS,REAL_TEST,SURROGATES,TRAIN,SYNTH,ARMS,CONFIGS,LLM_REPEATS,ARM_ID,parent_source,fold_for,seed
    from main_experiment.data import load_graph
    from main_experiment.observation import parse,features
    from main_experiment.sampling import Walk,history_panel_mask
    import pickle
    run=ROOT/CURRENT_RUN;rev=ROOT/CURRENT_REVISION
    report=read_json(run/'report.json');assert report['offline_ready']
    graph={k:load_graph(run/'graphs'/k) for k in MAIN_KEYS}
    invariant=[];crn=[]
    for parent in REAL_TEST:
        g=graph[parent];s=graph[parent+'__pwt']
        assert np.array_equal(g.ends,s.ends) and np.array_equal(g.u,s.u) and np.array_equal(g.v,s.v)
        assert np.array_equal(g.pair,s.pair) and g.horizon==s.horizon
        assert np.array_equal(np.sort(g.t),np.sort(s.t))
        assert np.array_equal(np.bincount(g.pair),np.bincount(s.pair))
        assert np.array_equal(g.m,s.m) and g.M==s.M
        assert read_json(run/'graphs'/s.key/'invariant_audit.json')['passed']
        # Independent rederivation of P[w,t] and event identity, no generator call.
        rs=np.random.Generator(np.random.PCG64(seed('pwt_productive',parent)))
        assert np.array_equal(rs.permutation(g.t),s.t)
        invariant.append(parent)
        bg=read_json(run/'calibration'/g.key/'budget.json');bs=read_json(run/'calibration'/s.key/'budget.json')
        assert bg['T']==.1*g.cells and bs['T']==.1*s.cells
        wg=Walk(g,run/'build');ws=Walk(s,run/'build')
        for ix in (1,2,3):
            for arm in ARMS:assert seed('sample',parent_source(g.key),ARM_ID[arm],ix)==seed('sample',parent_source(s.key),ARM_ID[arm],ix)
            L=min(bg['L'],bs['L'])
            a=wg.run([seed('sample',parent,ARM_ID['S'],ix)],L,True)[2]
            b=ws.run([seed('sample',parent,ARM_ID['S'],ix)],L,True)[2]
            assert np.array_equal(a,b)
            r=np.random.Generator(np.random.PCG64(seed('sample',parent,ARM_ID['R'],ix)))
            assert bg['n_panel']==bs['n_panel']
            u=r.random(g.M);ka=u<bg['p'];kb=u<bs['p']
            assert np.all(ka<=kb) if bg['p']<=bs['p'] else np.all(kb<=ka)
        crn.append(parent)
    obs=[read_json(p) for p in (run/'observations/sample').glob('*.json')]
    assert {r['graph_id'] for r in obs}==set(MAIN_KEYS)
    assert len([g for g in MAIN_KEYS if g in REAL_TEST])==8 and len(SURROGATES)==8 and len(SYNTH)==8
    primary=read_json(rev/'primary_baselines.json')['observations'];assert set(primary)=={r['id'] for r in obs}
    models={};manifests=[]
    for folder in ('models_pooled','models_real_only'):
        for fold in (*REAL_TEST,'synthetic'):
            path=rev/folder/fold;m=read_json(path/'manifest.json')
            assert sha(path/'model.pkl')==m['model_sha256']
            from main_experiment.training import FOREST_SEED
            assert m['parameters']['random_state']==FOREST_SEED
            allowed=set(TRAIN)-({fold} if fold in REAL_TEST else set())
            assert set(m['real_sources'])==allowed
            assert not set(m['sources'])&set(SURROGATES)
            assert not set(m['sources'])&set(SYNTH)
            if folder=='models_pooled':
                assert len(m['block_sources']['dar'])==200 and len(m['block_sources']['ad'])==200
            assert all('p888-20260921' in oid for oid in m['observations'])
            with (path/'model.pkl').open('rb') as f:models[folder,fold]=pickle.load(f)
            truths=[load_graph(run/'graphs'/s).truth for s in sorted(allowed)]
            np.testing.assert_array_equal(m['median'],np.median(truths,axis=0))
            manifests.append({'path':str(path),'sha256':sha(path/'model.pkl'),'excluded_parent':fold,'training_rows':m['training_rows']})
    for row in obs:
        fold=fold_for(row['graph_id']);o=parse(row['block']);b=primary[row['id']]
        assert b['truth']==row['truth'] and b['block_sha256']==row['block_sha256']
        for name in ('pooled','real_only'):
            expected=models['models_'+name,fold].predict([features(o)])[0] if o['D_obs'] else read_json(rev/('models_'+name)/fold/'manifest.json')['median']
            np.testing.assert_array_equal(b['extratrees_'+name]['prediction'],expected)
    requests=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()]
    assert len(requests)==len(obs)*len(CONFIGS)*LLM_REPEATS
    assert all(not r['started'] for r in requests)
    assert all(not r['production_dispatch_enabled'] for r in requests if r['config_id'] in ('sol','deepseek'))
    old=ROOT/'archive/pre_panel888_20260921/results/main_experiment/cells10_final_20260920/requests.jsonl'
    assert not {r['id'] for r in requests}&{json.loads(x)['id'] for x in old.read_text().splitlines()}
    prompt_files=['system.txt','user_prefix.txt','rule_R.txt','rule_S.txt','rule_H_time_v3.txt','rule_B.txt','aux_S.txt']
    for f in prompt_files:assert sha(ROOT/'config/main_experiment'/f)==sha(ROOT/'archive/pre_panel888_20260921/config/main_experiment'/f)
    for family,modes in (('dar',('a0','a08')),('ad',('memoryless','memory'))):
        for rep in (1,2):
            a,b=[read_json(run/'graphs'/f'{family}_{mode}_r{rep}'/'manifest.json') for mode in modes]
            assert a['shared_latents']==b['shared_latents']
    sensitivity={}
    for name in ('history_sensitivity','srw_diagnostics','control_diagnostics','census_diagnostics'):
        p=rev/name/'report.json';assert p.exists();sensitivity[name]=sha(p)
    control=read_json(rev/'control_diagnostics/report.json');assert control['null_shuffles']==792 and control['W_rows']==24*19
    assert all(r['invariants_passed'] for p in (rev/'control_diagnostics').glob('*_null.json') for r in read_json(p))
    history=list(csv.DictReader((rev/'history_sensitivity/sources.csv').open()));assert len(history)==72
    assert set(r['graph_id'] for r in history)==set(MAIN_KEYS)
    paths=list(csv.DictReader((rev/'srw_diagnostics/paths.csv').open()));assert len(paths)==24*32*2
    for stage in ('pool','train','dev','main','decompose','decompose_main'):assert (rev/f'report_{stage}.json').exists()
    seed_audit=read_json(rev/'seed_audit.json');assert seed_audit['collision_audit_passed'] and seed_audit['previous_generation_seed_overlap']==0
    assert sha(rev/'complete_seed_manifest.jsonl')==seed_audit['manifest_sha256']
    result={'seed_audit':seed_audit,'verified':True,'design_version':DESIGN_VERSION,'main_observations':len(obs),'requests':len(requests),
        'qwen_requests':sum(r['config_id'].startswith('qwen') for r in requests),
        'surrogate_invariant_pairs':invariant,'crn_pairs':crn,'fresh_reference_models':manifests,
        'sensitivity_reports':sensitivity,'prompt_templates_byte_identical':True,'old_request_ids_reused':0,
        'sol_deepseek_started':0,'checksums':{'requests':sha(run/'requests.jsonl'),'baselines':sha(rev/'primary_baselines.json')}}
    # Optional completed Qwen validation independently parses collected raw final text.
    qwen=run.with_name(run.name+'_qwen')
    if (qwen/'responses.jsonl').exists():
        response=[json.loads(x) for x in (qwen/'responses.jsonl').read_text().splitlines()]
        expected={r['id'] for r in requests if r['config_id'].startswith('qwen') and r['status']!='skipped_empty'}
        assert {r['id'] for r in response}==expected and len(response)==len(expected)
        er={r['id']:r for r in csv.DictReader((qwen/'evaluation/answer_errors.csv').open())};req={r['id']:r for r in requests};oi={o['id']:o for o in obs}
        valid=0
        for r in response:
            p=None if r.get('technical_error') else strict(r.get('final_text',''))
            assert (p is not None)==(er[r['id']]['valid']=='True')
            if p is not None:
                truth=oi[req[r['id']]['observation_id']]['truth'];diff=np.abs(np.array(p)-truth)
                assert abs(float(er[r['id']]['AE2'])-diff[0])<1e-14
                assert abs(float(er[r['id']]['ProfileAE'])-diff.mean())<1e-14
                valid+=1
        result['qwen']={'complete':True,'responses':len(response),'valid':valid,'invalid':len(response)-valid,'responses_sha256':sha(qwen/'responses.jsonl')}
    write_json(rev/'independent_audit.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
