#!/usr/bin/env python3
"""Strict final-only provenance audit; adapt baseline metadata, never predictions."""
import hashlib,json,pickle,sys,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,PREVIOUS_RUN,DESIGN_VERSION,REAL_TEST,TRAIN,sha,read_json,write_json,digest
from main_experiment.observation import parse,features,messages
from main_experiment.integrity import bind,validate_request
from main_experiment.requests import REVISION
from main_experiment.evaluation import parse_final
from run_qwen_engine import GENERATION_CONFIG

FREEZE='aaaac491006aa6981aeebc95b913b8847da76120'
def main():
    assert DESIGN_VERSION=='cells10-final-20260920'
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION; raw=ROOT/'results/imported_final_20260920'
    out=run.with_name(run.name+'_qwen'); out.mkdir(parents=True,exist_ok=True)
    for name in ['scripts/evaluate_main_responses.py','scripts/run_qwen_engine.py']+[str(p.relative_to(ROOT)) for p in (ROOT/'src/main_experiment').glob('*') if p.suffix in ('.py','.cpp')]:
        frozen=subprocess.check_output(['git','show',f'{FREEZE}:{name}'],cwd=ROOT)
        assert hashlib.sha256(frozen).hexdigest()==sha(ROOT/name),name
    preparation_drift=[]
    for name,h in read_json(run/'preparation_inputs.json')['sources'].items():
        if sha(ROOT/name)!=h:
            assert name=='docs/PROTOCOL_SRW_20260920.md',name
            historical=subprocess.check_output(['git','show',f'ff59f93:{name}'],cwd=ROOT)
            assert hashlib.sha256(historical).hexdigest()==h
            frozen=subprocess.check_output(['git','show',f'{FREEZE}:{name}'],cwd=ROOT)
            assert hashlib.sha256(frozen).hexdigest()==sha(ROOT/name)
            preparation_drift.append({'path':name,'prepared_sha256':h,'freeze_sha256':sha(ROOT/name),
                'scope':'protocol prose only; all executable and prompt preparation hashes match; not silently rebound'})
    checks=read_json(raw/'CHECKSUMS.json')
    for name,h in checks.items():
        assert (raw/name).resolve().is_relative_to(raw.resolve())
        assert sha(raw/name)==h,name
    assert (raw/'SPEC_COMMIT').read_text().strip()==FREEZE
    assert not read_json(raw/'ARCHIVE_REPORT.json')['readback_mismatches']
    assert sha(raw/'requests.jsonl')==sha(run/'requests.jsonl')
    binding=read_json(raw/'answers/engine_inputs.json'); identity=read_json(raw/'model_identity.json')
    assert binding['generation']==GENERATION_CONFIG and binding['runner_sha256']==sha(ROOT/'scripts/run_qwen_engine.py')
    assert binding['requests_sha256']==sha(run/'requests.jsonl') and identity['revision_pinned']==REVISION
    for name,h in identity.items():
        if name in binding['model_files']: assert binding['model_files'][name]==h
    for name,h in identity['safetensors'].items(): assert binding['model_files'][name]==h
    requests=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()]
    assert len(requests)==len({r['id'] for r in requests})==3360
    assert len({r['seed'] for r in requests})==3360
    oldids={r['id'] for r in map(json.loads,(PREVIOUS_RUN/'requests.jsonl').read_text().splitlines())}
    assert not oldids.intersection(r['id'] for r in requests)
    expected={r['id']:r for r in requests if r['config_id'].startswith('qwen_')}
    rendered={(r['observation_id'],r['mode']):r for r in map(json.loads,(raw/'rendered_prompts.jsonl').read_text().splitlines())}
    assert len(rendered)==560
    for r in requests:
        validate_request(r); assert r['id'].endswith('__'+DESIGN_VERSION)
        assert r['sample_index'] in range(1,6) and r['repeat_index'] in range(1,4)
        o=read_json(run/'observations/sample'/f"{r['observation_id']}.json")
        assert messages(o['block'])==o['messages'] and digest(o['messages'])==r['prompt_sha256']
        assert read_json(raw/'observations/sample'/f"{r['observation_id']}.json")==o
        if not r['config_id'].startswith('qwen_'):
            assert not r['production_dispatch_enabled'] and not r['started'] and r['status']=='not_started'
    counts=Counter((r['graph_id'],r['arm'],r['config_id'],r['sample_index'],r['repeat_index']) for r in requests)
    assert len(counts)==3360 and set(counts.values())=={1}
    files=list((raw/'answers').glob('*_r*/*.json')); ids=[read_json(p)['id'] for p in files]
    assert len(ids)==1680 and set(ids)==set(expected)
    attempts=list((raw/'answers').glob('*_r*/*.attempt'))
    assert len(attempts)==1680 and {read_json(p)['id'] for p in attempts}==set(expected)
    stats={}; invalid=[]; records=[]
    for p in files:
        d=read_json(p); r=expected[d['id']]; a=read_json(p.with_suffix('.attempt'))
        assert d['design_version']==DESIGN_VERSION and d['runner_sha256']==binding['runner_sha256']
        for k in ('id','arm','config_id','seed','observation_id','prompt_sha256','payload_sha256'): assert d[k]==r[k]
        for k in ('id','payload_sha256'): assert a[k]==r[k]
        rp=rendered[r['observation_id'],d['mode']]
        assert rp['prompt_sha256']==r['prompt_sha256']
        assert hashlib.sha256(rp['rendered'].encode()).hexdigest()==rp['rendered_sha256']
        assert a['input_tokens']==d['input_tokens']==d['engine_prompt_tokens']==rp['input_tokens']
        for q in (p,p.with_suffix('.attempt')): assert str(q.relative_to(raw)) in checks
        pred,reason=parse_final(d.get('final_text','')); technical=d.get('technical_error',d['status']!='completed')
        valid=pred is not None and not technical
        if not valid: invalid.append({'id':d['id'],'reason':reason,'final_text':d.get('final_text','')})
        for key in ('all',d['arm'],d['arm']+'/'+d['config_id']):
            c=stats.setdefault(key,Counter()); c['found']+=1; c['valid']+=valid; c['invalid']+=not valid
            c['technical_errors']+=technical; c['tokenlimits']+=d['end_state']=='output_limit'
            c['unclosed_reasoning']+=not d.get('reasoning_closed',True)
            c['output_tokens_sum']+=d['output_tokens']; c['output_tokens_max']=max(c['output_tokens_max'],d['output_tokens'])
        records.append({'id':d['id'],'sha256':sha(p),'attempt_sha256':sha(p.with_suffix('.attempt'))})
    for key,c in stats.items(): c.update(planned=1680 if key=='all' else 210 if '/' in key else 420,missing=0)
    # Exact feature invariance permits reference reuse, not generation reuse.
    primary=read_json(rev/'primary_baselines.json'); model_checks=0
    for domain in ('sample','training'):
        for p in (run/'observations'/domain).glob('*.json'):
            o=read_json(p); before=read_json(PREVIOUS_RUN/'observations'/domain/p.name)
            assert parse(o['block'])==parse(before['block']) and o['truth']==before['truth']
            np.testing.assert_array_equal(features(parse(o['block'])),features(parse(before['block'])))
            if domain=='sample':
                b=primary['observations'][o['id']]; assert b['block_sha256']==before['block_sha256'] and b['truth']==o['truth']
                b['block_sha256']=o['block_sha256']
    for folder,method in (('models_pooled','extratrees_pooled'),('models_real_only','extratrees_real_only')):
        for fold in (*REAL_TEST,'synthetic'):
            p=rev/folder/fold; m=read_json(p/'manifest.json'); assert sha(p/'model.pkl')==m['model_sha256']
            assert fold not in m['sources'] and not any('_dev_' in s for s in m['sources'])
            assert set(m['real_sources'])==set(TRAIN)-({fold} if fold in REAL_TEST else set())
            with (p/'model.pkl').open('rb') as f: model=pickle.load(f)
            for oid,b in primary['observations'].items():
                wanted=b['graph_id'] if b['stratum']=='real' else 'synthetic'
                if wanted!=fold: continue
                o=parse(read_json(run/'observations/sample'/f'{oid}.json')['block'])
                pred=model.predict([features(o)])[0] if o['D_obs'] else m['median']
                np.testing.assert_allclose(pred,b[method]['prediction'],atol=1e-14,rtol=0); model_checks+=1
    primary['design_version']=DESIGN_VERSION
    primary['final_binding']={'source_sha256':sha(rev/'primary_baselines.json'),'feature_invariance_observations':600,
        'new_fits':0,'prediction_changes':0,'recomputed_learned_predictions':model_checks}
    bind(out/'primary_baselines.json',primary)
    report={'freeze_commit':FREEZE,'design_version':DESIGN_VERSION,'verified':True,'counts':stats,'invalid_answers':invalid,
        'duplicates':0,'hash_mismatches':0,'archive_files_verified':len(checks),'old_answers_used':0,
        'archive_checksums_sha256':sha(raw/'CHECKSUMS.json'),'requests_sha256':sha(run/'requests.jsonl'),
        'preparation_document_drift':preparation_drift,
        'reference_binding':primary['final_binding'],'model_identity':identity,'engine_inputs':binding,
        'records':sorted(records,key=lambda r:r['id'])}
    bind(out/'provenance_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','model_identity','engine_inputs')},indent=2))

if __name__=='__main__': main()
