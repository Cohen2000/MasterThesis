#!/usr/bin/env python3
"""Integrate immutable raw archives only after per-request identity verification."""
import argparse,json,shutil,sys,hashlib
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,PREVIOUS_RUN,sha,read_json,write_json
from main_experiment.requests import REVISION
from run_qwen_engine import GENERATION_CONFIG

def lines(path):
    return [json.loads(s) for s in path.read_text().splitlines()]

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--arms',default='R,S,H,B')
    args=parser.parse_args(); arms=set(args.arms.split(',')); assert arms and arms<=set('RSHB')
    run=ROOT/CURRENT_RUN; out=run.with_name(run.name+'_qwen')
    original=ROOT/'archive/pre_h_time_20260920/results/main_experiment/cells10_json_20260918_qwen'
    sources={'R':original,'B':original,'H':ROOT/'results/imported_h_20260920',
             'S':ROOT/'results/imported_srw_20260920'}
    current=[r for r in lines(run/'requests.jsonl') if r['arm'] in arms and r['config_id'].startswith('qwen_')]
    archives={}; records=[]; common_model=None
    for source in sorted({sources[a] for a in arms}):
        checks=read_json(source/'CHECKSUMS.json')
        for name in ('answers/engine_inputs.json','requests.jsonl','rendered_prompts.jsonl','model_identity.json'):
            assert name in checks,(source,name)
        for name,h in checks.items():
            assert (source/name).resolve().is_relative_to(source.resolve()),name
            assert sha(source/name)==h,(source,name)
        assert not read_json(source/'ARCHIVE_REPORT.json')['readback_mismatches']
        binding=read_json(source/'answers/engine_inputs.json')
        assert binding['generation']==GENERATION_CONFIG
        assert binding['runner_sha256']==sha(ROOT/'scripts/run_qwen_engine.py')
        assert binding['requests_sha256']==sha(source/'requests.jsonl')
        identity=read_json(source/'model_identity.json'); assert identity['revision_pinned']==REVISION
        if common_model is None: common_model=binding['model_files']
        else: assert common_model==binding['model_files'],'model files differ across source archives'
        prior={r['id']:r for r in lines(source/'requests.jsonl')}
        rendered={(r['observation_id'],r['mode']):r for r in lines(source/'rendered_prompts.jsonl')}
        files=list((source/'answers').glob('*_r*/*.json'))
        answer_ids=[read_json(p)['id'] for p in files]; assert len(set(answer_ids))==len(answer_ids)
        attempts=[read_json(p)['id'] for p in (source/'answers').glob('*_r*/*.attempt')]
        assert len(set(attempts))==len(attempts) and set(attempts)==set(answer_ids)
        selected={r['id']:r for r in current if sources[r['arm']]==source}
        relevant=[p for p in files if read_json(p)['arm'] in {r['arm'] for r in selected.values()}]
        assert {read_json(p)['id'] for p in relevant}==set(selected),'missing or unexpected source answers'
        for path in relevant:
            answer=read_json(path); r=selected[answer['id']]; previous=prior[r['id']]
            for k in ('id','seed','payload','prompt_sha256','payload_sha256','observation_id','arm','config_id'):
                assert r[k]==previous[k],(r['id'],k)
            obs=read_json(run/'observations/sample'/f"{r['observation_id']}.json")
            before=read_json(source/'observations/sample'/f"{r['observation_id']}.json")
            assert f"observations/sample/{r['observation_id']}.json" in checks
            for k in ('block','block_sha256','messages','prompt_sha256','truth'): assert obs[k]==before[k],(r['id'],k)
            for k in ('id','seed','prompt_sha256','payload_sha256','arm','config_id','observation_id'):
                assert answer[k]==r[k],(r['id'],k)
            assert answer['runner_sha256']==binding['runner_sha256']
            rp=rendered[r['observation_id'],answer['mode']]
            assert rp['prompt_sha256']==r['prompt_sha256']
            assert hashlib.sha256(rp['rendered'].encode()).hexdigest()==rp['rendered_sha256']
            attempt=read_json(path.with_suffix('.attempt'))
            for k in ('id','payload_sha256'): assert attempt[k]==r[k]
            assert attempt['input_tokens']==rp['input_tokens']==answer['input_tokens']==answer['engine_prompt_tokens']
            for src in (path,path.with_suffix('.attempt')):
                assert str(src.relative_to(source)) in checks
                dst=out/src.relative_to(source); dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists(): assert sha(dst)==sha(src)
                else: shutil.copy2(src,dst)
            records.append({'id':r['id'],'arm':r['arm'],'config_id':r['config_id'],
                'source':str(path.relative_to(ROOT)),'answer_sha256':sha(path),
                'attempt_sha256':sha(path.with_suffix('.attempt')),
                'payload_sha256':r['payload_sha256'],'prompt_sha256':r['prompt_sha256']})
        archives[str(source.relative_to(ROOT))]={'checksums_sha256':sha(source/'CHECKSUMS.json'),
            'engine_inputs':binding,'model_identity':identity,'files_verified':len(checks)}
    assert len(records)==420*len(arms)
    result={'arms':sorted(arms),'complete':arms==set('RSHB'),'found':len(records),
        'counts':dict(Counter(r['arm'] for r in records)),'requests_sha256':sha(run/'requests.jsonl'),
        'script_sha256':sha(__file__),'archives':archives,'records':sorted(records,key=lambda r:r['id'])}
    suffix='' if result['complete'] else '_partial'
    write_json(out/f'integration_manifest{suffix}.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('archives','records')},indent=2))

if __name__=='__main__': main()
