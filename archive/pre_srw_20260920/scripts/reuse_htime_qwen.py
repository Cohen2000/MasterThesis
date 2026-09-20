#!/usr/bin/env python3
"""Copy only proven-identical R/S/B raw answers into the new study collection."""
import json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,PREVIOUS_RUN,sha,read_json,write_json,digest
from main_experiment.requests import REVISION
from run_qwen_engine import GENERATION_CONFIG


def main():
    old=PREVIOUS_RUN.with_name(PREVIOUS_RUN.name+'_qwen')
    run=ROOT/CURRENT_RUN; out=run.with_name(run.name+'_qwen')
    for name,h in read_json(old/'CHECKSUMS.json').items(): assert sha(old/name)==h,name
    prior={r['id']:r for r in map(json.loads,(PREVIOUS_RUN/'requests.jsonl').read_text().splitlines())}
    current=[r for r in map(json.loads,(run/'requests.jsonl').read_text().splitlines())]
    binding=read_json(old/'answers/engine_inputs.json')
    assert binding['generation']==GENERATION_CONFIG
    assert binding['runner_sha256']==sha(ROOT/'scripts/run_qwen_engine.py')
    assert read_json(old/'model_identity.json')['revision_pinned']==REVISION
    records=[]
    for r in current:
        if r['arm']=='H' or not r['config_id'].startswith('qwen'): continue
        p=prior[r['id']]
        for k in ('id','seed','payload','prompt_sha256','payload_sha256','observation_id','arm','config_id'):
            assert p[k]==r[k],(r['id'],k)
        o=read_json(run/'observations/sample'/f"{r['observation_id']}.json")
        before=read_json(PREVIOUS_RUN/'observations/sample'/f"{r['observation_id']}.json")
        for k in ('block','block_sha256','messages','prompt_sha256','truth'): assert o[k]==before[k]
        mode=r['config_id'].removeprefix('qwen_')
        rel=Path('answers')/f"{mode}_r{r['repeat_index']}"/f"{r['id']}.json"
        answer=read_json(old/rel)
        for k in ('id','seed','prompt_sha256','payload_sha256'): assert answer[k]==r[k]
        assert answer['runner_sha256']==binding['runner_sha256']
        for path in (rel,rel.with_suffix('.attempt')):
            target=out/path; target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists(): assert sha(target)==sha(old/path)
            else: shutil.copy2(old/path,target)
        records.append({'id':r['id'],'source':str((old/rel).relative_to(ROOT)),
            'answer_sha256':sha(old/rel),'attempt_sha256':sha((old/rel).with_suffix('.attempt')),
            'payload_sha256':r['payload_sha256'],'prompt_sha256':r['prompt_sha256']})
    assert len(records)==1260
    write_json(out/'reuse_manifest.json',{'reused':len(records),'arms':['R','S','B'],
        'old_requests_sha256':sha(PREVIOUS_RUN/'requests.jsonl'),
        'new_requests_sha256':sha(run/'requests.jsonl'),
        'source_archive_checksums_sha256':sha(old/'CHECKSUMS.json'),
        'engine_inputs':binding,'model_identity':read_json(old/'model_identity.json'),'records':records})
    print(f'{len(records)} R/S/B answers reused byte-for-byte; no H answers reused')


if __name__=='__main__': main()
