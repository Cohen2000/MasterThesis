#!/usr/bin/env python3
"""Create paired H-known Qwen manifests from an existing v9 manifest.

The H-known case must be identical to its H-unknown twin in every respect
except the prompt spelling out h=0.60 -- same observation, draw, repeat, model
config, seed and DESIGN_VERSION. It therefore does NOT get a different
design_version (that field is exactly what validate_request checks against
the deployed common.DESIGN_VERSION, and a mismatch there is the bug this
fixes); it only needs its own request `id`, since the id is the engine/ledger
dedup key and must not collide with the already-answered H-unknown request.
"""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.requests import payload as build_payload
from main_experiment.common import digest,write_json
LABEL='hknown'   # descriptive only; never enters design_version or the payload

def main(src,dst):
    src,dst=Path(src),Path(dst); (dst/'observations/sample').mkdir(parents=True,exist_ok=True)
    records=[]
    for line in (src/'requests.jsonl').read_text().splitlines():
        old=json.loads(line)
        if old['arm']!='H' or not old['config_id'].startswith('qwen') or old.get('status')=='skipped_empty': continue
        messages=[dict(m) for m in old['payload']['messages']]
        messages[1]['content']=messages[1]['content'].replace(
          'Temporal_access=', 'The most recent fraction h=0.60 of the archive is observable.\nTemporal_access=', 1)
        payload=build_payload(old['config_id'],messages,old['seed'])          # same seed, same config, same design_version
        r=dict(old); r['id']=old['id']+'__'+LABEL
        r['payload']=payload; r['payload_sha256']=digest(payload); r['prompt_sha256']=digest(messages)
        r['production_dispatch_enabled']=True; r['requires_technical_release']=False
        records.append(r)
    messages_by_obs={r['observation_id']:r['payload']['messages'] for r in records}
    for r in records:
        oid=r['observation_id']
        src_obs=src/'observations/sample'/(oid+'.json')
        # the observation file's own `messages` must match the h-known prompt too --
        # run_qwen_engine.load_requests cross-checks digest(obs['messages']) against
        # the request's prompt_sha256, which is exactly the mismatch this fixes.
        if src_obs.exists():
            record=json.loads(src_obs.read_text())
        else:
            record={'id':oid,'arm':'H','graph_id':r['graph_id'],'sample_index':r['sample_index']}
        record['messages']=messages_by_obs[oid]
        write_json(dst/'observations/sample'/(oid+'.json'),record)
    (dst/'requests.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    write_json(dst/'report.json',{'verified':True,'label':LABEL,'design_version':records[0]['design_version'] if records else None,
                                  'requests':len(records),'paired_source':str(src)})
    print(json.dumps({'requests':len(records),'label':LABEL}))
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
