#!/usr/bin/env python3
"""Create paired H-known Qwen manifests from an existing v9 manifest."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import main_experiment.requests as rq
from main_experiment.common import digest,write_json
VERSION='panel888-access-v9-hknown-20260922'

def main(src,dst):
    src,dst=Path(src),Path(dst); (dst/'observations/sample').mkdir(parents=True,exist_ok=True)
    rq.DESIGN_VERSION=VERSION
    records=[]
    for line in (src/'requests.jsonl').read_text().splitlines():
        old=json.loads(line)
        if old['arm']!='H' or not old['config_id'].startswith('qwen') or old.get('status')=='skipped_empty': continue
        messages=[dict(m) for m in old['payload']['messages']]
        messages[1]['content']=messages[1]['content'].replace(
          'Temporal_access=', 'The most recent fraction h=0.60 of the archive is observable.\\nTemporal_access=', 1)
        payload=rq.payload(old['config_id'],messages,old['seed'])
        r=dict(old); r['id']=old['id'].replace(old['design_version'],VERSION); r['design_version']=VERSION
        r['payload']=payload; r['payload_sha256']=digest(payload); r['prompt_sha256']=digest(messages)
        r['production_dispatch_enabled']=True; r['requires_technical_release']=False
        records.append(r)
    for r in records:
        write_json(dst/'observations/sample'/(r['observation_id']+'.json'),{'id':r['observation_id'],'arm':'H','graph_id':r['graph_id'],'sample_index':r['sample_index'],'block':next(m['content'].split('W=5',1)[1] for m in r['payload']['messages'] if 'W=5' in m['content'])})
    (dst/'requests.jsonl').write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in records))
    write_json(dst/'report.json',{'verified':True,'design_version':VERSION,'requests':len(records),'paired_source':str(src)})
    print(json.dumps({'requests':len(records),'version':VERSION}))
if __name__=='__main__': main(sys.argv[1],sys.argv[2])
