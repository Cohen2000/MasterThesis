#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
p=argparse.ArgumentParser();p.add_argument('--archive',required=True);a=p.parse_args();folder=Path(a.archive)
sums=read_json(folder/'CHECKSUMS.json')
for name,h in sums.items():
    assert not Path(name).is_absolute() and '..' not in Path(name).parts
    assert sha(folder/name)==h,name
run=ROOT/CURRENT_RUN
assert sha(folder/'requests.jsonl')==sha(run/'requests.jsonl')
for f in (folder/'observations/sample').glob('*.json'):assert sha(f)==sha(run/'observations/sample'/f.name)
requests={r['id']:r for r in map(json.loads,(run/'requests.jsonl').read_text().splitlines()) if r['config_id'].startswith('qwen') and r['status']!='skipped_empty'}
answers=[read_json(p) for p in (folder/'answers').glob('*_r*/*.json')]
assert len(answers)==len(requests) and {r['id'] for r in answers}==set(requests)
for r in answers:
    for k in ('prompt_sha256','payload_sha256','seed'):assert r[k]==requests[r['id']][k]
    assert r['runner_sha256']==sha(ROOT/'scripts/run_qwen_engine.py')
identity=read_json(folder/'model_identity.json');assert identity['revision_pinned']=='995ad96eacd98c81ed38be0c5b274b04031597b0'
cluster_record=read_json(ROOT/'docs/CLUSTER_PANEL888.json')
assert (folder/'SPEC_COMMIT').read_text().strip()==cluster_record['source_commit']
result={'verified':True,'checksums':len(sums),'answers':len(answers),'requests_sha256':sha(folder/'requests.jsonl'),
    'model_identity_sha256':sha(folder/'model_identity.json'),'source_commit':(folder/'SPEC_COMMIT').read_text().strip()}
write_json(folder.parent/(folder.name+'_verification.json'),result);print(json.dumps(result,indent=2))
