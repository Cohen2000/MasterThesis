#!/usr/bin/env python3
"""Enumerate every production/offline stream and audit seed-field collisions."""
import sys,json
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
from main_experiment.pool import pool_definition
run=ROOT/CURRENT_RUN;rev=ROOT/CURRENT_REVISION
for row in read_json(run/'seed_manifest.json'):
    assert row['fields'][0]==MASTER_SEED
    assert seed(*row['fields'][1:])==row['seed']
specs=pool_definition()['graphs']
for sp in specs:
    key=sp['key'];b=read_json(rev/'pool/calibration'/key/'budget.json')
    seed('pool',key)
    for i in range(1,257):seed('walk_calibration_cells',key,ARM_ID['S'],i)
    for i in range(1,b['validation_n']+1):seed('walk_validation_cells',key,ARM_ID['S'],i)
    for arm in ARMS:
        for i in range(1,draws_for(arm,b,'pool_'+sp['partition'])+1):seed('pool_'+sp['partition'],key,ARM_ID[arm],i)
for key in MAIN_KEYS:
    for i in range(1,33):seed('srw_diagnostic',key,ARM_ID['S'],i)
for key in REAL_TEST:
    for i in range(1,100):seed('pwt_null_diagnostic',key,sample_index=i)
    seed('pwt_productive',key)
counts=Counter(json.loads(v)[1] for v in SEEDS.values())
path=rev/'complete_seed_manifest.jsonl'
with path.open('w') as f:
    for s,fields in sorted(SEEDS.items()):f.write(json.dumps({'seed':s,'fields':json.loads(fields)},separators=(',',':'))+'\n')
old=read_json(PREVIOUS_RUN/'seed_manifest.json')
assert not set(SEEDS)&{r['seed'] for r in old}
write_json(rev/'seed_audit.json',{'master_seed':MASTER_SEED,'design_version':DESIGN_VERSION,'unique_streams':len(SEEDS),
    'domain_counts':dict(counts),'collision_audit_passed':True,'previous_generation_seed_overlap':0,
    'manifest_sha256':sha(path),'crn_aliases':{g:parent_source(g) for g in SURROGATES},
    'alias_scope':'sample R/H/S/B only; intentional same underlying stream, not accidental collision'})
print(json.dumps(read_json(rev/'seed_audit.json'),indent=2))
