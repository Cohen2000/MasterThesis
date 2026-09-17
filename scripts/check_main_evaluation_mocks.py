#!/usr/bin/env python3
"""Offline integration check with explicitly marked fake responses, never inference."""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import write_json,read_json
from evaluate_main_responses import evaluate

p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--out',required=True); p.add_argument('--baselines'); a=p.parse_args()
out=Path(a.out)
if 'mock' not in str(out).lower(): raise ValueError('explicit mock output location required')
out.mkdir(parents=True,exist_ok=True)
requests=[json.loads(x) for x in (Path(a.run)/'requests.jsonl').read_text().splitlines()]
records=[]
for i,r in enumerate(requests):
    record={'id':r['id'],'mock':True,'started':True,'terminal':True,
            'final_text':'{"rho_2":0.4,"rho_3":0.3,"rho_4":0.2,"rho_5":0.1}',
            'reasoning':'MOCK reasoning includes {"rho_2":999}; it must never be parsed.',
            'finish_reason':'mock_stop','limit_hit':i==2,'technical_error':False}
    if i==0: record['final_text']='MOCK invalid answer'
    if i==1: record['final_text']=''; record['technical_error']=True
    if i==3: record['started']=False; record['terminal']=False
    if i==4: record['terminal']=False
    records.append(record)
source=out/'mock_responses.jsonl'
source.write_text(''.join(json.dumps(r)+'\n' for r in records))
evaluate(a.run,source,out/'evaluation',True,a.baselines)
rows=list(csv.DictReader(open(out/'evaluation/answer_errors.csv')))
assert len(rows)==len(requests)==read_json(Path(a.run)/'report.json')['planned_logical_calls']
assert rows[0]['replacement']=='plugin' and rows[1]['replacement']=='plugin'
assert rows[2]['valid']=='True' and rows[2]['limit_hit']=='True'
assert rows[3]['status']=='not_started' and rows[3]['AE2']==''
assert rows[4]['status']=='in_progress' and rows[4]['AE2']==''
report=read_json(out/'evaluation/report.json')
assert report['mock'] and not report['complete_main_result']
# No accidental ingestion as genuine results.
try: evaluate(a.run,source,out/'forbidden_real',False)
except ValueError: pass
else: raise AssertionError('mock records accepted as real')
# Duplicate logical request IDs must not be counted twice.
duplicate=out/'duplicate_mock.jsonl'; duplicate.write_text(json.dumps(records[0])+'\n'+json.dumps(records[0])+'\n')
try: evaluate(a.run,duplicate,out/'forbidden_duplicate_mock',True)
except ValueError: pass
else: raise AssertionError('duplicate request ID accepted')
write_json(out/'check_report.json',{'mock_only':True,'inference_calls':0,'rows_checked':len(rows),
                                  'replacement_and_pending_states_checked':True,
                                  'mock_as_real_rejected':True,'duplicate_id_rejected':True})
print(json.dumps(read_json(out/'check_report.json'),indent=2))
