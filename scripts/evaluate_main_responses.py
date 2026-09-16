#!/usr/bin/env python3
"""Evaluate supplied final responses; never calls a model. Mocks require --mock."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import REAL_TEST,ARMS,CONFIGS,read_json,write_json,sha
from main_experiment.observation import parse
from main_experiment.baselines import corrector,plugin
from main_experiment.evaluation import resolve,errors,paired_summary
from main_experiment.pipeline import csv_write


def evaluate(run,response_file,out,mock=False):
    run=Path(run); out=Path(out)
    if mock and 'mock' not in str(out).lower(): raise ValueError('mock output directory must contain mock')
    if out.resolve()==run.resolve(): raise ValueError('evaluation must have its own output directory')
    planned=[json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
    known={r['id']:r for r in planned}; records={}
    if response_file:
        for line in Path(response_file).read_text().splitlines():
            r=json.loads(line)
            if bool(r.get('mock',False))!=mock: raise ValueError('mock and real responses must not mix')
            if r['id'] not in known or r['id'] in records: raise ValueError('unknown or duplicate logical request ID')
            if 'final_text' not in r and r.get('terminal') and not r.get('technical_error'):
                raise ValueError('terminal response lacks explicit final_text; reasoning is never parsed')
            records[r['id']]=r
    out.mkdir(parents=True,exist_ok=True)
    # Preserve source bytes, reasoning, finish reason and usage without modifying them.
    if response_file:
        dest=out/'raw_responses.jsonl'
        content=Path(response_file).read_bytes()
        if dest.exists() and dest.read_bytes()!=content: raise ValueError('immutable evaluation inputs changed')
        dest.write_bytes(content)
    rows=[]
    for request in planned:
        obs=read_json(run/'observations/sample'/(request['observation_id']+'.json'))
        fold=obs['graph_id'] if obs['stratum']=='real' else 'synthetic'
        median=read_json(run/'models'/fold/'manifest.json')['median']
        o=parse(obs['block']); outcome=resolve(o,median,records.get(request['id']))
        bases=read_json(run/'baselines'/(obs['id']+'.json'))
        base_error=errors(bases['corrector']['prediction'],obs['truth'])
        err=errors(outcome['prediction'],obs['truth'])
        rows.append({k:request[k] for k in ['id','graph_id','arm','sample_index','repeat_index','config_id','stratum']} |
                    outcome | err | {'empty':obs['empty'],'baseline_AE2':base_error['AE2'],
                    'baseline_ProfileAE':base_error['ProfileAE'],
                    'delta_AE2':None if err['AE2'] is None else err['AE2']-base_error['AE2'],
                    'delta_ProfileAE':None if err['ProfileAE'] is None else err['ProfileAE']-base_error['ProfileAE'],
                    'budget_matched':obs['budget_matched'],'mock':mock})
    csv_write(out/'answer_errors.csv',rows)
    summary=[]; source_rows=[]; conditional=[]
    for stratum in ['real','dar_a0','dar_a08','ad_memoryless','ad_memory']:
        for arm in ARMS:
            for config in CONFIGS:
                group=[r for r in rows if r['arm']==arm and r['config_id']==config and
                       (r['stratum']=='real' if stratum=='real' else r['graph_id'].startswith(stratum+'_r'))]
                if not group: continue
                base={'stratum':stratum,'arm':arm,'config_id':config,'mock':mock}
                begun=[r for r in group if r['started']]
                complete=all(r['prediction'] is not None for r in group)
                rates={name:(sum(bool(r.get(field)) for r in begun)/len(begun) if begun else None)
                       for name,field in [('valid_fraction','valid'),('limit_fraction','limit_hit'),('technical_failure_fraction','technical_error')]}
                rates['other_answer_error_fraction']=(sum(r['status']=='terminal' and not r['valid'] and
                       not r.get('technical_error') and not r.get('limit_hit') for r in begun)/len(begun) if begun else None)
                rates['replacement_fraction']=(sum(r['replacement'] is not None for r in begun)/len(begun) if begun else None)
                result={**base,**rates,'started_logical_requests':len(begun),
                        'not_started':sum(r['status']=='not_started' for r in group),
                        'in_progress':sum(r['status']=='in_progress' for r in group),
                        'empty_observations':sum(r['empty'] for r in group)//3,
                        'empty_fraction':sum(r['empty'] for r in group)/len(group),'complete':complete}
                sources=sorted({r['graph_id'] for r in group})
                if stratum=='real' and set(sources)!=set(REAL_TEST): complete=False; result['complete']=False
                for metric in ['AE2','ProfileAE','delta_AE2','delta_ProfileAE']:
                    cells={}
                    for source in sources:
                        a=np.full((1 if arm=='H' else 5,3),np.nan)
                        for r in group:
                            if r['graph_id']==source: a[r['sample_index']-1,r['repeat_index']-1]=r[metric] if r[metric] is not None else np.nan
                        if np.isfinite(a).all():
                            one=paired_summary({source:a})
                            source_group=[r for r in group if r['graph_id']==source]
                            source_begun=[r for r in source_group if r['started']]
                            source_rates={name:(sum(bool(r.get(field)) for r in source_begun)/len(source_begun) if source_begun else None)
                                for name,field in [('valid_fraction','valid'),('limit_fraction','limit_hit'),('technical_failure_fraction','technical_error')]}
                            source_rates['replacement_fraction']=sum(r['replacement'] is not None for r in source_begun)/len(source_begun) if source_begun else None
                            source_rates['empty_fraction']=sum(r['empty'] for r in source_group)/len(source_group)
                            source_rows.append({**base,'source':source,'metric':metric,'value':one['mean'],'MCSE':one['mcse'],**source_rates})
                        cells[source]=a
                    if complete:
                        estimated=paired_summary(cells); result[metric]=estimated['mean']; result[metric+'_MCSE']=estimated['mcse']
                    else: result[metric]=None; result[metric+'_MCSE']=None
                summary.append(result)
                # Explicitly conditional; average valid answers within observation,
                # then observations within source, then sources with any valid case.
                valid=[r for r in group if r['valid'] and r['status']=='terminal']
                per_source=[]
                for source in sources:
                    per_obs=[]
                    for ix in range(1,2 if arm=='H' else 6):
                        vs=[r for r in valid if r['graph_id']==source and r['sample_index']==ix]
                        if vs: per_obs.append([np.mean([r[k] for r in vs]) for k in ['AE2','baseline_AE2','delta_AE2','ProfileAE','baseline_ProfileAE']])
                    if per_obs: per_source.append(np.mean(per_obs,axis=0))
                values=np.mean(per_source,axis=0).tolist() if per_source else [None]*5
                conditional.append({**base,**rates,'valid_answers':len(valid),'sources_with_valid_answers':len(per_source),
                    **dict(zip(['MAE2','matched_baseline_MAE2','paired_delta','ProfileMAE','matched_baseline_ProfileMAE'],values)),
                    'interpretation':'conditional on valid raw answers; not the main result'})
    csv_write(out/'summary.csv',summary); csv_write(out/'source_results.csv',source_rows)
    csv_write(out/'valid_only.csv',conditional)
    write_json(out/'report.json',{'mock':mock,'logical_requests':len(rows),'input_sha256':sha(response_file) if response_file else None,
         'started':sum(r['started'] for r in rows),'not_started':sum(r['status']=='not_started' for r in rows),
         'empty_skips':sum(r['status']=='empty' for r in rows),'complete_main_result':all(r['complete'] for r in summary),
         'metric_label':'MAE2 with fixed replacement rule'})

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--responses')
    p.add_argument('--out',required=True); p.add_argument('--mock',action='store_true')
    a=p.parse_args(); evaluate(a.run,a.responses,a.out,a.mock)
