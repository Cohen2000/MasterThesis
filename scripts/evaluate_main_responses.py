#!/usr/bin/env python3
"""Validity and conditional accuracy, with no imputation and no model calls."""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (REAL_TEST,ARMS,CONFIGS,DESIGN_VERSION,read_json,
                                    write_json,sha,digest,LLM_REPEATS)
from main_experiment.evaluation import resolve,errors,paired_summary,conditional_summary,EVALUATION_VERSION
from main_experiment.observation import parse
from main_experiment.pipeline import csv_write,binding

STRATA=['real','dar_a0','dar_a08','ad_memoryless','ad_memory']
REFERENCES={'baseline':'primary_corrector','plugin':'plugin',
            'extratrees':'extratrees_pooled','median':'median'}


def _in_stratum(r,s):
    return r['stratum']=='real' if s=='real' else r['graph_id'].startswith(s+'_r')


def evaluate(run,response_file,out,mock=False,baselines=None):
    run=Path(run); out=Path(out)
    if not baselines: raise ValueError('decided primary baselines are required')
    if mock and 'mock' not in str(out).lower(): raise ValueError('mock directory must contain mock')
    if out.resolve()==run.resolve(): raise ValueError('evaluation needs a separate directory')
    if read_json(run/'report.json')['design_version']!=DESIGN_VERSION:
        raise ValueError('historical run: use its archived evaluator')
    primary=read_json(baselines)
    if primary['design_version']!=DESIGN_VERSION: raise ValueError('baseline design mismatch')
    planned=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()]
    known={r['id']:r for r in planned}
    if len(known)!=len(planned): raise ValueError('duplicate planned ID')
    observations={}
    for r in planned:
        if r['payload_sha256']!=digest(r['payload']): raise ValueError('payload hash mismatch')
        oid=r['observation_id']
        if oid not in observations:
            observations[oid]=read_json(run/'observations/sample'/f'{oid}.json')
        o=observations[oid]
        if digest(o['messages'])!=r['prompt_sha256']: raise ValueError('prompt hash mismatch')
        if o['block_sha256']!=digest(o['block']): raise ValueError('block hash mismatch')
        b=primary['observations'][oid]
        if b['block_sha256']!=o['block_sha256'] or b['truth']!=o['truth']:
            raise ValueError('baseline observation mismatch')
    records={}
    if response_file:
        for line in Path(response_file).read_text().splitlines():
            r=json.loads(line); rid=r['id']
            if rid not in known or rid in records: raise ValueError('unknown or duplicate response ID')
            if bool(r.get('mock',False))!=mock: raise ValueError('mock / production mismatch')
            if r.get('extraction'): raise ValueError('post-hoc extraction is not a main response')
            for key in ('prompt_sha256','payload_sha256'):
                if r.get(key)!=known[rid][key]: raise ValueError(f'{key} missing or different: {rid}')
            if r.get('terminal') and 'final_text' not in r and not r.get('technical_error'):
                raise ValueError('terminal response lacks final_text')
            records[rid]=r
    out.mkdir(parents=True,exist_ok=True)
    binding(out/'evaluation_inputs.json',{
        'evaluation_version':EVALUATION_VERSION,'mock':mock,
        'requests_sha256':sha(run/'requests.jsonl'),'baselines_sha256':sha(baselines),
        'responses_sha256':sha(response_file) if response_file else None,
        'observations_sha256':digest(observations),
        'evaluator_sha256':sha(__file__),
        'parser_sha256':sha(Path(__file__).resolve().parents[1]/'src/main_experiment/evaluation.py')})
    if response_file: (out/'raw_responses.jsonl').write_bytes(Path(response_file).read_bytes())
    rows=[]
    for r in planned:
        obs=observations[r['observation_id']]; b=primary['observations'][obs['id']]
        outcome=resolve(parse(obs['block']),record=records.get(r['id']))
        row={k:r[k] for k in ('id','graph_id','arm','sample_index','repeat_index','config_id','stratum')}
        row.update(outcome); row.update(errors(outcome['prediction'],obs['truth']))
        row.update(observation_id=obs['id'],empty=obs['empty'],mock=mock,
                   prediction_json=outcome['prediction'],budget_matched=obs['budget_matched'],
                   reference_name=b['primary_corrector_name'],
                   reference_status=b['primary_corrector']['status'],
                   reference_fallback=b['primary_corrector'].get('fallback',''))
        for prefix,key in REFERENCES.items():
            err=errors(b[key]['prediction'],obs['truth'])
            for metric in ('AE2','ProfileAE'):
                row[f'{prefix}_{metric}_all']=err[metric]
                row[f'{prefix}_{metric}_matched']=err[metric] if row['valid'] else None
                row[f'delta_{metric}_vs_{prefix}']=(row[metric]-err[metric]
                    if row[metric] is not None and err[metric] is not None else None)
        rows.append(row)
    csv_write(out/'answer_errors.csv',rows)
    summary=[]; source_rows=[]
    for stratum in STRATA:
        for arm in ARMS:
            for config in CONFIGS:
                group=[r for r in rows if r['arm']==arm and r['config_id']==config and _in_stratum(r,stratum)]
                if not group: continue
                sources=sorted({r['graph_id'] for r in group})
                expected={s:len({r['sample_index'] for r in group if r['graph_id']==s}) for s in sources}
                cells={s:np.full((expected[s],LLM_REPEATS),np.nan) for s in sources}
                slots=set()
                for r in group:
                    slot=(r['graph_id'],r['sample_index'],r['repeat_index'])
                    if slot in slots: raise ValueError('duplicate sampling slot')
                    slots.add(slot)
                shape_complete=len(slots)==sum(expected.values())*LLM_REPEATS
                if stratum=='real': shape_complete &= set(sources)==set(REAL_TEST)
                complete=bool(shape_complete and all(r['terminal'] for r in group))
                base={'stratum':stratum,'arm':arm,'config_id':config,'mock':mock}
                result={**base,'complete':complete,'planned':len(group),
                        'started':sum(r['started'] for r in group),
                        'terminal':sum(r['terminal'] for r in group),
                        'valid_answers':sum(bool(r['valid']) for r in group),
                        'invalid_terminal':sum(r['terminal'] and not r['valid'] for r in group),
                        'not_started':sum(r['status']=='not_started' for r in group),
                        'in_progress':sum(r['status']=='in_progress' for r in group),
                        'empty_skips':sum(r['empty'] for r in group),
                        'technical_failures':sum(bool(r.get('technical_error')) for r in group),
                        'limit_hits':sum(bool(r.get('limit_hit')) for r in group),
                        'fenced_valid':sum(r.get('validation_reason')=='valid_after_fence' for r in group),
                        'accuracy_condition':'valid answers only; no imputation; equal source weights'}
                metrics=['valid_fraction','AE2','ProfileAE','signed_rho2']+[
                    key for key in group[0] if key.endswith(('_all','_matched')) or key.startswith('delta_')]
                for metric in metrics:
                    for a in cells.values(): a.fill(np.nan)
                    for r in group:
                        value=float(bool(r['valid'])) if metric=='valid_fraction' and r['terminal'] else r.get(metric)
                        if value is not None: cells[r['graph_id']][r['sample_index']-1,r['repeat_index']-1]=value
                    full_metric=metric=='valid_fraction' or metric.endswith('_all')
                    # Baseline values can be shown before inference, but no interim LLM accuracy.
                    reportable=complete or (metric.endswith('_all') and shape_complete)
                    est=(paired_summary(cells,expected) if full_metric else conditional_summary(cells,expected)) if reportable else None
                    result[metric]=est['mean'] if est else None
                    result[metric+'_MCSE']=est['mcse'] if est else None
                    if metric=='AE2': result['sources_with_valid_answers']=est['sources_with_valid_answers'] if est else 0
                    if est:
                        for source,v in est['sources'].items():
                            source_rows.append({**base,'source':source,'metric':metric,
                                                'value':v['mean'],'MCSE':v['mcse'],**{k:v[k] for k in ('draws','valid_answers','planned_answers')}})
                result['numeric_model_estimate']=result['AE2'] is not None
                summary.append(result)
    csv_write(out/'summary.csv',summary); csv_write(out/'source_results.csv',source_rows)
    write_json(out/'report.json',{'mock':mock,'design_version':DESIGN_VERSION,'evaluation_version':EVALUATION_VERSION,
        'logical_requests':len(rows),'primary_baselines_sha256':sha(baselines),
        'complete_main_result':bool(summary) and all(r['complete'] for r in summary),
        'metric_label':'validity and conditional MAE; no replacement',
        'accuracy_undefined_cells':[f"{r['stratum']}/{r['arm']}/{r['config_id']}" for r in summary if r['complete'] and not r['numeric_model_estimate']]})


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--baselines',required=True)
    p.add_argument('--responses'); p.add_argument('--out',required=True); p.add_argument('--mock',action='store_true')
    a=p.parse_args(); evaluate(a.run,a.responses,a.out,a.mock,a.baselines)
