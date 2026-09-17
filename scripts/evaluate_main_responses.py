#!/usr/bin/env python3
"""Evaluate supplied final responses; never calls a model. Mocks require --mock.

Main result: pipeline MAE2 under parser rule v2 (parse_final) with the fixed
replacement rule, paired against the decided references in primary_baselines.json.
Secondary, all labelled as such in the output:
  * the fold-specific real training median as a visible constant reference;
  * a fixed 50/50 shrinkage of each answer towards the plug-in and, separately,
    towards the training median -- post-hoc diagnostics, never tuned;
  * the componentwise median of the three answers of one observation;
  * the spread of the three answers of one observation;
  * per-source results.
A cell in which no answer is valid is reported without any numeric model value:
its pipeline value would be the plug-in replacement, not an estimate of the model.
Parser rule v3 is produced by collecting with --extract-trailing-json and
evaluating into a separate directory; it never changes this rule.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (REAL_TEST,ARMS,CONFIGS,read_json,write_json,sha,LLM_REPEATS)
from main_experiment.observation import parse
from main_experiment.baselines import plugin
from main_experiment.evaluation import resolve,errors,paired_summary
from main_experiment.pipeline import csv_write

STRATA=['real','dar_a0','dar_a08','ad_memoryless','ad_memory']
# Metrics that describe the model's own answers. They are withheld for a cell
# whose answers were all replaced, because there they would describe the plug-in.
MODEL_METRICS=['AE2','ProfileAE','delta_AE2','delta_ProfileAE','delta_AE2_vs_plugin',
               'delta_AE2_vs_extratrees','delta_AE2_vs_median',
               'shrink_plugin_AE2','shrink_median_AE2','signed_rho2']
REFERENCE_METRICS=['baseline_AE2','plugin_AE2','extratrees_pooled_AE2','median_AE2']


def _in_stratum(r,stratum):
    return r['stratum']=='real' if stratum=='real' else r['graph_id'].startswith(stratum+'_r')


def evaluate(run,response_file,out,mock=False,baselines=None):
    run=Path(run); out=Path(out)
    if mock and 'mock' not in str(out).lower(): raise ValueError('mock output directory must contain mock')
    if out.resolve()==run.resolve(): raise ValueError('evaluation must have its own output directory')
    # The decided references. Without them the comparison would use
    # run/baselines/*.json, which holds development references.
    primary=read_json(baselines) if baselines else None
    if primary and primary.get('design_version') not in (None,read_json(run/'report.json').get('design_version')):
        raise ValueError('primary baselines belong to a different design version')
    planned=[json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
    known={r['id']:r for r in planned}; records={}
    if response_file:
        for line in Path(response_file).read_text().splitlines():
            r=json.loads(line)
            if bool(r.get('mock',False))!=mock: raise ValueError('mock and real responses must not mix')
            if r['id'] not in known or r['id'] in records: raise ValueError('unknown or duplicate logical request ID')
            if 'final_text' not in r and r.get('terminal') and not r.get('technical_error'):
                raise ValueError('terminal response lacks explicit final_text; reasoning is never parsed')
            if primary is not None and r.get('prompt_sha256') not in (None,known[r['id']]['prompt_sha256']):
                raise ValueError(f'response {r["id"]} was generated for a different prompt')
            records[r['id']]=r
    out.mkdir(parents=True,exist_ok=True)
    if response_file:
        dest=out/'raw_responses.jsonl'
        content=Path(response_file).read_bytes()
        if dest.exists() and dest.read_bytes()!=content: raise ValueError('immutable evaluation inputs changed')
        dest.write_bytes(content)
    observations={}
    for request in planned:
        oid=request['observation_id']
        if oid not in observations: observations[oid]=read_json(run/'observations/sample'/(oid+'.json'))
    # Distinct sampler draws per (graph, arm), from the run itself.
    draws={}
    for o in observations.values(): draws[(o['graph_id'],o['arm'])]=draws.get((o['graph_id'],o['arm']),0)+1
    rows=[]
    for request in planned:
        obs=observations[request['observation_id']]
        fold=obs['graph_id'] if obs['stratum']=='real' else 'synthetic'
        median=read_json(run/'models'/fold/'manifest.json')['median']
        o=parse(obs['block']); outcome=resolve(o,median,records.get(request['id']))
        bases=read_json(run/'baselines'/(obs['id']+'.json'))
        bounds=None
        if primary:
            e=primary['observations'][obs['id']]
            ref=e['primary_corrector']['prediction']; ref_name=e['primary_corrector_name']
            trained=e['extratrees_pooled']['prediction']
            if e['median']['prediction']!=list(median): raise ValueError('median mismatch between run and primary baselines')
            bounds=e.get('h_bounds')
        else:
            ref=bases['corrector']['prediction']; ref_name='corrector'; trained=None
        truth=obs['truth']
        plug=bases['plugin']['prediction']
        base_error=errors(ref,truth); plug_error=errors(plug,truth)
        med_error=errors(median,truth)
        trained_error=errors(trained,truth) if trained else {'AE2':None,'ProfileAE':None}
        pred=outcome['prediction']
        err=errors(pred,truth)
        shrink_p=errors(None if pred is None else [(a+b)/2 for a,b in zip(pred,plug)],truth)
        shrink_m=errors(None if pred is None else [(a+b)/2 for a,b in zip(pred,median)],truth)
        d=lambda x,y: None if x is None or y is None else x-y
        inside=None
        if bounds and outcome.get('valid'):
            inside=bool(bounds['lower'][0]<=pred[0]<=bounds['upper'][0])
        rows.append({k:request[k] for k in ['id','graph_id','arm','sample_index','repeat_index','config_id','stratum']} |
                    outcome | err | {'empty':obs['empty'],'deterministic_draw':draws[(obs['graph_id'],obs['arm'])]==1,
                    'observation_id':obs['id'],'prediction_json':None if pred is None else json.dumps(pred),
                    'truth_rho2':truth[0],
                    'baseline_AE2':base_error['AE2'],'baseline_ProfileAE':base_error['ProfileAE'],
                    'delta_AE2':d(err['AE2'],base_error['AE2']),
                    'delta_ProfileAE':d(err['ProfileAE'],base_error['ProfileAE']),
                    'budget_matched':obs['budget_matched'],'mock':mock,
                    'reference_name':ref_name,
                    'plugin_AE2':plug_error['AE2'],
                    'delta_AE2_vs_plugin':d(err['AE2'],plug_error['AE2']),
                    'median_AE2':med_error['AE2'],
                    'delta_AE2_vs_median':d(err['AE2'],med_error['AE2']),
                    'extratrees_pooled_AE2':trained_error['AE2'],
                    'delta_AE2_vs_extratrees':d(err['AE2'],trained_error['AE2']),
                    'shrink_plugin_AE2':shrink_p['AE2'],'shrink_median_AE2':shrink_m['AE2'],
                    'rho2_inside_h_bounds':inside})
    csv_write(out/'answer_errors.csv',rows)
    summary=[]; source_rows=[]; conditional=[]; secondary=[]
    for stratum in STRATA:
        for arm in ARMS:
            for config in CONFIGS:
                group=[r for r in rows if r['arm']==arm and r['config_id']==config and _in_stratum(r,stratum)]
                if not group: continue
                base={'stratum':stratum,'arm':arm,'config_id':config,'mock':mock}
                begun=[r for r in group if r['started']]
                complete=all(r['prediction'] is not None for r in group)
                rates=_rates(begun)
                valid_answers=sum(bool(r.get('valid')) for r in begun)
                # No numeric model value where every answer was replaced.
                fallback_only=bool(begun) and valid_answers==0
                result={**base,**rates,'started_logical_requests':len(begun),
                        'valid_answers':valid_answers,'fallback_only':fallback_only,
                        'numeric_model_estimate':bool(complete and not fallback_only),
                        'not_started':sum(r['status']=='not_started' for r in group),
                        'in_progress':sum(r['status']=='in_progress' for r in group),
                        'empty_observations':sum(r['empty'] for r in group)//LLM_REPEATS,
                        'empty_fraction':sum(r['empty'] for r in group)/len(group),
                        'deterministic_draw_sources':sorted({r['graph_id'] for r in group if r['deterministic_draw']}),
                        'complete':complete}
                sources=sorted({r['graph_id'] for r in group})
                if stratum=='real' and set(sources)!=set(REAL_TEST): complete=False; result['complete']=False
                expected={s:draws[(s,arm)] for s in sources}
                for metric in MODEL_METRICS+REFERENCE_METRICS:
                    cells={}
                    for source in sources:
                        a=np.full((expected[source],LLM_REPEATS),np.nan)
                        for r in group:
                            if r['graph_id']==source and r[metric] is not None:
                                a[r['sample_index']-1,r['repeat_index']-1]=r[metric]
                        if np.isfinite(a).all():
                            one=paired_summary({source:a},{source:expected[source]})
                            src=one['sources'][source]
                            source_group=[r for r in group if r['graph_id']==source]
                            srates=_rates([r for r in source_group if r['started']])
                            srates['empty_fraction']=sum(r['empty'] for r in source_group)/len(source_group)
                            withheld=metric in MODEL_METRICS and srates['valid_fraction']==0
                            source_rows.append({**base,'source':source,'metric':metric,
                                'value':None if withheld else one['mean'],
                                'MCSE':None if withheld else one['mcse'],
                                'MCSE_model_repeats':None if withheld else src['mcse_model_repeats'],
                                'MCSE_sampler':None if withheld else src['mcse_sampler'],
                                'draws':expected[source],'withheld_fallback_only':withheld,**srates})
                        cells[source]=a
                    # A metric whose reference is absent (no primary baselines given)
                    # stays empty instead of failing the whole evaluation.
                    finite=all(np.isfinite(c).all() for c in cells.values())
                    if complete and finite and not (fallback_only and metric in MODEL_METRICS):
                        est=paired_summary(cells,expected)
                        result[metric]=est['mean']; result[metric+'_MCSE']=est['mcse']
                        result[metric+'_MCSE_model_repeats']=est['mcse_model_repeats']
                        result[metric+'_MCSE_sampler']=est['mcse_sampler']
                        result[metric+'_between_source_SE']=est['between_source_se']
                    else:
                        result[metric]=None; result[metric+'_MCSE']=None
                if arm=='H':
                    inside=[r['rho2_inside_h_bounds'] for r in group if r['rho2_inside_h_bounds'] is not None]
                    result['valid_rho2_inside_h_bounds_share']=float(np.mean(inside)) if inside else None
                summary.append(result)
                secondary.append({**base,**_secondary(group,sources,expected,complete and not fallback_only)})
                # Conditional side analysis: valid answers only, averaged within
                # observation, then observations within source, then sources.
                valid=[r for r in group if r['valid'] and r['status']=='terminal']
                per_source=[]
                for source in sources:
                    per_obs=[]
                    for ix in range(1,expected[source]+1):
                        vs=[r for r in valid if r['graph_id']==source and r['sample_index']==ix]
                        if vs: per_obs.append([np.mean([r[k] for r in vs]) for k in ['AE2','baseline_AE2','delta_AE2','ProfileAE','baseline_ProfileAE']])
                    if per_obs: per_source.append(np.mean(per_obs,axis=0))
                values=np.mean(per_source,axis=0).tolist() if per_source else [None]*5
                conditional.append({**base,**rates,'valid_answers':len(valid),'sources_with_valid_answers':len(per_source),
                    **dict(zip(['MAE2','matched_baseline_MAE2','paired_delta','ProfileMAE','matched_baseline_ProfileMAE'],values)),
                    'interpretation':'conditional on valid raw answers; not the main result'})
    csv_write(out/'summary.csv',summary); csv_write(out/'source_results.csv',source_rows)
    csv_write(out/'valid_only.csv',conditional); csv_write(out/'secondary.csv',secondary)
    write_json(out/'report.json',{'mock':mock,'logical_requests':len(rows),'input_sha256':sha(response_file) if response_file else None,
         'design_version':read_json(run/'report.json').get('design_version'),
         'primary_baselines':str(baselines) if baselines else None,
         'started':sum(r['started'] for r in rows),'not_started':sum(r['status']=='not_started' for r in rows),
         'empty_skips':sum(r['status']=='empty' for r in rows),'complete_main_result':all(r['complete'] for r in summary),
         'metric_label':'MAE2 with fixed replacement rule (parser v2)',
         'secondary_label':'post-hoc: 50/50 shrinkage, median of three, answer spread; not main results',
         'fallback_only_cells':[f"{r['stratum']}/{r['arm']}/{r['config_id']}" for r in summary if r['fallback_only']]})


def _rates(begun):
    rates={name:(sum(bool(r.get(field)) for r in begun)/len(begun) if begun else None)
           for name,field in [('valid_fraction','valid'),('limit_fraction','limit_hit'),('technical_failure_fraction','technical_error')]}
    # Share of answers that needed a whole-answer markdown fence removed first.
    rates['fence_fraction']=(sum(r.get('validation_reason')=='valid_after_fence' for r in begun)/len(begun) if begun else None)
    rates['other_answer_error_fraction']=(sum(r['status']=='terminal' and not r['valid'] and
           not r.get('technical_error') and not r.get('limit_hit') for r in begun)/len(begun) if begun else None)
    rates['replacement_fraction']=(sum(r['replacement'] is not None for r in begun)/len(begun) if begun else None)
    return rates


def _secondary(group,sources,expected,reportable):
    """Median of three answers and answer spread, per observation, then source.

    Both use the pipeline predictions (replacements included) for the median, and
    valid answers only for the spread. Nothing is reported for a cell whose
    answers were all replaced.
    """
    if not reportable:
        return {'median3_MAE2':None,'median3_MCSE':None,'spread_rho2_sd_valid':None,
                'spread_rho2_range_valid':None,'observations_with_three_valid':0}
    cells={}; sd=[]; rng=[]; three=0
    for source in sources:
        a=np.full((expected[source],1),np.nan); s_sd=[]; s_rng=[]
        for ix in range(1,expected[source]+1):
            rs=sorted((r for r in group if r['graph_id']==source and r['sample_index']==ix),key=lambda r:r['repeat_index'])
            preds=np.array([json.loads(r['prediction_json']) for r in rs if r['prediction_json']])
            if len(preds)==LLM_REPEATS:
                # Componentwise median; monotone because the median is monotone
                # in each argument.
                a[ix-1,0]=abs(float(np.median(preds[:,0]))-rs[0]['truth_rho2'])
            vals=[json.loads(r['prediction_json'])[0] for r in rs if r.get('valid')]
            if len(vals)==LLM_REPEATS:
                three+=1; s_sd.append(float(np.std(vals,ddof=1))); s_rng.append(float(np.ptp(vals)))
        cells[source]=a
        if s_sd: sd.append(np.mean(s_sd)); rng.append(np.mean(s_rng))
    ok=all(np.isfinite(v).all() for v in cells.values())
    est=paired_summary(cells,expected) if ok and cells else None
    return {'median3_MAE2':est['mean'] if est else None,'median3_MCSE':est['mcse'] if est else None,
            'spread_rho2_sd_valid':float(np.mean(sd)) if sd else None,
            'spread_rho2_range_valid':float(np.mean(rng)) if rng else None,
            'observations_with_three_valid':three}


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',required=True); p.add_argument('--baselines'); p.add_argument('--responses')
    p.add_argument('--out',required=True); p.add_argument('--mock',action='store_true')
    a=p.parse_args(); evaluate(a.run,a.responses,a.out,a.mock,a.baselines)
