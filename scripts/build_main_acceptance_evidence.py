#!/usr/bin/env python3
"""Derive the compact, reviewable acceptance evidence from a finished offline run.

Reads only the artifacts of an existing run directory. Performs no sampling, no
training and no LLM call; it reformats what the pipeline already produced.
"""
import argparse,csv,io,json,pathlib,sys

def rows(p):
    return list(csv.DictReader(io.StringIO(p.read_text())))

def jnum(x):
    return json.loads(x) if x else None

def data_volume(run,ev):
    out=[]
    for r in rows(run/'data_summary.csv'):
        truth=jnum(r['truth'])
        out.append({'graph_id':r['key'],'stratum':'synthetic' if r['source_family']=='' else 'real',
                    'N_full':int(r['N_full']),'D_full':int(r['D_full']),'M_full':int(r['M_full']),
                    'B':int(r['B']),'events_per_window':jnum(r['events_per_window']),
                    'rho_true':{f'rho_{k}':v for k,v in zip((2,3,4,5),truth)},
                    'canonical_sha256':r['canonical_sha256'],'graph_sha256':r['graph_sha256']})
    (ev/'data_and_ground_truth.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out

def budget(run,ev):
    out=[]
    for r in rows(run/'budget_summary.csv'):
        out.append({'graph_id':r['graph_id'],'B':int(r['B']),'p':float(r['p']),
                    'n_panel':int(r['n_panel']),
                    'node_relative_budget_error':float(r['node_relative_budget_error']),
                    'L':int(r['L']),'C':int(r['C']),
                    'calibration_mean':float(r['calibration_mean']),
                    'validation_n':int(r['validation_n']),
                    'validation_mean':float(r['validation_mean']),
                    'validation_relative_error':float(r['validation_relative_error']),
                    'validation_mcse_over_B':float(r['validation_mcse'])/int(r['B']),
                    'search_limit_reached_without_budget':r['search_limit_reached_without_budget']=='True',
                    'budget_matched':r['budget_matched']=='True',
                    'unmatched_reasons':jnum(r['unmatched_reasons'])})
    (ev/'budget_deviations.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out

def observations(run,ev):
    rs=rows(run/'observation_status.csv')
    empty=[r for r in rs if r['empty']=='True']
    summary={'planned_observations':len(rs),
             'prepared_not_started':sum(r['status']=='prepared_not_started' for r in rs),
             'by_status':{},'by_arm':{},
             'planned_logical_calls':sum(int(r['planned_logical_calls']) for r in rs),
             'actual_calls':sum(int(r['actual_calls']) for r in rs),
             'empty_observations':len(empty),
             'empty_observation_ids':[r['id'] for r in empty]}
    for r in rs:
        summary['by_status'][r['status']]=summary['by_status'].get(r['status'],0)+1
        a=summary['by_arm'].setdefault(r['arm'],{'observations':0,'planned_logical_calls':0,'empty':0})
        a['observations']+=1; a['planned_logical_calls']+=int(r['planned_logical_calls'])
        a['empty']+=r['empty']=='True'
    (ev/'observation_status.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

def baselines(run,ev):
    out=[]
    for r in rows(run/'baseline_summary.csv'):
        out.append({'stratum':r['stratum'],'arm':r['arm'],'method':r['method'],
                    'MAE2':float(r['MAE2']),'MCSE':float(r['MCSE']),
                    'ProfileMAE':float(r['ProfileMAE']),
                    'signed_plugin_error':float(r['signed_plugin_error']) if r['signed_plugin_error'] else None,
                    'valid_fraction':float(r['valid_fraction']),
                    'replacement_fraction':float(r['replacement_fraction']),
                    'empty_fraction':float(r['empty_fraction'])})
    (ev/'baseline_results.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
    return out

def training(run,ev):
    folds=[]
    for m in sorted((run/'models').glob('*/manifest.json')):
        d=json.loads(m.read_text())
        folds.append({'fold':m.parent.name,'test_source':d['test_source'],
                      'training_rows':d['training_rows'],
                      'independent_sources':d['independent_sources'],
                      'n_features':len(d['feature_names']),
                      'frozen_training_median':d['median'],
                      'sklearn_version':d['sklearn_version'],
                      'parameters':d['parameters'],
                      'model_sha256':d['model_sha256'],
                      'features_sha256':d['features_sha256'],
                      'labels_sha256':d['labels_sha256']})
    (ev/'training_folds.json').write_text(json.dumps(folds,indent=2,sort_keys=True)+'\n')
    return folds

def prompts(run,ev):
    rs=rows(run/'prompt_sizes.csv')
    cols=['qwen_thinking','qwen_nonthinking','deepseek_message_texts','sol_o200k_message_proxy']
    summary={'n_observations':len(rs),'hard_input_limit_tokens':4096,'per_counter':{}}
    for c in cols:
        v=[int(r[c]) for r in rs]
        summary['per_counter'][c]={'min':min(v),'max':max(v),'mean':round(sum(v)/len(v),2),
                                   'over_limit':sum(x>4096 for x in v),
                                   'headroom_at_max':4096-max(v)}
    summary['qwen_exact_template_checked']=all(r['qwen_exact_template_checked']=='True' for r in rs)
    summary['provider_template_verified']={
        'deepseek':all(r['deepseek_provider_template_verified']=='True' for r in rs),
        'sol':all(r['sol_provider_template_verified']=='True' for r in rs)}
    summary['note']=rs[0]['api_template_note'] if rs else ''
    (ev/'prompt_sizes.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--out',required=True)
    a=ap.parse_args()
    run=pathlib.Path(a.run); ev=pathlib.Path(a.out); ev.mkdir(parents=True,exist_ok=True)
    rep=json.loads((run/'report.json').read_text())
    if not rep.get('offline_ready'):
        sys.exit('run is not marked offline_ready; refusing to build acceptance evidence')
    d=data_volume(run,ev); b=budget(run,ev); o=observations(run,ev)
    bl=baselines(run,ev); tr=training(run,ev); pr=prompts(run,ev)
    index={'run':str(run),'report':rep,
           'execution_policy':json.loads((run/'execution_policy.json').read_text()),
           'counts':{'graphs':len(d),'calibrated_graphs':len(b),
                     'observations':o['planned_observations'],
                     'planned_logical_calls':o['planned_logical_calls'],
                     'started_calls':o['actual_calls'],
                     'empty_observations':o['empty_observations'],
                     'baseline_rows':len(bl),'models':len(tr),
                     'prompt_sizes_checked':pr['n_observations']},
           'budget_matched_all':all(x['budget_matched'] for x in b),
           'max_validation_relative_error':max(abs(x['validation_relative_error']) for x in b),
           'max_validation_mcse_over_B':max(x['validation_mcse_over_B'] for x in b),
           'max_prompt_tokens':max(v['max'] for v in pr['per_counter'].values()),
           'files':sorted(p.name for p in ev.glob('*.json'))}
    (ev/'index.json').write_text(json.dumps(index,indent=2,sort_keys=True)+'\n')
    print(json.dumps(index['counts'],indent=2))

if __name__=='__main__': main()
