#!/usr/bin/env python3
"""Parent-matched temporal-control metrics and within-r synthetic mode contrasts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
from main_experiment.evaluation import resolve,conditional_summary
from main_experiment.observation import parse
from main_experiment.pipeline import csv_write
from main_experiment.integrity import bind


def delta_metrics(parent_truth,surrogate_truth,parent_prediction,surrogate_prediction):
    truth=np.asarray(surrogate_truth)-parent_truth
    if parent_prediction is None or surrogate_prediction is None:
        return {'delta_rho':truth.tolist(),'delta_rhohat':None,'AE_Delta_2':None,'Delta_ProfileAE':None,'signed_delta_error_2':None,'sign_agreement_2':None}
    pred=np.asarray(surrogate_prediction)-parent_prediction;err=pred-truth
    return {'delta_rho':truth.tolist(),'delta_rhohat':pred.tolist(),'AE_Delta_2':float(abs(err[0])),
        'Delta_ProfileAE':float(np.abs(err).mean()),'signed_delta_error_2':float(err[0]),
        'sign_agreement_2':bool(np.sign(pred[0])==np.sign(truth[0]))}


def evaluate(run,baselines,out,responses=None):
    run=Path(run);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bind(out/'inputs.json',{'requests':sha(run/'requests.jsonl'),'baselines':sha(baselines),
        'responses':sha(responses) if responses else None,'script':sha(__file__)})
    obs=[read_json(p) for p in sorted((run/'observations/sample').glob('*.json'))]
    by={(o['graph_id'],o['arm'],o['sample_index']):o for o in obs}
    base=read_json(baselines)['observations'];methods=['plugin','median','extratrees_pooled','extratrees_real_only','primary_corrector']
    records={r['id']:r for r in map(json.loads,Path(responses).read_text().splitlines())} if responses else {}
    planned=[json.loads(l) for l in (run/'requests.jsonl').read_text().splitlines()]
    requests={(r['observation_id'],r['config_id'],r['repeat_index']):r for r in planned}
    configs=sorted({r['config_id'] for r in planned if r['id'] in records})
    def prediction(o,method,repeat):
        if method in methods:return base[o['id']][method]['prediction']
        req=requests[o['id'],method,repeat]
        return resolve(parse(o['block']),record=records.get(req['id']))['prediction']
    pairs=[]
    for parent in REAL_TEST:
        surrogate=parent+'__pwt'
        for arm in ARMS:
            ni={g:sum(o['graph_id']==g and o['arm']==arm for o in obs) for g in (parent,surrogate)}
            for i in range(1,max(ni.values())+1):
                a=by[parent,arm,1 if ni[parent]==1 else i];b=by[surrogate,arm,1 if ni[surrogate]==1 else i]
                for method in methods+configs:
                    for repeat in range(1,2 if method in methods else LLM_REPEATS+1):
                        pa=prediction(a,method,repeat);pb=prediction(b,method,repeat)
                        pairs.append({'parent':parent,'arm':arm,'method':method,'sample_index':i,'repeat_index':repeat,
                            'parent_observation':a['id'],'surrogate_observation':b['id'],
                            'one_side_deterministic':min(ni.values())==1 and max(ni.values())>1,
                            'valid_pair':pa is not None and pb is not None,**delta_metrics(a['truth'],b['truth'],pa,pb)})
    csv_write(out/'paired_observations.csv',pairs);source=[];summary=[]
    for arm in ARMS:
        for method in methods+configs:
            group=[r for r in pairs if r['arm']==arm and r['method']==method]
            result={'arm':arm,'method':method,'parent_pairs':len(REAL_TEST),'planned_pairs':len(group),'valid_pairs':sum(r['valid_pair'] for r in group)}
            for metric in ('AE_Delta_2','Delta_ProfileAE','signed_delta_error_2','sign_agreement_2'):
                cells={};expected={}
                for parent in REAL_TEST:
                    sel=[r for r in group if r['parent']==parent];n=max(r['sample_index'] for r in sel);rep=max(r['repeat_index'] for r in sel)
                    arr=np.full((n,rep),np.nan)
                    for r in sel:
                        if r[metric] is not None:arr[r['sample_index']-1,r['repeat_index']-1]=float(r[metric])
                    # conditional_summary requires 3 repeat columns; deterministic baselines
                    # are duplicated across columns, preserving draw-level variance exactly.
                    if rep==1:arr=np.repeat(arr,LLM_REPEATS,axis=1)
                    cells[parent]=arr;expected[parent]=n
                    source.append({'parent':parent,'arm':arm,'method':method,'metric':metric,'value':float(np.nanmean(arr)) if np.isfinite(arr).any() else None,'valid_pairs':sum(r['valid_pair'] for r in sel),'planned_pairs':len(sel)})
                estimate=conditional_summary(cells,expected)
                result[metric]=estimate['mean']
                result[metric+'_MCSE']=None if any(r['one_side_deterministic'] for r in group) else estimate['mcse']
            valid=[r for r in group if r['valid_pair']]
            truth=[];pred=[]
            for parent in REAL_TEST:
                sel=[r for r in valid if r['parent']==parent]
                if sel:truth.append(sel[0]['delta_rho'][0]);pred.append(np.mean([r['delta_rhohat'][0] for r in sel]))
            result['descriptive_parent_mean_delta_correlation']=float(np.corrcoef(truth,pred)[0,1]) if len(truth)>1 and np.ptp(truth)>0 and np.ptp(pred)>0 else None
            summary.append(result)
    csv_write(out/'paired_sources.csv',source);csv_write(out/'paired_summary.csv',summary)
    contrasts=[]
    for family,modes in [('dar',('a0','a08')),('ad',('memoryless','memory'))]:
        for r in (1,2):
            for arm in ARMS:
                for method in methods+configs:
                    vals=[]
                    for mode in modes:
                        key=f'{family}_{mode}_r{r}';selected=[o for o in obs if o['graph_id']==key and o['arm']==arm];errors=[]
                        for o in selected:
                            for repeat in range(1,2 if method in methods else LLM_REPEATS+1):
                                p=prediction(o,method,repeat)
                                if p is not None:errors.append(float(abs(p[0]-o['truth'][0])))
                        vals.append(float(np.mean(errors)) if errors else None)
                    contrasts.append({'family':family,'outer_replicate':r,'arm':arm,'method':method,'mode_a':modes[0],'mode_b':modes[1],
                        'MAE2_a':vals[0],'MAE2_b':vals[1],'paired_MAE2_b_minus_a':vals[1]-vals[0] if all(v is not None for v in vals) else None})
    csv_write(out/'synthetic_within_replicate_contrasts.csv',contrasts)
    write_json(out/'report.json',{'design_version':DESIGN_VERSION,'pair_rows':len(pairs),'summary':summary,
        'primary':'AE_Delta_2','secondary':'Delta_ProfileAE','source_weighting':'eight parents equally',
        'validity':'both predictions required; no imputation','synthetic_contrast_rows':len(contrasts)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default=CURRENT_RUN);p.add_argument('--baselines',default=CURRENT_REVISION+'/primary_baselines.json');p.add_argument('--out',required=True);p.add_argument('--responses');a=p.parse_args();evaluate(a.run,a.baselines,a.out,a.responses)
