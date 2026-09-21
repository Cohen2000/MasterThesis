#!/usr/bin/env python3
"""Offline h sensitivity and oracle diagnostics, fixed before new Qwen results."""
import sys,pickle
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import MAIN_KEYS,fold_for,graph_stratum,ARM_ID,ROOT,CURRENT_RUN,CURRENT_REVISION,REAL_TEST,SYNTH,H_SENSITIVITY,draws_for,read_json,write_json,sha
from main_experiment.data import load_graph
from main_experiment.sampling import h_parameters,history_panel_mask,draw
from main_experiment.observation import make,features
from main_experiment.baselines import plugin,h_extrapolator
from main_experiment.history_diagnostics import decompose
from main_experiment.pipeline import csv_write
from main_experiment.integrity import bind,code_binding


def main():
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION; out=rev/'history_sensitivity'
    out.mkdir(parents=True,exist_ok=True)
    bind(out/'inputs.json',{'code':code_binding(),'script_sha256':sha(__file__),
        'baselines_sha256':sha(rev/'primary_baselines.json'),'h_values':list(H_SENSITIVITY),
        'graph_hashes':{g:sha(run/'graphs'/g/'graph.npz') for g in MAIN_KEYS},
        'model_hashes':{str(p.relative_to(rev)):sha(p) for p in rev.glob('models_*/*/model.pkl')}})
    rows=[]; models={}
    for gid in MAIN_KEYS:
        g=load_graph(run/'graphs'/gid); fold=fold_for(gid)
        if fold not in models:
            models[fold]={}
            for name in ('pooled','real_only'):
                with (rev/f'models_{name}'/fold/'model.pkl').open('rb') as f: models[fold][name]=pickle.load(f)
        median=read_json(rev/'models_pooled'/fold/'manifest.json')['median']
        for h in H_SENSITIVITY:
            b=h_parameters(g,.1*g.cells,h)
            for i in range(1,draws_for('H',b)+1):
                c,_=draw(g,'H',i,'sample',b); panel=history_panel_mask(g,i,'sample',b)
                d=decompose(g.counts,c,panel)
                row=dict(graph_id=gid,stratum=graph_stratum(gid),h=h,sample_index=i,
                    n_panel=b['n_panel_history'],N_full=g.N,node_share=b['n_panel_history']/g.N,
                    expected_coverage=b['h_expected_cells']/g.cells,realized_coverage=int((c>0).sum())/g.cells,
                    saturated=b['h_saturated'],target_unreachable=b['h_target_unreachable'],
                    budget_matched=b['h_within_tolerance'],**d)
                o=make(g,'H',b,c,None)
                preds={'median':median}
                if o['D_obs']:
                    fit=h_extrapolator(o); row.update(fitted_q=fit['q'],fit_status=fit['status'])
                    preds.update(plugin=plugin(o),homogeneous=fit['prediction'])
                    preds.update({f'extratrees_{name}':m.predict([features(o)])[0] for name,m in models[fold].items()})
                else:
                    row.update(fitted_q=None,fit_status='empty_training_median')
                    preds.update({k:median for k in ('plugin','homogeneous','extratrees_pooled','extratrees_real_only')})
                for name,pred in preds.items():
                    ae=np.abs(np.asarray(pred)-g.truth)
                    row[name+'_AE2']=float(ae[0]); row[name+'_ProfileAE']=float(ae.mean())
                rows.append(row)
    csv_write(out/'observations.csv',rows)
    source=[]
    metrics=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and not isinstance(v,bool)
             and k not in ('h','sample_index','N_full','n_panel')]
    for h in H_SENSITIVITY:
        for gid in MAIN_KEYS:
            r=[x for x in rows if x['graph_id']==gid and x['h']==h]
            d={'h':h,'graph_id':gid,'stratum':r[0]['stratum'],'draws':len(r),
               'defined_draws':sum(x['defined'] for x in r),'n_panel':r[0]['n_panel'],
               'budget_matched':all(x['budget_matched'] for x in r)}
            for k in metrics:
                v=[x[k] for x in r if x.get(k) is not None]
                d[k]=float(np.mean(v)) if len(v)==len(r) else None
            for suffix in ('abs_rho2','profile_abs'):
                a=d.get('node_selection_'+suffix); b=d.get('net_history_'+suffix)
                d['history_share_'+suffix]=b/(a+b) if a is not None and b is not None and a+b else None
                diffs=[x['net_history_'+suffix]-x['node_selection_'+suffix] for x in r if x['defined']]
                d['history_minus_selection_'+suffix]=float(np.mean(diffs)) if len(diffs)==len(r) else None
                d['history_minus_selection_'+suffix+'_MCSE']=float(np.std(diffs,ddof=1)/np.sqrt(len(diffs))) if len(diffs)>1 else 0. if diffs else None
            source.append(d)
    csv_write(out/'sources.csv',source)
    summary=[]
    for h in H_SENSITIVITY:
        for stratum in ('real','surrogate','dar_a0','dar_a08','ad_memoryless','ad_memory'):
            group=[r for r in source if r['h']==h and (r['stratum']==stratum if stratum in ('real','surrogate') else r['graph_id'].startswith(stratum+'_r'))]
            d={'h':h,'stratum':stratum,'sources':len(group),
               'budget_matched_sources':sum(r['budget_matched'] for r in group),
               'undefined_sources':sum(r['defined_draws']!=r['draws'] for r in group)}
            for k in metrics:
                d[k]=float(np.mean([r[k] for r in group])) if all(r.get(k) is not None for r in group) else None
            for suffix in ('abs_rho2','profile_abs'):
                a=d.get('node_selection_'+suffix); b=d.get('net_history_'+suffix)
                d['history_share_'+suffix]=b/(a+b) if a is not None and b is not None and a+b else None
                d['sources_history_larger_'+suffix]=sum(r.get('history_share_'+suffix,0) is not None and r['history_share_'+suffix]>.5 for r in group)
                k='history_minus_selection_'+suffix
                d[k]=float(np.mean([r[k] for r in group])) if not d['undefined_sources'] else None
                d[k+'_MCSE']=float(np.sqrt(sum(r[k+'_MCSE']**2 for r in group))/len(group)) if not d['undefined_sources'] else None
            summary.append(d)
    csv_write(out/'summary.csv',summary)
    write_json(out/'report.json',{'h_primary':.6,'qwen_sensitivity_generated':False,
        'learned_reference':'trained at primary h=0.60; h=0.40/0.80 are cross-h sensitivity',
        'diagnosis':'net history includes dyad disappearance; descriptive fixed-panel comparisons only',
        'rows':summary})
    print([(r['h'],r['stratum'],r['history_share_abs_rho2'],r['homogeneous_AE2']) for r in summary])


if __name__=='__main__': main()
