#!/usr/bin/env python3
"""Fixed offline null, window-count and mixture-bound diagnostics. No inference."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
from main_experiment.data import load_graph
from main_experiment.surrogates import shuffle,audit
from main_experiment.pipeline import csv_write
from main_experiment.integrity import bind,code_binding
from dataset_census import WINDOWS,THRESHOLDS,relative_cutoff
from main_experiment.observation import parse
from main_experiment.mixtures import bound_sensitivity
from run_baseline_revision import _homogeneous_start


def main():
    run=ROOT/CURRENT_RUN;out=ROOT/CURRENT_REVISION/'control_diagnostics';out.mkdir(parents=True,exist_ok=True)
    bind(out/'inputs.json',{'code':code_binding(),'script_sha256':sha(__file__),
        'graph_hashes':{k:sha(run/'graphs'/k/'graph.npz') for k in MAIN_KEYS},'null_shuffles':99,
        'W':list(WINDOWS),'relative_thresholds':list(THRESHOLDS),'bound_decades':2})
    null=[];summary=[];windows=[]
    for key in MAIN_KEYS:
        g=load_graph(run/'graphs'/key)
        for W in WINDOWS:
            cuts=np.array([g.horizon[0]+(g.horizon[1]-g.horizon[0])*j/W for j in range(1,W)])
            w=np.searchsorted(cuts,g.t,side='right')
            c=np.bincount(g.pair*W+w,minlength=g.D*W).reshape(-1,W)>0;K=c.sum(1)
            persistence=float(np.sum(c[:,:-1]&c[:,1:])/np.sum(c[:,:-1])) if np.sum(c[:,:-1]) else None
            row={'graph_id':key,'stratum':graph_stratum(key),'W':W,'mean_occupancy':float(K.mean()/W),
                 'median_occupancy':float(np.median(K/W)),'C_one_step':persistence}
            row.update({f'rho_{k}':float(np.mean(K>=k)) for k in range(2,W+1)})
            row.update({f'rho_relative_{t}':float(np.mean(K>=relative_cutoff(t,W))) for t in THRESHOLDS})
            row.update({f'event_upper_bound_{k}':float(np.mean(g.m>=k)) for k in range(2,W+1)})
            windows.append(row)
        if key not in REAL_TEST:continue
        path=out/(key+'_null.json')
        if path.exists():records=read_json(path)
        else:
            records=[]
            for ix in range(1,100):
                s=shuffle(g,ix);check=audit(g,s)
                records.append({'parent':key,'shuffle':ix,'seed':seed('pwt_null_diagnostic',key,sample_index=ix),
                    'rho':s.truth,'delta_rho':(np.array(s.truth)-g.truth).tolist(),
                    'collisions':check['surrogate_collisions'],'invariants_passed':check['passed']})
            write_json(path,records)
        null.extend(records);prod=load_graph(run/'graphs'/(key+'__pwt'));values=np.array([r['rho'] for r in records])
        summary.append({'parent':key,'productive_rho':prod.truth,'productive_delta_rho':(np.array(prod.truth)-g.truth).tolist(),
            'null_mean':values.mean(0).tolist(),'null_min':values.min(0).tolist(),'null_max':values.max(0).tolist(),
            'productive_lower_rank_fraction':((1+(values<=prod.truth).sum(0))/100).tolist(),
            'productive_upper_rank_fraction':((1+(values>=prod.truth).sum(0))/100).tolist(),
            'productive_selection_changed':False})
        print(key,'null complete',flush=True)
    csv_write(out/'window_sensitivity.csv',windows)
    # Historical W={4,5,8} comparisons, separately within evidence blocks.
    wr=[]
    from scipy.stats import spearmanr
    for block in ('real','surrogate','dar_a0','dar_a08','ad_memoryless','ad_memory'):
        keys=[k for k in MAIN_KEYS if (graph_stratum(k)==block if block in ('real','surrogate') else k.startswith(block+'_r'))]
        for W in (4,8):
            for metric in [*(f'rho_{k}' for k in range(2,min(W,5)+1)),'mean_occupancy','C_one_step']:
                a=np.array([next(x[metric] for x in windows if x['graph_id']==k and x['W']==5) for k in keys],float)
                b=np.array([next(x[metric] for x in windows if x['graph_id']==k and x['W']==W) for k in keys],float)
                corr=float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else None
                wr.append({'block':block,'W':W,'metric':metric,'sources':len(keys),'mean_absolute_change':float(np.abs(a-b).mean()),'max_absolute_change':float(np.abs(a-b).max()),'spearman':corr})
    csv_write(out/'W_4_5_8_comparisons.csv',wr)
    bounds=[]
    for p in sorted((run/'observations/sample').glob('*.json')):
        r=read_json(p)
        if r['arm']!='B':continue
        o=parse(r['block'])
        if not o['D_obs']:continue
        cache=out/'bounds'/p.name
        if cache.exists():entry=read_json(cache)
        else:
            mu,lam=_homogeneous_start(o);d=bound_sensitivity(o,mu,lam,decades=2.)
            entry={'observation':r['id'],'graph_id':r['graph_id'],**d}
            entry={k:None if isinstance(v,float) and not np.isfinite(v) else v for k,v in entry.items()}
            write_json(cache,entry)
        bounds.append(entry)
    csv_write(out/'mixture_bound_sensitivity.csv',bounds)
    write_json(out/'report.json',{'null_shuffles':len(null),'parents':len(summary),'null_summary':summary,
        'W_rows':len(windows),'W_grid':list(WINDOWS),'historical_W_comparisons':len(wr),'bound_rows':len(bounds),
        'null_diagnostic_only':True,'production_selection_changed':False})
    write_json(out/'seed_manifest.json',[{'seed':s,'fields':__import__('json').loads(v)} for s,v in sorted(SEEDS.items())])
if __name__=='__main__':main()
