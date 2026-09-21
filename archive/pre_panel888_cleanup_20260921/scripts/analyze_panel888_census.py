#!/usr/bin/env python3
"""Retained census diagnostics on canonical event multisets; never LLM variants."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
from main_experiment.data import load_graph
from main_experiment.pipeline import csv_write
from main_experiment.integrity import bind,code_binding
from dataset_census import TARGETS,count_feasibility
from census import W_GRID,ALPHA
run=ROOT/CURRENT_RUN;out=ROOT/CURRENT_REVISION/'census_diagnostics';out.mkdir(parents=True,exist_ok=True)
bind(out/'inputs.json',{'code':code_binding(),'script_sha256':sha(__file__),'grid':list(W_GRID),'alpha':ALPHA,
    'count_feasibility_targets':list(TARGETS),'graphs':{g:sha(run/'graphs'/g/'graph.npz') for g in MAIN_KEYS}})
rows=[];bounds=[]
for key in MAIN_KEYS:
    g=load_graph(run/'graphs'/key);lo,hi=g.horizon;T=hi-lo
    order=np.lexsort((np.arange(g.M),g.t,g.pair));pair=g.pair[order];t=g.t[order]
    start=np.r_[0,np.flatnonzero(pair[1:]!=pair[:-1])+1];end=np.r_[start[1:]-1,len(pair)-1]
    life=t[end]-t[start];same=pair[1:]==pair[:-1];gap=np.diff(t)[same];gp=pair[1:][same]
    mu=float(gap.mean()) if len(gap) else 0.;sd=float(gap.std()) if len(gap) else 0.
    n=np.bincount(gp,minlength=g.D);sm=np.bincount(gp,weights=gap,minlength=g.D);sq=np.bincount(gp,weights=gap**2,minlength=g.D)
    eligible=n>=2;means=sm[eligible]/n[eligible];std=np.sqrt(np.maximum(0,sq[eligible]/n[eligible]-means**2));den=std+means
    burst=(std[den>0]-means[den>0])/den[den>0]
    shifted=np.searchsorted(lo+(np.arange(5)+.5)*T/5,g.t,side='right');ks=np.bincount(np.unique(g.pair*6+shifted)//6,minlength=g.D)
    # Same stable rank rule as census.equal_event_windows, canonical dyad/time order.
    global_order=np.argsort(t,kind='stable');rank=np.empty(g.M,dtype=np.int64);rank[global_order]=(np.arange(g.M)*5)//g.M
    ke=np.bincount(np.unique(pair*5+rank)//5,minlength=g.D)
    distinct=np.bincount(pair[np.r_[True,(pair[1:]!=pair[:-1])|(t[1:]!=t[:-1])]],minlength=g.D)
    median_gap=float(np.median(gap)) if len(gap) else None
    row={'graph_id':key,'stratum':graph_stratum(key),'rho2':g.truth[0],'rho_shifted_half_window':float(np.mean(ks>=2)),
        'rho_equal_event_rank_windows':float(np.mean(ke>=2)),'rho_event_weighted':float(g.m[g.K>=2].sum()/g.M),
        'lifetime_mean':float(life.mean()),'lifetime_median':float(np.median(life)),'lifetime_mean_over_horizon':float(life.mean()/T),
        'median_interevent_time':median_gap,'burstiness_pooled':(sd-mu)/(sd+mu) if sd+mu else None,
        'burstiness_pair_median':float(np.median(burst)) if len(burst) else None,
        'censoring_share_last_window':float(np.mean(t[start]>=lo+4*T/5)),
        'window_length_over_median_interevent_time':T/5/median_gap if median_gap else None,
        'share_single_event_dyads':float(np.mean(g.m==1)),'repeat_rate':float(np.mean(g.m>1))}
    rows.append(row);bounds.append(count_feasibility(key,g.m,g.truth[0],distinct))
csv_write(out/'census_sensitivities.csv',rows);csv_write(out/'count_feasibility.csv',bounds)
write_json(out/'report.json',{'graphs':len(rows),'inherited_sources':['src/census.py','src/dataset_census.py'],
    'main_window_convention':'fixed archive horizon; source-time cutpoints, side=right, no epsilon',
    'rank_window_ties':'inherited stable event-rank rule may split tied timestamps; offline diagnostic only',
    'surrogate_event_multiplicity_retained':True,'target_driven_timing_allocator_run':False})
print('CENSUS_DIAGNOSTICS_COMPLETE',len(rows))
