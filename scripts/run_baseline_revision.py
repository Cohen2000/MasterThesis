#!/usr/bin/env python3
"""Baseline revision: synthetic training/development pool, refitted ExtraTrees,
and the bounded development check of the two mixture correctors.

Runs offline only. No LLM call, no API access, no paid job. The frozen main run
in results/main_experiment/frozen_20260916 is never written to.
"""
import argparse,math,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import write_json,read_json,digest
from main_experiment import pool as poolmod
from main_experiment.training import fit_folds,TRAINING_REVISION
from main_experiment.observation import parse,features,FEATURE_NAMES
from main_experiment.baselines import plugin,corrector,activity,bisect
from main_experiment import mixtures

FROZEN=Path('results/main_experiment/frozen_20260916')


def load_real_training():
    rows=[read_json(f) for f in sorted((FROZEN/'observations'/'training').glob('*.json'))]
    truth={r['source_family']:r['truth'] for r in rows}
    for r in rows: r['block_group']='real'
    return rows,truth


def load_pool(out,partition):
    rows=[];truth={};meta={}
    for f in sorted((out/'pool'/'observations').glob('*.json')):
        d=read_json(f)
        if d['partition']!=partition: continue
        truth[d['key']]=d['truth']; meta[d['key']]=d
        for r in d['observations']: rows.append({**r,'block_group':d['family']})
    return rows,truth,meta


def stage_train(out):
    real,real_truth=load_real_training()
    pool_rows,pool_truth,_=load_pool(out,'train')
    lock=FROZEN/'environment.lock.txt'
    pooled,_=fit_folds(real,real_truth,out/'models_pooled',lock,pool_rows,pool_truth)
    realonly,_=fit_folds([dict(r) for r in real],real_truth,out/'models_real_only',lock)
    return {'revision':TRAINING_REVISION,'n_features':len(FEATURE_NAMES),
            'real_rows':len(real),'pool_rows':len(pool_rows),
            'pool_graphs':len(pool_truth),'folds':sorted(pooled)}


def _homogeneous_start(o):
    """mu and lambda of the existing homogeneous corrector, used as fit starts."""
    D=o['D_obs']; S=sum(p.count('1')*d for p,d,e in o['table']); M=o['M_obs']
    n=3 if o['arm']=='H' else 5
    mu=activity(S/D,n)
    if o['arm']!='B': return mu,float('nan')
    p=o['parameter']; mean=M/S
    r=0. if mean<=1 else bisect(lambda x:1. if x==0 else x/-math.expm1(-x),mean,0.,mean)
    return mu,(r/p if p>0 else float('nan'))


def stage_pool(out):
    definition=poolmod.pool_definition()
    path=out/'pool_definition.json'
    if path.exists():
        old=read_json(path)
        if old['version']!=definition['version'] or old['graphs']!=definition['graphs']:
            raise ValueError('frozen pool definition changed; use a new output directory')
    else:
        write_json(path,definition)
    return poolmod.build_pool(out/'pool',definition['graphs'])



def _predict(o,models,medians,graph_truth,arm):
    """Every baseline for one development observation, plus the new candidate."""
    out={}
    D=o['D_obs']
    if D==0:
        for m in ('plugin','corrector','median','extratrees_pooled','extratrees_real_only','candidate'):
            out[m]={'prediction':list(medians),'status':'empty_sample'}
        return out
    out['plugin']={'prediction':plugin(o),'status':'ok'}
    try: out['corrector']={'prediction':corrector(o),'status':'ok'}
    except (ArithmeticError,FloatingPointError,OverflowError,ValueError) as e:
        out['corrector']={'prediction':plugin(o),'status':f'fallback:{type(e).__name__}'}
    out['median']={'prediction':list(medians),'status':'ok'}
    x=features(o).reshape(1,-1)
    out['extratrees_pooled']={'prediction':list(map(float,models['pooled'].predict(x)[0])),'status':'ok'}
    out['extratrees_real_only']={'prediction':list(map(float,models['real_only'].predict(x)[0])),'status':'ok'}
    if arm in ('H','B'):
        mu0,lam0=_homogeneous_start(o)
        fit=mixtures.fit_suffix(o,mu0) if arm=='H' else mixtures.fit_events(o,mu0,lam0)
        out['candidate']={'prediction':list(fit.prediction),'status':fit.status,
                          'seconds':fit.seconds,'kappa':fit.kappa,'mu':fit.mu,
                          'lam':fit.lam,'lam_event_only':fit.lam_event_only,
                          'flat_per_decade':fit.flat_per_decade,'nll_spread':fit.nll_spread}
    return out


def stage_dev(out):
    import csv,math,pickle,time
    import numpy as np
    rows,truth,meta=load_pool(out,'dev')
    models={}
    for name,folder in (('pooled','models_pooled'),('real_only','models_real_only')):
        with open(out/folder/'synthetic'/'model.pkl','rb') as f: models[name]=pickle.load(f)
    medians=read_json(out/'models_pooled'/'synthetic'/'manifest.json')['median']
    recs=[]; t0=time.perf_counter()
    cache=out/'development_observations.csv'
    if cache.exists():
        # Per-observation records are the expensive part; reuse them on a rerun.
        with open(cache,newline='') as f:
            for d in csv.DictReader(f):
                for k in ('AE2','ProfileAE','signed_rho2','seconds'): d[k]=float(d[k])
                for k in ('kappa','flat_per_decade','lam','lam_event_only'):
                    d[k]=float(d[k]) if d[k] not in ('','nan') else float('nan')
                d['sample_index']=int(d['sample_index'])
                recs.append(d)
    for r in ([] if recs else rows):
        o=parse(r['block']); g=r['graph_id']; tr=truth[g]
        pred=_predict(o,models,medians,tr,r['arm'])
        for method,d in pred.items():
            e=np.asarray(d['prediction'],float)-np.asarray(tr,float)
            recs.append({'graph_id':g,'family':meta[g]['family'],'arm':r['arm'],
                         'sample_index':r['sample_index'],'method':method,
                         'AE2':float(abs(e[0])),'ProfileAE':float(np.mean(np.abs(e))),
                         'signed_rho2':float(e[0]),'status':d['status'],
                         'seconds':d.get('seconds',0.),'kappa':d.get('kappa',''),
                         'flat_per_decade':d.get('flat_per_decade',''),
                         'lam':d.get('lam',''),'lam_event_only':d.get('lam_event_only','')})
    for r in recs: r['budget_matched']=bool(meta[r['graph_id']]['budget_matched'])
    if not cache.exists():
        with open(cache,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    summary=_aggregate(recs)
    write_json(out/'development_summary.json',summary)
    # Union of keys: only the comparison rows carry the paired columns.
    fields=list(dict.fromkeys(k for row in summary['rows'] for k in row))
    with open(out/'development_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,restval=''); w.writeheader(); w.writerows(summary['rows'])
    return {'observations':len(rows),'records':len(recs),'graphs':len(truth),
            'elapsed_seconds':time.perf_counter()-t0,'diagnostics':summary['diagnostics']}


def _per_graph(recs,key,pick):
    """Per-graph means, so every development graph counts once."""
    import numpy as np
    acc={}
    for r in recs:
        if not pick(r): continue
        acc.setdefault(r['graph_id'],[]).append(r[key])
    return {g:float(np.mean(v)) for g,v in acc.items()}


def _mse(vals):
    import numpy as np
    a=np.asarray(list(vals),float)
    if len(a)<2: return float(np.mean(a)) if len(a) else float('nan'),float('nan')
    return float(a.mean()),float(a.std(ddof=1)/math.sqrt(len(a)))


def _aggregate(recs):
    methods=['plugin','corrector','median','extratrees_pooled','extratrees_real_only','candidate']
    rows=[];
    for fam in ('all','dar','ad'):
        for arm in ('R','S','H','B'):
            for method in methods:
                sel=lambda r,f=fam,a=arm,m=method:(f=='all' or r['family']==f) and r['arm']==a and r['method']==m
                if not any(sel(r) for r in recs): continue
                ae=_per_graph(recs,'AE2',sel); pe=_per_graph(recs,'ProfileAE',sel)
                sg=_per_graph(recs,'signed_rho2',sel)
                mae,se=_mse(ae.values()); pm,pse=_mse(pe.values()); sm,sse=_mse(sg.values())
                row={'family':fam,'arm':arm,'method':method,'graphs':len(ae),
                     'MAE2':mae,'MAE2_se':se,'ProfileMAE':pm,'ProfileMAE_se':pse,
                     'signed_rho2':sm,'signed_rho2_se':sse}
                for ref in ('plugin','corrector'):
                    if method==ref: continue
                    base=_per_graph(recs,'AE2',lambda r,f=fam,a=arm,m=ref:(f=='all' or r['family']==f) and r['arm']==a and r['method']==m)
                    common=sorted(set(ae)&set(base))
                    if common:
                        d=[ae[g]-base[g] for g in common]
                        dm,dse=_mse(d)
                        row[f'paired_vs_{ref}']=dm; row[f'paired_vs_{ref}_se']=dse
                rows.append(row)
    diag={}
    cand=[r for r in recs if r['method']=='candidate']
    for arm in ('H','B'):
        sub=[r for r in cand if r['arm']==arm]
        if not sub: continue
        st={}
        for r in sub: st[r['status']]=st.get(r['status'],0)+1
        secs=[r['seconds'] for r in sub]
        flat=[r['flat_per_decade'] for r in sub if isinstance(r['flat_per_decade'],float) and r['flat_per_decade']==r['flat_per_decade']]
        diag[arm]={'n':len(sub),'status_counts':st,'seconds_total':float(sum(secs)),
                   'seconds_mean':float(sum(secs)/len(sub)),'seconds_max':float(max(secs)),
                   'flat_per_decade_median':float(sorted(flat)[len(flat)//2]) if flat else None,
                   'weakly_identified_or_boundary':sum(v for k,v in st.items()
                        if k.startswith('boundary') or k in ('weakly_identified','not_converged','starts_disagree'))}
    # Does a non-converged fit predict worse? Reported rather than assumed.
    by_status=[]
    for arm in ('H','B'):
        sub=[r for r in cand if r['arm']==arm]
        groups={}
        for r in sub: groups.setdefault(r['status'],[]).append(r['AE2'])
        for st,v in sorted(groups.items(),key=lambda kv:-len(kv[1])):
            by_status.append({'arm':arm,'status':st,'n':len(v),'MAE2':float(sum(v)/len(v))})
        for label,pick in (('converged',lambda r:r['status']=='converged'),
                           ('not_converged',lambda r:r['status']!='converged')):
            v=[r['AE2'] for r in sub if pick(r)]
            if v: by_status.append({'arm':arm,'status':f'ALL_{label}','n':len(v),
                                    'MAE2':float(sum(v)/len(v))})
    # Pre-registered sensitivity: the pool keeps graphs whose walk budget did not
    # match, so the development numbers are also reported without them.
    sens=[]
    for arm in ('R','S','H','B'):
        for m in methods:
            sub=[r for r in recs if r['arm']==arm and r['method']==m]
            if not sub: continue
            a=_per_graph(recs,'AE2',lambda r,x=arm,y=m:r['arm']==x and r['method']==y)
            b=_per_graph(recs,'AE2',lambda r,x=arm,y=m:r['arm']==x and r['method']==y and r['budget_matched'])
            if not b: continue
            sens.append({'arm':arm,'method':m,'MAE2_all':_mse(a.values())[0],
                         'MAE2_budget_matched':_mse(b.values())[0],
                         'graphs_all':len(a),'graphs_matched':len(b)})
    return {'rows':rows,'diagnostics':diag,'by_status':by_status,
            'budget_matched_sensitivity':sens}



def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--stage',default='all',choices=['pool','train','dev','all'])
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); report={}
    if a.stage in ('pool','all'):
        report['pool']=stage_pool(out)
        print('pool:',report['pool'],flush=True)
    if a.stage in ('train','all'):
        report['train']=stage_train(out)
        print('train:',report['train'],flush=True)
    if a.stage in ('dev','all'):
        report['dev']=stage_dev(out)
        print('dev:',report['dev'],flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    write_json(out/f'report_{a.stage}.json',report)
    print(report,flush=True)

if __name__=='__main__': main()
