#!/usr/bin/env python3
"""Baseline revision: synthetic training/development pool, refitted ExtraTrees,
development check of the fixed references, and the main-panel baselines.

Runs offline only. No LLM call, no API access, no paid job. Earlier runs
(frozen_20260916, budget10_20261001 and their baseline revisions) are never
written to; every revision writes into its own output directory.

Current H uses uniform nodes and a common time suffix. Its fixed reference is
the homogeneous zero-truncated Binomial extrapolator. B keeps its fixed mixture.
"""
import argparse,math,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (write_json,read_json,digest,LEGACY_H,DESIGN_VERSION,draws_for,
                                    CURRENT_RUN,PREVIOUS_REVISION,sha,fold_for,graph_stratum,MAIN_KEYS)
from main_experiment.common import REAL_TEST as REAL_TEST_LIST, SYNTH as SYNTH_LIST
from main_experiment import pool as poolmod
from main_experiment.training import fit_folds,TRAINING_REVISION
from main_experiment.observation import parse,features,FEATURE_NAMES,FEATURE_VERSION
from main_experiment.baselines import plugin,corrector,activity,bisect,h_extrapolator
from main_experiment import mixtures

# Current design run. frozen_20260916 and budget10_20261001 hold superseded
# designs and are kept as the development history, not read here.
FROZEN=Path(os.environ.get('MAIN_RUN',CURRENT_RUN))
PREVIOUS_POOL=Path(os.environ.get('PREVIOUS_POOL',str(PREVIOUS_REVISION/'pool/observations')))
_FOLD={}


def load_real_training():
    rows=[read_json(f) for f in sorted((FROZEN/'observations'/'training').glob('*.json'))]
    truth={r['source_family']:r['truth'] for r in rows}
    for r in rows: r['block_group']='real'
    return rows,truth


def load_pool(out,partition):
    rows=[];truth={};meta={}
    for f in sorted((out/'pool'/'observations').glob('*.json')):
        checksum=f.with_suffix('.sha256')
        if not checksum.exists() or checksum.read_text().strip()!=sha(f): raise ValueError(f'pool checksum: {f}')
        d=read_json(f)
        if d['partition']!=partition: continue
        truth[d['key']]=d['truth']; meta[d['key']]=d
        for r in d['observations']: rows.append({**r,'block_group':d['family']})
    expected=sum(v[partition] for v in poolmod.COUNTS.values())
    if len(truth)!=expected: raise ValueError(f'incomplete pool {partition}: {len(truth)}/{expected}')
    return rows,truth,meta


def stage_train(out):
    real,real_truth=load_real_training()
    pool_rows,pool_truth,_=load_pool(out,'train')
    lock=FROZEN/'environment.lock.txt'
    pooled,_=fit_folds(real,real_truth,out/'models_pooled',lock,pool_rows,pool_truth)
    realonly,_=fit_folds([dict(r) for r in real],real_truth,out/'models_real_only',lock)
    return {'revision':TRAINING_REVISION,'n_features':len(FEATURE_NAMES),'feature_version':FEATURE_VERSION,
            'real_rows':len(real),'pool_rows':len(pool_rows),
            'pool_graphs':len(pool_truth),'folds':sorted(pooled)}


def _homogeneous_start(o):
    """mu and lambda of the existing homogeneous corrector, used as fit starts."""
    D=o['D_obs']; S=sum(r[0].count('1')*r[1] for r in o['table']); M=o['M_obs']
    n=3 if o['arm']==LEGACY_H else 5
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
    result=poolmod.build_pool(out/'pool',definition['graphs'])
    if result['failures']: raise ValueError('pool incomplete; do not train')
    result['previous_pool_comparison']=compare_previous_pool(out)
    return result


def compare_previous_pool(out):
    return {'status':'new versioned pool seeds and observations; all models fitted fresh'}


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
    if arm=='H':
        out['corrector']=h_extrapolator(o)
    if arm=='S':
        out['corrector'].update(model='srw_traversal_frequency',working_model=True)
    if arm=='B':
        mu0,lam0=_homogeneous_start(o)
        fit=mixtures.fit_events(o,mu0,lam0)
        pred=list(fit.prediction); fallback=''
        if mixtures.is_unreliable(fit.status,fit.flags):
            # Pre-registered: an unreliable fit does not contribute a mixture
            # prediction at all; it falls back to the homogeneous corrector. The
            # condition reads the independent flags, so a disagreement that
            # coincides with a boundary hit is not masked by the summary label.
            pred=list(out['corrector']['prediction']); fallback='homogeneous_corrector'
        out['candidate']={'prediction':pred,'status':fit.status,'fallback':fallback,
                          'flags':dict(fit.flags),
                          'seconds':fit.seconds,'kappa':fit.kappa,'mu':fit.mu,
                          'lam':fit.lam,'lam_event_only':fit.lam_event_only,
                          'flat_per_decade':fit.flat_per_decade,'nll_spread':fit.nll_spread}
    return out


def stage_dev(out):
    import csv,math,pickle,time
    import numpy as np
    rows,truth,meta=load_pool(out,'dev')
    from main_experiment.integrity import bind,files
    bind(out/'development_inputs.json',{'observations':digest(rows),'truth':digest(truth),
         'models':files(list((out/'models_pooled').glob('*/model.pkl'))+list((out/'models_real_only').glob('*/model.pkl')))})
    models={}
    for name,folder in (('pooled','models_pooled'),('real_only','models_real_only')):
        with open(out/folder/'synthetic'/'model.pkl','rb') as f: models[name]=pickle.load(f)
    medians=read_json(out/'models_pooled'/'synthetic'/'manifest.json')['median']
    recs=[]; t0=time.perf_counter()
    cache=out/'development_observations.csv'
    if cache.exists():
        if not cache.with_suffix('.sha256').exists() or cache.with_suffix('.sha256').read_text().strip()!=sha(cache):
            raise ValueError('development cache checksum mismatch')
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
        pred.pop('h_bounds',None)
        for method,d in pred.items():
            e=np.asarray(d['prediction'],float)-np.asarray(tr,float)
            recs.append({'graph_id':g,'family':meta[g]['family'],'arm':r['arm'],
                         'sample_index':r['sample_index'],'method':method,
                         'AE2':float(abs(e[0])),'ProfileAE':float(np.mean(np.abs(e))),
                         'signed_rho2':float(e[0]),'status':d['status'],
                         'fallback':d.get('fallback',''),
                         'flag_starts_disagree':bool(d.get('flags',{}).get('starts_disagree')),
                         'flag_boundary':bool(d.get('flags',{}).get('boundary_homogeneous') or d.get('flags',{}).get('boundary_other')),
                         'flag_flatness_unavailable':bool(d.get('flags',{}).get('flatness_unavailable')),
                         'seconds':d.get('seconds',0.),'kappa':d.get('kappa',''),
                         'flat_per_decade':d.get('flat_per_decade',''),
                         'lam':d.get('lam',''),'lam_event_only':d.get('lam_event_only','')})
    for r in recs: r['budget_matched']=bool(meta[r['graph_id']]['budget_matched_by_arm'][r['arm']])
    if not cache.exists():
        with open(cache,'w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
        cache.with_suffix('.sha256').write_text(sha(cache)+'\n')
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
        diag[arm]={'n':len(sub),'status_counts':st,
                   'fallbacks_to_homogeneous':sum(1 for r in sub if r.get('fallback')),'seconds_total':float(sum(secs)),
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
    # Pre-registered sensitivity: the pool keeps graphs whose budget did not match
    # for an arm, so the development numbers are also reported without them.
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



def stage_main(out):
    """All baselines on the main run: six real sources and eight synthetic instances.

    Runs only after the development check has fixed the corrector choice. The
    ranking here must not feed back into that choice; it is reported as is.
    """
    import csv, pickle, time
    import numpy as np
    rows=[read_json(f) for f in sorted((FROZEN/'observations'/'sample').glob('*.json'))]
    man={m['key']:m for m in (read_json(f) for f in (FROZEN/'graphs').glob('*/manifest.json'))}
    models={}
    for name,folder in (('pooled','models_pooled'),('real_only','models_real_only')):
        with open(out/folder/'synthetic'/'model.pkl','rb') as f: models[name]=pickle.load(f)
    med_syn=read_json(out/'models_pooled'/'synthetic'/'manifest.json')['median']
    global _FOLD
    fold_models={}; fold_med={}
    for src in REAL_TEST_LIST:
        for name,folder in (('pooled','models_pooled'),('real_only','models_real_only')):
            with open(out/folder/src/'model.pkl','rb') as f: fold_models[(src,name)]=pickle.load(f)
        fold_med[src]=read_json(out/'models_pooled'/src/'manifest.json')['median']
    _FOLD=fold_models
    recs=[]; t0=time.time()
    for r in rows:
        o=parse(r['block']); g=r['graph_id']; truth=man[g]['truth']
        fold=fold_for(g); real=fold!='synthetic'
        # Leave-one-source-out for a real test source; the all-real model otherwise.
        m={'pooled':fold_models[(fold,'pooled')] if real else models['pooled'],
           'real_only':fold_models[(fold,'real_only')] if real else models['real_only']}
        med=fold_med[fold] if real else med_syn
        pred=_predict(o,m,med,truth,r['arm'])
        pred.pop('h_bounds',None)
        S_full=man[g]['D_full']*(1+sum(truth))
        S_obs=sum(row[0].count('1')*row[1] for row in o['table'])
        for method,d in pred.items():
            e=np.asarray(d['prediction'],float)-np.asarray(truth,float)
            recs.append({'graph_id':g,'stratum':graph_stratum(g),
                'arm':r['arm'],'sample_index':r['sample_index'],'method':method,
                'AE2':float(abs(e[0])),'ProfileAE':float(np.mean(np.abs(e))),
                'signed_rho2':float(e[0]),'status':d['status'],
                'fallback':d.get('fallback',''),
                'D_obs':o['D_obs'],'M_obs':o['M_obs'],
                'dyad_coverage':o['D_obs']/man[g]['D_full'],
                'window_coverage':S_obs/S_full,
                'event_coverage':o['M_obs']/man[g]['M_full'],
                'seconds':d.get('seconds',0.),'budget_matched':bool(r['budget_matched'])})
    with open(out/'main_observations.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    write_json(out/'primary_baselines.json',_primary_baselines(rows,man,models,fold_med,med_syn))
    summary=_aggregate_main(recs)
    write_json(out/'main_summary.json',summary)
    fields=list(dict.fromkeys(k for row in summary['rows'] for k in row))
    with open(out/'main_summary.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,restval=''); w.writeheader(); w.writerows(summary['rows'])
    return {'observations':len(rows),'records':len(recs),'seconds':time.time()-t0,
            'coverage':summary['coverage']}


# Which corrector is primary for which arm. R, S and B as fixed in
# results/baseline_revision_20261001/CORRECTOR_DECISION.md. For the new H the
# H is the homogeneous zero-truncated Binomial working-model extrapolator.
PRIMARY_CORRECTOR={'R':'corrector','S':'corrector','H':'corrector','B':'candidate'}
PRIMARY_NAME={'R':'plugin_equivalent_corrector','S':'srw_traversal_frequency','H':'homogeneous_zero_truncated_binomial','B':'beta_ztp_mixture'}


def _primary_baselines(rows,man,models,fold_med,med_syn):
    """Per-observation predictions of every baseline, with the primary one named.

    The response evaluator needs the *decided* reference, not whichever baseline
    happened to be stored in the run's own baselines directory. That directory
    holds the homogeneous corrector and the real-only ExtraTrees, which are
    development references here, so comparing the LLM against them would silently
    use a reference the decision record does not designate.
    """
    import numpy as np
    out={'decision':'docs/PROTOCOL_PANEL888_20260921.md; S traversal-frequency working reference; R/B/H unchanged',
         'design_version':DESIGN_VERSION,
         'protocol':'docs/PROTOCOL_PANEL888_20260921.md',
         'primary_corrector_by_arm':PRIMARY_CORRECTOR,
         'primary_corrector_meaning':PRIMARY_NAME,
         'primary_trained_reference':'extratrees_pooled',
         'observations':{}}
    for r in rows:
        o=parse(r['block']); g=r['graph_id']; truth=man[g]['truth']
        fold=fold_for(g); real=fold!='synthetic'
        m={'pooled':models['pooled'] if not real else None,'real_only':models['real_only'] if not real else None}
        # stage_main already resolved the right fold model; recompute identically
        from main_experiment.common import REAL_TEST as RT
        if real:
            import pickle
            m={'pooled':_FOLD[(fold,'pooled')],'real_only':_FOLD[(fold,'real_only')]}
        med=fold_med[fold] if real else med_syn
        pred=_predict(o,m,med,truth,r['arm'])
        bounds=pred.pop('h_bounds',None)
        entry={k:{field:v[field] for field in ('prediction','status','fallback','flags') if field in v} for k,v in pred.items()}
        entry['block_sha256']=r['block_sha256']
        if bounds: entry['h_bounds']=bounds
        entry['primary_corrector']=dict(entry[PRIMARY_CORRECTOR[r['arm']]])
        entry['primary_corrector_name']=PRIMARY_NAME[r['arm']]
        entry['arm']=r['arm']; entry['stratum']=graph_stratum(g)
        entry['sample_index']=r['sample_index']; entry['graph_id']=g
        entry['deterministic_draw']=bool(r.get('deterministic_draw',False))
        entry['truth']=list(truth)
        out['observations'][r['id']]=entry
    return out


def _aggregate_main(recs):
    """Hierarchical aggregation: observations within a source, then sources equally.

    Repeats and dyads are never counted as extra independent graphs; a source
    contributes exactly once to its stratum mean.
    """
    import numpy as np
    methods=['plugin','corrector','median','extratrees_pooled','extratrees_real_only','candidate']
    rows=[]
    for stratum in ('real','surrogate','dar_a0','dar_a08','ad_memoryless','ad_memory'):
        for arm in ('R','S','H','B'):
            for method in methods:
                sel=[r for r in recs if (r['stratum']==stratum if stratum in ('real','surrogate') else r['graph_id'].startswith(stratum+'_r')) and r['arm']==arm and r['method']==method]
                if not sel: continue
                per={}
                for r in sel: per.setdefault(r['graph_id'],[]).append(r)
                src_ae=[np.mean([x['AE2'] for x in v]) for v in per.values()]
                src_pe=[np.mean([x['ProfileAE'] for x in v]) for v in per.values()]
                src_sg=[np.mean([x['signed_rho2'] for x in v]) for v in per.values()]
                row={'stratum':stratum,'arm':arm,'method':method,'sources':len(per),
                     'MAE2':float(np.mean(src_ae)),
                     'MAE2_se':float(np.std(src_ae,ddof=1)/np.sqrt(len(src_ae))) if len(src_ae)>1 else float('nan'),
                     'ProfileMAE':float(np.mean(src_pe)),
                     'signed_rho2':float(np.mean(src_sg)),
                     'worst_source_AE2':float(np.max(src_ae)),
                     'fallbacks':sum(1 for r in sel if r['fallback'])}
                for ref in ('plugin','corrector'):
                    if method==ref: continue
                    base={}
                    for r in recs:
                        if (r['stratum']==stratum if stratum in ('real','surrogate') else r['graph_id'].startswith(stratum+'_r')) and r['arm']==arm and r['method']==ref:
                            base.setdefault(r['graph_id'],[]).append(r['AE2'])
                    common=sorted(set(per)&set(base))
                    if common:
                        d=[np.mean([x['AE2'] for x in per[g]])-np.mean(base[g]) for g in common]
                        row[f'paired_vs_{ref}']=float(np.mean(d))
                        row[f'paired_vs_{ref}_se']=float(np.std(d,ddof=1)/np.sqrt(len(d))) if len(d)>1 else float('nan')
                rows.append(row)
    cov=[]
    for stratum in ('real','surrogate','dar_a0','dar_a08','ad_memoryless','ad_memory'):
        for arm in ('R','S','H','B'):
            sel=[r for r in recs if (r['stratum']==stratum if stratum in ('real','surrogate') else r['graph_id'].startswith(stratum+'_r')) and r['arm']==arm and r['method']=='plugin']
            if not sel: continue
            cov.append({'stratum':stratum,'arm':arm,
                'dyad_coverage_median':float(np.median([r['dyad_coverage'] for r in sel])),
                'window_coverage_median':float(np.median([r['window_coverage'] for r in sel])),
                'event_coverage_median':float(np.median([r['event_coverage'] for r in sel])),
                'D_obs_median':float(np.median([r['D_obs'] for r in sel])),
                'D_obs_min':int(np.min([r['D_obs'] for r in sel])),
                'M_obs_median':float(np.median([r['M_obs'] for r in sel]))})
    return {'rows':rows,'coverage':cov}


def stage_decompose_main(out):
    """The same selection/history split on the main observations.

    The development version regenerates pool graphs; here the graphs are on disk,
    so the draw is replayed from the stored seeds instead. Real and synthetic
    strata are kept apart. Full histories are used for evaluation only.
    """
    import csv, time
    import numpy as np
    from main_experiment.data import load_graph
    from main_experiment.sampling import calibrate, draw
    rows=[]; t0=time.time()
    for gd in sorted((FROZEN/'graphs').iterdir()):
        if not (gd/'manifest.json').exists(): continue
        key=gd.name
        if key not in MAIN_KEYS: continue
        g=load_graph(gd)
        budget,walk=calibrate(g,FROZEN/'calibration'/key,FROZEN/'build')
        truth=np.array(g.truth,float)
        for arm in ('R','S','H','B'):
            for ix in range(1,draws_for(arm,budget)+1):
                counts,_=draw(g,arm,ix,'sample',budget,walk)
                seen=counts.sum(1)>0
                if not seen.any(): continue
                Kf=(g.counts[seen]>0).sum(1); Ko=(counts[seen]>0).sum(1)
                oracle=np.array([float((Kf>=k).mean()) for k in range(2,6)])
                plug=np.array([float((Ko>=k).mean()) for k in range(2,6)])
                rows.append({'graph_id':key,
                    'stratum':graph_stratum(key),
                    'arm':arm,'sample_index':ix,'D_full':g.D,'D_obs':int(seen.sum()),
                    'dyad_coverage':float(seen.mean()),
                    'truth_rho2':float(truth[0]),'oracle_rho2':float(oracle[0]),
                    'plugin_rho2':float(plug[0]),
                    'selection_rho2':float(oracle[0]-truth[0]),
                    'history_rho2':float(plug[0]-oracle[0]),
                    'total_rho2':float(plug[0]-truth[0])})
    with open(out/'error_decomposition_main.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for stratum in ('real','surrogate','dar_a0','dar_a08','ad_memoryless','ad_memory'):
        for arm in ('R','S','H','B'):
            sel=[r for r in rows if r['arm']==arm and (r['stratum']==stratum if stratum in ('real','surrogate') else r['graph_id'].startswith(stratum+'_r'))]
            if not sel: continue
            per={}
            for r in sel: per.setdefault(r['graph_id'],[]).append(r)
            agg=lambda k: float(np.mean([np.mean([x[k] for x in v]) for v in per.values()]))
            abs_sel=float(np.mean([abs(x['selection_rho2']) for x in sel]))
            abs_his=float(np.mean([abs(x['history_rho2']) for x in sel]))
            abs_tot=float(np.mean([abs(x['total_rho2']) for x in sel]))
            summary.append({'stratum':stratum,'arm':arm,'graphs':len(per),
                'dyad_coverage':agg('dyad_coverage'),
                'selection_rho2':agg('selection_rho2'),'history_rho2':agg('history_rho2'),
                'total_rho2':agg('total_rho2'),
                'abs_selection':abs_sel,'abs_history':abs_his,'abs_total':abs_tot,
                # Share of the SUM OF ABSOLUTE COMPONENTS, not an additive share of
                # the total: selection and history have opposite signs on H and B and
                # partly cancel, so |selection| + |history| exceeds |total| there.
                'history_share_of_abs_components':abs_his/(abs_sel+abs_his) if (abs_sel+abs_his) else None,
                'cancellation_ratio':(abs_sel+abs_his)/abs_tot if abs_tot else None})
    write_json(out/'error_decomposition_main.json',
               {'formula':'selection = oracle - truth; history = plugin - oracle; '
                          'total = selection + history exactly. history_share_of_abs_components '
                          '= |history| / (|selection| + |history|) is a share of the summed '
                          'absolute components, NOT an additive share of |total|; the two '
                          'components have opposite signs on H and B and partly cancel, '
                          'which cancellation_ratio = (|sel|+|hist|)/|total| quantifies.',
                'rows':summary})
    return {'observations':len(rows),'graphs':len({r['graph_id'] for r in rows}),
            'seconds':time.time()-t0}


def stage_decompose(out):
    """Split the plug-in error into a selection part and a lost-history part.

    For one observation, let
      truth  = the profile over all dyads of the graph,
      oracle = the profile over the *observed* dyads but computed from their
               complete five-window histories,
      plugin = the profile actually computable from the observation.
    Then oracle - truth is the error from which dyads the mechanism happened to
    reach, and plugin - oracle is the error from what it lost about the dyads it
    did reach. Full histories are used here for evaluation only; nothing in this
    function feeds an estimator.
    """
    import csv, time
    import numpy as np
    from main_experiment.pool import pool_definition
    from main_experiment.synthetic import generate_one
    from main_experiment.sampling import calibrate, draw
    specs={g['key']:g for g in pool_definition()['graphs'] if g['partition']=='dev'}
    rows=[]; t0=time.time()
    for key,sp in sorted(specs.items()):
        params={k:v for k,v in sp['parameters'].items() if v is not None and k!='mean_degree'}
        g,_,_=generate_one(key,sp['family'],params,domain='pool')
        budget,walk=calibrate(g,out/'pool'/'calibration'/key,out/'pool'/'build')
        truth=np.array(g.truth,float)
        for arm in ('R','S','H','B'):
            for ix in range(1,draws_for(arm,budget,'pool_dev')+1):
                counts,_=draw(g,arm,ix,'pool_dev',budget,walk)
                seen=counts.sum(1)>0
                if not seen.any(): continue
                Kf=(g.counts[seen]>0).sum(1)              # full histories, observed dyads
                Ko=(counts[seen]>0).sum(1)                # what the observation shows
                oracle=np.array([float((Kf>=k).mean()) for k in range(2,6)])
                plug=np.array([float((Ko>=k).mean()) for k in range(2,6)])
                rows.append({'graph_id':key,'family':sp['family'],'arm':arm,'sample_index':ix,
                    'D_full':g.D,'D_obs':int(seen.sum()),'dyad_coverage':float(seen.mean()),
                    'truth_rho2':float(truth[0]),'oracle_rho2':float(oracle[0]),
                    'plugin_rho2':float(plug[0]),
                    'selection_rho2':float(oracle[0]-truth[0]),
                    'history_rho2':float(plug[0]-oracle[0]),
                    'total_rho2':float(plug[0]-truth[0]),
                    'selection_profile':float(np.mean(oracle-truth)),
                    'history_profile':float(np.mean(plug-oracle)),
                    'total_profile':float(np.mean(plug-truth))})
    path=out/'error_decomposition.csv'
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary=[]
    for fam in ('all','dar','ad'):
        for arm in ('R','S','H','B'):
            sel=[r for r in rows if r['arm']==arm and (fam=='all' or r['family']==fam)]
            if not sel: continue
            per={}
            for r in sel: per.setdefault(r['graph_id'],[]).append(r)
            agg=lambda k:float(np.mean([np.mean([x[k] for x in v]) for v in per.values()]))
            summary.append({'family':fam,'arm':arm,'graphs':len(per),
                'dyad_coverage':agg('dyad_coverage'),
                'selection_rho2':agg('selection_rho2'),'history_rho2':agg('history_rho2'),
                'total_rho2':agg('total_rho2'),
                'abs_selection':float(np.mean([abs(x['selection_rho2']) for x in sel])),
                'abs_history':float(np.mean([abs(x['history_rho2']) for x in sel]))})
    write_json(out/'error_decomposition.json',{'rows':summary})
    return {'observations':len(rows),'graphs':len(specs),'seconds':time.time()-t0}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--stage',default='all',choices=['pool','train','dev','decompose','decompose_main','main','all'])
    a=ap.parse_args()
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    from main_experiment.integrity import bind,code_binding
    bind(out/'revision_inputs.json',{'code':code_binding(),'runner_sha256':sha(__file__),
         'main_inputs':read_json(FROZEN/'preparation_inputs.json'),
         'pool_definition':poolmod.pool_definition()})
    import fcntl
    lock=open(out/'revision.lock','a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter(); report={}
    if a.stage in ('pool','all'):
        report['pool']=stage_pool(out)
        print('pool:',report['pool'],flush=True)
    if a.stage in ('train','all'):
        report['train']=stage_train(out)
        print('train:',report['train'],flush=True)
    if a.stage=='decompose_main':
        report['decompose_main']=stage_decompose_main(out)
        print('decompose_main:',report['decompose_main'],flush=True)
    if a.stage=='main':
        report['main']=stage_main(out)
        print('main:',report['main'],flush=True)
    if a.stage in ('decompose','all'):
        report['decompose']=stage_decompose(out)
        print('decompose:',report['decompose'],flush=True)
    if a.stage in ('dev','all'):
        report['dev']=stage_dev(out)
        print('dev:',report['dev'],flush=True)
    report['elapsed_seconds']=time.perf_counter()-start
    write_json(out/f'report_{a.stage}.json',report)
    print(report,flush=True)

if __name__=='__main__': main()
