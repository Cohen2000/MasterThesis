import sys,json,hashlib,re,collections,gc,math,pickle
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[3]; sys.path.insert(0,str(ROOT/'src'))
from main_experiment.common import REAL_TEST,TRAIN,SYNTH,ANSWER_REGEX,digest
from main_experiment.observation import parse,features
from main_experiment.pool import pool_definition
RUN=ROOT/'results/main_experiment/cells10_20260917'
REV=ROOT/'results/baseline_revision_cells10_20260917'
Q=ROOT/'results/main_experiment/cells10_20260917_qwen'
report={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
obs={p.stem:read(p) for p in (RUN/'observations/sample').glob('*.json')}
req={r['id']:r for r in map(json.loads,(RUN/'requests.jsonl').read_text().splitlines())}
base=read(REV/'primary_baselines.json')['observations']
# Recompute all scores directly from raw JSON and source truth; no repository parser or error reducer.
rows=[]; anomalies=[]; bad=[]; seeds=[]
for p in sorted((Q/'answers').rglob('*.json')):
 r=read(p); plan=req[r['id']]; o=obs[r['observation_id']]; b=base[o['id']]
 assert r['prompt_sha256']==plan['prompt_sha256']==digest(o['messages'])
 assert r['seed']==plan['seed'] and r['config_id']==plan['config_id']
 assert r['runner_sha256']==sha(ROOT/'scripts/run_qwen_engine.py')
 assert r['structured_output']['regex']==ANSWER_REGEX
 assert r['engine_prompt_tokens']==r['input_tokens']
 assert r['finish_reason']=='stop' and r['reasoning_closed'] and r['status']=='completed'
 assert re.fullmatch(ANSWER_REGEX,r['final_text'])
 if r['mode']=='thinking':assert r['raw_text'].partition('</think>')[2].strip()==r['final_text']
 else: assert r['raw_text'].strip()==r['final_text']
 z=json.loads(r['final_text']); values=[z[f'rho_{k}'] for k in range(2,6)]
 valid=(set(z)=={f'rho_{k}' for k in range(2,6)} and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in values) and all(values[i]>=values[i+1] for i in range(3)))
 if not valid:bad.append(r['id']); values=b['plugin']['prediction']
 truth=np.array(o['truth']); assert b['truth']==o['truth']
 err=np.abs(np.array(values)-truth)
 rows.append(dict(id=r['id'],graph_id=o['graph_id'],arm=o['arm'],sample_index=o['sample_index'],repeat_index=r['repeat_index'],config_id=r['config_id'],stratum=o['stratum'],AE2=err[0],ProfileAE=err.mean(),delta_AE2=err[0]-abs(b['primary_corrector']['prediction'][0]-truth[0]),valid=valid,output_tokens=r['output_tokens']))
 seeds.append(r['seed']%2**31)
frame=pd.DataFrame(rows); stored=pd.read_csv(Q/'evaluation_v2/answer_errors.csv').set_index('id')
for row in rows:
 for k in ['AE2','ProfileAE','delta_AE2']:
  assert abs(row[k]-stored.loc[row['id'],k])<1e-14
report['raw_answers']={'n':len(rows),'invalid':bad,'mod31_unique_seeds':len(set(seeds)),'raw_hash_seed_runner_regex_and_score_checks':'passed'}
means=frame[frame.stratum=='real'].groupby(['config_id','arm','graph_id'])[['AE2','ProfileAE','delta_AE2']].mean().groupby(['config_id','arm']).mean()
report['real_scores']=means.reset_index().to_dict('records')
# Compare reported variance decomposition to the data, independently of cell_variance.
varrows=[]
for (cfg,arm,g),gdf in frame.groupby(['config_id','arm','graph_id']):
 for metric in ['AE2','delta_AE2']:
  a=gdf.pivot(index='sample_index',columns='repeat_index',values=metric).to_numpy()
  assert a.shape==(5,3)
  total=float(a.mean(1).var(ddof=1)/5)
  model=float(a.var(axis=1,ddof=1).mean()/15)
  residual=total-model
  varrows.append(dict(config=cfg,arm=arm,graph=g,metric=metric,total=total,model=model,raw_sampler=residual,reported_sum=model+max(0.,residual),real=g in REAL_TEST))
v=pd.DataFrame(varrows); report['variance']={'negative_residual_cells':int((v.raw_sampler < -1e-15).sum()),'cells':len(v),'negative_real_AE2':int(((v.raw_sampler < -1e-15)&v.real&(v.metric=='AE2')).sum()),'real_AE2_cells':int((v.real&(v.metric=='AE2')).sum()),'examples':v[(v.raw_sampler< -1e-15)&v.real].sort_values('raw_sampler').head(8).to_dict('records')}
summary=pd.read_csv(Q/'evaluation_v2/summary.csv')
report['aggregate_mcse']=summary[(summary.stratum=='real')&summary.config_id.str.startswith('qwen')][['arm','config_id','AE2_MCSE','AE2_MCSE_model_repeats','AE2_MCSE_sampler','delta_AE2_MCSE','delta_AE2_MCSE_model_repeats','delta_AE2_MCSE_sampler']].to_dict('records')
# All 14 baseline models: hashes, full held-out source exclusion, independent pool partition,
# prediction agreement with the primary reference for every observation.
model_checks=[]
trainpool={s['key'] for s in pool_definition()['graphs'] if s['partition']=='train'}
devpool={s['key'] for s in pool_definition()['graphs'] if s['partition']=='dev'}
for folder,method in [('models_pooled','extratrees_pooled'),('models_real_only','extratrees_real_only')]:
 for fold in list(REAL_TEST)+['synthetic']:
  d=REV/folder/fold; m=read(d/'manifest.json')
  assert sha(d/'model.pkl')==m['model_sha256']
  allowed=set(TRAIN)-({fold} if fold in REAL_TEST else set())
  assert set(m['real_sources'])==allowed
  assert not (set(m['sources'])&devpool)
  assert set(m['sources'])==allowed|(trainpool if folder=='models_pooled' else set())
  assert not any(oid.startswith(fold+'__') for oid in m['observations'])
  eligible=[o for o in obs.values() if (o['graph_id']==fold if fold!='synthetic' else o['stratum']=='synthetic')]
  with open(d/'model.pkl','rb') as f:model=pickle.load(f)
  pred=model.predict(np.array([features(parse(o['block'])) for o in eligible]))
  assert np.allclose(pred,[base[o['id']][method]['prediction'] for o in eligible],rtol=0,atol=1e-14)
  model_checks.append(dict(folder=folder,fold=fold,n_observations=len(eligible),n_training_sources=len(m['sources'])))
  del model;gc.collect()
report['model_checks']=model_checks
# Check the exact distribution induced by the deterministic train/dev split.
sp=pool_definition()['graphs']; report['pool_split']={}
assert read(REV/'pool_definition.json')['graphs']==sp
for spec in sp:
 actual=read(REV/'pool/observations'/f"{spec['key']}.json")
 assert actual['parameters']==spec['parameters'] and actual['partition']==spec['partition']
for f in ['dar','ad']:
 for split in ['train','dev']:
  pars=[s['parameters'] for s in sp if s['family']==f and s['partition']==split]
  report['pool_split'][f+'_'+split]={'N':dict(collections.Counter(p['N'] for p in pars))}
  if f=='ad': report['pool_split'][f+'_'+split]['rounds']=dict(collections.Counter(p['rounds'] for p in pars))
# Raw-file identity checks for every empirical source.
report['raw_file_hashes']={}
for g in TRAIN:
 m=read(RUN/'graphs'/g/'manifest.json')
 ok=sha(ROOT/'data/raw'/m['raw_file'])==m['raw_sha256']; assert ok
 report['raw_file_hashes'][g]=ok
# Independently reconstruct the six real truths using pandas and integer source IDs,
# without canonical(), Graph.truth or census helpers.
ind=[]
for g in REAL_TEST:
 m=read(RUN/'graphs'/g/'manifest.json'); fmt=m['format']; path=ROOT/'data/raw'/m['raw_file']
 raw=pd.read_csv(path,sep=r'\s+' if fmt['delimiter']=='whitespace' else fmt['delimiter'],header=None,skiprows=fmt.get('skiprows',0),comment='#',usecols=sorted(fmt['columns'].values()))
 raw.columns=[next(k for k,v in fmt['columns'].items() if v==i) for i in sorted(fmt['columns'].values())]
 raw=raw[['u','v','t']]; raw=raw[raw.u!=raw.v].copy()
 uv=np.sort(raw[['u','v']].to_numpy(dtype=np.int64),axis=1); raw['u']=uv[:,0];raw['v']=uv[:,1]
 if g.startswith('sp_') or g=='copenhagen_bluetooth':raw=raw.drop_duplicates(['u','v','t'])
 lo,hi=raw.t.min(),raw.t.max(); t=raw.t.to_numpy()
 raw['window']=sum((t>=lo+(hi-lo)*j/5).astype(np.int64) for j in range(1,5))
 k=raw.groupby(['u','v']).window.nunique(); truth=[float((k>=j).mean()) for j in range(2,6)]
 assert truth==m['truth'] and len(raw)==m['M_full'] and len(k)==m['D_full']
 ind.append(dict(graph=g,events=len(raw),dyads=len(k),truth=truth,horizon_days=float((hi-lo)/86400)))
 del raw;gc.collect()
report['independent_raw_reconstruction']=ind
# Same current source code bound to the offline preparation, or explicitly list differences.
manifest=read(RUN/'preparation_inputs.json'); report['changed_source_hashes']=[p for p,h in manifest['sources'].items() if sha(ROOT/p)!=h]
Path(__file__).with_name('audit_results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['model_checks','aggregate_mcse','raw_file_hashes']},indent=2))
