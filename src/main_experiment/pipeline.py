import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import numpy as np
import yaml
from .common import ROOT,REAL_TEST,TRAIN,SYNTH,ARMS,SEEDS,seed,sha,digest,read_json,write_json,verify_immutable_checkpoints
from .data import prepare_real,load_graph,save_graph
from .synthetic import generate_pair
from .sampling import calibrate,Walk,draw
from .observation import make,serialize,parse,features,messages
from .training import fit_folds
from .baselines import all_baselines
from .evaluation import errors,paired_summary
from .requests import planned,EXECUTION_POLICY
from .token_sizes import TokenCounters


def csv_write(path,rows):
    if not rows: return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    tmp=Path(path).with_suffix('.tmp')
    with open(tmp,'w') as f:
        w=csv.DictWriter(f,fieldnames=keys); w.writeheader()
        for r in rows: w.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    tmp.replace(path)


def binding(path,inputs):
    path=Path(path)
    if path.exists():
        if read_json(path)!=inputs: raise ValueError(f'resume dependency changed: {path}; use a new output directory')
    else: write_json(path,inputs)


def source_inventory():
    paths=[*Path(__file__).parent.glob('*.py'),*Path(__file__).parent.glob('*.cpp'),
           * (ROOT/'config/main_experiment').glob('*.txt'),ROOT/'config/study.yaml',
           ROOT/'config/datasets.yaml',ROOT/'scripts/run_main_offline.py',ROOT/'scripts/evaluate_main_responses.py',
           ROOT/'src/dataset_census.py',ROOT/'src/census.py',ROOT/'docs/MAIN_FREEZE_SOURCE.txt',ROOT/'docs/MAIN_EXPERIMENT_IMPLEMENTATION.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def run(args):
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter()
    verify_immutable_checkpoints(out)
    binding(out/'preparation_inputs.json',{'sources':source_inventory(),'master_seed':20260916})
    lockfile=out/'environment.lock.txt'
    packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    text='\n'.join(packages)+'\n'
    if lockfile.exists() and lockfile.read_text()!=text: raise ValueError('environment changed on resume')
    lockfile.write_text(text)
    write_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'lock_sha256':sha(lockfile),'inference_performed':False,
            'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0]})
    graph_keys=list(TRAIN)+list(SYNTH)
    if args.graph:
        if args.graph not in graph_keys: raise ValueError('unknown graph')
        graph_keys=[args.graph]
    # Each source is an independent resumable stage. Failures remain explicit.
    failures={}; all_obs=[]; train_rows=[]; truths={}; data_rows=[]; budgets=[]
    for key in graph_keys:
        gd=out/'graphs'/key
        try:
            if key in TRAIN:
                spec=yaml.safe_load((ROOT/'config/datasets.yaml').read_text())['datasets'][key]
                input_file=Path(args.raw_dir)/spec['file']
                raw_bind={'raw_sha256':sha(input_file)}
                if key=='copenhagen_bluetooth': raw_bind['original_sha256']=sha(Path(args.raw_dir)/'bt_symmetric.csv')
                binding(gd/'raw_binding.json',raw_bind)
                g=load_graph(gd) if (gd/'manifest.json').exists() else prepare_real(key,args.raw_dir,gd)
            else:
                seed('graph',key.split('_')[0]+'_pair_r'+key[-1])
                if not (gd/'manifest.json').exists():
                    family=key.split('_')[0]; replicate=int(key[-1])
                    for g0,x,meta in generate_pair(family,replicate):
                        dest=out/'graphs'/g0.key
                        if not (dest/'manifest.json').exists(): save_graph(dest,g0,meta,x)
                g=load_graph(gd)
            truths[key]=g.truth; data_rows.append(read_json(gd/'manifest.json'))
            budget,walk=calibrate(g,out/'calibration'/key,out/'build')
            budgets.append({'graph_id':key,**budget})
            print(f'{key}: N={g.N} D={g.D} M={g.M} B={g.B} L={budget["L"]} matched={budget["budget_matched"]}',flush=True)
            for domain in (['training'] if key in TRAIN and key not in REAL_TEST else
                           ['training','sample'] if key in REAL_TEST else ['sample']):
                for arm in ARMS:
                    for ix in range(1,2 if arm=='H' else 6):
                        oid=f'{key}__{arm}__s{ix}'; dest=out/'observations'/domain/(oid+'.json')
                        # Register seeds also on resume; suffix is deterministic.
                        if arm!='H': seed(domain,key,arm,ix)
                        if dest.exists(): row=read_json(dest)
                        else:
                            counts,re=draw(g,arm,ix,domain,budget,walk)
                            obs=make(g,arm,budget,counts,re); block=serialize(obs)
                            parsed=parse(block)
                            if not np.array_equal(features(obs),features(parsed)): raise AssertionError('serialization changes features')
                            msg=messages(block)
                            row={'id':oid,'graph_id':key,'source_family':key,'stratum':'real' if key in TRAIN else 'synthetic',
                                 'arm':arm,'sample_index':ix,'domain':domain,'empty':parsed['D_obs']==0,
                                 'block':block,'block_sha256':digest(block),'messages':msg,'prompt_sha256':digest(msg),
                                 'truth':g.truth,'budget_matched':budget['budget_matched'],
                                 'internal_evaluation':{'observed_event_fraction':parsed['M_obs']/g.M,
                                    'observed_dyad_fraction':parsed['D_obs']/g.D}}
                            write_json(dest,row)
                        if digest(row['block'])!=row['block_sha256'] or digest(row['messages'])!=row['prompt_sha256']:
                            raise ValueError('observation checksum mismatch')
                        (train_rows if domain=='training' else all_obs).append(row)
        except (ValueError,FileNotFoundError) as e:
            failures[key]=str(e); print(f'BLOCKED {key}: {e}',flush=True)
    csv_write(out/'data_summary.csv',data_rows); csv_write(out/'budget_summary.csv',budgets)
    write_json(out/'failures.json',failures)
    if args.graph:
        write_json(out/'partial_status.json',{'graph':args.graph,'failures':failures,'elapsed_seconds':time.perf_counter()-start})
        return
    complete_sources=set(truths)==set(TRAIN)|set(SYNTH)
    baseline_rows=[]
    if set(TRAIN)<=set(truths):
        models,medians=fit_folds(train_rows,truths,out/'models',lockfile)
        for row in all_obs:
            fold=row['graph_id'] if row['stratum']=='real' else 'synthetic'
            o=parse(row['block']); extra=models[fold].predict([features(o)])[0]
            b=all_baselines(o,medians[fold],extra)
            write_json(out/'baselines'/(row['id']+'.json'),b)
            for method,result in b.items():
                baseline_rows.append({k:row[k] for k in ['id','graph_id','stratum','arm','sample_index','empty','budget_matched']} |
                    {'method':method,**result,**errors(result['prediction'],row['truth'])})
    csv_write(out/'baseline_observations.csv',baseline_rows)
    summaries=[]
    for stratum in ['real','dar_a0','dar_a08','ad_memoryless','ad_memory']:
        rs=[r for r in baseline_rows if (r['stratum']=='real' if stratum=='real' else r['graph_id'].startswith(stratum+'_r'))]
        for arm in ARMS:
            for method in ['plugin','corrector','median','extratrees']:
                group=[r for r in rs if r['arm']==arm and r['method']==method]
                if not group: continue
                cells={}
                for gid in sorted({r['graph_id'] for r in group}):
                    cells[gid]=[[r['AE2']]*3 for r in sorted(group,key=lambda x:x['sample_index']) if r['graph_id']==gid]
                estimate=paired_summary(cells)
                summaries.append({'stratum':stratum,'arm':arm,'method':method,'MAE2':estimate['mean'],'MCSE':estimate['mcse'],
                    'ProfileMAE':float(np.mean([r['ProfileAE'] for r in group])),
                    'signed_plugin_error':float(np.mean([r['signed_rho2'] for r in group])) if method=='plugin' else None,
                    'valid_fraction':float(np.mean([r['valid'] for r in group])),
                    'replacement_fraction':float(np.mean([r['replacement'] is not None for r in group])),
                    'empty_fraction':float(np.mean([r['empty'] for r in group])),
                    'sources':estimate['sources']})
    csv_write(out/'baseline_summary.csv',summaries)
    prompt_failures={}; sizes=[]
    try:
        counters=TokenCounters(args.tokenizers)
        for row in all_obs:
            try: sizes.append({'id':row['id'],**counters.count(row['messages'])})
            except ValueError as e: prompt_failures[row['id']]=str(e)
    except (ValueError,FileNotFoundError,OSError,ImportError) as e: prompt_failures['tokenizers']=str(e)
    csv_write(out/'prompt_sizes.csv',sizes)
    write_json(out/'tokenizer_manifest.json',{name:read_json(Path(args.tokenizers)/name/'provenance.json') for name in ['qwen','deepseek'] if (Path(args.tokenizers)/name/'provenance.json').exists()})
    requests=planned(all_obs)
    write_json(out/'execution_policy.json',EXECUTION_POLICY)
    path=out/'requests.jsonl'; tmp=path.with_suffix('.tmp')
    with open(tmp,'w') as f:
        # sort_keys: byte-identical manifest whether a stage was just computed or reloaded.
        for r in requests: f.write(json.dumps(r,separators=(',',':'),allow_nan=False,sort_keys=True)+'\n')
    tmp.replace(path)
    obs_index={r['id']:r for r in all_obs}
    status=[]
    for key in [*REAL_TEST,*SYNTH]:
        for arm in ARMS:
            for ix in range(1,2 if arm=='H' else 6):
                oid=f'{key}__{arm}__s{ix}'; row=obs_index.get(oid)
                status.append({'id':oid,'graph_id':key,'arm':arm,'sample_index':ix,
                    'status':'blocked_preparation' if row is None else 'empty_no_calls' if row['empty'] else 'prepared_not_started',
                    'planned_logical_calls':12,'actual_calls':0,'empty':row['empty'] if row else None})
    csv_write(out/'observation_status.csv',status)
    write_json(out/'seed_manifest.json',[{'seed':s,'fields':json.loads(v)} for s,v in sorted(SEEDS.items())])
    report={'planned_observations':224,'prepared_observations':len(all_obs),'training_observations':len(train_rows),
        'planned_logical_calls':2688,'request_manifest_rows':len(requests),
        'empty_observations':sum(r['empty'] for r in all_obs),
        'eligible_unstarted_calls':sum(not r['empty'] for r in all_obs)*12,'started_calls':0,'failed_responses':0,
        'baseline_rows':len(baseline_rows),'fitted_models':7 if baseline_rows else 0,
        'prompt_sizes_checked':len(sizes),'prompt_failures':prompt_failures,'source_failures':failures,
        'offline_ready':len(train_rows)==256 and complete_sources and len(all_obs)==224 and len(baseline_rows)==896 and len(sizes)==224 and not prompt_failures,
        'provider_release_ready':False,'llm_study_conducted':False,
        'technical_start_conditions':['API authentication and current prices/reserves',
          'Provider framing: exact DeepSeek and Sol input size verification',
          'Qwen BF16 vLLM 0.20.1 serving and reasoning/JSON separation',
          'Authorized technical inference tests, streaming, usage and transport recovery'],
        'elapsed_seconds_this_invocation':time.perf_counter()-start}
    write_json(out/'report.json',report)
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()
           and p.name not in ['checksums.json','run.lock'] and 'build' not in p.parts and not p.name.endswith('.tmp')}
    write_json(out/'checksums.json',files)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description='Frozen main experiment, strictly offline; no LLM transport')
    p.add_argument('--out',default=str(ROOT/'results/main_experiment/run'))
    p.add_argument('--raw-dir',default=str(ROOT/'data/raw'))
    p.add_argument('--tokenizers',default=str(ROOT/'data/tokenizers'))
    p.add_argument('--graph',help='Prepare one source only, resumable checkpoint for CPU jobs')
    run(p.parse_args())
