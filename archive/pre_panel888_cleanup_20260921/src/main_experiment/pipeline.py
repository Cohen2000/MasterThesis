from .common import SURROGATES, SURROGATE_PARENT, MAIN_KEYS, MASTER_SEED, parent_source, graph_stratum, fold_for
from .surrogates import prepare_surrogate
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
from .common import (ROOT,REAL_TEST,TRAIN,SYNTH,ARMS,SEEDS,CONFIGS,LLM_REPEATS,ARM_ID,DESIGN_VERSION,
                     H_VARIANT,MATCHED_QUANTITY,COVERAGE_FRACTION,CURRENT_RUN,
                     seed,sha,digest,read_json,write_json,verify_immutable_checkpoints,
                     draws_for,observation_id,planned_sizes)
from .data import prepare_real,load_graph,save_graph
from .synthetic import generate_pair
from .sampling import calibrate,Walk,draw
from .observation import make,serialize,parse,features,messages,FEATURE_VERSION
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
           ROOT/'src/dataset_census.py',ROOT/'src/census.py',ROOT/'docs/PROTOCOL_PANEL888_20260921.md']
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(paths)}


def run(args):
    out=Path(args.out).resolve(); out.mkdir(parents=True,exist_ok=True)
    lock=open(out/'run.lock','w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    start=time.perf_counter()
    verify_immutable_checkpoints(out)
    binding(out/'preparation_inputs.json',{'sources':source_inventory(),'master_seed':MASTER_SEED,
                                           'design_version':DESIGN_VERSION,'h_variant':H_VARIANT,
                                           'feature_version':FEATURE_VERSION})
    lockfile=out/'environment.lock.txt'
    packages=sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    text='\n'.join(packages)+'\n'
    if lockfile.exists() and lockfile.read_text()!=text: raise ValueError('environment changed on resume')
    lockfile.write_text(text)
    write_json(out/'environment.json',{'python':platform.python_version(),'platform':platform.platform(),
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'lock_sha256':sha(lockfile),'inference_performed':False,'design_version':DESIGN_VERSION,
            'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0]})
    graph_keys=list(TRAIN)+list(SYNTH)+list(SURROGATES)
    if args.graph:
        if args.graph not in graph_keys: raise ValueError('unknown graph')
        graph_keys=[args.graph]
    # Each source is an independent resumable stage. Failures remain explicit.
    failures={}; all_obs=[]; train_rows=[]; truths={}; data_rows=[]; budgets=[]; budget_by_graph={}
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
            elif key in SURROGATES:
                parent=load_graph(out/'graphs'/SURROGATE_PARENT[key])
                g=prepare_surrogate(parent,gd)
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
            budgets.append({'graph_id':key,**budget}); budget_by_graph[key]=budget
            print(f'{key}: N={g.N} D={g.D} M={g.M} W={g.cells} T={budget["T"]:.1f} n={budget["n_panel"]} '
                  f'L={budget["L"]} n_H={budget["n_panel_history"]} p={budget["p"]:.5f} '
                  f'H_saturated={budget["h_saturated"]} matched={budget["budget_matched"]}',flush=True)
            for domain in (['training'] if key in TRAIN and key not in REAL_TEST else
                           ['training','sample'] if key in REAL_TEST else ['sample']):
                for arm in ARMS:
                    # A saturated H draw is deterministic and carried once.
                    for ix in range(1,draws_for(arm,budget,domain)+1):
                        oid=observation_id(key,arm,ix); dest=out/'observations'/domain/(oid+'.json')
                        # Register seeds also on resume.
                        seed(domain,parent_source(key),ARM_ID[arm],ix)
                        if dest.exists(): row=read_json(dest)
                        else:
                            counts,re=draw(g,arm,ix,domain,budget,walk)
                            obs=make(g,arm,budget,counts,re); block=serialize(obs)
                            parsed=parse(block)
                            if not np.array_equal(features(obs),features(parsed)): raise AssertionError('serialization changes features')
                            msg=messages(block)
                            row={'id':oid,'graph_id':key,'source_family':key,'stratum':graph_stratum(key),'parent_source':parent_source(key),
                                 'arm':arm,'sample_index':ix,'domain':domain,'empty':parsed['D_obs']==0,
                                 'block':block,'block_sha256':digest(block),'messages':msg,'prompt_sha256':digest(msg),
                                 'truth':g.truth,'design_version':DESIGN_VERSION,
                                 'h_variant':H_VARIANT if arm=='H' else None,
                                 'deterministic_draw':draws_for(arm,budget,domain)==1,
                                 'budget_matched':budget['budget_matched_by_arm'][arm],
                                 'budget_matched_all_arms':budget['budget_matched'],
                                 'internal_evaluation':{'observed_event_fraction':parsed['M_obs']/g.M,
                                    'observed_dyad_fraction':parsed['D_obs']/g.D,
                                    'observed_cell_fraction':int((counts>0).sum())/g.cells}}
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
    complete_sources=set(truths)==set(TRAIN)|set(SYNTH)|set(SURROGATES)
    baseline_rows=[]
    if set(TRAIN)<=set(truths):
        models,medians=fit_folds(train_rows,truths,out/'models',lockfile)
        for row in all_obs:
            fold=fold_for(row['graph_id'])
            o=parse(row['block']); extra=models[fold].predict([features(o)])[0]
            b=all_baselines(o,medians[fold],extra)
            write_json(out/'baselines'/(row['id']+'.json'),b)
            for method,result in b.items():
                baseline_rows.append({k:row[k] for k in ['id','graph_id','stratum','arm','sample_index','empty','budget_matched']} |
                    {'method':method,**result,**errors(result['prediction'],row['truth'])})
    csv_write(out/'baseline_observations.csv',baseline_rows)
    summaries=[]
    for stratum in ['real','surrogate','dar_a0','dar_a08','ad_memoryless','ad_memory']:
        rs=[r for r in baseline_rows if (r['stratum']==stratum if stratum in ('real','surrogate') else r['graph_id'].startswith(stratum+'_r'))]
        for arm in ARMS:
            for method in ['plugin','corrector','median','extratrees']:
                group=[r for r in rs if r['arm']==arm and r['method']==method]
                if not group: continue
                cells={}; expected={}
                for gid in sorted({r['graph_id'] for r in group}):
                    # A baseline is deterministic given the observation: one column.
                    cells[gid]=[[r['AE2']] for r in sorted(group,key=lambda x:x['sample_index']) if r['graph_id']==gid]
                    expected[gid]=draws_for(arm,budget_by_graph[gid])
                estimate=paired_summary(cells,expected)
                summaries.append({'stratum':stratum,'arm':arm,'method':method,'MAE2':estimate['mean'],'MCSE':estimate['mcse'],
                    'between_source_SD':estimate['between_source_sd'],
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
    for key in MAIN_KEYS:
        for arm in ARMS:
            # Draw counts come from the calibrated design of each graph; a graph
            # whose preparation failed is listed with the default draw count.
            n=draws_for(arm,budget_by_graph[key]) if key in budget_by_graph else None
            for ix in range(1,(n or 3)+1):
                oid=observation_id(key,arm,ix); row=obs_index.get(oid)
                status.append({'id':oid,'graph_id':key,'arm':arm,'sample_index':ix,
                    'status':'blocked_preparation' if row is None else 'empty_no_calls' if row['empty'] else 'prepared_not_started',
                    'deterministic_draw':None if n is None else n==1,
                    'planned_logical_calls':len(CONFIGS)*LLM_REPEATS,'actual_calls':0,'empty':row['empty'] if row else None})
    csv_write(out/'observation_status.csv',status)
    write_json(out/'seed_manifest.json',[{'seed':s,'fields':json.loads(v)} for s,v in sorted(SEEDS.items())])
    main_keys=MAIN_KEYS
    sizes_ok=set(main_keys)|set(TRAIN)<=set(budget_by_graph)
    design=planned_sizes(budget_by_graph,main_keys,TRAIN) if sizes_ok else None
    MAIN=design['main_observations'] if design else None
    report={'design_version':DESIGN_VERSION,'h_variant':H_VARIANT,'feature_version':FEATURE_VERSION,
        'matched_quantity':MATCHED_QUANTITY,'coverage_fraction':COVERAGE_FRACTION,
        'planned_observations':MAIN,'prepared_observations':len(all_obs),
        'planned_training_observations':design['training_observations'] if design else None,
        'training_observations':len(train_rows),
        'deterministic_h_graphs':sorted(k for k,b in budget_by_graph.items() if b['h_saturated']),
        'h_target_unreachable_graphs':sorted(k for k,b in budget_by_graph.items() if b['h_target_unreachable']),
        'budget_unmatched_graphs':{k:b['unmatched_reasons'] for k,b in budget_by_graph.items() if not b['budget_matched']},
        'planned_logical_calls':design['planned_calls'] if design else None,'request_manifest_rows':len(requests),
        'empty_observations':sum(r['empty'] for r in all_obs),
        'eligible_unstarted_calls':sum(not r['empty'] for r in all_obs)*len(CONFIGS)*LLM_REPEATS,'started_calls':0,'failed_responses':0,
        'baseline_rows':len(baseline_rows),'fitted_models':len(REAL_TEST)+1 if baseline_rows else 0,
        'prompt_sizes_checked':len(sizes),'prompt_failures':prompt_failures,'source_failures':failures,
        'offline_ready':bool(design and len(train_rows)==design['training_observations'] and complete_sources
                         and len(all_obs)==MAIN and len(sizes)==MAIN
                         and len(requests)==design['planned_calls']
                         and len(baseline_rows)==MAIN*4 and not prompt_failures),
        'provider_release_ready':False,'llm_study_conducted':False,
        'technical_start_conditions':['API authentication and current prices/reserves',
          'Provider framing: exact DeepSeek and Sol input size verification',
          'Qwen BF16 serving under vLLM 0.29.0 (verified to know the architecture)',
          'Authorized technical inference tests, streaming, usage and transport recovery'],
        'elapsed_seconds_this_invocation':time.perf_counter()-start}
    write_json(out/'report.json',report)
    files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()
           and p.name not in ['checksums.json','run.lock'] and 'build' not in p.parts and not p.name.endswith('.tmp')}
    write_json(out/'checksums.json',files)
    print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description='Frozen main experiment, strictly offline; no LLM transport')
    p.add_argument('--out',default=str(ROOT/CURRENT_RUN))
    p.add_argument('--raw-dir',default=str(ROOT/'data/raw'))
    p.add_argument('--tokenizers',default=str(ROOT/'data/tokenizers'))
    p.add_argument('--graph',help='Prepare one source only, resumable checkpoint for CPU jobs')
    run(p.parse_args())
