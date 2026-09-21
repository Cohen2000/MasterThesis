#!/usr/bin/env python3
"""Independent raw JSON/error cross-check and compact, hash-bound result packet."""
import csv,json,math,re,sys,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,DESIGN_VERSION,sha,read_json,write_json

def strict(raw):
    def pairs(items):
        d={}
        for k,v in items:
            if k in d: raise ValueError('duplicate')
            d[k]=v
        return d
    raw=raw.strip()
    if raw.startswith('```'):
        m=re.fullmatch(r'```[A-Za-z0-9_+-]*\s*\n(.*?)\n?```\s*',raw,re.S)
        if not m: return None
        raw=m[1]
    try:
        d=json.loads(raw,object_pairs_hook=pairs)
        if type(d)!=dict or set(d)!={f'rho_{k}' for k in range(2,6)}: return None
        v=[d[f'rho_{k}'] for k in range(2,6)]
        if any(type(x) not in (int,float) or not math.isfinite(x) or not 0<=x<=1 for x in v): return None
        return v if all(a>=b for a,b in zip(v,v[1:])) else None
    except (ValueError,TypeError,OverflowError): return None

def main():
    run=ROOT/CURRENT_RUN; rev=ROOT/CURRENT_REVISION; qwen=run.with_name(run.name+'_qwen')
    evaluation=qwen/'evaluation'; manifest=read_json(qwen/'integration_manifest.json'); assert manifest['complete']
    assert read_json(rev/'revision_audit.json')['verified']
    ledger=read_json(run.with_name(run.name+'_api')/'ledger.json')
    assert ledger['requests']=={} and ledger['batches']=={}
    planned={r['id']:r for r in map(json.loads,(run/'requests.jsonl').read_text().splitlines()) if r['config_id'].startswith('qwen_')}
    errors={r['id']:r for r in csv.DictReader((evaluation/'answer_errors.csv').open())}
    collected={r['id']:r for r in map(json.loads,(qwen/'responses.jsonl').read_text().splitlines())}
    assert set(collected)==set(planned)=={r['id'] for r in manifest['records']}
    counts=defaultdict(Counter); metric=defaultdict(list); outputs=defaultdict(list)
    for r in manifest['records']:
        source=ROOT/r['source']; raw=read_json(source); request=planned[r['id']]
        assert sha(source)==r['answer_sha256']
        local=qwen/'answers'/f"{raw['mode']}_r{request['repeat_index']}"/source.name
        assert sha(local)==sha(source)
        pred=strict(raw.get('final_text',''))
        if raw.get('technical_error',raw.get('status')!='completed'): pred=None
        row=errors[r['id']]; valid=pred is not None; assert valid==(row['valid']=='True')
        obs=read_json(run/'observations/sample'/f"{request['observation_id']}.json")
        for key in ('all',request['arm'],request['arm']+'/'+request['config_id']):
            c=counts[key]; c['found']+=1; c['valid' if valid else 'invalid']+=1
            c['technical_errors']+=bool(raw.get('technical_error',raw.get('status')!='completed'))
            c['tokenlimits']+=raw.get('end_state')=='output_limit'
            c['unclosed_reasoning']+=not raw.get('reasoning_closed',True)
            c['empty_final']+=not bool(raw.get('final_text','').strip())
            outputs[key].append(raw.get('output_tokens') or 0)
        if valid:
            diff=np.abs(np.array(pred)-obs['truth']); ae=float(diff[0]); pe=float(diff.mean())
            assert abs(ae-float(row['AE2']))<1e-14 and abs(pe-float(row['ProfileAE']))<1e-14
            stratum='real' if request['stratum']=='real' else request['graph_id'].rsplit('_r',1)[0]
            metric[stratum,request['arm'],request['config_id'],request['graph_id']].append((ae,pe))
    summaries=list(csv.DictReader((evaluation/'summary.csv').open())); checked=0
    for row in summaries:
        if not row['config_id'].startswith('qwen_'): continue
        assert row['complete']=='True'
        key=(row['stratum'],row['arm'],row['config_id'])
        means=[np.mean(v,axis=0) for k,v in metric.items() if k[:3]==key]
        means=np.mean(means,axis=0)
        assert abs(means[0]-float(row['AE2']))<1e-14 and abs(means[1]-float(row['ProfileAE']))<1e-14
        checked+=1
    for key,c in counts.items():
        c['planned']=1680 if key=='all' else 420 if '/' not in key else 210
        c['missing']=c['planned']-c['found']; c['invalid']+=0
        c['output_tokens_sum']=sum(outputs[key]); c['output_tokens_max']=max(outputs[key])
    assert counts['all']['found']==1680 and checked==40
    packet=ROOT/'docs/results/srw_20260920'; packet.mkdir(parents=True,exist_ok=True)
    files={
        'offline_report.json':run/'report.json','offline_verification.json':ROOT/'results/srw_20260920_logs/verify.log',
        'revision_audit.json':rev/'revision_audit.json',
        'h_sensitivity.json':rev/'history_sensitivity/report.json',
        'srw_diagnostics.json':rev/'srw_diagnostics/report.json',
        'h_sources.csv':rev/'history_sensitivity/sources.csv',
        'srw_components.csv':rev/'srw_diagnostics/components.csv',
        'srw_summary.csv':rev/'srw_diagnostics/summary.csv',
        'tests.txt':ROOT/'results/srw_20260920_logs/tests.log',
        'bundle_SHA256SUMS':ROOT/'results/srw_bundle_provenance/BUNDLE_SHA256SUMS',
        'bundle_WORKTREE_STATUS':ROOT/'results/srw_bundle_provenance/WORKTREE_STATUS',
        'bundle_SPEC_COMMIT':ROOT/'results/srw_bundle_provenance/SPEC_COMMIT',
        'slurm_final.psv':ROOT/'results/srw_20260920_logs/slurm_final.psv',
        'h_cluster_status.json':ROOT/'results/imported_h_20260920/status_final.json',
        's_cluster_status.json':ROOT/'results/imported_srw_20260920/status_final.json',
        'qwen_summary.csv':evaluation/'summary.csv','qwen_source_results.csv':evaluation/'source_results.csv',
        'evaluation_report.json':evaluation/'report.json','evaluation_inputs.json':evaluation/'evaluation_inputs.json'}
    for name,p in files.items():
        if not p.exists(): raise FileNotFoundError(p)
        shutil.copy2(p,packet/name)
    result={'design_version':DESIGN_VERSION,'qwen_complete':True,'counts':dict(counts),
        'duplicates':0,'hash_mismatches':0,'independent_metric_cells':checked,'api_calls':0,
        'all_providers_complete':False,'not_run':['sol','deepseek'],
        'inputs':{str(p.relative_to(ROOT)):sha(p) for p in (qwen/'integration_manifest.json',qwen/'responses.jsonl',
            run/'requests.jsonl',rev/'primary_baselines.json',ROOT/'scripts/audit_srw_results.py')},
        'source_archives':{k:{'checksums_sha256':v['checksums_sha256'],'files_verified':v['files_verified']} for k,v in manifest['archives'].items()}}
    write_json(packet/'qwen_audit.json',result)
    write_json(packet/'CHECKSUMS.json',{p.name:sha(p) for p in sorted(packet.iterdir()) if p.is_file() and p.name!='CHECKSUMS.json'})
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
