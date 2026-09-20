#!/usr/bin/env python3
"""Independently recompute source-equal errors/cluster MCSE and export evidence."""
import csv,json,sys,shutil,math
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,read_json,write_json,sha
from audit_srw_results import strict

def main():
    run=ROOT/CURRENT_RUN; q=run.with_name(run.name+'_qwen'); ev=q/'evaluation'; rev=ROOT/CURRENT_REVISION
    packet=ROOT/'docs/results/final_20260920'; packet.mkdir(parents=True,exist_ok=True)
    audit=read_json(q/'provenance_audit.json'); baselines=read_json(q/'primary_baselines.json')
    ledger=read_json(run.with_name(run.name+'_api')/'ledger.json'); assert ledger['requests']=={} and ledger['batches']=={}
    planned={r['id']:r for r in map(json.loads,(run/'requests.jsonl').read_text().splitlines())}
    responses={r['id']:r for r in map(json.loads,(q/'responses.jsonl').read_text().splitlines())}
    errors={r['id']:r for r in csv.DictReader((ev/'answer_errors.csv').open())}
    groups=defaultdict(lambda:defaultdict(lambda:np.full((5,3),np.nan)))
    for rid,r in responses.items():
        req=planned[rid]; o=read_json(run/'observations/sample'/f"{req['observation_id']}.json")
        pred=strict(r['final_text']); assert (pred is not None)==(errors[rid]['valid']=='True')
        s='real' if req['stratum']=='real' else req['graph_id'].rsplit('_r',1)[0]
        if pred is not None:
            diff=np.abs(np.asarray(pred)-o['truth'])
            for metric,value in [('AE2',diff[0]),('ProfileAE',diff.mean())]:
                assert abs(float(errors[rid][metric])-value)<1e-14
                groups[s,req['arm'],req['config_id'],metric][req['graph_id']][req['sample_index']-1,req['repeat_index']-1]=value
    rows=list(csv.DictReader((ev/'summary.csv').open())); checked=0
    for row in rows:
        if not row['config_id'].startswith('qwen_'): continue
        assert row['complete']=='True'
        for metric in ('AE2','ProfileAE'):
            data=groups[row['stratum'],row['arm'],row['config_id'],metric]
            assert len(data)==(6 if row['stratum']=='real' else 2)
            means=[]; variances=[]
            for x in data.values():
                n=np.isfinite(x).sum(); mu=np.nansum(x)/n; counts=np.isfinite(x).sum(1)
                residual=np.nansum(x,axis=1)-mu*counts
                means.append(mu); variances.append(5/4*np.sum(residual**2)/n**2)
            assert abs(np.mean(means)-float(row[metric]))<1e-14
            assert abs(math.sqrt(sum(variances))/len(variances)-float(row[metric+'_MCSE']))<1e-14
        checked+=1
    assert checked==40
    compact={k:v for k,v in audit.items() if k not in ('records','engine_inputs','model_identity')}
    compact.update(independent_error_and_MCSE_cells=checked,api_calls=0,
        audit_script_sha256=sha(ROOT/'scripts/audit_final_study.py'),
        report_script_sha256=sha(__file__),
        evaluator_sha256=sha(ROOT/'scripts/evaluate_main_responses.py'),
        full_provenance_audit_sha256=sha(q/'provenance_audit.json'),
        limitations=['fixed sources/training/calibration; conditional accuracy, no imputation',
            'protocol preparation hash predates final appendix; executable/prompt hashes unchanged from freeze',
            'SRW finite mixing/component bias; H homogeneous/stationary working-model assumptions',
            'Sol/DeepSeek require separate explicit technical release before dispatch'])
    write_json(packet/'audit.json',compact)
    files={'summary.csv':ev/'summary.csv','source_results.csv':ev/'source_results.csv',
        'evaluation_inputs.json':ev/'evaluation_inputs.json','evaluation_report.json':ev/'report.json',
        'prompt_audit.json':run/'prompt_freeze_audit.json','tests.txt':Path('/tmp/final_study_tests.log'),
        'offline_verification.json':Path('/tmp/final_offline_verify.json'),'slurm.psv':Path('/tmp/final_slurm.psv'),
        'cluster_status.json':ROOT/'results/imported_final_20260920/status_final.json',
        'archive_report.json':ROOT/'results/imported_final_20260920/ARCHIVE_REPORT.json',
        'mock_check.json':Path('/tmp/final_mock.log'),
        'bundle_SHA256SUMS':ROOT/'results/final_bundle_provenance/BUNDLE_SHA256SUMS',
        'bundle_WORKTREE_STATUS':ROOT/'results/final_bundle_provenance/WORKTREE_STATUS',
        'bundle_SPEC_COMMIT':ROOT/'results/final_bundle_provenance/SPEC_COMMIT',
        'h_sensitivity.json':rev/'history_sensitivity/report.json','srw_diagnostics.json':rev/'srw_diagnostics/report.json'}
    for name,p in files.items():
        if p.suffix=='.csv': (packet/name).write_text(p.read_text())
        else: shutil.copy2(p,packet/name)
    text=['# Final Qwen results: cells10-final-20260920','',
        'Frozen evaluator; conditional valid-answer MAE2 / ProfileMAE. Equal source weights.',
        'All values below are errors (lower is better). Full MCSE and source rows: [evidence](results/final_20260920/summary.csv).',
        '','| Stratum | Arm | Thinking | Nonthinking | Statistical | Plugin | Median | ExtraTrees |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for s in ('real','dar_a0','dar_a08','ad_memoryless','ad_memory'):
        for arm in 'RSHB':
            t=next(r for r in rows if (r['stratum'],r['arm'],r['config_id'])==(s,arm,'qwen_thinking'))
            n=next(r for r in rows if (r['stratum'],r['arm'],r['config_id'])==(s,arm,'qwen_nonthinking'))
            fmt=lambda r,a,b:f'{float(r[a]):.4f} / {float(r[b]):.4f}'
            cells=[fmt(t,'AE2','ProfileAE'),fmt(n,'AE2','ProfileAE')]+[fmt(t,p+'_AE2_all',p+'_ProfileAE_all') for p in ('baseline','plugin','median','extratrees')]
            text.append('| '+' | '.join([s,arm]+cells)+' |')
    text+=['','References are evaluated on all observations; paired valid-subset comparisons are in the CSV.',
        'Two invalid Thinking answers occur only in synthetic cells; no imputation or trailing-JSON extraction.',
        'All-provider complete_main_result remains false because Sol/DeepSeek are intentionally unstarted.','']
    (ROOT/'docs/RESULTS_FINAL_20260920.md').write_text('\n'.join(text))
    write_json(packet/'CHECKSUMS.json',{p.name:sha(p) for p in packet.iterdir() if p.name!='CHECKSUMS.json'})
    print(json.dumps({'verified_metric_cells':checked,'counts':audit['counts']},indent=2))

if __name__=='__main__': main()
