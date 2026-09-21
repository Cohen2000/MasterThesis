#!/usr/bin/env python3
"""Publish compact evidence and immutable hashes only after all offline checks."""
import json,sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import *
run=ROOT/CURRENT_RUN;rev=ROOT/CURRENT_REVISION;out=ROOT/'docs/results/panel888_20260921';out.mkdir(parents=True,exist_ok=True)
assert read_json(rev/'independent_audit.json')['verified']
assert read_json(run/'prompt_freeze_audit.json')['audit_passed']
assert read_json(run.with_name(run.name+'_mock')/'check_report.json')['mock_only']
ledger=read_json(run.with_name(run.name+'_api')/'ledger.json');assert ledger['requests']=={} and ledger['batches']=={}
tests=(ROOT/'results/panel888_logs/tests.log').read_text();assert '\nOK' in tests and 'FAILED (' not in tests
verify=json.loads((ROOT/'results/panel888_logs/verify.log').read_text());assert verify['verified']
artifacts={'census_sensitivities.csv':rev/'census_diagnostics/census_sensitivities.csv','count_feasibility.csv':rev/'census_diagnostics/count_feasibility.csv','seed_audit.json':rev/'seed_audit.json','offline_report.json':run/'report.json','budget_summary.csv':run/'budget_summary.csv','data_summary.csv':run/'data_summary.csv',
    'prompt_freeze_audit.json':run/'prompt_freeze_audit.json','prompt_sizes.csv':run/'prompt_sizes.csv','independent_audit.json':rev/'independent_audit.json',
    'main_baseline_summary.csv':rev/'main_summary.csv','development_summary.json':rev/'development_summary.json',
    'history_summary.csv':rev/'history_sensitivity/summary.csv','history_sources.csv':rev/'history_sensitivity/sources.csv',
    'srw_report.json':rev/'srw_diagnostics/report.json','null_report.json':rev/'control_diagnostics/report.json',
    'window_sensitivity.csv':rev/'control_diagnostics/window_sensitivity.csv','W_4_5_8_comparisons.csv':rev/'control_diagnostics/W_4_5_8_comparisons.csv',
    'mixture_bound_sensitivity.csv':rev/'control_diagnostics/mixture_bound_sensitivity.csv',
    'paired_baseline_summary.csv':rev/'paired_control/paired_summary.csv','synthetic_baseline_contrasts.csv':rev/'paired_control/synthetic_within_replicate_contrasts.csv',
    'decomposition_main.json':rev/'error_decomposition_main.json','decomposition_dev.json':rev/'error_decomposition.json',
    'verification.json':ROOT/'results/panel888_logs/verify.log','tests.txt':ROOT/'results/panel888_logs/tests.log'}
for name,source in artifacts.items():
    dest=out/name
    if dest.exists() and sha(dest)!=sha(source):raise ValueError(f'sealed evidence changed: {dest}')
    shutil.copy2(source,dest)
files={}
for base in (run,rev,run.with_name(run.name+'_api')):
    for p in sorted(base.rglob('*')):
        if p.is_file() and 'build' not in p.parts and not p.name.endswith(('.lock','.tmp')):
            files[str(p.relative_to(ROOT))]=sha(p)
write_json(out/'ARTIFACT_CHECKSUMS.json',files)
source_files=[*ROOT.glob('src/main_experiment/*.py'),*ROOT.glob('src/main_experiment/*.cpp'),*ROOT.glob('config/main_experiment/*.txt'),
    ROOT/'config/study.yaml',ROOT/'config/datasets.yaml',*ROOT.glob('scripts/*.py'),*ROOT.glob('scripts/*panel888*.sh'),*ROOT.glob('cluster/*')]
source_files=[p for p in source_files if p.is_file()]
write_json(out/'SOURCE_CHECKSUMS.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(source_files)})
write_json(out/'FREEZE.json',{'design_version':DESIGN_VERSION,'offline_ready':True,'production_submitted':False,
    'evidence_sha256':{str(p.relative_to(out)):sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='FREEZE.json'},
    'sol_deepseek_started':False,'raw_data_location':'data/raw; hashes in graph raw bindings; not duplicated',
    'inference':'Qwen only, pending one new production chain','not_retroactive_preregistration':True})
print('OFFLINE_FREEZE_READY',len(files),'artifact hashes')
