#!/usr/bin/env python3
"""Offline freeze: compact committed evidence plus hashes of every artifact and source.

  seal_offline.py           after all offline stages, audits and tests have passed:
                            copies the compact evidence to docs/results/panel888_offline,
                            writes ARTIFACT_CHECKSUMS.json, SOURCE_CHECKSUMS.json, FREEZE.json.
  seal_offline.py --verify  before bundling for the cluster: every sealed source must
                            equal the working tree AND the committed HEAD, and every
                            sealed artifact and evidence file must be unchanged.
"""
import argparse
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (AUDIT, DESIGN_VERSION, DIAGNOSTICS, PREPARED, REFERENCES, RESULTS, ROOT,
                                    read_json, sha, write_json)

EVIDENCE = ROOT/'docs/results/panel888_offline'
LOGS = RESULTS/'logs'
COMPACT = {
    'prepared_report.json': PREPARED/'report.json', 'budget_summary.csv': PREPARED/'budget_summary.csv',
    'data_summary.csv': PREPARED/'data_summary.csv', 'prompt_sizes.csv': PREPARED/'prompt_sizes.csv',
    'offline_audit.json': AUDIT/'offline_audit.json', 'mock_check.json': AUDIT/'mock_evaluation/check_report.json',
    'development_summary.json': REFERENCES/'development_summary.json',
    'main_baseline_summary.csv': REFERENCES/'main_summary.csv', 'main_coverage.csv': REFERENCES/'main_coverage.csv',
    'paired_baseline_summary.csv': RESULTS/'paired_control_references/paired_summary.csv',
    'synthetic_baseline_contrasts.csv': RESULTS/'paired_control_references/synthetic_within_replicate_contrasts.csv',
    'decomposition.json': DIAGNOSTICS/'decomposition/report.json',
    'history_summary.csv': DIAGNOSTICS/'history/summary.csv', 'history_sources.csv': DIAGNOSTICS/'history/sources.csv',
    'srw_summary.csv': DIAGNOSTICS/'srw/summary.csv', 'srw_components.csv': DIAGNOSTICS/'srw/components.csv',
    'null_model.json': DIAGNOSTICS/'null_model/report.json',
    'window_sensitivity.csv': DIAGNOSTICS/'windows/window_sensitivity.csv',
    'W_4_5_8_comparisons.csv': DIAGNOSTICS/'windows/W_4_5_8_comparisons.csv',
    'census_sensitivities.csv': DIAGNOSTICS/'windows/census_sensitivities.csv',
    'count_feasibility.csv': DIAGNOSTICS/'windows/count_feasibility.csv',
    'mixture_bound_sensitivity.csv': DIAGNOSTICS/'mixture_bounds/mixture_bound_sensitivity.csv',
    'tests.txt': LOGS/'tests.log'}


def source_files():
    patterns = ['src/main_experiment/*.py', 'src/main_experiment/*.cpp', 'src/census.py', 'src/dataset_census.py',
                'config/main_experiment/*.txt', 'config/*.yaml', 'scripts/*.py', 'scripts/*.sh', 'cluster/*.py',
                'cluster/*.sh', 'cluster/*.sbatch', 'docs/PROTOCOL_PANEL888_20260921.md']
    return sorted(p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file())


def artifact_files():
    """Every offline artifact except the compiled kernel and logs."""
    return sorted(p for p in RESULTS.rglob('*') if p.is_file()
                  and not {'build', 'logs', 'qwen'} & set(p.relative_to(RESULTS).parts[:1]))


def seal():
    assert read_json(AUDIT/'offline_audit.json')['verified']
    assert read_json(AUDIT/'mock_evaluation/check_report.json')['mock_only']
    ledger = read_json(RESULTS/'api/ledger.json')
    assert ledger['requests'] == {} and ledger['batches'] == {}             # Sol/DeepSeek not started
    tests = (LOGS/'tests.log').read_text()
    assert '\nOK' in tests and 'FAILED' not in tests
    if EVIDENCE.exists(): shutil.rmtree(EVIDENCE)
    EVIDENCE.mkdir(parents=True)
    for name, source in COMPACT.items(): shutil.copy2(source, EVIDENCE/name)
    write_json(EVIDENCE/'ARTIFACT_CHECKSUMS.json', {str(p.relative_to(ROOT)): sha(p) for p in artifact_files()})
    write_json(EVIDENCE/'SOURCE_CHECKSUMS.json', {str(p.relative_to(ROOT)): sha(p) for p in source_files()})
    write_json(EVIDENCE/'FREEZE.json', {
        'design_version': DESIGN_VERSION, 'offline_ready': True, 'production_submitted': False,
        'evidence_sha256': {p.name: sha(p) for p in sorted(EVIDENCE.iterdir()) if p.name != 'FREEZE.json'},
        'sol_deepseek_started': False, 'raw_data': 'data/raw; hashes in graph manifests; not duplicated',
        'inference': 'Qwen only, one new production chain after this freeze', 'not_retroactive_preregistration': True})
    print('OFFLINE_FREEZE_SEALED', len(read_json(EVIDENCE/'ARTIFACT_CHECKSUMS.json')), 'artifacts')


def verify():
    freeze = read_json(EVIDENCE/'FREEZE.json')
    for name, h in freeze['evidence_sha256'].items(): assert sha(EVIDENCE/name) == h, name
    for name, h in read_json(EVIDENCE/'SOURCE_CHECKSUMS.json').items():
        assert sha(ROOT/name) == h, ('source changed after seal', name)
        committed = subprocess.check_output(['git', 'show', 'HEAD:'+name], cwd=ROOT)
        assert hashlib.sha256(committed).hexdigest() == h, ('source not committed', name)
    current = {str(p.relative_to(ROOT)) for p in source_files()}
    assert current == set(read_json(EVIDENCE/'SOURCE_CHECKSUMS.json')), 'source file set changed after seal'
    for name, h in read_json(EVIDENCE/'ARTIFACT_CHECKSUMS.json').items(): assert sha(ROOT/name) == h, name
    print('SEAL_VERIFIED', subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    if parser.parse_args().verify:
        verify()
    else:
        seal()
