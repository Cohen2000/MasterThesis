#!/usr/bin/env python3
"""Copy the compact Qwen result evidence into docs/results/panel888_qwen and hash it.

Separate from the offline freeze, which stays unchanged.
"""
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import AUDIT, DESIGN_VERSION, QWEN, RESULTS, ROOT, read_json, read_jsonl, sha, write_json

OUT = ROOT/'docs/results/panel888_qwen'
EVIDENCE = {'summary.csv': QWEN/'evaluation/summary.csv', 'source_results.csv': QWEN/'evaluation/source_results.csv',
            'evaluation_report.json': QWEN/'evaluation/report.json',
            'paired_summary.csv': QWEN/'paired_control/paired_summary.csv',
            'paired_sources.csv': QWEN/'paired_control/paired_sources.csv',
            'synthetic_within_replicate_contrasts.csv': QWEN/'paired_control/synthetic_within_replicate_contrasts.csv',
            'collection_report.json': QWEN/'collection_report.json', 'archive_verification.json': RESULTS/'qwen_archive_verification.json',
            'qwen_audit.json': AUDIT/'qwen_audit.json'}


def main():
    audit = read_json(AUDIT/'qwen_audit.json')
    assert audit['verified']
    OUT.mkdir(parents=True, exist_ok=True)
    for name, source in EVIDENCE.items(): shutil.copy2(source, OUT/name)
    responses = read_jsonl(QWEN/'responses.jsonl')
    write_json(OUT/'RESULT_FREEZE.json', {
        'design_version': DESIGN_VERSION, 'qwen_audit': audit,
        'technical_errors': sum(bool(r.get('technical_error')) for r in responses),
        'output_limit_hits': sum(bool(r.get('limit_hit')) for r in responses),
        'offline_freeze_sha256': sha(ROOT/'docs/results/panel888_offline/FREEZE.json'),
        'evidence_sha256': {n: sha(OUT/n) for n in EVIDENCE},
        'artifact_checksums': {str(p.relative_to(ROOT)): sha(p) for p in sorted(QWEN.rglob('*')) if p.is_file()},
        'sol_deepseek_started': False})
    print(json.dumps({k: v for k, v in read_json(OUT/'RESULT_FREEZE.json').items() if k != 'artifact_checksums'}, indent=1))


if __name__ == '__main__':
    main()
