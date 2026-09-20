#!/usr/bin/env python3
"""Read-only checks of the finished JSON Qwen run; export compact evidence.

Does not generate answers, fit models, or change evaluation rules. Requires the
cluster archive plus bundle_provenance, and an evaluation by the current code.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from main_experiment.common import sha, digest, read_json, write_json, DESIGN_VERSION
from main_experiment.evaluation import parse_final
from run_qwen_engine import GENERATION_CONFIG


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines()]


def export_csv(path, data):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(data)


def audit(run, archive, baselines, out):
    checksums = read_json(archive / 'CHECKSUMS.json')
    for name, expected in checksums.items():
        assert sha(archive / name) == expected, name
    assert sha(run / 'requests.jsonl') == sha(archive / 'requests.jsonl')
    for path in (archive / 'observations/sample').glob('*.json'):
        assert sha(path) == sha(run / 'observations/sample' / path.name), path.name
    runner_hash = sha(ROOT / 'scripts/run_qwen_engine.py')
    assert sha(archive / 'run_qwen_engine.py') == runner_hash
    bundle = archive / 'bundle_provenance'
    source_hashes = {}
    bundle_count = 0
    for line in (bundle / 'BUNDLE_SHA256SUMS').read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.removeprefix('./')
        if name.startswith('mainexp/run/'):
            path = run / name.removeprefix('mainexp/run/')
        elif name.startswith('mainexp/'):
            basename = name.removeprefix('mainexp/')
            path = bundle / basename if basename in ('status.py', 'build_archive.py') else archive / basename
        else:
            path = bundle / name
        assert sha(path) == expected, name
        if name.startswith('src/') or name == 'mainexp/run_qwen_engine.py':
            local = name if name.startswith('src/') else 'scripts/run_qwen_engine.py'
            frozen = subprocess.check_output(['git', 'show', f'778d043:{local}'], cwd=ROOT)
            assert hashlib.sha256(frozen).hexdigest() == expected, local
            assert sha(ROOT / local) == expected, local
            source_hashes[local] = expected
        bundle_count += 1
    binding = read_json(archive / 'answers/engine_inputs.json')
    assert binding['runner_sha256'] == runner_hash
    assert binding['requests_sha256'] == sha(run / 'requests.jsonl')
    assert binding['generation'] == GENERATION_CONFIG
    identity = read_json(archive / 'model_identity.json')
    for name, expected in binding['model_files'].items():
        recorded = identity.get(name, identity['safetensors'].get(name))
        if recorded is not None:
            assert recorded == expected, name
        if name.endswith('.json'):
            assert sha(archive / 'model_metadata' / name) == expected, name
    assert sha(archive / 'model_metadata/chat_template.jinja') == identity['chat_template.jinja']
    rendered = {(r['observation_id'], r['mode']): r for r in rows(archive / 'rendered_prompts.jsonl')}
    assert len(rendered) == 560
    for r in rendered.values():
        assert hashlib.sha256(r['rendered'].encode()).hexdigest() == r['rendered_sha256']
    planned = {r['id']: r for r in rows(run / 'requests.jsonl') if r['config_id'].startswith('qwen')}
    files = sorted((archive / 'answers').glob('*_r*/*.json'))
    answers = [read_json(p) for p in files]
    assert len({r['id'] for r in answers}) == len(answers), 'duplicate answers'
    assert {r['id'] for r in answers} == set(planned), 'missing or extra answers'
    attempts = list((archive / 'answers').glob('*_r*/*.attempt'))
    assert len(attempts) == len(planned)
    evidence = []
    for path, answer in zip(files, answers):
        r = planned[answer['id']]
        for key in ('id', 'prompt_sha256', 'payload_sha256', 'seed', 'config_id', 'repeat_index'):
            assert answer[key] == r[key], (r['id'], key)
        assert digest(r['payload']) == r['payload_sha256']
        assert answer['runner_sha256'] == runner_hash
        assert answer['design_version'] == DESIGN_VERSION
        assert answer['structured_output'] == GENERATION_CONFIG['structured_output']
        assert answer['mode'] == r['config_id'].removeprefix('qwen_')
        assert path.parent.name == f"{answer['mode']}_r{r['repeat_index']}"
        attempt = read_json(path.with_suffix('.attempt'))
        assert attempt['id'] == r['id'] and attempt['payload_sha256'] == r['payload_sha256']
        prompt = rendered[r['observation_id'], answer['mode']]
        assert prompt['prompt_sha256'] == r['prompt_sha256']
        assert prompt['input_tokens'] == answer['input_tokens'] == answer['engine_prompt_tokens'] == attempt['input_tokens']
        prediction, reason = parse_final(answer.get('final_text', ''))
        valid = prediction is not None and answer['terminal'] and not answer.get('technical_error', False)
        evidence.append(dict(id=r['id'], config_id=r['config_id'], answer_sha256=sha(path),
                             attempt_sha256=sha(path.with_suffix('.attempt')), valid=valid,
                             validation_reason=reason, end_state=answer['end_state'],
                             input_tokens=answer['input_tokens'], output_tokens=answer['output_tokens']))
    evaluation = archive / 'evaluation'
    inputs = read_json(evaluation / 'evaluation_inputs.json')
    assert inputs['requests_sha256'] == sha(run / 'requests.jsonl')
    assert inputs['responses_sha256'] == sha(archive / 'responses.jsonl')
    assert inputs['baselines_sha256'] == sha(baselines)
    assert inputs['evaluator_sha256'] == sha(ROOT / 'scripts/evaluate_main_responses.py')
    assert inputs['parser_sha256'] == sha(ROOT / 'src/main_experiment/evaluation.py')
    tables = {}
    for name in ('summary', 'source_results', 'answer_errors'):
        with (evaluation / f'{name}.csv').open() as f:
            tables[name] = [r for r in csv.DictReader(f) if r['config_id'].startswith('qwen')]
    assert {r['id'] for r in tables['answer_errors']} == set(planned)
    assert all((r['valid'] == 'True') == e['valid'] for r, e in
               zip(sorted(tables['answer_errors'], key=lambda r: r['id']), sorted(evidence, key=lambda r: r['id'])))
    # Independently recompute each conditional panel mean from raw final JSON.
    errors = defaultdict(lambda: defaultdict(list))
    for answer in answers:
        r = planned[answer['id']]
        pred, _ = parse_final(answer['final_text'])
        obs = read_json(run / 'observations/sample' / (r['observation_id'] + '.json'))
        ae = [abs(x - y) for x, y in zip(pred, obs['truth'])]
        stratum = 'real' if r['stratum'] == 'real' else r['graph_id'].rsplit('_r', 1)[0]
        errors[stratum, r['arm'], r['config_id']][r['graph_id']].append((ae[0], statistics.mean(ae)))
    for r in tables['summary']:
        assert r['complete'] == 'True' and float(r['valid_fraction']) == 1
        sources = errors[r['stratum'], r['arm'], r['config_id']]
        for i, metric in enumerate(('AE2', 'ProfileAE')):
            value = statistics.mean(statistics.mean(x[i] for x in values) for values in sources.values())
            assert abs(value - float(r[metric])) < 1e-12
    per_mode = {}
    for config in sorted({r['config_id'] for r in answers}):
        group = [r for r in answers if r['config_id'] == config]
        valid = sum(e['valid'] for e in evidence if e['config_id'] == config)
        per_mode[config] = dict(planned=sum(r['config_id'] == config for r in planned.values()),
            found=len(group), valid=valid, invalid=len(group)-valid, missing=0,
            technical_errors=sum(bool(r.get('technical_error')) or r['status'] != 'completed' for r in group),
            token_limit_hits=sum(r['end_state'] == 'output_limit' for r in group),
            unclosed_reasoning=sum(not r['reasoning_closed'] for r in group),
            output_tokens_sum=sum(r['output_tokens'] for r in group),
            output_tokens_median=statistics.median(r['output_tokens'] for r in group),
            output_tokens_max=max(r['output_tokens'] for r in group))
    result = dict(design_version=DESIGN_VERSION, pre_result_freeze='778d043',
        bundle_spec_commit=(archive / 'SPEC_COMMIT').read_text().strip(),
        bundle_dirty=True, bundle_files_verified=bundle_count, archive_files_verified=len(checksums),
        archive_answer_json_files=len(list((archive / 'answers').rglob('*.json'))),
        archive_extra_json='answers/engine_inputs.json (configuration binding, not an answer)',
        source_hashes_matching_freeze_and_current=source_hashes,
        model_identity=identity, evaluation_inputs=inputs, per_mode=per_mode,
        planned=len(planned), found=len(answers), valid=sum(e['valid'] for e in evidence),
        invalid=sum(not e['valid'] for e in evidence), missing=0, duplicates=0, unexpected=0,
        attempts=len(attempts), rendered_prompts=len(rendered),
        end_states=dict(Counter(r['end_state'] for r in answers)),
        per_pass=dict(Counter(p.parent.name for p in files)),
        per_arm=dict(Counter(r['arm'] for r in planned.values())),
        independently_recomputed_accuracy_cells=len(tables['summary']),
        complete_qwen=True, complete_four_configuration_main=False,
        api_requests_started=0,
        artifact_hashes={name: sha(archive / name) for name in
            ('CHECKSUMS.json', 'ARCHIVE_REPORT.json', 'status_final.json', 'responses.jsonl',
             'answers/engine_inputs.json', 'bundle_provenance/BUNDLE_SHA256SUMS',
             'bundle_provenance/WORKTREE_STATUS')}, audit_script_sha256=sha(__file__))
    ledger = read_json(run.with_name(run.name + '_api') / 'ledger.json')
    assert ledger['requests'] == {} and ledger['batches'] == {}
    with (archive / 'slurm_final.psv').open() as f:
        jobs = [r for r in csv.DictReader(f, delimiter='|') if '.' not in r['JobID']]
    expected_jobs = {'7035304'} | {f'{job}_{i}' for job in ('7035301', '7035302', '7035303') for i in range(6)}
    assert {r['JobID'] for r in jobs} == expected_jobs
    assert all(r['State'] == 'COMPLETED' and r['ExitCode'] == '0:0' for r in jobs)
    result['slurm_jobs'] = jobs
    result['slurm_sha256'] = sha(archive / 'slurm_final.psv')
    result['generation_completed_in_round1'] = True
    for job in ('7035302', '7035303'):
        logs = list((archive / 'logs').glob(f'*_{job}_*.out'))
        assert len(logs) == 6
        assert all('280 already done, 0 to run' in p.read_text() for p in logs)
    logs = ROOT / 'results/json_revision_20260918_logs'
    test_log = (logs / 'qwen_final_tests.log').read_text()
    assert '\nOK\n' in test_log and 'Ran 125 tests' in test_log
    offline = json.loads((logs / 'qwen_final_offline.log').read_text())
    protocol = read_json(logs / 'qwen_final_protocol.json')
    mock = read_json(run.with_name(run.name + '_qwen_final_mock') / 'check_report.json')
    assert offline['verified'] and protocol['verified'] and mock['inference_calls'] == 0
    result['verification'] = dict(unit_tests=125, failures=0, skips=0,
                                  offline=offline, protocol=protocol, mock=mock)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'tests.txt').write_text(test_log)
    write_json(out / 'audit.json', result)
    export_csv(out / 'answer_manifest.csv', evidence)
    for name in ('summary', 'source_results'):
        export_csv(out / f'{name}.csv', tables[name])
    write_json(out / 'checksums.json', {p.name: sha(p) for p in sorted(out.iterdir())
                                       if p.is_file() and p.name != 'checksums.json'})
    print(json.dumps({k: result[k] for k in ('planned', 'found', 'valid', 'invalid', 'missing', 'per_mode')}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--baselines', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    audit(a.run, a.archive, a.baselines, a.out)
