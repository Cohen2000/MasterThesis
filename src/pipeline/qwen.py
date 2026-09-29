"""Qwen requests for the stage-2 observations and strict evaluation of the answers.

In plain words: builds the list of Qwen prompts for the stage-2 graphs, installs it for the
GPU jobs on the cluster and reads the answers back. Requests follow the stage-1 protocol: requests.planned for thinking and
non-thinking with three repeats, the frozen payloads and, for released R/H
observations, the generation seed keyed by the hidden sampler identity (as in
scripts/add_panel_size.prepare). The bundle is installed once into
PIPELINE_DIR/qwen and is never replaced; the unchanged runner skips every answered request.
"""
import json
import shutil
from study.common import ROOT, digest, read_json, read_jsonl, seed, sha, write_json
from study.answer_format import parse_final
from study.model_requests import payload, planned, protocol_version, validate_request
from .core import PIPELINE_DIR, STAGE2_SOURCES, PANEL_ARMS

QWEN_DIR = PIPELINE_DIR/'qwen'          # experiment directory 'pipeline/qwen' for cluster/qwen_engine.sbatch
MODES = ('qwen_thinking', 'qwen_nonthinking')


# Qwen requests (thinking and non-thinking, three repeats) for a list of stored observations.
def requests_for(rows):
    by_id = {r['id']: r for r in rows}
    requests = [r for r in planned(rows) if r['config_id'] in MODES]
    for r in requests:
        obs = by_id[r['observation_id']]
        if obs['arm'] in PANEL_ARMS:
            version = protocol_version(r['arm'])
            hidden_sampler = obs['paired_hidden_id'].rsplit('__', 2)[1]
            r['seed'] = seed('llm', r['graph_id'], hidden_sampler, r['sample_index'], r['repeat_index'],
                             r['config_id']+':'+version)
            r['payload'] = payload(r['config_id'], obs['messages'], r['seed'], version)
            r['payload_sha256'] = digest(r['payload'])
        validate_request(r)
    if len(requests) != len(rows)*len(MODES)*3 or len({r['id'] for r in requests}) != len(requests):
        raise ValueError('incomplete Qwen request grid')
    if len({r['seed'] for r in requests}) != len(requests): raise ValueError('Qwen seed collision')
    return requests


# Stage 'qwen_bundle': the self-contained folder the GPU jobs run from (requests, runner, config).
def bundle(task, out, inputs):
    rows = [read_json(p) for source in STAGE2_SOURCES
            for p in sorted((inputs[f'source:{source}']/'observations').glob('*.json'))]
    requests = requests_for(rows)
    run = out/'run'
    for r in rows: write_json(run/'observations/sample'/f'{r["id"]}.json', r)
    (run/'requests.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in requests))
    install(out)
    write_json(out/'bundle.json', {'observations': len(rows), 'requests': len(requests),
                                   'requests_sha256': sha(run/'requests.jsonl'), 'installed': str(QWEN_DIR)})
    print('QWEN_BUNDLE', len(rows), 'observations', len(requests), 'requests', flush=True)


def install(out, dest=None):
    """Copy the bundle to dest (default QWEN_DIR) once; an existing bundle must be identical."""
    dest = dest or QWEN_DIR
    new = sha(out/'run/requests.jsonl')
    existing = dest/'mainexp/run/requests.jsonl'
    if existing.exists():
        if sha(existing) != new: raise RuntimeError('an installed Qwen bundle with different requests exists; not replacing it')
        return
    # No requests file means no generation can have started: a partial install is completed in place.
    for folder in ('src/study', 'config/prompts'):
        shutil.copytree(ROOT/folder, dest/folder, ignore=shutil.ignore_patterns('__pycache__'), dirs_exist_ok=True)
    for name in ('study.yaml', 'datasets.yaml'): shutil.copy2(ROOT/'config'/name, dest/'config'/name)
    main = dest/'mainexp'
    (main/'logs').mkdir(parents=True, exist_ok=True)
    if (main/'answers').exists() and any((main/'answers').iterdir()):
        raise RuntimeError('answers exist without an installed request manifest')
    shutil.copy2(ROOT/'scripts/run_qwen_engine.py', main/'run_qwen_engine.py')
    shutil.copy2(ROOT/'cluster/qwen_engine.sbatch', main/'qwen_engine.sbatch')
    if (main/'run').exists(): shutil.rmtree(main/'run')
    shutil.copytree(out/'run', main/'run')


def answers(folder=QWEN_DIR):
    """Strictly evaluated answers; fails on unknown, duplicate or identity-mismatched files."""
    requests = {r['id']: r for r in read_jsonl(folder/'mainexp/run/requests.jsonl')}
    found = {}
    for path in (folder/'mainexp/answers').glob('*_r*/*.json'):
        a = read_json(path)
        if a['id'] not in requests or a['id'] in found: raise ValueError(f'unknown/duplicate Qwen answer {a["id"]}')
        if any(a.get(k) != requests[a['id']][k] for k in ('seed', 'prompt_sha256', 'payload_sha256')):
            raise ValueError(f'Qwen answer identity mismatch {a["id"]}')
        found[a['id']] = a
    rows = []
    for rid, req in sorted(requests.items()):
        a = found.get(rid)
        if a is None:
            rows.append({'id': rid, 'observation_id': req['observation_id'], 'method': req['config_id'],
                         'repeat_index': req['repeat_index'], 'prediction': None, 'valid': False,
                         'status': 'missing', 'validation_reason': 'missing'})
            continue
        values, reason = parse_final(a.get('final_text', ''))
        if a.get('status') != 'completed' or a.get('technical_error'):
            values, reason = None, a.get('end_state', 'technical_error')
        rows.append({'id': rid, 'observation_id': req['observation_id'], 'method': req['config_id'],
                     'repeat_index': req['repeat_index'], 'prediction': values, 'valid': values is not None,
                     'status': a.get('status'), 'validation_reason': reason, 'end_state': a.get('end_state'),
                     'output_tokens': a.get('output_tokens'), 'seconds': a.get('seconds')})
    return rows
