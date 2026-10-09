"""The cluster task that collected every prediction into PREDICTIONS.csv.

In plain words: report() is the last task of the cluster pipeline. It collects every prediction
(offline methods, ExtraTrees fits, Qwen, API answers) into one table, checks that no ExtraTrees
fit saw the network it predicts, and writes the tables. The table functions themselves are in
src/study/tables.py; scripts/evaluate.py rebuilds the published tables from the frozen
PREDICTIONS.csv. S_obs is drawn for reproducibility only and is not reported.
"""
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
import numpy as np
from study.common import ROOT, read_json, write_csv, write_json
from study.answer_format import valid_profile
from study.tables import (QWEN, GROUPS, REPORTED_ARMS, errors, family, fixed_input_variability, history_markdown,
                          history_summary, main_markdown, paired_families, paired_methods, per_source, s_design_appendix,
                          summary, variability, variability_markdown, walk_markdown)
from .core import CFG, MLE_ANCHOR_ARMS, STAGE2_SOURCES, PANEL_RUN, STAGE1, WORK
from .surrogates_and_checks import QWEN_SUR_DIR, SURROGATES
from .qwen import QWEN_DIR, answers as qwen_answers

# Inputs of the report task: the stage-1 tables (score_stage1.py --out) and the scored API
# answers (API_PREDICTIONS.csv, one row per answer; scripts/evaluate.py derives the same rows today).
STAGE1_TABLES = ROOT/'results/stage1_tables'
API = ROOT/'results/api_evaluation'


# ---------------------------------------------------------------- collection
# Every test observation of the panel (32 graphs x 4 arms x 3 draws = 384).
def observations(inputs):
    obs = {}
    for folder in (STAGE1/'prepared/observations/sample', PANEL_RUN/'mainexp/run/observations/sample'):
        for p in folder.glob('*.json'):
            r = read_json(p); obs[r['id']] = r
    for parent in STAGE2_SOURCES:
        for name in (f'source:{parent}', f'surrogate:{parent}'):
            for p in (inputs[name]/'observations').glob('*.json'):
                r = read_json(p); obs[r['id']] = r
    return {i: {'id': i, 'source': r['graph_id'], 'group': r['stratum'], 'arm': r['arm'],
                'sample_index': r['sample_index'], 'truth': r['truth']}
            for i, r in obs.items() if r['arm'] in REPORTED_ARMS and
            (r['arm'] not in ('R', 'H') or '-panel-release__' in i)}     # R/H: the released draws only


# One prediction row as it appears in PREDICTIONS.csv, with its errors.
def row(o, method, p, replicate=None, repeat=None, valid=None, origin='', reason=''):
    if method == 'et' and p is not None:
        valid = len(p) == 4 and all(math.isfinite(x) and 0 <= x <= 1 for x in p) and all(a >= b for a, b in zip(p, p[1:]))
    return {'observation_id': o['id'], 'source': o['source'], 'family': family(o['source']), 'group': o['group'],
            'arm': o['arm'], 'sample_index': o['sample_index'], 'method': method, 'replicate': replicate,
            'repeat_index': repeat, 'prediction': p, 'valid': p is not None if valid is None else valid,
            'validation_reason': reason, 'origin': origin, 'truth_rho2': o['truth'][0], **errors(p, o['truth'])}


# Gather every prediction of the panel: plugin/median/MLE and Qwen, ExtraTrees (production
# fit 0 plus the replicate fits) and the evaluated API answers.
def collect(inputs, replicates):
    obs = observations(inputs)
    rows = []
    with (STAGE1_TABLES/'PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            if r['observation_id'] not in obs: continue
            if r['method'] in ('plugin', 'median', 'mle', *QWEN):
                p = json.loads(r['prediction']) if r['prediction'] else None
                rows.append(row(obs[r['observation_id']], r['method'], p,
                                repeat=int(r['repeat_index']) if r['repeat_index'] else None,
                                valid=r['valid'] == 'True', origin='v11', reason=r.get('validation_reason', '')))
            if r['method'] == 'et' and r['arm'] not in MLE_ANCHOR_ARMS:
                obs[r['observation_id']]['sealed_et'] = json.loads(r['prediction'])
    for parent in STAGE2_SOURCES:
        for name in (f'source:{parent}', f'surrogate:{parent}'):
            for x in read_json(inputs[name]/'offline.json'):
                if x['id'] in obs and x['method'] in ('plugin', 'median', 'mle'):
                    rows.append(row(obs[x['id']], x['method'], x['prediction'], origin='panel'))
    for folder in (QWEN_DIR, QWEN_SUR_DIR):
        for a in qwen_answers(folder):
            if a['observation_id'] in obs:
                if a['status'] != 'completed': raise RuntimeError(f'Qwen answer missing: {a["id"]}')
                rows.append(row(obs[a['observation_id']], a['method'], a['prediction'], repeat=a['repeat_index'],
                                valid=a['valid'], origin='panel', reason=a['validation_reason']))
    et, outputs = {}, []
    for name, folder in inputs.items():
        if name.startswith(('train:', 'train_sur:')):
            d = read_json(folder/'predictions.json'); outputs.append((name, d))
            for r in d['observations']:
                if r['id'] in obs:
                    if (d['replicate'], r['id']) in et: raise ValueError('duplicate ET prediction')
                    et[d['replicate'], r['id']] = r['prediction']
    missing = [(k, i) for k in range(replicates+1) for i in obs if (k, i) not in et]
    if missing: raise ValueError(f'ET predictions missing: {missing[:3]}')
    diff = max(float(np.max(np.abs(np.asarray(et[0, i])-np.asarray(o['sealed_et']))))
               for i, o in obs.items() if 'sealed_et' in o)
    if diff != 0.0: raise AssertionError(f'R/H/B replicate 0 differs from stage 1: {diff}')
    for (k, i), p in et.items(): rows.append(row(obs[i], 'et', valid_profile(p), replicate=k, origin='panel'))
    with (API/'API_PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            p = json.loads(r['prediction']) if r['prediction'] else None
            rows.append(row(obs[r['observation_id']], r['method'], p, repeat=int(r['repeat_index']),
                            valid=r['valid'] == 'True', origin='api_v11', reason=r['validation_reason']))
    return rows, obs, outputs


# ---------------------------------------------------------------- leakage audit
# Safety check: no ExtraTrees fit may have seen the source family it predicts.
def leakage(outputs, obs):
    """No fit may train or select on the family (original or surrogate) of its test graphs."""
    for name, d in outputs:
        fold = d['fold']
        families = {family(obs[r['id']]['source']) for r in d['observations'] if r['id'] in obs}
        inner = set(d['choice'].get('inner_source_scores', {}))
        bad = (families & set(d['train_sources'])) | (families & inner) | ({fold} & (set(d['train_sources']) | inner))
        if bad: raise AssertionError(f'{name}: test family in training or selection: {bad}')
        if fold != 'synthetic' and families - {fold}: raise AssertionError(f'{name}: test graphs outside fold {fold}')
    return {'fits_checked': len(outputs), 'leaks': 0,
            'rule': 'the test family (original and surrogate) is absent from the real training rows and from '
                    'the inner selection sources of every fit, including the nr_radoslaw_email fold'}


# ---------------------------------------------------------------- wall times
# Computing time per stage, read from the task records.
def wall_times():
    jobs = []
    for path in sorted((WORK/'plans').glob('*/jobs.json')): jobs += read_json(path)
    if not jobs: return []
    out = subprocess.run(['sacct', '-j', ','.join(j['job'] for j in jobs), '-X', '-P', '-n',
                          '--format=JobName,Start,End,State,AllocCPUS,ElapsedRaw'], capture_output=True, text=True).stdout
    by = defaultdict(list)
    for line in out.splitlines():
        name, start, end, state, cpus, seconds = line.split('|')
        by[name].append((start, end, state, int(cpus or 0), int(seconds or 0)))
    return [{'array': n, 'tasks': len(v), 'states': ','.join(sorted({x[2] for x in v})),
             'first_start': min(x[0] for x in v), 'last_end': max(x[1] for x in v),
             'cpu_hours': sum(x[3]*x[4] for x in v)/3600} for n, v in sorted(by.items())]


# ---------------------------------------------------------------- stage
# Cluster task 'report': collect everything, check leakage and write the full table set.
def report(task, out, inputs):
    replicates = task.params['replicates']
    rows, obs, outputs = collect(inputs, replicates)
    write_json(out/'TRUTH.json', {o['source']: o['truth'] for o in obs.values()})
    write_csv(out/'PREDICTIONS.csv', [{**r, 'prediction': json.dumps(r['prediction']) if r['prediction'] else ''}
                                      for r in rows])
    main = [r for r in rows if r['replicate'] in (None, 0)]
    table, infer, fam, per_src = summary(main), paired_methods(main), paired_families(main), per_source(main)
    write_csv(out/'SUMMARY.csv', table); write_csv(out/'PAIRED_METHODS.csv', infer)
    write_csv(out/'PAIRED_FAMILIES.csv', fam); write_csv(out/'PER_SOURCE.csv', per_src)
    s_paths = list((STAGE1/'prepared/observations/sample').glob('*__S-*.json'))
    s_paths += [p for name in inputs if name.startswith(('source:', 'surrogate:'))
                for p in (inputs[name]/'observations').glob('*__S-*.json')]
    s_design = s_design_appendix(s_paths, main)
    write_csv(out/'S_DESIGN_APPENDIX.csv', s_design)
    (out/'MAIN_RESULTS.md').write_text(main_markdown(table, infer, fam, per_src, s_design))
    training, sampling, response = variability(rows, replicates)
    write_csv(out/'VARIABILITY_TRAINING.csv', training); write_csv(out/'VARIABILITY_SAMPLING.csv', sampling)
    write_csv(out/'VARIABILITY_RESPONSE.csv', response)
    (out/'VARIABILITY.md').write_text(variability_markdown(training, sampling, response, replicates,
                                                            fixed_input_variability(rows)))
    losses, errs = history_summary(inputs['history']/'HISTORY.csv')
    (out/'HISTORY.csv').write_text((inputs['history']/'HISTORY.csv').read_text())
    write_csv(out/'HISTORY_LOSSES.csv', losses); write_csv(out/'HISTORY_ERRORS.csv', errs)
    (out/'HISTORY.md').write_text(history_markdown(losses, errs))
    walks = [read_json(inputs[f'walkdiag:{k}']/'walk.json') for k in (*STAGE2_SOURCES, *SURROGATES)]
    write_csv(out/'WALK.csv', walks)
    (out/'WALK.md').write_text(walk_markdown(walks))
    leak = leakage(outputs, obs)
    write_csv(out/'ET_CHOICES.csv', sorted(({'replicate': d['replicate'], 'arm': d['arm'], 'fold': d['fold'],
                                             'anchor': d['choice']['anchor'],
                                             'min_samples_leaf': d['choice']['min_samples_leaf'],
                                             'max_features': d['choice']['max_features'],
                                             'inner_MAE_2': d['choice']['mean_inner_source_MAE2']}
                                            for n, d in outputs if n.startswith('train:')),
                                           key=lambda r: (r['replicate'], r['arm'], r['fold'])))
    graphs = [read_json(inputs[f'{s}:{p}']/'summary.json') for p in STAGE2_SOURCES for s in ('source', 'surrogate')]
    write_csv(out/'STAGE2_GRAPHS.csv', [{'graph': g['source'], 'N': g['N'], 'D': g['D'], 'M': g['M'],
                                        **{f'rho_{k}': g['truth'][k-2] for k in range(2, 6)},
                                        'budget_matched_by_arm': json.dumps({a: g['budget_matched_by_arm'][a] for a in REPORTED_ARMS}),
                                        'h_saturated': g['h_saturated'], 'n_panel': g['n_panel'],
                                        'n_panel_history': g['n_panel_history'], 'L': g['L'], 'p': g['p']} for g in graphs])
    freeze = read_json(inputs['api_freeze']/'API_FREEZE_EXT.json')
    write_json(out/'API_FREEZE_EXT.json', freeze)
    write_csv(out/'WALL_TIMES.csv', wall_times())
    qwen_new = [r for r in rows if r['method'] in QWEN and r['origin'] == 'panel']
    write_json(out/'REPORT.json', {
        'version': CFG['version'], 'arms': list(REPORTED_ARMS), 'replicates': replicates,
        'blocks': {g: sorted({r['source'] for r in rows if r['group'] == g}) for g in GROUPS},
        'leakage': leak, 'r_h_b_replicate0_equals_v11': True,
        'train_sur_models': Counter(d['check']['model'] for n, d in outputs if n.startswith('train_sur:')),
        'qwen_stage2_answers': len(qwen_new), 'qwen_stage2_valid': sum(r['valid'] for r in qwen_new),
        'qwen_stage2_invalid_reasons': Counter(r['validation_reason'] for r in qwen_new if not r['valid']),
        'api_pending_observations': freeze['observations'], 'rows': len(rows)})
    print('REPORT', len(rows), 'rows', flush=True)
