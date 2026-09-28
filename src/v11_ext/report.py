"""Result tables of the v11 extension.

Real-12 (eight v11 real sources plus the four new ones) is the headline; real-8,
new-4 and per-source tables follow, and surrogate and synthetic graphs are
reported only as separate blocks. v11 rows are copied from the sealed v11
PREDICTIONS.csv; the real-8 summary recomputed here must equal the v11 summary.
Accuracy is equal-source MAE_2 over valid predictions, as in v11.
"""
import csv
import json
import math
import subprocess
from collections import Counter, defaultdict
from statistics import median
import numpy as np
from main_experiment.common import ROOT, read_json, write_csv, write_json
from .core import ARMS, CFG, NEW_SOURCES, RH, V10, WORK
from .qwen import answers as qwen_answers

V11 = ROOT/'docs/results/panel888_v11_main_20260923'
API = ROOT/'docs/results/api_v11_20260923'
METHODS = ('plugin', 'median', 'design', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
REF_METHOD = {'R': 'plugin', 'S': 'design', 'S_obs': 'design', 'H': 'mle', 'B': 'mle'}
GROUPS = {'real12': ('real8', 'new4'), 'real8': ('real8',), 'new4': ('new4',),
          'surrogate': ('surrogate',), 'synthetic': ('synthetic',)}
GROUP_TITLE = {'real12': 'Real-12 (headline: 8 v11 real + 4 new real sources)', 'real8': 'Real-8 (v11 real sources)',
               'new4': 'New-4 (additional real sources)', 'surrogate': 'Surrogate-8 (separate block)',
               'synthetic': 'Synthetic-8 (separate block)'}


def errors(p, truth):
    if p is None: return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    d = np.asarray(p, float)-np.asarray(truth, float)
    return {'AE2': float(abs(d[0])), 'ProfileAE': float(np.abs(d).mean()), 'signed_rho2': float(d[0])}


def fmt(x, digits=4):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{digits}f}'


# ---------------------------------------------------------------- collection
def v11_rows():
    truth = {}
    for folder in (V10/'prepared/observations/sample', RH/'mainexp/run/observations/sample'):
        for path in folder.glob('*.json'):
            r = read_json(path); truth[r['id']] = r['truth']
    rows = []
    with (V11/'PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            p = json.loads(r['prediction']) if r['prediction'] else None
            group = 'real8' if r['stratum'] == 'real' else r['stratum']
            rows.append({'observation_id': r['observation_id'], 'source': r['source'], 'group': group,
                         'arm': r['arm'], 'sample_index': int(r['sample_index']), 'method': r['method'],
                         'replicate': 0 if r['method'] == 'et' else None,
                         'repeat_index': int(r['repeat_index']) if r['repeat_index'] else None,
                         'prediction': p, 'valid': r['valid'] == 'True', 'origin': 'v11_sealed',
                         'truth_rho2': truth[r['observation_id']][0], **errors(p, truth[r['observation_id']])})
    if len({r['observation_id'] for r in rows}) != 360: raise ValueError('v11 predictions incomplete')
    return rows, truth


def new_rows(inputs, et, qwen):
    rows, observations = [], {}
    for source in NEW_SOURCES:
        folder = inputs[f'source:{source}']
        for path in (folder/'observations').glob('*.json'):
            r = read_json(path); observations[r['id']] = r
        for x in read_json(folder/'offline.json'):
            o = observations[x['id']]
            rows.append(_row(o, x['method'], x['prediction'], None, None))
    for (k, oid), p in et.items():
        if oid in observations: rows.append(_row(observations[oid], 'et', p, k, None))
    for a in qwen:
        rows.append({**_row(observations[a['observation_id']], a['method'], a['prediction'], None, a['repeat_index']),
                     'valid': a['valid'], 'validation_reason': a['validation_reason']})
    return rows, observations


def _row(o, method, p, replicate, repeat):
    return {'observation_id': o['id'], 'source': o['graph_id'], 'group': 'new4', 'arm': o['arm'],
            'sample_index': o['sample_index'], 'method': method, 'replicate': replicate, 'repeat_index': repeat,
            'prediction': p, 'valid': p is not None, 'origin': 'v11_ext', 'truth_rho2': o['truth'][0],
            **errors(p, o['truth'])}


def et_replicates(inputs, replicates):
    out = {}
    for name, folder in inputs.items():
        if not name.startswith('train:'): continue
        d = read_json(folder/'predictions.json')
        for r in d['observations']:
            if (d['replicate'], r['id']) in out: raise ValueError('duplicate ET prediction')
            out[d['replicate'], r['id']] = r['prediction']
    reps = {k for k, _ in out}
    if reps != set(range(replicates+1)): raise ValueError('ET replicates incomplete')
    counts = {k: sum(1 for kk, _ in out if kk == k) for k in reps}
    if len(set(counts.values())) != 1: raise ValueError(f'unequal ET coverage per replicate: {counts}')
    return out


# ---------------------------------------------------------------- summaries
def equal_source(rows, field='AE2'):
    by = defaultdict(list)
    for r in rows:
        if r[field] is not None: by[r['source']].append(r[field])
    return {s: float(np.mean(v)) for s, v in by.items()}


def summary_row(rows, sources):
    ae, pe, sg = equal_source(rows), equal_source(rows, 'ProfileAE'), equal_source(rows, 'signed_rho2')
    complete = set(ae) == set(sources)
    return {'MAE_2': float(np.mean(list(ae.values()))) if complete else None,
            'ProfileMAE': float(np.mean(list(pe.values()))) if complete else None,
            'signed_rho_2': float(np.mean(list(sg.values()))) if complete else None,
            'validity': float(np.mean([r['valid'] for r in rows])) if rows else None,
            'sources': len(sources), 'sources_with_valid': len(ae)}


def main_tables(rows):
    sources = {g: sorted({r['source'] for r in rows if r['group'] in members}) for g, members in GROUPS.items()}
    table = []
    for g, members in GROUPS.items():
        for arm in ARMS:
            for method in METHODS:
                sel = [r for r in rows if r['group'] in members and r['arm'] == arm and r['method'] == method
                       and r['replicate'] in (None, 0)]
                if not sel: continue
                table.append({'group': g, 'arm': arm, 'method': method,
                              'reference': method == REF_METHOD[arm], **summary_row(sel, sources[g])})
    return table, sources


def per_source(rows):
    out = []
    key = lambda r: (r['group'], r['source'], r['arm'], r['method'])
    grouped = defaultdict(list)
    for r in rows:
        if r['replicate'] in (None, 0): grouped[key(r)].append(r)
    for (group, source, arm, method), sel in sorted(grouped.items()):
        valid = [r for r in sel if r['AE2'] is not None]
        out.append({'group': group, 'source': source, 'arm': arm, 'method': method,
                    'MAE_2': float(np.mean([r['AE2'] for r in valid])) if valid else None,
                    'ProfileMAE': float(np.mean([r['ProfileAE'] for r in valid])) if valid else None,
                    'signed_rho_2': float(np.mean([r['signed_rho2'] for r in valid])) if valid else None,
                    'validity': len(valid)/len(sel)})
    return out


def spread(values):
    a = np.asarray(values, float)
    return float(np.std(a, ddof=1)), float(np.ptp(a))


def replicate_table(rows, api, replicates):
    """ET replicate variability next to LLM repeat variability, per arm and group."""
    out = []
    sources = {g: sorted({r['source'] for r in rows if r['group'] in m}) for g, m in GROUPS.items()}
    et = [r for r in rows if r['method'] == 'et']
    for g, members in GROUPS.items():
        for arm in ARMS:
            sel = [r for r in et if r['group'] in members and r['arm'] == arm]
            if not sel: continue
            maes = [summary_row([r for r in sel if r['replicate'] == k], sources[g])['MAE_2'] for k in range(replicates+1)]
            per_obs = defaultdict(list)
            for r in sel: per_obs[r['observation_id']].append(r['prediction'][0])
            if any(len(v) != replicates+1 for v in per_obs.values()): raise ValueError('ET replicate grid incomplete')
            sd_range = [spread(v) for v in per_obs.values()]
            entry = {'group': g, 'arm': arm, 'observations': len(per_obs), 'replicates': replicates+1,
                     'ET_MAE_2_mean': float(np.mean(maes)), 'ET_MAE_2_SD': float(np.std(maes, ddof=1)),
                     'ET_MAE_2_min': min(maes), 'ET_MAE_2_max': max(maes), 'ET_MAE_2_production': maes[0],
                     'ET_median_obs_SD': median(s for s, _ in sd_range),
                     'ET_share_range_gt_0.01': float(np.mean([r > .01 for _, r in sd_range])),
                     'MLE_obs_SD': 0.0}
            for label, source_rows in (('qwen_thinking', rows), ('qwen_nonthinking', rows),
                                       ('gpt_6_sol', api), ('gpt_6_sol_tools', api)):
                reps = defaultdict(list)
                for r in source_rows:
                    if r['method'] == label and r['group'] in members and r['arm'] == arm and r['valid']:
                        reps[r['observation_id']].append(r['prediction'][0])
                full = [spread(v) for v in reps.values() if len(v) == 3]
                entry[f'{label}_obs_with_3_valid'] = len(full)
                entry[f'{label}_median_obs_SD'] = median(s for s, _ in full) if full else None
                entry[f'{label}_share_range_gt_0.01'] = float(np.mean([r > .01 for _, r in full])) if full else None
            out.append(entry)
    return out


def api_rows():
    rows = []
    with (API/'API_PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            rows.append({'observation_id': r['observation_id'], 'method': r['method'], 'arm': r['arm'],
                         'group': 'real8' if r['stratum'] == 'real' else r['stratum'],
                         'valid': r['valid'] == 'True',
                         'prediction': json.loads(r['prediction']) if r['prediction'] else None})
    return rows


# ---------------------------------------------------------------- markdown
def md_table(header, rows):
    return ['| '+' | '.join(header)+' |', '|'+'|'.join('---' if i < 2 else '---:' for i in range(len(header)))+'|',
            *('| '+' | '.join(map(str, r))+' |' for r in rows)]


def main_markdown(table, per_src, sources, qwen_complete):
    lines = ['# v11 extension: main results', '',
             'Equal-source MAE_2 (lower is better); ProfileMAE and signed rho_2 error are secondary.',
             'LLM accuracy uses valid final answers only; nothing is clipped or repaired. ET is the production',
             'fit (replicate 0); its replicate variability is in [ET_REPLICATES.md](ET_REPLICATES.md).',
             'Real-8, surrogate and synthetic rows are the sealed v11 predictions. Paid API models were not run',
             f'on the new sources. Qwen complete for the new sources: {qwen_complete}.', '']
    for g in GROUPS:
        lines += [f'## {GROUP_TITLE[g]}', '', f'Sources: {", ".join(sources[g])}.', '']
        rows = [[r['arm'], r['method']+(' (ref.)' if r['reference'] else ''), fmt(r['MAE_2']), fmt(r['ProfileMAE']),
                 fmt(r['signed_rho_2']), fmt(r['validity'], 3), f"{r['sources_with_valid']}/{r['sources']}"]
                for r in table if r['group'] == g]
        lines += md_table(['Arm', 'Method', 'MAE_2', 'ProfileMAE', 'Signed rho_2', 'Validity', 'Sources'], rows)+['']
    lines += ['## Per source (real-12, MAE_2)', '']
    cols = ('plugin', 'median', 'design', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking')
    index = {(r['source'], r['arm'], r['method']): r['MAE_2'] for r in per_src}
    rows = [[f"{s} ({'new' if g == 'new4' else 'v11'})", arm, *(fmt(index.get((s, arm, m))) for m in cols)]
            for g in ('real8', 'new4') for s in sources[g] for arm in ARMS]
    lines += md_table(['Source', 'Arm', *cols], rows)+['', 'All groups and metrics: `PER_SOURCE.csv`.', '']
    return '\n'.join(lines)


def replicate_markdown(rep, replicates, verification):
    lines = ['# ET replicates', '',
             f'Replicate 0 is the v11 production fit; replicates 1-{replicates} redraw every real training',
             'observation and every pool-train observation on the fixed pool graphs, reseed the forests and rerun',
             'the nested leave-one-real-training-source-out selection. Test observations are fixed.',
             f'Statistics use all {replicates+1} replicates. Per-observation SD is the sample SD of the rho_2',
             'prediction; LLM columns use observations with three valid repeats (GPT-6 Sol: v11 R/S/H/B only).',
             'MLE is deterministic at fixed input (SD 0).', '',
             f'Replicate-0 reproduction: {verification}', '']
    header = ['Group', 'Arm', 'ET MAE_2 mean ± SD', 'ET prod.', 'ET med. obs SD', 'ET share range>0.01',
              'Qwen-T med. SD', 'Qwen-T share', 'Qwen-NT med. SD', 'Qwen-NT share', 'GPT med. SD', 'GPT share',
              'GPT+Py med. SD', 'GPT+Py share', 'MLE SD']
    rows = [[r['group'], r['arm'], f"{fmt(r['ET_MAE_2_mean'])} ± {fmt(r['ET_MAE_2_SD'])}", fmt(r['ET_MAE_2_production']),
             fmt(r['ET_median_obs_SD']), fmt(r['ET_share_range_gt_0.01'], 3),
             fmt(r['qwen_thinking_median_obs_SD']), fmt(r['qwen_thinking_share_range_gt_0.01'], 3),
             fmt(r['qwen_nonthinking_median_obs_SD']), fmt(r['qwen_nonthinking_share_range_gt_0.01'], 3),
             fmt(r['gpt_6_sol_median_obs_SD']), fmt(r['gpt_6_sol_share_range_gt_0.01'], 3),
             fmt(r['gpt_6_sol_tools_median_obs_SD']), fmt(r['gpt_6_sol_tools_share_range_gt_0.01'], 3), '0'] for r in rep]
    return '\n'.join(lines+md_table(header, rows)+[''])


def qwen_summary(rows, answers):
    out = []
    for arm in ARMS:
        for mode in ('qwen_thinking', 'qwen_nonthinking'):
            sel = [(a, r) for a, r in answers if r['arm'] == arm and a['method'] == mode]
            out.append({'arm': arm, 'method': mode, 'requests': len(sel),
                        'completed': sum(a['status'] == 'completed' for a, _ in sel),
                        'valid': sum(a['valid'] for a, _ in sel),
                        'invalid_reasons': json.dumps(dict(sorted(Counter(
                            a['validation_reason'] for a, _ in sel if not a['valid']).items()))),
                        'median_output_tokens': median([a['output_tokens'] for a, _ in sel if a.get('output_tokens')] or [0]),
                        'median_seconds': median([a['seconds'] for a, _ in sel if a.get('seconds')] or [0])})
    return out


# ---------------------------------------------------------------- wall times
def wall_times():
    jobs = []
    for path in sorted((WORK/'plans').glob('*/jobs.json')):
        jobs += read_json(path)
    ids = ','.join(j['job'] for j in jobs)
    if not ids: return {}, []
    out = subprocess.run(['sacct', '-j', ids, '-X', '-P', '-n', '--format=JobID,JobName,Elapsed,Start,End,State,AllocCPUS,ElapsedRaw'],
                         capture_output=True, text=True).stdout
    rows = [dict(zip(['job', 'name', 'elapsed', 'start', 'end', 'state', 'cpus', 'seconds'], line.split('|')))
            for line in out.splitlines() if line]
    by = defaultdict(list)
    for r in rows: by[r['name']].append(r)
    summary = []
    for name, group in sorted(by.items()):
        starts = [g['start'] for g in group if g['start'] not in ('Unknown', 'None')]
        ends = [g['end'] for g in group if g['end'] not in ('Unknown', 'None')]
        summary.append({'array': name, 'tasks': len(group), 'states': ','.join(sorted({g['state'] for g in group})),
                        'first_start': min(starts) if starts else '', 'last_end': max(ends) if ends else '',
                        'max_task_seconds': max(int(g['seconds'] or 0) for g in group),
                        'cpu_hours': sum(int(g['seconds'] or 0)*int(g['cpus'] or 0) for g in group)/3600})
    return {'first_start': min(s['first_start'] for s in summary if s['first_start']),
            'last_end': max(s['last_end'] for s in summary if s['last_end'])}, summary


# ---------------------------------------------------------------- stage
def report(task, out, inputs):
    replicates = task.params['replicates']
    old, truth = v11_rows()
    et = et_replicates(inputs, replicates)
    # Replicate 0 of every v11 observation must equal the sealed v11 ET prediction.
    sealed = {r['observation_id']: r['prediction'] for r in old if r['method'] == 'et'}
    rep0_diff = max(float(np.max(np.abs(np.asarray(et[0, oid])-np.asarray(p)))) for oid, p in sealed.items())
    if rep0_diff != 0.0: raise AssertionError(f'replicate 0 differs from v11 ET: {rep0_diff}')
    answers = qwen_answers()
    qwen_complete = all(a['status'] == 'completed' for a in answers)
    if not qwen_complete:
        missing = sum(a['status'] == 'missing' for a in answers)
        raise RuntimeError(f'Qwen incomplete: {missing} missing answers of {len(answers)}')
    new, observations = new_rows(inputs, et, answers)
    obs_meta = {r['observation_id']: r for r in old}
    reps = []
    for (k, oid), p in sorted(et.items()):
        if k == 0 or oid not in obs_meta: continue
        m = obs_meta[oid]
        reps.append({**{f: m[f] for f in ('observation_id', 'source', 'group', 'arm', 'sample_index', 'truth_rho2')},
                     'method': 'et', 'replicate': k, 'repeat_index': None, 'prediction': p, 'valid': True,
                     'origin': 'v11_ext', **errors(p, truth[oid])})
    rows = old+reps+new
    write_csv(out/'PREDICTIONS.csv', [{**r, 'prediction': json.dumps(r['prediction']) if r['prediction'] else ''}
                                      for r in rows])
    table, sources = main_tables(rows)
    # The recomputed real-8 table must equal the sealed v11 summary.
    with (V11/'SUMMARY.csv').open() as f:
        sealed_summary = {(r['arm'], r['method']): float(r['MAE_2']) for r in csv.DictReader(f)
                          if r['stratum'] == 'real' and r['MAE_2']}
    worst = max(abs(r['MAE_2']-sealed_summary[r['arm'], r['method']]) for r in table
                if r['group'] == 'real8' and (r['arm'], r['method']) in sealed_summary)
    if worst > 1e-12: raise AssertionError(f'real-8 summary differs from v11: {worst}')
    write_csv(out/'MAIN_SUMMARY.csv', table)
    per_src = per_source(rows)
    write_csv(out/'PER_SOURCE.csv', per_src)
    (out/'MAIN_RESULTS.md').write_text(main_markdown(table, per_src, sources, qwen_complete))
    rep = replicate_table(rows, api_rows(), replicates)
    write_csv(out/'ET_REPLICATES.csv', rep)
    trains = [read_json(folder/'predictions.json') for name, folder in inputs.items() if name.startswith('train:')]
    ver = [t['verification'] for t in trains if 'verification' in t]
    verification = {'folds_verified': len(ver),
                    'refit_vs_v11_max_abs': max(v['refit_vs_v11_max_abs'] for v in ver),
                    'sealed_model_vs_v11_max_abs': max(v['sealed_model_vs_v11_max_abs'] for v in ver),
                    'refit_vs_sealed_model_new_rows_max_abs': max(v['refit_vs_sealed_model_new_rows_max_abs'] or 0 for v in ver),
                    'replicate0_equals_v11_predictions': rep0_diff == 0.0}
    choices = [{'replicate': t['replicate'], 'arm': t['arm'], 'fold': t['fold'], 'anchor': t['choice']['anchor'],
                'min_samples_leaf': t['choice']['min_samples_leaf'], 'max_features': t['choice']['max_features'],
                'inner_MAE_2': t['choice']['mean_inner_source_MAE2'], 'n_train': t['n_train']} for t in trains]
    write_csv(out/'ET_CHOICES.csv', sorted(choices, key=lambda r: (r['replicate'], r['arm'], r['fold'])))
    (out/'ET_REPLICATES.md').write_text(replicate_markdown(rep, replicates, json.dumps(verification)))
    qs = qwen_summary(new, [(a, observations[a['observation_id']]) for a in answers])
    write_csv(out/'QWEN_NEW_SOURCES.csv', qs)
    src = [read_json(inputs[f'source:{s}']/'summary.json') for s in NEW_SOURCES]
    write_csv(out/'SOURCES.csv', [{'source': s['source'], 'label': s['label'], 'fold': s['fold'], 'N': s['N'],
                                   'D': s['D'], 'M': s['M'], **{f'rho_{k}': s['truth'][k-2] for k in range(2, 6)},
                                   'window_shares': json.dumps([round(x, 4) for x in s['window_check']['shares']]),
                                   'window_flagged': s['window_check']['flagged'],
                                   'trimmed_records': s['window_check']['trimmed_records'],
                                   'n_panel': s['n_panel'], 'n_panel_history': s['n_panel_history'], 'L': s['L'],
                                   'p': s['p'], 'h_saturated': s['h_saturated'],
                                   'budget_matched_by_arm': json.dumps(s['budget_matched_by_arm']),
                                   'unmatched_reasons': json.dumps(s['unmatched_reasons']),
                                   'observations': s['observations']} for s in src])
    walls, arrays = wall_times()
    write_csv(out/'WALL_TIMES.csv', arrays)
    feature_check = read_json(inputs['testset']/'feature_check.json')
    write_json(out/'REPORT.json', {'version': CFG['version'], 'replicates': replicates, 'et_verification': verification,
                                   'feature_recomputation_check': feature_check, 'wall': walls,
                                   'rows': len(rows), 'new_observations': len(observations),
                                   'qwen_new_requests': len(answers), 'qwen_new_valid': sum(a['valid'] for a in answers),
                                   'real8_summary_max_abs_difference_to_v11': worst,
                                   'sources': {s['source']: {'window_check': s['window_check'],
                                                             'unmatched_reasons': s['unmatched_reasons']} for s in src}})
    print('REPORT', len(rows), 'rows', flush=True)
