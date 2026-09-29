"""Final result tables: 12 real sources, 12 paired surrogates, 8 synthetic graphs, arms R/S/H/B.

Four estimators per sampler (plugin, median, MLE, ExtraTrees) plus the LLMs. The reference
is plugin for R and the MLE otherwise. Real sources are the main analysis (all twelve,
equally weighted); surrogates and synthetic graphs are separate blocks; original-surrogate
contrasts are paired by source family. A method missing any source of a block is marked
pending. S_obs is drawn for reproducibility only and is not reported.

In plain words: report() is the cluster task that collects every prediction (offline
methods, ExtraTrees fits, Qwen, API answers) into one table and scores it; finalize() is
the local step that rebuilds all published tables in docs/results/final from that table
and the evaluated API answers.
"""
import csv
import itertools
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
import numpy as np
from study.common import ROOT, read_json, write_csv, write_json
from study.answer_format import valid_profile
from study.observation import parse
from study.estimators import design_estimate
from .core import CFG, MLE_ANCHOR_ARMS, STAGE2_SOURCES, PANEL_RUN, STAGE1, WORK
from .surrogates_and_checks import REPORTED_ARMS, QWEN_SUR_DIR, SURROGATES, family
from .qwen import QWEN_DIR, answers as qwen_answers

# Inputs of the report task: stage-1 tables (scripts/score_stage1.py --out) and the
# API evaluation (scripts/evaluate_api.py --out). Both are regenerated, not committed.
STAGE1_TABLES = ROOT/'results/stage1_tables'
API = ROOT/'results/api_evaluation'
OFFLINE = ('plugin', 'median', 'mle', 'et')
QWEN = ('qwen_thinking', 'qwen_nonthinking')
APIM = ('deepseek_flash', 'gpt_6_sol', 'gpt_6_sol_tools')
METHODS = OFFLINE+QWEN+APIM
REFERENCE = {'R': 'plugin', 'S': 'mle', 'H': 'mle', 'B': 'mle'}
GROUPS = ('real', 'surrogate', 'synthetic')
TITLE = {'real': 'Real networks (main analysis, 12 networks)',
         'surrogate': 'Time-shuffled copies of the real networks (12, separate block)',
         'synthetic': 'Synthetic networks (8, separate block)'}


# AE2 = |rho_2 error|, ProfileAE = mean |error| over rho_2..rho_5, signed_rho2 = rho_2 error with sign.
def errors(p, truth):
    if p is None: return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    d = np.asarray(p, float)-np.asarray(truth, float)
    return {'AE2': float(abs(d[0])), 'ProfileAE': float(np.abs(d).mean()), 'signed_rho2': float(d[0])}


def fmt(x, digits=4):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{digits}f}'


# Markdown table helper for the .md reports.
def md_table(header, rows):
    return ['| '+' | '.join(header)+' |', '|'+'|'.join('---' if i < 2 else '---:' for i in range(len(header)))+'|',
            *('| '+' | '.join(map(str, r))+' |' for r in rows)]


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


# ---------------------------------------------------------------- accuracy
# Mean of a field per source (graph); rows without a score (invalid answers) are skipped.
def by_source(rows, field='AE2'):
    out = defaultdict(list)
    for r in rows:
        if r[field] is not None: out[r['source']].append(r[field])
    return {s: float(np.mean(v)) for s, v in out.items()}


# SUMMARY.csv: equal-source MAE_2 per block, arm and method. Only ExtraTrees fit 0 enters
# here (the caller passes replicate 0 only).
def summary(rows):
    sources = {g: sorted({r['source'] for r in rows if r['group'] == g}) for g in GROUPS}
    table = []
    for g in GROUPS:
        for arm in REPORTED_ARMS:
            for m in METHODS:
                sel = [r for r in rows if r['group'] == g and r['arm'] == arm and r['method'] == m]
                ae = by_source(sel)
                complete = bool(ae) and set(ae) == set(sources[g])
                entry = {'group': g, 'arm': arm, 'method': m, 'reference': m == REFERENCE[arm],
                         'sources': len(sources[g]), 'sources_with_valid': len(ae),
                         'status': f'{len(ae)}/{len(sources[g])}' if complete else f'pending ({len(ae)}/{len(sources[g])})',
                         'MAE_2': None, 'ProfileMAE': None, 'signed_rho_2': None,
                         'validity': float(np.mean([r['valid'] for r in sel])) if sel else None}
                if complete:
                    entry.update(MAE_2=float(np.mean(list(ae.values()))),
                                 ProfileMAE=float(np.mean(list(by_source(sel, 'ProfileAE').values()))),
                                 signed_rho_2=float(np.mean(list(by_source(sel, 'signed_rho2').values()))))
                table.append(entry)
    return table


# Exact sign-flip p value of a mean of per-source differences.
# Reference: sign-flip permutation inference, Winkler et al. (2014).
def signflip(values):
    a = np.asarray(values, float)
    observed = abs(a.mean())
    return float(np.mean([abs(np.mean(a*np.asarray(s))) >= observed-1e-15
                          for s in itertools.product((-1, 1), repeat=len(a))]))


# PAIRED_METHODS.csv: per-source head-to-head comparisons within a block.
def paired_methods(rows):
    """Within a block: first minus second source-level MAE_2 (both complete on the same sources)."""
    out = []
    for g in ('real', 'surrogate'):
        for arm in REPORTED_ARMS:
            ref = REFERENCE[arm]
            for first, second in dict.fromkeys([('et', ref), ('qwen_thinking', ref), (ref, 'plugin'),
                                                ('et', 'plugin'), ('qwen_thinking', 'plugin')]):
                if first == second: continue
                a, b = (by_source([r for r in rows if r['group'] == g and r['arm'] == arm and r['method'] == m])
                        for m in (first, second))
                if set(a) != set(b): continue
                d = np.array([a[s]-b[s] for s in sorted(a)])
                out.append({'group': g, 'arm': arm, 'comparison': f'{first} vs {second}', 'sources': len(d),
                            'mean_difference': float(d.mean()), 'first_better': int((d < 0).sum()),
                            'exact_signflip_p': signflip(d)})
    return out


# PAIRED_FAMILIES.csv: each real source against its own surrogate.
def paired_families(rows):
    """Original minus surrogate source-level MAE_2, one pair per source family."""
    out = []
    for arm in REPORTED_ARMS:
        for m in OFFLINE+QWEN:
            real = by_source([r for r in rows if r['group'] == 'real' and r['arm'] == arm and r['method'] == m])
            sur = by_source([r for r in rows if r['group'] == 'surrogate' and r['arm'] == arm and r['method'] == m])
            fams = sorted(f for f in real if f+'__pwt' in sur)
            d = np.array([real[f]-sur[f+'__pwt'] for f in fams])
            out.append({'arm': arm, 'method': m, 'families': len(d),
                        'original_MAE_2': float(np.mean([real[f] for f in fams])),
                        'surrogate_MAE_2': float(np.mean([sur[f+'__pwt'] for f in fams])),
                        'mean_difference': float(d.mean()), 'original_better': int((d < 0).sum()),
                        'exact_signflip_p': signflip(d)})
    return out


# PER_SOURCE.csv: scores per single graph.
def per_source(rows):
    grouped = defaultdict(list)
    for r in rows: grouped[r['group'], r['source'], r['arm'], r['method']].append(r)
    out = []
    for (g, s, arm, m), sel in sorted(grouped.items()):
        scored = [r for r in sel if r['AE2'] is not None]
        out.append({'group': g, 'source': s, 'arm': arm, 'method': m, 'answers': len(sel),
                    'valid': sum(r['valid'] for r in sel),
                    'MAE_2': float(np.mean([r['AE2'] for r in scored])) if scored else None,
                    'ProfileMAE': float(np.mean([r['ProfileAE'] for r in scored])) if scored else None,
                    'signed_rho_2': float(np.mean([r['signed_rho2'] for r in scored])) if scored else None})
    return out


# ---------------------------------------------------------------- variability
def sd(values):
    return float(np.std(np.asarray(values, float), ddof=1))


# Three separate kinds of spread (VARIABILITY.md):
# - training: how much ExtraTrees' MAE moves across its 11 fits;
# - sampling: how much a method's rho_2 moves across the 3 observation draws of a graph;
# - response: how much an LLM's answer moves across its 3 repeats of the same observation.
# Reference: separating sources of variance, Bouthillier et al. (2021).
def variability(rows, replicates):
    """Training (ET fits), sampling (sampler draws) and LLM answer variability, kept apart."""
    training, sampling, response = [], [], []
    for g in GROUPS:
        n_graphs = len({r['source'] for r in rows if r['group'] == g})
        for arm in REPORTED_ARMS:
            et = [r for r in rows if r['group'] == g and r['arm'] == arm and r['method'] == 'et']
            maes = [float(np.mean(list(by_source([r for r in et if r['replicate'] == k]).values())))
                    for k in range(replicates+1)]
            per_obs = defaultdict(list)
            for r in et: per_obs[r['observation_id']].append(r['prediction'][0])
            training.append({'group': g, 'arm': arm, 'fits': replicates+1, 'ET_MAE_2_mean': float(np.mean(maes)),
                             'ET_MAE_2_SD': sd(maes), 'ET_MAE_2_min': min(maes), 'ET_MAE_2_max': max(maes),
                             'median_observation_SD_rho2': median(sd(v) for v in per_obs.values())})
            for m in METHODS:
                cells = defaultdict(lambda: defaultdict(list))
                reps = defaultdict(list)
                for r in rows:
                    if (r['group'], r['arm'], r['method']) == (g, arm, m) and (r['valid'] or m == 'et') and r['replicate'] in (None, 0):
                        cells[r['source']][r['sample_index']].append(r['prediction'][0])
                        reps[r['observation_id']].append(r['prediction'][0])
                draw_sd = [sd([np.mean(v) for v in draws.values()]) for draws in cells.values() if len(draws) >= 2]
                if draw_sd:
                    sampling.append({'group': g, 'arm': arm, 'method': m, 'graphs': len(draw_sd), 'graphs_in_block': n_graphs,
                                     'median_graph_SD_rho2_across_draws': median(draw_sd)})
                full = [sd(v) for v in reps.values() if len(v) == 3] if m in QWEN+APIM else []
                if full:
                    response.append({'group': g, 'arm': arm, 'method': m, 'observations_with_3_valid': len(full),
                                     'median_observation_SD_rho2': median(full)})
    return training, sampling, response


# Spread at identical input: 3 LLM repeats vs 3 randomly chosen ET fits, for a fair comparison.
def fixed_input_variability(rows):
    """Median observation SD at fixed input, with ET fit count matched to LLM repeats."""
    rng = np.random.default_rng(20260928)
    out = []
    for arm in REPORTED_ARMS:
        for method in ('mle', 'et', 'qwen_thinking', 'deepseek_flash', 'gpt_6_sol', 'gpt_6_sol_tools'):
            cells = defaultdict(list)
            for r in rows:
                if r['group'] == 'real' and r['arm'] == arm and r['method'] == method and (r['valid'] or method == 'et'):
                    cells[r['observation_id']].append(r['prediction'][0])
            if method == 'mle':
                values = [0. for v in cells.values() if v]
            elif method == 'et':
                values = [float(np.mean([sd(np.asarray(v)[rng.choice(len(v), 3, replace=False)])
                                         for _ in range(200)])) for v in cells.values() if len(v) == 11]
            else:
                values = [sd(v) for v in cells.values() if len(v) == 3]
            out.append({'arm': arm, 'method': method, 'observations': len(values),
                        'median_observation_SD_rho2': median(values) if values else None})
    return out


# S_DESIGN_APPENDIX.csv: the walk's inverse-event ratio estimate, for reference only (not a reported method).
def s_design_appendix(block_paths, rows):
    """Descriptive S ratio from the released traversal weights and frozen answers."""
    design = {}
    for path in block_paths:
        item = read_json(path)
        if item['arm'] == 'S':
            design[item['id']] = design_estimate(item if 'table' in item else parse(item['block']))
    if len(design) != 96: raise ValueError(f'expected 96 frozen S observations, got {len(design)}')
    srows = [r for r in rows if r['arm'] == 'S' and r['observation_id'] in design]
    out = []
    for group in GROUPS:
        cells = defaultdict(list)
        seen = set()
        for r in srows:
            if r['group'] == group and r['observation_id'] not in seen:
                cells[r['source']].append(abs(design[r['observation_id']][0] - r['truth_rho2']))
                seen.add(r['observation_id'])
        out.append({'group': group, 'arm': 'S', 'model': 'design', 'sources': len(cells),
                    'MAE_2': float(np.mean([np.mean(v) for v in cells.values()])),
                    'valid_answers': '', 'share_plugin': '', 'share_design': '', 'share_mle': ''})
    base = {(r['observation_id'], r['method']): r['prediction'][0] for r in srows
            if r['method'] in ('plugin', 'mle') and r['replicate'] in (None, 0)}
    for method in QWEN+APIM:
        answers = [r for r in srows if r['group'] == 'real' and r['method'] == method and r['valid']]
        out.append({'group': 'real', 'arm': 'S', 'model': method, 'sources': len({r['source'] for r in answers}),
                    'MAE_2': '', 'valid_answers': len(answers),
                    **{f'share_{name}': float(np.mean([abs(r['prediction'][0] -
                             (design[r['observation_id']][0] if name == 'design' else
                              base[r['observation_id'], name])) <= .005 for r in answers]))
                       for name in ('plugin', 'design', 'mle')}})
    return out


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


# ---------------------------------------------------------------- diagnostics
# HISTORY tables: how much of arm H's error is caused by the truncated time axis.
def history_summary(path):
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        for k in ('numerator_loss', 'denominator_loss', 'plugin_error', 'mle_error'):
            r[k] = float(r[k]) if r.get(k) not in (None, '') else None
        r['k'] = int(r['k'])
    losses, errs = [], []
    for g in ('real', 'surrogate'):
        pop = [r for r in rows if r['group'] == g and r['level'] == 'population']
        for window in ('last60', 'first60'):
            for k in range(2, 6):
                sel = [r for r in pop if r['window'] == window and r['k'] == k]
                losses.append({'group': g, 'window': window, 'k': k, 'graphs': len(sel),
                               'numerator_loss': float(np.mean([r['numerator_loss'] for r in sel])),
                               'denominator_loss': float(np.mean([r['denominator_loss'] for r in sel])),
                               'plugin_signed_error': float(np.mean([r['plugin_error'] for r in sel])),
                               'graphs_abs_error_le_0.01': sum(abs(r['plugin_error']) <= .01 for r in sel)})
        for level, window in (('population', 'last60'), ('population', 'first60'), ('H_sampled', 'last60')):
            for est in ('plugin', 'mle'):
                per_graph = defaultdict(lambda: defaultdict(list))
                for r in rows:
                    if (r['group'], r['level'], r['window']) == (g, level, window):
                        per_graph[r['graph_id']][r['k']].append(r[f'{est}_error'])
                e = {gid: [[float(x) for x in v[k]] for k in range(2, 6)] for gid, v in per_graph.items()}
                errs.append({'group': g, 'estimator': est, 'truncation': window, 'node_sampling': level == 'H_sampled',
                             'graphs': len(e), 'signed_rho2': float(np.mean([np.mean(v[0]) for v in e.values()])),
                             'MAE_2': float(np.mean([np.mean(np.abs(v[0])) for v in e.values()])),
                             'ProfileMAE': float(np.mean([np.mean([np.mean(np.abs(k)) for k in v]) for v in e.values()]))})
    return losses, errs


# ---------------------------------------------------------------- markdown
# Plain names used in every table (the CSV files keep the short method codes).
LABEL = {'plugin': 'Observed share (plug-in)', 'median': 'Training median', 'mle': 'Statistical model (MLE)',
         'et': 'ExtraTrees (trained model)', 'qwen_thinking': 'Qwen, thinking', 'qwen_nonthinking': 'Qwen, no thinking',
         'deepseek_flash': 'DeepSeek Flash', 'gpt_6_sol': 'GPT-6 Sol', 'gpt_6_sol_tools': 'GPT-6 Sol + Python'}
ARM_LABEL = {'R': 'R (random nodes)', 'S': 'S (random walk)', 'H': 'H (random nodes, last 60% of time)',
             'B': 'B (random event loss)'}
GROUP_LABEL = {'real': 'real', 'surrogate': 'time-shuffled copy', 'synthetic': 'synthetic'}
PART_LABEL = {'last60': 'last 60%', 'first60': 'first 60%'}
TERMS = 'Terms are explained in the [glossary](../../DESIGN.md#glossary).'


def label(method):
    """Plain name of a method, or of a comparison 'a vs b'."""
    return ' vs '.join(LABEL.get(m, m) for m in method.split(' vs '))


# MAIN_RESULTS.md: the tables above in readable form.
def main_markdown(table, infer, fam, per_src, s_design=None):
    lines = ['# Main results', '',
             'How to read this: each number is an average error in estimating rho_2, the share of interacting',
             'pairs that are active in at least two of five time windows. Lower is better; 0.03 means the estimate',
             'is off by 3 percentage points on average. Errors are averaged per network first, then over networks,',
             'so every network counts equally. The 12 real networks are the main analysis. ' + TERMS, '',
             '- **MAE_2**: mean absolute error of rho_2 (main measure). **ProfileMAE**: the same over rho_2 to rho_5.',
             '- **Signed error**: average error with its sign; positive means the method overestimates.',
             '- **Valid**: share of answers that follow the answer format; only valid answers are scored.',
             '- **(reference)**: the standard estimator for that sampling arm, which the others are compared with.',
             '- Language-model answers are never corrected. ExtraTrees output is limited to a valid profile',
             '  (values between 0 and 1, never increasing from rho_2 to rho_5).', '']
    for g in GROUPS:
        lines += [f'## {TITLE[g]}', '']
        lines += md_table(['Sampling arm', 'Method', 'MAE_2', 'ProfileMAE', 'Signed error', 'Valid', 'Networks'],
                          [[ARM_LABEL[r['arm']], LABEL[r['method']]+(' (reference)' if r['reference'] else ''),
                            fmt(r['MAE_2']), fmt(r['ProfileMAE']), fmt(r['signed_rho_2']), fmt(r['validity'], 3),
                            r['status']] for r in table if r['group'] == g])+['']
        if g == 'synthetic':
            lines += ['The eight synthetic networks come from four generators, two networks each, so these results',
                      'describe the generators and are not tested for significance.', '']
    lines += ['## Real network versus its time-shuffled copy', '',
              'For each of the 12 real networks: the error on the real network minus the error on its',
              'time-shuffled copy (negative: the real network is easier). "p" is an exact sign-flip test over',
              'the 12 pairs: the chance of a difference at least this large if real and copy were alike.', '']
    lines += md_table(['Sampling arm', 'Method', 'Real', 'Shuffled copy', 'Difference', 'Real better', 'p'],
                      [[ARM_LABEL[r['arm']], LABEL[r['method']], fmt(r['original_MAE_2']), fmt(r['surrogate_MAE_2']),
                        fmt(r['mean_difference']), f"{r['original_better']}/{r['families']}", fmt(r['exact_signflip_p'])]
                       for r in fam])+['']
    lines += ['## Method against method', '',
              'Error of the first method minus error of the second, per network and then averaged',
              '(negative: the first method is better). "First better" counts the networks it wins; p is the',
              'exact sign-flip test over networks.', '']
    lines += md_table(['Networks', 'Sampling arm', 'Comparison', 'Difference', 'First better', 'p'],
                      [[GROUP_LABEL[r['group']], ARM_LABEL[r['arm']], label(r['comparison']), fmt(r['mean_difference']),
                        f"{r['first_better']}/{r['sources']}", fmt(r['exact_signflip_p'])] for r in infer])+['']
    index = {(r['source'], r['arm'], r['method']): r['MAE_2'] for r in per_src}
    reals = sorted({r['source'] for r in per_src if r['group'] == 'real'})
    lines += ['## Error per real network (MAE_2)', '']
    lines += md_table(['Network', 'Sampling arm', *(LABEL[m] for m in OFFLINE+QWEN)],
                      [[s, arm, *(fmt(index.get((s, arm, m))) for m in OFFLINE+QWEN)] for s in reals for arm in REPORTED_ARMS])
    lines += ['', 'All networks, the API models and all measures: `PER_SOURCE.csv`.', '']
    if s_design is not None:
        lines += ['## Appendix: re-weighted walk estimate (arm S)', '',
                  'The random walk visits busy pairs more often. Re-weighting each observed pair by the inverse of its',
                  'number of events gives a simple corrected share (the "design ratio"). It is shown for comparison',
                  'only and is not one of the four estimators. The second table shows how often the valid answers of',
                  'each language model lie within 0.005 of the observed share, the design ratio or the MLE.', '']
        lines += md_table(['Networks', 'Design ratio MAE_2'],
                          [[GROUP_LABEL[r['group']], fmt(r['MAE_2'])] for r in s_design if r['model'] == 'design'])+['']
        lines += md_table(['Model (real networks, arm S)', 'Valid answers', 'Near observed share', 'Near design ratio',
                           'Near MLE'],
                          [[LABEL.get(r['model'], r['model']), r['valid_answers'], fmt(r['share_plugin'], 3),
                            fmt(r['share_design'], 3), fmt(r['share_mle'], 3)] for r in s_design if r['model'] != 'design'])
        lines += ['', 'Per group and model: `S_DESIGN_APPENDIX.csv`.', '']
    return '\n'.join(lines)


# VARIABILITY.md: how much results move for three different reasons.
def variability_markdown(training, sampling, response, replicates, fixed=None):
    lines = ['# Variability', '',
             'How to read this: a result can change for three separate reasons, reported separately.',
             'SD is the standard deviation (typical size of the variation) of the rho_2 estimate. ' + TERMS, '',
             f'1. **Training** (ExtraTrees only, {replicates+1} fits): fit 0 is the one reported everywhere; fits 1-10 are',
             '   trained on newly drawn training observations with new random seeds. The test observations stay the same.',
             '2. **Sampling**: each network is observed three times per arm; how much does the estimate change',
             '   between these three observations?',
             '3. **Answers**: each language model answered each observation three times; how much do the answers differ?', '',
             f'## 1. Training (ExtraTrees, {replicates+1} fits)', '']
    lines += md_table(['Networks', 'Sampling arm', 'MAE_2 mean', 'MAE_2 SD', 'MAE_2 min', 'MAE_2 max',
                       'Median SD per observation'],
                      [[GROUP_LABEL[r['group']], ARM_LABEL[r['arm']], fmt(r['ET_MAE_2_mean']), fmt(r['ET_MAE_2_SD']),
                        fmt(r['ET_MAE_2_min']), fmt(r['ET_MAE_2_max']), fmt(r['median_observation_SD_rho2'])]
                       for r in training])
    lines += ['', '## 2. Sampling', '',
              'SD of the rho_2 estimate across the three observations of a network (language models: mean of their',
              'valid answers per observation; ExtraTrees: fit 0), median over networks. The simple estimators give',
              'the same number for the same observation, so this is their only source of variation. Networks whose',
              'H arm covers every node have a single observation there and are left out.', '']
    lines += md_table(['Networks', 'Sampling arm', 'Method', 'Networks used', 'Median SD across observations'],
                      [[GROUP_LABEL[r['group']], ARM_LABEL[r['arm']], LABEL[r['method']],
                        f"{r['graphs']}/{r['graphs_in_block']}", fmt(r['median_graph_SD_rho2_across_draws'])]
                       for r in sampling])
    lines += ['', '## 3. Answers of the language models', '',
              'SD of rho_2 across the three answers to the same observation, median over observations with three',
              'valid answers.', '']
    lines += md_table(['Networks', 'Sampling arm', 'Model', 'Observations', 'Median SD across answers'],
                      [[GROUP_LABEL[r['group']], ARM_LABEL[r['arm']], LABEL[r['method']], r['observations_with_3_valid'],
                        fmt(r['median_observation_SD_rho2'])] for r in response])
    if fixed is not None:
        lines += ['', '## Same input, three outputs (real networks)', '',
                  'For a fair comparison with the three answers of a language model, ExtraTrees is also looked at',
                  'through three of its fits (200 random choices of three out of 11). The MLE always gives the same',
                  'number, so its SD is 0.', '']
        lines += md_table(['Sampling arm', 'Method', 'Observations', 'Median SD'],
                          [[ARM_LABEL[r['arm']], LABEL[r['method']], r['observations'], fmt(r['median_observation_SD_rho2'])]
                           for r in fixed])
    return '\n'.join(lines+[''])


# HISTORY.md: why arm H is hard.
def history_markdown(losses, errs):
    lines = ['# Seeing only part of the time axis (arm H)', '',
             'How to read this: arm H sees only the last 60% of the time axis (3 of 5 windows). A pair active in',
             'two early windows looks active in fewer windows, and pairs active only early disappear. These tables',
             'measure both effects on the complete networks, without any node sampling. ' + TERMS, '',
             '- K = active windows of a pair among all five; J = among the three visible ones.',
             '- **Numerator loss**: share of pairs with K >= k that no longer show J >= k.',
             '- **Denominator loss**: share of all interacting pairs that become invisible (J = 0).',
             '- The observed share is correct only when both losses are equal. With three visible windows, J >= 4',
             '  is impossible, so the observed share of rho_4 and rho_5 is always 0.',
             '- "Last 60%" is what arm H sees; "first 60%" is shown for comparison. "With node sampling" uses the',
             '  actual H observations (time cut plus random nodes).',
             '- Averages over the 12 real networks and over their 12 time-shuffled copies.', '', '## Losses', '']
    lines += md_table(['Networks', 'Visible part', 'k', 'Numerator loss', 'Denominator loss',
                       'Error of the observed share', 'Networks with |error| <= 0.01'],
                      [[GROUP_LABEL[r['group']], PART_LABEL.get(r['window'], r['window']), r['k'], fmt(r['numerator_loss']), fmt(r['denominator_loss']),
                        fmt(r['plugin_signed_error']), f"{r['graphs_abs_error_le_0.01']}/{r['graphs']}"] for r in losses])
    lines += ['', '## Errors', '']
    lines += md_table(['Networks', 'Estimator', 'Visible part', 'With node sampling', 'Signed error', 'MAE_2', 'ProfileMAE'],
                      [[GROUP_LABEL[r['group']], LABEL.get(r['estimator'], r['estimator']), PART_LABEL.get(r['truncation'], r['truncation']),
                        'yes' if r['node_sampling'] else 'no',
                        fmt(r['signed_rho2']), fmt(r['MAE_2']), fmt(r['ProfileMAE'])] for r in errs])
    return '\n'.join(lines+['', 'Per network and k: `HISTORY.csv`.', ''])


# WALK.md: the walk audit of all graphs.
def walk_markdown(walks):
    lines = ['# Random-walk check (arm S)', '',
             'How to read this: for every network, 1000 random walks of the arm-S length L were simulated. The',
             'table shows how far the observed share lands from the truth on average (bias) and how much it',
             'varies between walks (SD), for the plain observed share and for the re-weighted design ratio.',
             'Busy pairs are visited more often, which pushes the observed share up. ' + TERMS, '',
             '- **Shift**: how much the walk\'s preference for busy pairs moves rho_2 on its own.',
             '- **Check needed**: yes when the shift is larger than 0.05. **Correctable**: yes when the design ratio',
             '  removes at least 90% of that bias and halves the error. A "no" is reported, and the network stays in',
             '  every table.',
             '- **Effective walks (weights / ratio)**: how many independent observations the walk is worth.',
             '- **Revisits**: share of steps that return to an already seen pair. **Pairs**: distinct pairs seen per walk.', '']
    lines += md_table(['Network', 'Group', 'L', 'Shift', 'Observed share bias', 'Observed share SD', 'Design ratio bias',
                       'Design ratio SD', 'Check needed', 'Correctable', 'Effective walks (weights)',
                       'Effective walks (ratio)', 'Revisits', 'Pairs'],
                      [[w['graph_id'], GROUP_LABEL[w['stratum']], w['L'], fmt(float(w['stationary_shift_rho2'])),
                        fmt(float(w['plugin_rho2_bias'])), fmt(float(w['plugin_rho2_sd'])),
                        fmt(float(w['design_S_rho2_bias'])), fmt(float(w['design_S_rho2_sd'])),
                        w['gate_applicable'], w['gate_pass'], fmt(float(w['weight_ess_mean']), 1),
                        fmt(float(w['ratio_ess_mean']), 1), fmt(float(w['revisit_rate_mean']), 3),
                        fmt(float(w['distinct_dyads_mean']), 1)] for w in walks])
    return '\n'.join(lines+[''])


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


# Local step after all API answers are evaluated: add every LLM repeat to PREDICTIONS.csv
# and rebuild all published tables in `out` (docs/results/final).
def finalize(out, api_dir, observations=None):
    """Rebuild the consolidated tables from frozen predictions and collected API answers."""
    out, api_dir = Path(out), Path(api_dir)
    path = out/'PREDICTIONS.csv'
    raw = list(csv.DictReader(path.open()))
    by_obs = {r['observation_id']: r for r in raw}
    api = [r for r in csv.DictReader((api_dir/'API_PREDICTIONS.csv').open())
           if r['method'] == 'deepseek_flash']
    if len(api) != 384*3: raise ValueError(f'expected 1152 DeepSeek repeat slots, got {len(api)}')
    old_first = {r['observation_id']: r for r in raw
                 if r['method'] == 'deepseek_flash' and r['repeat_index'] == '1'}
    if len(old_first) != 384: raise ValueError('missing frozen DeepSeek first repeats')
    for r in api:
        if r['repeat_index'] == '1' and (r['prediction'], r['valid']) != (
                old_first[r['observation_id']]['prediction'], old_first[r['observation_id']]['valid']):
            raise ValueError('frozen DeepSeek first repeat changed')
    raw = [r for r in raw if r['method'] != 'deepseek_flash' or r['repeat_index'] == '1']
    # ExtraTrees output is always turned into a valid profile (unchanged if already valid),
    # and its errors are recomputed from the truth.
    truth = read_json(out/'TRUTH.json')
    for r in raw:
        if r['method'] == 'et' and r['prediction']:
            fixed = valid_profile(json.loads(r['prediction']))
            r.update(prediction=json.dumps(fixed), valid='True', validation_reason='',
                     **errors(fixed, truth[r['source']]))
    for r in api:
        if r['repeat_index'] == '1': continue
        base = by_obs[r['observation_id']]
        raw.append({'observation_id': r['observation_id'], 'source': r['source'],
                    'family': base['family'], 'group': r['stratum'], 'arm': r['arm'],
                    'sample_index': r['sample_index'], 'method': 'deepseek_flash', 'replicate': '',
                    'repeat_index': r['repeat_index'], 'prediction': r['prediction'],
                    'valid': r['valid'], 'validation_reason': r['validation_reason'],
                    'truth_rho2': base['truth_rho2'], 'AE2': r['AE2'],
                    'ProfileAE': r['ProfileAE'], 'signed_rho2': r['signed_rho2']})
    columns = [k for k in raw[0] if k != 'origin']
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator='\n')
        writer.writeheader()
        writer.writerows({k: r.get(k, '') for k in columns} for r in raw)
    rows = []
    for source in raw:
        r = source.copy()
        r['prediction'] = json.loads(r['prediction']) if r['prediction'] else None
        r['valid'] = r['valid'] == 'True'
        r['replicate'] = int(r['replicate']) if r['replicate'] else None
        for key in ('truth_rho2', 'AE2', 'ProfileAE', 'signed_rho2'):
            r[key] = float(r[key]) if r[key] else None
        rows.append(r)
    main = [r for r in rows if r['replicate'] in (None, 0)]
    table, infer, fam, per_src = summary(main), paired_methods(main), paired_families(main), per_source(main)
    write_csv(out/'SUMMARY.csv', table); write_csv(out/'PAIRED_METHODS.csv', infer)
    write_csv(out/'PAIRED_FAMILIES.csv', fam); write_csv(out/'PER_SOURCE.csv', per_src)
    observations = Path(observations or Path.home()/'.local/share/masterthesis/api_observations')
    blocks = list(observations.glob('*.json'))
    design = s_design_appendix(blocks, main)
    write_csv(out/'S_DESIGN_APPENDIX.csv', design)
    (out/'MAIN_RESULTS.md').write_text(main_markdown(table, infer, fam, per_src, design))
    training, sampling, response = variability(rows, 10)
    write_csv(out/'VARIABILITY_TRAINING.csv', training)
    write_csv(out/'VARIABILITY_SAMPLING.csv', sampling)
    write_csv(out/'VARIABILITY_RESPONSE.csv', response)
    (out/'VARIABILITY.md').write_text(variability_markdown(training, sampling, response, 10,
                                                            fixed_input_variability(rows)))
    losses, errs = history_summary(out/'HISTORY.csv')
    write_csv(out/'HISTORY_LOSSES.csv', losses); write_csv(out/'HISTORY_ERRORS.csv', errs)
    (out/'HISTORY.md').write_text(history_markdown(losses, errs))
    (out/'WALK.md').write_text(walk_markdown(list(csv.DictReader((out/'WALK.csv').open()))))
    return rows
