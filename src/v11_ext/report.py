"""Final result tables: 12 real sources, 12 paired surrogates, 8 synthetic graphs, arms R/S/H/B.

Four estimators per sampler (plugin, median, MLE, ExtraTrees) plus the LLMs. The reference
is plugin for R and the MLE otherwise. Real sources are the main analysis (all twelve,
equally weighted); surrogates and synthetic graphs are separate blocks; original-surrogate
contrasts are paired by source family. Paid API answers exist for the v11 graphs only and
are marked pending wherever sources are missing. S_obs is historical and not reported.
"""
import csv
import itertools
import json
import math
import subprocess
from collections import Counter, defaultdict
from statistics import median
import numpy as np
from main_experiment.common import ROOT, read_json, write_csv, write_json
from .core import CFG, MLE_ANCHOR_ARMS, NEW_SOURCES, RH, V10, WORK
from .panel12 import ARMS4, QWEN_SUR_DIR, SURROGATES, family
from .qwen import QWEN_DIR, answers as qwen_answers

V11 = ROOT/'docs/results/panel888_v11_main_20260923'
API = ROOT/'docs/results/api_v11_20260923'
OFFLINE = ('plugin', 'median', 'mle', 'et')
QWEN = ('qwen_thinking', 'qwen_nonthinking')
APIM = ('deepseek_flash', 'gpt_6_sol', 'gpt_6_sol_tools')
METHODS = OFFLINE+QWEN+APIM
REF_METHOD = {'R': 'plugin', 'S': 'mle', 'H': 'mle', 'B': 'mle'}
GROUPS = ('real', 'surrogate', 'synthetic')
TITLE = {'real': 'Real sources (main analysis, 12 sources)', 'surrogate': 'Surrogates (12, separate block)',
         'synthetic': 'Synthetic graphs (8, separate block)'}


def errors(p, truth):
    if p is None: return {'AE2': None, 'ProfileAE': None, 'signed_rho2': None}
    d = np.asarray(p, float)-np.asarray(truth, float)
    return {'AE2': float(abs(d[0])), 'ProfileAE': float(np.abs(d).mean()), 'signed_rho2': float(d[0])}


def fmt(x, digits=4):
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else f'{x:.{digits}f}'


def md_table(header, rows):
    return ['| '+' | '.join(header)+' |', '|'+'|'.join('---' if i < 2 else '---:' for i in range(len(header)))+'|',
            *('| '+' | '.join(map(str, r))+' |' for r in rows)]


# ---------------------------------------------------------------- collection
def observations(inputs):
    obs = {}
    for folder in (V10/'prepared/observations/sample', RH/'mainexp/run/observations/sample'):
        for p in folder.glob('*.json'):
            r = read_json(p); obs[r['id']] = r
    for parent in NEW_SOURCES:
        for name in (f'source:{parent}', f'surrogate:{parent}'):
            for p in (inputs[name]/'observations').glob('*.json'):
                r = read_json(p); obs[r['id']] = r
    return {i: {'id': i, 'source': r['graph_id'], 'group': r['stratum'], 'arm': r['arm'],
                'sample_index': r['sample_index'], 'truth': r['truth']}
            for i, r in obs.items() if r['arm'] in ARMS4 and
            (r['arm'] not in ('R', 'H') or '-panel-release__' in i)}     # R/H: the released draws only


def row(o, method, p, replicate=None, repeat=None, valid=None, origin='', reason=''):
    return {'observation_id': o['id'], 'source': o['source'], 'family': family(o['source']), 'group': o['group'],
            'arm': o['arm'], 'sample_index': o['sample_index'], 'method': method, 'replicate': replicate,
            'repeat_index': repeat, 'prediction': p, 'valid': p is not None if valid is None else valid,
            'validation_reason': reason, 'origin': origin, 'truth_rho2': o['truth'][0], **errors(p, o['truth'])}


def collect(inputs, replicates):
    obs = observations(inputs)
    rows = []
    with (V11/'PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            if r['observation_id'] not in obs: continue
            if r['method'] in ('plugin', 'median', 'mle', *QWEN):
                p = json.loads(r['prediction']) if r['prediction'] else None
                rows.append(row(obs[r['observation_id']], r['method'], p,
                                repeat=int(r['repeat_index']) if r['repeat_index'] else None,
                                valid=r['valid'] == 'True', origin='v11', reason=r.get('validation_reason', '')))
            if r['method'] == 'et' and r['arm'] not in MLE_ANCHOR_ARMS:
                obs[r['observation_id']]['sealed_et'] = json.loads(r['prediction'])
    for parent in NEW_SOURCES:
        for name in (f'source:{parent}', f'surrogate:{parent}'):
            for x in read_json(inputs[name]/'offline.json'):
                if x['id'] in obs and x['method'] in ('plugin', 'median', 'mle'):
                    rows.append(row(obs[x['id']], x['method'], x['prediction'], origin='panel12'))
    for folder in (QWEN_DIR, QWEN_SUR_DIR):
        for a in qwen_answers(folder):
            if a['observation_id'] in obs:
                if a['status'] != 'completed': raise RuntimeError(f'Qwen answer missing: {a["id"]}')
                rows.append(row(obs[a['observation_id']], a['method'], a['prediction'], repeat=a['repeat_index'],
                                valid=a['valid'], origin='panel12', reason=a['validation_reason']))
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
    if diff != 0.0: raise AssertionError(f'R/H/B replicate 0 differs from v11: {diff}')
    for (k, i), p in et.items(): rows.append(row(obs[i], 'et', p, replicate=k, origin='panel12'))
    # The joint API evaluation (v11 + extension runs) supersedes the v11-only file once it exists.
    joint = ROOT/'docs/results/final_20260928/api/API_PREDICTIONS.csv'
    with (joint if joint.exists() else API/'API_PREDICTIONS.csv').open() as f:
        for r in csv.DictReader(f):
            p = json.loads(r['prediction']) if r['prediction'] else None
            rows.append(row(obs[r['observation_id']], r['method'], p, repeat=int(r['repeat_index']),
                            valid=r['valid'] == 'True', origin='api_v11', reason=r['validation_reason']))
    return rows, obs, outputs


# ---------------------------------------------------------------- accuracy
def by_source(rows, field='AE2'):
    out = defaultdict(list)
    for r in rows:
        if r[field] is not None: out[r['source']].append(r[field])
    return {s: float(np.mean(v)) for s, v in out.items()}


def summary(rows):
    sources = {g: sorted({r['source'] for r in rows if r['group'] == g}) for g in GROUPS}
    table = []
    for g in GROUPS:
        for arm in ARMS4:
            for m in METHODS:
                sel = [r for r in rows if r['group'] == g and r['arm'] == arm and r['method'] == m]
                ae = by_source(sel)
                complete = bool(ae) and set(ae) == set(sources[g])
                entry = {'group': g, 'arm': arm, 'method': m, 'reference': m == REF_METHOD[arm],
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


def signflip(values):
    a = np.asarray(values, float)
    observed = abs(a.mean())
    return float(np.mean([abs(np.mean(a*np.asarray(s))) >= observed-1e-15
                          for s in itertools.product((-1, 1), repeat=len(a))]))


def paired_methods(rows):
    """Within a block: first minus second source-level MAE_2 (both complete on the same sources)."""
    out = []
    for g in GROUPS:
        for arm in ARMS4:
            ref = REF_METHOD[arm]
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


def paired_families(rows):
    """Original minus surrogate source-level MAE_2, one pair per source family."""
    out = []
    for arm in ARMS4:
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


def per_source(rows):
    grouped = defaultdict(list)
    for r in rows: grouped[r['group'], r['source'], r['arm'], r['method']].append(r)
    out = []
    for (g, s, arm, m), sel in sorted(grouped.items()):
        valid = [r for r in sel if r['AE2'] is not None]
        out.append({'group': g, 'source': s, 'arm': arm, 'method': m, 'answers': len(sel), 'valid': len(valid),
                    'MAE_2': float(np.mean([r['AE2'] for r in valid])) if valid else None,
                    'ProfileMAE': float(np.mean([r['ProfileAE'] for r in valid])) if valid else None,
                    'signed_rho_2': float(np.mean([r['signed_rho2'] for r in valid])) if valid else None})
    return out


# ---------------------------------------------------------------- variability
def sd(values):
    return float(np.std(np.asarray(values, float), ddof=1))


def variability(rows, replicates):
    """Training (ET fits), sampling (sampler draws) and LLM answer variability, kept apart."""
    training, sampling, response = [], [], []
    for g in GROUPS:
        n_graphs = len({r['source'] for r in rows if r['group'] == g})
        for arm in ARMS4:
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
                    if (r['group'], r['arm'], r['method']) == (g, arm, m) and r['valid'] and r['replicate'] in (None, 0):
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


# ---------------------------------------------------------------- leakage audit
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
                e = {gid: [float(np.mean(v[k])) for k in range(2, 6)] for gid, v in per_graph.items()}
                errs.append({'group': g, 'estimator': est, 'truncation': window, 'node_sampling': level == 'H_sampled',
                             'graphs': len(e), 'signed_rho2': float(np.mean([v[0] for v in e.values()])),
                             'MAE_2': float(np.mean([abs(v[0]) for v in e.values()])),
                             'ProfileMAE': float(np.mean([np.mean(np.abs(v)) for v in e.values()]))})
    return losses, errs


WALK_COLUMNS = ('graph_id', 'L', 'stationary_shift_rho2', 'plugin_rho2_bias', 'plugin_rho2_sd', 'design_S_rho2_bias',
                'design_S_rho2_sd', 'mle_S_rho2_bias', 'mle_S_rho2_sd', 'mle_S_failures', 'gate_applicable', 'gate_pass',
                'weight_ess_mean', 'ratio_ess_mean', 'revisit_rate_mean', 'distinct_dyads_mean', 'components',
                'largest_component_cell_share', 'walk_cell_share_of_start_component', 'communities',
                'communities_touched_share', 'walk_cell_share_of_touched_communities')


# ---------------------------------------------------------------- markdown
def main_markdown(table, infer, fam, per_src):
    lines = ['# Main results', '',
             'Arms R, S, H and B; estimators plugin, median, MLE and ExtraTrees (ET, production fit), plus Qwen and',
             'the paid APIs. Reference (ref.): plugin for R, MLE for S, H and B. Equal-source MAE_2 over valid',
             'answers is primary, ProfileMAE secondary; nothing is clipped or repaired. The twelve real sources are',
             'the main analysis and are weighted equally. A method missing any source of a block is marked',
             '**pending** and not ranked. S_obs is a historical ablation and is not reported here.', '']
    for g in GROUPS:
        lines += [f'## {TITLE[g]}', '']
        lines += md_table(['Arm', 'Method', 'MAE_2', 'ProfileMAE', 'Signed rho_2', 'Validity', 'Sources'],
                          [[r['arm'], r['method']+(' (ref.)' if r['reference'] else ''), fmt(r['MAE_2']),
                            fmt(r['ProfileMAE']), fmt(r['signed_rho_2']), fmt(r['validity'], 3), r['status']]
                           for r in table if r['group'] == g])+['']
    lines += ['## Paired original minus surrogate (12 source families)', '',
              'Source-level MAE_2 of the original minus that of its surrogate; each family is one pair.', '']
    lines += md_table(['Arm', 'Method', 'Original', 'Surrogate', 'Mean difference', 'Original better', 'Sign-flip p'],
                      [[r['arm'], r['method'], fmt(r['original_MAE_2']), fmt(r['surrogate_MAE_2']),
                        fmt(r['mean_difference']), f"{r['original_better']}/{r['families']}", fmt(r['exact_signflip_p'])]
                       for r in fam])+['']
    lines += ['## Paired methods within a block', '',
              'First minus second source-level MAE_2 (negative: first better); exact sign-flip over sources.', '']
    lines += md_table(['Group', 'Arm', 'Comparison', 'Mean difference', 'First better', 'Sign-flip p'],
                      [[r['group'], r['arm'], r['comparison'], fmt(r['mean_difference']),
                        f"{r['first_better']}/{r['sources']}", fmt(r['exact_signflip_p'])] for r in infer])+['']
    index = {(r['source'], r['arm'], r['method']): r['MAE_2'] for r in per_src}
    reals = sorted({r['source'] for r in per_src if r['group'] == 'real'})
    lines += ['## Real sources (MAE_2 per source)', '']
    lines += md_table(['Source', 'Arm', *OFFLINE, *QWEN],
                      [[s, arm, *(fmt(index.get((s, arm, m))) for m in OFFLINE+QWEN)] for s in reals for arm in ARMS4])
    return '\n'.join(lines+['', 'All groups, API methods and metrics: `PER_SOURCE.csv`.', ''])


def variability_markdown(training, sampling, response, replicates):
    lines = ['# Variability', '',
             'Three different sources of variation, reported separately and not compared with one another',
             f'(different quantities and repetition counts: {replicates+1} ET fits, 3 sampler draws, 3 LLM answers).', '',
             f'## Training variability (ExtraTrees, {replicates+1} fits)', '',
             'Fit 0 is the production fit; fits 1-10 redraw the real training and pool observations, reseed the',
             'forests and rerun the nested selection. Test observations are fixed.', '']
    lines += md_table(['Group', 'Arm', 'MAE_2 mean', 'MAE_2 SD', 'MAE_2 min', 'MAE_2 max', 'Median obs. SD'],
                      [[r['group'], r['arm'], fmt(r['ET_MAE_2_mean']), fmt(r['ET_MAE_2_SD']), fmt(r['ET_MAE_2_min']),
                        fmt(r['ET_MAE_2_max']), fmt(r['median_observation_SD_rho2'])] for r in training])
    lines += ['', '## Sampling variability', '',
              'SD of the rho_2 estimate across the three sampler draws of a graph (LLM: mean of the valid answers per',
              'draw; ET: production fit), median over graphs. Plugin, median and MLE are deterministic given an',
              'observation; their uncertainty is this sampling variability, which is not zero. Graphs with a',
              'saturated (single-draw) H panel are excluded. API methods: v11 graphs only.', '']
    lines += md_table(['Group', 'Arm', 'Method', 'Graphs', 'Median SD across draws'],
                      [[r['group'], r['arm'], r['method'], f"{r['graphs']}/{r['graphs_in_block']}",
                        fmt(r['median_graph_SD_rho2_across_draws'])] for r in sampling])
    lines += ['', '## LLM answer variability', '',
              'SD of the rho_2 answer across the three repeats of one observation, median over observations with',
              'three valid answers. DeepSeek has one repeat.', '']
    lines += md_table(['Group', 'Arm', 'Method', 'Observations', 'Median SD across repeats'],
                      [[r['group'], r['arm'], r['method'], r['observations_with_3_valid'],
                        fmt(r['median_observation_SD_rho2'])] for r in response])
    return '\n'.join(lines+[''])


def history_markdown(losses, errs):
    lines = ['# History truncation (arm H)', '',
             'Population level: every dyad of the graph, no node sampling. K counts active windows among all five,',
             'J among the three retained ones (last 60% = windows 3-5, first 60% = windows 1-3). For k = 2..5 the',
             'numerator loss is 1 - #(J>=k)/#(K>=k) and the denominator loss 1 - #(J>=1)/#(K>=1); the truncated',
             'plugin equals rho_k (1 - numerator loss)/(1 - denominator loss), so the losses cancel only when',
             'they are equal. With three windows J >= 4 is impossible, so the truncated plugin is 0 for k = 4, 5.',
             '"H sampled" uses the actual H draws (last 60% plus the node panel): truncation plus node sampling.',
             'Equal-graph means over the 12 real sources and the 12 surrogates.', '', '## Losses', '']
    lines += md_table(['Group', 'Window', 'k', 'Numerator loss', 'Denominator loss', 'Plugin signed error',
                       '|error| <= 0.01'],
                      [[r['group'], r['window'], r['k'], fmt(r['numerator_loss']), fmt(r['denominator_loss']),
                        fmt(r['plugin_signed_error']), f"{r['graphs_abs_error_le_0.01']}/{r['graphs']}"] for r in losses])
    lines += ['', '## Errors', '']
    lines += md_table(['Group', 'Estimator', 'Truncation', 'Node sampling', 'Signed rho_2', 'MAE_2', 'ProfileMAE'],
                      [[r['group'], r['estimator'], r['truncation'], 'yes' if r['node_sampling'] else 'no',
                        fmt(r['signed_rho2']), fmt(r['MAE_2']), fmt(r['ProfileMAE'])] for r in errs])
    return '\n'.join(lines+['', 'Per graph and k: `HISTORY.csv`.', ''])


def walk_markdown(walks):
    lines = ['# Walk diagnostics of the added graphs', '',
             'The unchanged 1000-walk audit of `scripts/audit_v10_walk.py` (same seeds, calibration and gate rule:',
             'applicable when |stationary shift| > 0.05; pass when the design ratio removes 90% of the plugin bias',
             'and halves its RMSE) on the four added originals and their surrogates. Bias and SD of rho_2 are',
             'reported separately for plugin, the design ratio (gate criterion only) and the MLE (S reference).',
             'ESS: weight ESS of the traversal weights and ratio ESS. Coverage: discovered cells per cell of the',
             'start component, and per cell of the Louvain communities touched (a dyad counts for the community',
             'of its first endpoint). MLE failures: walks whose histogram the production MLE rejects (e.g. only',
             'dyads active in all five windows discovered); MLE bias and SD use the remaining walks. The 24 v11 graphs: `docs/results/panel888_v10_walk_gate_20260923`.', '']
    lines += md_table(['Graph', 'L', 'Shift', 'Plugin bias', 'Plugin SD', 'Design bias', 'Design SD', 'MLE bias',
                       'MLE SD', 'MLE failures', 'Gate appl.', 'Gate pass', 'Weight ESS', 'Ratio ESS', 'Revisit rate', 'Distinct dyads',
                       'Components', 'Largest comp.', 'Comp. coverage', 'Communities', 'Comm. touched', 'Comm. coverage'],
                      [[w['graph_id'], w['L'], fmt(w['stationary_shift_rho2']), fmt(w['plugin_rho2_bias']),
                        fmt(w['plugin_rho2_sd']), fmt(w['design_S_rho2_bias']), fmt(w['design_S_rho2_sd']),
                        fmt(w['mle_S_rho2_bias']), fmt(w['mle_S_rho2_sd']), f"{w['mle_S_failures']}/1000", w['gate_applicable'], w['gate_pass'],
                        fmt(w['weight_ess_mean'], 1), fmt(w['ratio_ess_mean'], 1), fmt(w['revisit_rate_mean'], 3),
                        fmt(w['distinct_dyads_mean'], 1), w['components'], fmt(w['largest_component_cell_share'], 3),
                        fmt(w['walk_cell_share_of_start_component'], 3), w['communities'],
                        fmt(w['communities_touched_share'], 3), fmt(w['walk_cell_share_of_touched_communities'], 3)]
                       for w in walks])
    return '\n'.join(lines+[''])


# ---------------------------------------------------------------- wall times
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
    (out/'MAIN_RESULTS.md').write_text(main_markdown(table, infer, fam, per_src))
    training, sampling, response = variability(rows, replicates)
    write_csv(out/'VARIABILITY_TRAINING.csv', training); write_csv(out/'VARIABILITY_SAMPLING.csv', sampling)
    write_csv(out/'VARIABILITY_RESPONSE.csv', response)
    (out/'VARIABILITY.md').write_text(variability_markdown(training, sampling, response, replicates))
    losses, errs = history_summary(inputs['history']/'HISTORY.csv')
    (out/'HISTORY.csv').write_text((inputs['history']/'HISTORY.csv').read_text())
    write_csv(out/'HISTORY_LOSSES.csv', losses); write_csv(out/'HISTORY_ERRORS.csv', errs)
    (out/'HISTORY.md').write_text(history_markdown(losses, errs))
    walks = [read_json(inputs[f'walkdiag:{k}']/'walk.json') for k in (*NEW_SOURCES, *SURROGATES)]
    write_csv(out/'WALK.csv', walks)
    (out/'WALK.md').write_text(walk_markdown([{c: w[c] for c in WALK_COLUMNS} for w in walks]))
    leak = leakage(outputs, obs)
    write_csv(out/'ET_CHOICES.csv', sorted(({'replicate': d['replicate'], 'arm': d['arm'], 'fold': d['fold'],
                                             'anchor': d['choice']['anchor'],
                                             'min_samples_leaf': d['choice']['min_samples_leaf'],
                                             'max_features': d['choice']['max_features'],
                                             'inner_MAE_2': d['choice']['mean_inner_source_MAE2']}
                                            for n, d in outputs if n.startswith('train:')),
                                           key=lambda r: (r['replicate'], r['arm'], r['fold'])))
    graphs = [read_json(inputs[f'{s}:{p}']/'summary.json') for p in NEW_SOURCES for s in ('source', 'surrogate')]
    write_csv(out/'ADDED_GRAPHS.csv', [{'graph': g['source'], 'N': g['N'], 'D': g['D'], 'M': g['M'],
                                        **{f'rho_{k}': g['truth'][k-2] for k in range(2, 6)},
                                        'budget_matched_by_arm': json.dumps({a: g['budget_matched_by_arm'][a] for a in ARMS4}),
                                        'h_saturated': g['h_saturated'], 'n_panel': g['n_panel'],
                                        'n_panel_history': g['n_panel_history'], 'L': g['L'], 'p': g['p']} for g in graphs])
    freeze = read_json(inputs['api_freeze']/'API_FREEZE_EXT.json')
    write_json(out/'API_FREEZE_EXT.json', freeze)
    write_csv(out/'WALL_TIMES.csv', wall_times())
    qwen_new = [r for r in rows if r['method'] in QWEN and r['origin'] == 'panel12']
    write_json(out/'REPORT.json', {
        'version': CFG['version'], 'arms': list(ARMS4), 'replicates': replicates,
        'blocks': {g: sorted({r['source'] for r in rows if r['group'] == g}) for g in GROUPS},
        'leakage': leak, 'r_h_b_replicate0_equals_v11': True,
        'train_sur_models': Counter(d['check']['model'] for n, d in outputs if n.startswith('train_sur:')),
        'qwen_added_answers': len(qwen_new), 'qwen_added_valid': sum(r['valid'] for r in qwen_new),
        'qwen_added_invalid_reasons': Counter(r['validation_reason'] for r in qwen_new if not r['valid']),
        'api_pending_observations': freeze['observations'], 'rows': len(rows)})
    print('REPORT', len(rows), 'rows', flush=True)
