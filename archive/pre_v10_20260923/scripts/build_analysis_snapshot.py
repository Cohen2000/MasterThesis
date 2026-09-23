#!/usr/bin/env python3
"""Analysis-ready snapshot across the full coverage grid (Main = 0.10 plus the
six budget-sensitivity levels) for the five main methods -- plugin, shared_mle,
extratrees (mixed-budget), qwen_thinking, qwen_nonthinking -- plus the Main
mechanism-aware secondary reference and shared-MLE fit/adequacy diagnostics.

Reuses already-computed files; refits nothing and runs no inference:
  plugin            computed on the fly from each level's observations/sample
  shared_mle        results/panel888_shared_mle/main_and_budget_observations.csv
  extratrees        results/panel888_et_mixed/evaluation.csv (build_mixed_budget_et.py)
  qwen_*            results/panel888_shared_mle_main_quicklook/qwen_predictions.csv (Main),
                    results/panel888_budget_sensitivity/<b>/qwen_predictions.csv (budgets)
  secondary ref     results/panel888_shared_mle_main_quicklook/predictions.csv (Main only)
Accuracy is the equal-graph mean (each real source / surrogate / synthetic
instance counts once within its block); LLM accuracy uses formally valid
answers only. A method whose source file is missing is omitted and flagged.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.baselines import plugin
from main_experiment.common import BUDGET_GRID, BUDGET_SENSITIVITY, COVERAGE_FRACTION, PREPARED, ROOT, write_csv
from main_experiment.observation import parse
from main_experiment.references import mean_and_se, per_graph_mean

QUICKLOOK = ROOT/'results/panel888_shared_mle_main_quicklook'
SHARED_MLE = ROOT/'results/panel888_shared_mle'
ET_EVAL = ROOT/'results/panel888_et_mixed/evaluation.csv'
OUT = ROOT/'results/panel888_analysis_snapshot'
METHODS = ('plugin', 'shared_mle', 'extratrees', 'qwen_thinking', 'qwen_nonthinking')
ARM_ORDER = ('R', 'S1', 'H', 'B')
STRATA = ('real', 'surrogate', 'synthetic')


def level_dir(fraction):
    return PREPARED if fraction == COVERAGE_FRACTION else BUDGET_SENSITIVITY/f'b{round(fraction*1000):03d}'


def qwen_file(fraction):
    return QUICKLOOK/'qwen_predictions.csv' if fraction == COVERAGE_FRACTION else level_dir(fraction)/'qwen_predictions.csv'


def read_csv(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else []


def error_record(prediction, truth):
    e = np.asarray(prediction, float)-np.asarray(truth, float)
    return {'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0])}


def load():
    """(scored rows, qwen attempt rows, meta) over the whole grid."""
    rows = []; attempts = []; meta = {}
    for fraction in BUDGET_GRID:
        for p in sorted((level_dir(fraction)/'observations/sample').glob('*.json')):
            r = json.loads(p.read_text()); o = parse(r['block'])
            meta[(fraction, r['graph_id'])] = (r['stratum'], r['source_family'])
            if o['D_obs'] == 0: continue
            rows.append({'budget': fraction, 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                         'source_family': r['source_family'], 'method': 'plugin', **error_record(plugin(o), r['truth'])})
        for r in read_csv(qwen_file(fraction)):
            attempts.append({'budget': fraction, 'arm': r['arm'], 'stratum': r['stratum'], 'method': r['config'],
                             'valid': r['valid'] == 'True'})
            if r['valid'] != 'True': continue
            rows.append({'budget': fraction, 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                         'source_family': r['source_family'], 'method': r['config'],
                         'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])})
    for r in read_csv(SHARED_MLE/'main_and_budget_observations.csv'):
        b = float(r['budget']); stratum, source = meta[(b, r['graph_id'])]
        rows.append({'budget': b, 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': stratum, 'source_family': source,
                     'method': 'shared_mle', 'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']),
                     'signed_rho2': float(r['signed_rho2']), 'fallback_used': r['fallback_used'] == 'True'})
    for r in read_csv(ET_EVAL):
        rows.append({'budget': float(r['budget']), 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                     'source_family': r['source_family'], 'method': 'extratrees', 'AE2': float(r['AE2']),
                     'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2']),
                     'et_valid': r['in_unit_interval'] == 'True' and r['monotone'] == 'True'})
    return rows, attempts


def summarize(rows, attempts, group_fields, methods=METHODS):
    table = []
    keys = sorted({tuple(r[f] for f in group_fields) for r in rows},
                  key=lambda k: tuple(ARM_ORDER.index(v) if v in ARM_ORDER else v for v in k))
    for key in keys:
        for method in methods:
            chosen = [r for r in rows if r['method'] == method and all(r[f] == v for f, v in zip(group_fields, key))]
            if not chosen: continue
            ae = per_graph_mean(chosen, 'AE2'); pe = per_graph_mean(chosen, 'ProfileAE'); se = per_graph_mean(chosen, 'signed_rho2')
            mae2, mae2_se = mean_and_se(ae.values()); pmae, _ = mean_and_se(pe.values()); signed, _ = mean_and_se(se.values())
            row = dict(zip(group_fields, key))
            row.update(method=method, graphs=len(ae), n_evaluable=len(chosen), MAE2=mae2, MAE2_se=mae2_se,
                       ProfileMAE=pmae, signed_rho2=signed)
            if method.startswith('qwen'):
                att = [a for a in attempts if a['method'] == method and all(a[f] == v for f, v in zip(group_fields, key))]
                row.update(n_planned=len(att), validity=sum(a['valid'] for a in att)/len(att) if att else None)
            if method == 'shared_mle':
                row.update(fallback_count=sum(r['fallback_used'] for r in chosen),
                           fallback_rate=sum(r['fallback_used'] for r in chosen)/len(chosen))
            if method == 'extratrees':
                row.update(valid_profile_rate=sum(r['et_valid'] for r in chosen)/len(chosen))
            table.append(row)
    return table


def pivot(table, row_fields, value='MAE2'):
    """Markdown table: one line per row_fields key, one column per main method."""
    head = '| '+' | '.join(row_fields+list(METHODS))+' |'
    lines = [head, '|'+'---|'*(len(row_fields)+len(METHODS))]
    keys = []
    for r in table:
        k = tuple(r[f] for f in row_fields)
        if k not in keys: keys.append(k)
    for k in keys:
        cells = {r['method']: r[value] for r in table if tuple(r[f] for f in row_fields) == k}
        fmt = [f'{v:.3f}' if f == 'budget' else str(v) for f, v in zip(row_fields, k)]
        lines.append('| '+' | '.join(fmt+[f'{cells[m]:.4f}' if m in cells else '—' for m in METHODS])+' |')
    return lines


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, attempts = load()
    main_rows = [r for r in rows if r['budget'] == COVERAGE_FRACTION]
    main_att = [a for a in attempts if a['budget'] == COVERAGE_FRACTION]
    by_block = {s: summarize([r for r in main_rows if r['stratum'] == s], [a for a in main_att if a['stratum'] == s], ('arm',))
                for s in STRATA}
    write_csv(OUT/'MAIN_METHOD_COMPARISON.csv', summarize(main_rows, main_att, ('stratum', 'arm')))
    write_csv(OUT/'MAIN_REAL_BY_SOURCE.csv',
              summarize([r for r in main_rows if r['stratum'] == 'real'], [], ('source_family', 'arm')))
    write_csv(OUT/'SURROGATE_COMPARISON.csv', by_block['surrogate'])
    write_csv(OUT/'SYNTHETIC_COMPARISON.csv', by_block['synthetic'])
    budget = summarize(rows, attempts, ('stratum', 'budget', 'arm'))
    write_csv(OUT/'BUDGET_METHOD_COMPARISON.csv', budget)
    budget_real = [r for r in budget if r['stratum'] == 'real']

    secondary = [{'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'], 'method': 'mechanism_aware_reference',
                  'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])}
                 for r in read_csv(QUICKLOOK/'predictions.csv') if r['method'] == 'mechanism_aware_reference']
    write_csv(OUT/'MAIN_SECONDARY_REFERENCE.csv', summarize(secondary, [], ('stratum', 'arm'), ('mechanism_aware_reference',)))

    smle = [r for r in read_csv(SHARED_MLE/'main_and_budget_observations.csv')]
    write_csv(OUT/'FIT_DIAGNOSTICS.csv', [{k: r[k] for k in ('budget', 'graph_id', 'arm', 'evidence_block', 'fit_status',
                                                            'fallback_used', 'objective')} for r in smle])
    adequacy = {name: read_csv(SHARED_MLE/f'adequacy_{name}.csv') for name in ('full_data', 'h_like', 'b_like')}

    et_ready = ET_EVAL.exists()
    lines = ['# Analysis snapshot: final v9, Main + coverage grid', '',
             'Five main methods: plugin, shared_mle, extratrees (mixed-budget), qwen_thinking, qwen_nonthinking.',
             'MAE2 = equal-graph mean absolute error of rho_2 (each real source counts once; surrogates and',
             'synthetic instances are separate blocks, never extra real sources). LLM: formally valid answers only.',
             'The mechanism-aware secondary reference is in MAIN_SECONDARY_REFERENCE.csv, not among the five.', '']
    if not et_ready: lines += ['**extratrees missing: results/panel888_et_mixed/evaluation.csv not found.**', '']
    for s, title in (('real', 'A) Main, real sources'), ('surrogate', 'C) Main, surrogates'), ('synthetic', 'D) Main, synthetic')):
        lines += [f'## {title} — MAE2', ''] + pivot(by_block[s], ['arm']) + ['']
        lines += [f'ProfileMAE:', ''] + pivot(by_block[s], ['arm'], 'ProfileMAE') + ['']
    lines += ['## B) Coverage grid, real sources — MAE2', ''] + pivot(budget_real, ['budget', 'arm']) + ['']
    lines += ['## LLM validity (share of formally valid answers), all blocks', '',
              '| coverage | qwen_thinking | qwen_nonthinking |', '|---|---|---|']
    for b in BUDGET_GRID:
        v = {m: [a['valid'] for a in attempts if a['budget'] == b and a['method'] == m] for m in ('qwen_thinking', 'qwen_nonthinking')}
        lines.append(f"| {b:.3f} | "+' | '.join(f'{np.mean(v[m]):.4f} ({len(v[m])-sum(v[m])} invalid)' for m in v)+' |')
    lines += ['', '## E) shared_mle fallback and adequacy', '', '| coverage | observations | fallbacks | rate |', '|---|---|---|---|']
    for b in BUDGET_GRID:
        x = [r for r in smle if float(r['budget']) == b]; fb = sum(r['fallback_used'] == 'True' for r in x)
        lines.append(f'| {b:.3f} | {len(x)} | {fb} | {fb/len(x):.3f} |')
    lines += ['', 'Adequacy (development/training material only; model not changed on these numbers):', '']
    for name, recs in adequacy.items():
        if recs:
            fb = sum(r['fallback_used'] == 'True' for r in recs)
            lines.append(f"- {name}: n={len(recs)}, MAE2={np.mean([float(r['AE2']) for r in recs]):.4f}, fallback {fb}/{len(recs)}")
    if et_ready:
        et = [r for r in rows if r['method'] == 'extratrees']
        lines += ['', '## F) extratrees profile validity (prediction in [0,1] and non-increasing)', '',
                  '| coverage | predictions | valid profiles | rate |', '|---|---|---|---|']
        for b in BUDGET_GRID:
            x = [r for r in et if r['budget'] == b]; ok = sum(r['et_valid'] for r in x)
            lines.append(f'| {b:.3f} | {len(x)} | {ok} | {ok/len(x):.3f} |')
    (OUT/'ANALYSIS_SNAPSHOT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
