#!/usr/bin/env python3
"""Tables of the H history-fraction ablation (local, no refitting of anything
but the closed-form/shared-MLE estimators on the stored observations).

h=.60 is the existing final v9 H arm (sealed observations, mixed-budget ET,
Qwen answers); other h come from build_h_ablation.py (observations, H
ExtraTrees evaluation, feasibility) and, once collected, their Qwen answers
(results/panel888_h_ablation/h040/qwen_predictions.csv). Aggregation rules are
those of build_analysis_snapshot.summarize.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_analysis_snapshot import error_record, level_dir, qwen_file, read_csv, summarize   # noqa: E402
from main_experiment.baselines import h_extrapolator, plugin
from main_experiment.common import BUDGET_GRID, COVERAGE_FRACTION, ROOT, write_csv
from main_experiment.observation import parse
from main_experiment.shared_mle import fit

ABL = ROOT/'results/panel888_h_ablation'
OUT = ABL
METHODS = ('plugin', 'shared_mle', 'extratrees', 'qwen_thinking', 'qwen_nonthinking')
REF = ('mechanism_aware_reference',)
LEVELS = {.40: 'h040', .60: 'h060'}


def tests(h):
    for b in BUDGET_GRID:
        d = level_dir(b)/'observations/sample' if h == .60 else ABL/LEVELS[h]/'observations'/f'b{round(b*1000):03d}'
        for p in sorted(d.glob('*__H-*.json')):
            r = json.loads(p.read_text()); r['budget'] = b
            yield r


def load(h):
    rows = []; attempts = []
    for r in tests(h):
        o = parse(r['block'])
        if o['D_obs'] == 0: continue
        base = {'h': h, 'budget': r['budget'], 'graph_id': r['graph_id'], 'arm': 'H', 'stratum': r['stratum'],
                'source_family': r['source_family']}
        res = fit(o)
        rows += [{**base, 'method': 'plugin', **error_record(plugin(o), r['truth'])},
                 {**base, 'method': 'shared_mle', **error_record(res.rho, r['truth']), 'fallback_used': res.fallback_used},
                 {**base, 'method': 'mechanism_aware_reference', **error_record(h_extrapolator(o)['prediction'], r['truth'])}]
    et = ROOT/'results/panel888_et_mixed/evaluation.csv' if h == .60 else ABL/LEVELS[h]/'et/evaluation.csv'
    for r in read_csv(et):
        if r['arm'] != 'H': continue
        rows.append({'h': h, 'budget': float(r['budget']), 'graph_id': r['graph_id'], 'arm': 'H', 'stratum': r['stratum'],
                     'source_family': r['source_family'], 'method': 'extratrees', 'AE2': float(r['AE2']),
                     'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2']),
                     'et_valid': r['in_unit_interval'] == 'True' and r['monotone'] == 'True'})
    qfiles = [(b, qwen_file(b)) for b in BUDGET_GRID] if h == .60 else [(None, ABL/LEVELS[h]/'qwen_predictions.csv')]
    for b, path in qfiles:
        for r in read_csv(path):
            if r['arm'] != 'H': continue
            budget = b if b is not None else float(r['budget'])
            attempts.append({'h': h, 'budget': budget, 'arm': 'H', 'stratum': r['stratum'], 'method': r['config'],
                             'valid': r['valid'] == 'True'})
            if r['valid'] == 'True':
                rows.append({'h': h, 'budget': budget, 'graph_id': r['graph_id'], 'arm': 'H', 'stratum': r['stratum'],
                             'source_family': r['source_family'], 'method': r['config'], 'AE2': float(r['AE2']),
                             'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])})
    return rows, attempts


def pivot(table, keys, value, methods):
    lines = ['| '+' | '.join(keys+list(methods))+' |', '|'+'---|'*(len(keys)+len(methods))]
    seen = []
    for r in table:
        k = tuple(r[f] for f in keys)
        if k in seen: continue
        seen.append(k)
        cells = {x['method']: x[value] for x in table if tuple(x[f] for f in keys) == k}
        lines.append('| '+' | '.join([f'{v:.3f}' if isinstance(v, float) else str(v) for v in k] +
                                     [f'{cells[m]:+.4f}' if value == 'signed_rho2' and m in cells else
                                      f'{cells[m]:.4f}' if m in cells else '—' for m in methods])+' |')
    return lines


def main():
    rows = []; attempts = []
    for h in LEVELS:
        r, a = load(h); rows += r; attempts += a
    allm = METHODS+REF
    main_rows = [r for r in rows if r['budget'] == COVERAGE_FRACTION]
    main_att = [a for a in attempts if a['budget'] == COVERAGE_FRACTION]
    table_main = summarize(main_rows, main_att, ('stratum', 'h'), allm)
    table_budget = summarize(rows, attempts, ('stratum', 'budget', 'h'), allm)
    write_csv(OUT/'H_ABLATION_MAIN.csv', table_main)
    write_csv(OUT/'H_ABLATION_BUDGET.csv', table_budget)
    write_csv(OUT/'H_ABLATION_BY_SOURCE.csv',
              summarize([r for r in main_rows if r['stratum'] == 'real'], [], ('source_family', 'h'), allm))
    validity = summarize(rows, attempts, ('budget', 'h'), ('shared_mle', 'extratrees', 'qwen_thinking', 'qwen_nonthinking'))
    write_csv(OUT/'H_ABLATION_VALIDITY.csv', [{k: v for k, v in r.items() if k in
              ('budget', 'h', 'method', 'n_evaluable', 'n_planned', 'validity', 'fallback_count', 'fallback_rate',
               'valid_profile_rate')} for r in validity])
    feas = []
    for h, d in LEVELS.items():
        f = read_csv(ABL/d/'feasibility.csv')
        for b in BUDGET_GRID:
            x = [r for r in f if float(r['budget']) == b and r['graph_id'] in {q['graph_id'] for q in rows}]
            feas.append({'h': h, 'budget': b, 'graphs': len(x),
                         'target_unreachable': sum(r['h_target_unreachable'] == 'True' for r in x),
                         'outside_tolerance': sum(r['h_within_tolerance'] != 'True' for r in x),
                         'saturated_panel': sum(r['h_saturated'] == 'True' for r in x),
                         'unreachable_graphs': ';'.join(r['graph_id'] for r in x if r['h_target_unreachable'] == 'True')})
    write_csv(OUT/'H_ABLATION_FEASIBILITY.csv', feas)

    md = ['# H history-fraction ablation (arm H only; h=0.60 is the final v9 main arm)', '',
          'h=0.40 and higher coverage levels are stress tests, not design choices. h=0.50 is not run: its cutoff',
          'falls inside window 3, which the 0/1 Temporal_access contract cannot express (see CURRENT_STATE.md).',
          'MAE2 = equal-graph mean |error of rho_2|; LLM on formally valid answers only; the mechanism-aware',
          'reference (homogeneous ZT-Binomial on the visible windows) is shown separately.',
          'shared_mle at h=0.40 is NOT identified: with m=2 visible windows the zero-truncated Beta-Binomial has one',
          'free cell share but two parameters, so all starts reach the same likelihood with different W=5',
          'extrapolations (rho_2 spread up to ~0.13 on one observation); its h=0.40 numbers are an optimizer artefact.',
          'The model is used unchanged, as specified. Qwen h=0.40 columns fill in once its chain is archived.', '']
    for s, title in (('real', 'A) Main (coverage 0.10), real'), ('surrogate', 'C1) Main, surrogate'), ('synthetic', 'C2) Main, synthetic')):
        t = [r for r in table_main if r['stratum'] == s]
        for value in ('MAE2', 'ProfileMAE', 'signed_rho2'):
            md += [f'## {title} — {value}', ''] + pivot(t, ['h'], value, allm) + ['']
    md += ['## B) Coverage x h, real — MAE2', ''] + pivot([r for r in table_budget if r['stratum'] == 'real'], ['budget', 'h'], 'MAE2', allm) + ['']
    md += ['## D) Feasibility (24 main graphs per cell)', '', '| h | coverage | target unreachable | outside ±5% | saturated panel |',
           '|---|---|---|---|---|'] + [f"| {r['h']:.2f} | {r['budget']:.3f} | {r['target_unreachable']} | {r['outside_tolerance']} | {r['saturated_panel']} |" for r in feas] + ['']
    md += ['## E) Plugin signed rho_2 error (real / surrogate / synthetic)', '', '| coverage | h | real | surrogate | synthetic |', '|---|---|---|---|---|']
    for b in BUDGET_GRID:
        for h in LEVELS:
            v = {r['stratum']: r['signed_rho2'] for r in table_budget if r['method'] == 'plugin' and r['budget'] == b and r['h'] == h}
            md.append(f"| {b:.3f} | {h:.2f} | "+' | '.join(f'{v[s]:+.4f}' if s in v else '—' for s in ('real', 'surrogate', 'synthetic'))+' |')
    md += ['', '## Validity', '', '| coverage | h | method | evaluable | validity / fallback / valid-profile rate |', '|---|---|---|---|---|']
    for r in validity:
        rate = r.get('validity', r.get('fallback_rate', r.get('valid_profile_rate')))
        md.append(f"| {r['budget']:.3f} | {r['h']:.2f} | {r['method']} | {r['n_evaluable']} | {rate:.4f} |")
    (OUT/'H_ABLATION.md').write_text('\n'.join(md)+'\n')
    print('\n'.join(md))


if __name__ == '__main__':
    main()
