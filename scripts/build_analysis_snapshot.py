#!/usr/bin/env python3
"""Analysis-ready snapshot across the full coverage grid (Main + all six
budget-sensitivity levels): plugin, shared_mle, mixed-budget ExtraTrees (once
its evaluation file exists), Qwen thinking/non-thinking, plus the Main
mechanism-aware secondary reference and shared-MLE fit/adequacy diagnostics.

Reuses already-computed files; does not refit or re-run inference:
  - plugin: computed on the fly from each level's observations/sample (no
    trained model needed);
  - shared_mle: results/panel888_shared_mle/main_and_budget_observations.csv
    (scripts/build_shared_mle.py);
  - qwen_thinking/qwen_nonthinking: results/panel888_shared_mle_main_quicklook/
    qwen_predictions.csv (Main) and results/panel888_budget_sensitivity/<b>/
    qwen_predictions.csv (budgets), from scripts/build_qwen_main_quicklook.py
    on an already-collected responses.jsonl;
  - extratrees (mixed-budget): results/panel888_et_mixed_evaluation.csv
    (scripts/evaluate_mixed_budget_et.py) if it exists yet;
  - mechanism_aware_reference (secondary, Main only so far):
    results/panel888_shared_mle_main_quicklook/predictions.csv.
A column/table is simply omitted (and noted in ANALYSIS_SNAPSHOT.md) while its
source file does not exist yet.
"""
import csv
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.baselines import plugin
from main_experiment.common import BUDGET_GRID, COVERAGE_FRACTION, PREPARED, ROOT, BUDGET_SENSITIVITY, write_csv
from main_experiment.observation import parse
from main_experiment.references import mean_and_se, per_graph_mean

QUICKLOOK = ROOT/'results/panel888_shared_mle_main_quicklook'
SHARED_MLE = ROOT/'results/panel888_shared_mle'
ET_EVAL = ROOT/'results/panel888_et_mixed_evaluation.csv'
OUT = ROOT/'results/panel888_analysis_snapshot'
METHODS = ('plugin', 'shared_mle', 'extratrees', 'qwen_thinking', 'qwen_nonthinking')
METHOD_LABEL = {'mechanism_aware_reference': 'mechanism_aware_reference (secondary)'}


def budget_dir(fraction):
    return PREPARED if fraction == COVERAGE_FRACTION else BUDGET_SENSITIVITY/f'b{round(fraction*1000):03d}'


def read_csv(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else None


def error_record(prediction, truth):
    e = np.asarray(prediction, float)-np.asarray(truth, float)
    return {'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0])}


def plugin_rows(fraction):
    rows = []
    for p in sorted((budget_dir(fraction)/'observations/sample').glob('*.json')):
        row = json.loads(p.read_text())
        o = parse(row['block'])
        if o['D_obs'] == 0: continue
        rec = error_record(plugin(o), row['truth'])
        rows.append({'budget': fraction, 'graph_id': row['graph_id'], 'arm': row['arm'], 'stratum': row['stratum'],
                    'source_family': row['source_family'], 'method': 'plugin', **rec})
    return rows


def shared_mle_rows():
    rows = read_csv(SHARED_MLE/'main_and_budget_observations.csv') or []
    out = []
    for r in rows:
        out.append({'budget': float(r['budget']), 'graph_id': r['graph_id'], 'arm': r['arm'],
                    'stratum': r['evidence_block'], 'source_family': '', 'method': 'shared_mle',
                    'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2']),
                    'fallback_used': r['fallback_used'] == 'True', 'fit_status': r['fit_status']})
    # source_family/stratum for real graphs: recover from the observations we already parsed for plugin.
    return out


def qwen_rows(fraction):
    path = QUICKLOOK/'qwen_predictions.csv' if fraction == COVERAGE_FRACTION else budget_dir(fraction)/'qwen_predictions.csv'
    rows = read_csv(path) or []
    out = []
    for r in rows:
        if r['valid'] != 'True': continue
        out.append({'budget': fraction, 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                    'source_family': r.get('source_family', ''), 'method': r['config'],
                    'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])})
    return out


def qwen_validity(fraction, config):
    path = QUICKLOOK/'qwen_predictions.csv' if fraction == COVERAGE_FRACTION else budget_dir(fraction)/'qwen_predictions.csv'
    rows = read_csv(path)
    if not rows: return None
    chosen = [r for r in rows if r['config'] == config]
    return sum(r['valid'] == 'True' for r in chosen)/len(chosen) if chosen else None


def et_rows():
    rows = read_csv(ET_EVAL) or []
    out = []
    for r in rows:
        out.append({'budget': float(r['budget']), 'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                    'source_family': r.get('source_family', ''), 'method': 'extratrees',
                    'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])})
    return out


def load_all():
    rows = []
    graph_meta = {}   # graph_id -> (stratum, source_family), filled from plugin_rows (always available)
    for fraction in BUDGET_GRID:
        p = plugin_rows(fraction)
        rows += p
        for r in p: graph_meta[(fraction, r['graph_id'])] = (r['stratum'], r['source_family'])
        rows += qwen_rows(fraction)
    smle = shared_mle_rows()
    for r in smle:
        stratum, source_family = graph_meta.get((r['budget'], r['graph_id']), (r['stratum'], r['source_family']))
        r['stratum'] = stratum; r['source_family'] = source_family
    rows += smle
    rows += et_rows()
    return rows


def summarize(rows, group_fields, methods=METHODS):
    table = []
    groups = sorted({tuple(r[f] for f in group_fields) for r in rows if all(r[f] not in (None, '') for f in group_fields)})
    for group in groups:
        for method in methods:
            chosen = [r for r in rows if all(r[f] == g for f, g in zip(group_fields, group)) and r['method'] == method]
            if not chosen: continue
            ae = per_graph_mean(chosen, 'AE2'); pe = per_graph_mean(chosen, 'ProfileAE'); se = per_graph_mean(chosen, 'signed_rho2')
            mae2, mae2_se = mean_and_se(ae.values()); pmae, pmae_se = mean_and_se(pe.values())
            signed, signed_se = mean_and_se(se.values())
            fb = [r for r in chosen if 'fallback_used' in r]
            row = dict(zip(group_fields, group))
            row.update(method=METHOD_LABEL.get(method, method), graphs=len(ae), n=len(chosen),
                      MAE2=mae2, MAE2_se=mae2_se, ProfileMAE=pmae, ProfileMAE_se=pmae_se,
                      signed_rho2=signed, signed_rho2_se=signed_se)
            if fb: row.update(fallback_n=len(fb), fallback_rate=sum(r['fallback_used'] for r in fb)/len(fb))
            table.append(row)
    return table


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_all()
    main_rows = [r for r in rows if r['budget'] == COVERAGE_FRACTION]

    write_csv(OUT/'MAIN_METHOD_COMPARISON.csv', summarize(main_rows, ('arm',)))
    real_rows = [r for r in main_rows if r['stratum'] == 'real']
    write_csv(OUT/'MAIN_REAL_BY_SOURCE.csv', summarize(real_rows, ('source_family',)))
    write_csv(OUT/'SURROGATE_COMPARISON.csv', summarize([r for r in main_rows if r['stratum'] == 'surrogate'], ('arm',)))
    write_csv(OUT/'SYNTHETIC_COMPARISON.csv', summarize([r for r in main_rows if r['stratum'] == 'synthetic'], ('arm',)))
    write_csv(OUT/'BUDGET_METHOD_COMPARISON.csv', summarize(rows, ('budget', 'arm')))
    write_csv(OUT/'BUDGET_REAL_METHOD_COMPARISON.csv', summarize([r for r in rows if r['stratum'] == 'real'], ('budget', 'arm')))

    # Main's secondary mechanism-aware reference, kept out of the five main methods.
    secondary = read_csv(QUICKLOOK/'predictions.csv') or []
    secondary = [{'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'], 'source_family': r.get('source_family', ''),
                 'method': 'mechanism_aware_reference', 'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']),
                 'signed_rho2': float(r['signed_rho2'])} for r in secondary if r['method'] == 'mechanism_aware_reference']
    if secondary:
        write_csv(OUT/'MAIN_SECONDARY_REFERENCE.csv', summarize(secondary, ('arm',), methods=('mechanism_aware_reference',)))

    fit_diag = read_csv(QUICKLOOK/'fit_diagnostics.csv')
    if fit_diag: write_csv(OUT/'FIT_DIAGNOSTICS.csv', fit_diag)

    adequacy = read_csv(SHARED_MLE/'adequacy_full_data.csv')
    adequacy_h = read_csv(SHARED_MLE/'adequacy_h_like.csv')
    adequacy_b = read_csv(SHARED_MLE/'adequacy_b_like.csv')

    def adequacy_summary(records, name):
        if not records: return f'{name}: not available'
        ae = [float(r['AE2']) for r in records]; fb = sum(r['fallback_used'] == 'True' for r in records)
        return f'{name}: n={len(records)} MAE2={np.mean(ae):.4f} fallback={fb}/{len(records)}'

    et_ready = ET_EVAL.exists()
    lines = ['# Analysis snapshot (Main + full coverage grid)', '',
            'Five main methods: plugin, shared_mle, extratrees (mixed-budget), qwen_thinking,',
            'qwen_nonthinking. The mechanism-aware secondary reference and the S-oracle/',
            'design_reference construct-validity diagnostic are reported separately, never as a',
            'sixth normal competitor.', '',
            f"mixed-budget ExtraTrees: {'included' if et_ready else 'NOT YET AVAILABLE (mixed-budget ET job still running)'}",
            '', '## Main, by arm', '', '| arm | method | graphs | MAE2 | ProfileMAE | signed_rho2 |', '|---|---|---|---|---|---|']
    for r in summarize(main_rows, ('arm',)):
        lines.append(f"| {r['arm']} | {r['method']} | {r['graphs']} | {r['MAE2']:.4f} | {r['ProfileMAE']:.4f} | {r['signed_rho2']:+.4f} |")
    lines += ['', '## Budget grid, Real sources, by coverage x arm', '',
             '| coverage | arm | method | graphs | MAE2 | ProfileMAE |', '|---|---|---|---|---|---|']
    for r in summarize([r for r in rows if r['stratum'] == 'real'], ('budget', 'arm')):
        lines.append(f"| {r['budget']:.3f} | {r['arm']} | {r['method']} | {r['graphs']} | {r['MAE2']:.4f} | {r['ProfileMAE']:.4f} |")
    lines += ['', '## Qwen validity (fraction of formally valid final answers), by coverage', '']
    for fraction in BUDGET_GRID:
        for c in ('qwen_thinking', 'qwen_nonthinking'):
            v = qwen_validity(fraction, c)
            lines.append(f'- {fraction:.3f} {c}: {v:.3f}' if v is not None else f'- {fraction:.3f} {c}: not available')
    lines += ['', '## Shared-MLE adequacy (development/training material only; model not tuned on these numbers)', '',
             '- '+adequacy_summary(adequacy, 'full_data_representation'),
             '- '+adequacy_summary(adequacy_h, 'h_like_extrapolation'),
             '- '+adequacy_summary(adequacy_b, 'b_like_observation')]
    if not et_ready:
        lines += ['', '## Pending', '', '- mixed-budget ExtraTrees: needs scripts/evaluate_mixed_budget_et.py',
                 '  run against the trained models once the mixed-budget ET job (build_mixed_budget_et.py) completes.']
    (OUT/'ANALYSIS_SNAPSHOT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
