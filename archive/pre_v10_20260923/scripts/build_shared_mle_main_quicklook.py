#!/usr/bin/env python3
"""Fast Main-only shared_mle quicklook: no Qwen, no ET refit, no full budget sweep.

Reads the already-prepared main observations (results/panel888/prepared) and
the already-computed plugin/corrector baselines (results/panel888/references),
fits shared_mle on each, and writes compact CSV/MD next to them. Safe to run
against a partial offline run: only the `prepare` and `references` stages are
needed, not `train`'s mixed-budget ET or anything after it.

usage: build_shared_mle_main_quicklook.py [--prepared DIR] [--references DIR] [--out DIR]
"""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import PREPARED, REFERENCES, ROOT, write_csv, write_json
from main_experiment.observation import parse
from main_experiment.references import mean_and_se, per_graph_mean
from main_experiment.shared_mle import fit

OUT = ROOT/'results/panel888_shared_mle_main_quicklook'


def error_record(prediction, truth):
    e = np.asarray(prediction, float)-np.asarray(truth, float)
    return {'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0])}


def load_baselines_csv(path):
    """method -> list[dict] rows of the existing main_observations.csv (plugin/corrector/...)."""
    rows = list(csv.DictReader(open(path)))
    for r in rows:
        for k in ('AE2', 'ProfileAE', 'signed_rho2'): r[k] = float(r[k]) if r[k] != '' else None
        r['sample_index'] = int(r['sample_index'])
    return rows


def main(prepared, references, out):
    obs_dir = prepared/'observations/sample'
    files = sorted(obs_dir.glob('*.json'))
    if not files: raise FileNotFoundError(f'no main observations under {obs_dir}')
    baseline_rows = load_baselines_csv(references/'main_observations.csv')

    predictions = []; fit_rows = []; source_family = {}
    for path in files:
        import json
        row = json.loads(path.read_text())
        source_family[row['graph_id']] = row['source_family']
        o = parse(row['block'])
        rec = {'graph_id': row['graph_id'], 'stratum': row['stratum'], 'arm': row['arm'],
               'sample_index': row['sample_index'], 'source_family': row['source_family']}
        if o['D_obs'] == 0:
            predictions.append({**rec, 'method': 'shared_mle', 'status': 'empty_sample'})
            fit_rows.append({**rec, 'fit_status': 'empty_sample', 'fallback_used': None, 'objective': None})
            continue
        try:
            result = fit(o)
            err = error_record(result.rho, row['truth'])
            predictions.append({**rec, 'method': 'shared_mle', 'status': result.fit_status,
                                'fallback_used': result.fallback_used, **err})
            fit_rows.append({**rec, 'fit_status': result.fit_status, 'fallback_used': result.fallback_used,
                             'objective': result.objective, 'alpha': result.alpha, 'beta': result.beta,
                             'lam': result.lam})
        except ValueError as e:
            predictions.append({**rec, 'method': 'shared_mle', 'status': f'error:{e}'})
            fit_rows.append({**rec, 'fit_status': f'error:{e}', 'fallback_used': None, 'objective': None})

    # merge in the existing plugin / corrector (mechanism-aware secondary reference) rows
    for r in baseline_rows:
        if r['method'] not in ('plugin', 'corrector'): continue
        predictions.append({'graph_id': r['graph_id'], 'stratum': r['stratum'], 'arm': r['arm'],
                            'sample_index': r['sample_index'], 'source_family': source_family.get(r['graph_id'], ''),
                            'method': 'plugin' if r['method'] == 'plugin' else 'mechanism_aware_reference',
                            'status': r['status'], 'AE2': r['AE2'], 'ProfileAE': r['ProfileAE'],
                            'signed_rho2': r['signed_rho2']})

    write_csv(out/'predictions.csv', predictions)
    write_csv(out/'fit_diagnostics.csv', fit_rows)

    methods = ('plugin', 'mechanism_aware_reference', 'shared_mle')
    scored = [p for p in predictions if p.get('AE2') is not None]

    def summarize(rows, group_field):
        table = []
        groups = sorted({r[group_field] for r in rows})
        for group in groups:
            for method in methods:
                chosen = [r for r in rows if r[group_field] == group and r['method'] == method]
                if not chosen: continue
                ae = per_graph_mean(chosen, 'AE2'); pe = per_graph_mean(chosen, 'ProfileAE')
                se = per_graph_mean(chosen, 'signed_rho2')
                mae2, mae2_se = mean_and_se(ae.values()); pmae, pmae_se = mean_and_se(pe.values())
                signed, signed_se = mean_and_se(se.values())
                table.append({group_field: group, 'method': method, 'graphs': len(ae), 'n': len(chosen),
                              'MAE2': mae2, 'MAE2_se': mae2_se, 'ProfileMAE': pmae, 'ProfileMAE_se': pmae_se,
                              'signed_rho2': signed, 'signed_rho2_se': signed_se})
        return table

    by_arm = summarize(scored, 'arm')
    by_block = summarize(scored, 'stratum')
    write_csv(out/'summary_by_arm.csv', by_arm)
    write_csv(out/'summary_by_block.csv', by_block)

    real_rows = [r for r in scored if r['stratum'] == 'real']
    by_source = summarize(real_rows, 'source_family')
    write_csv(out/'summary_real_by_source.csv', by_source)

    fit_status_counts = {}
    for r in fit_rows:
        fit_status_counts[r['fit_status']] = fit_status_counts.get(r['fit_status'], 0)+1
    fallback_by_arm = {}
    for arm in sorted({r['arm'] for r in fit_rows}):
        chosen = [r for r in fit_rows if r['arm'] == arm and r['fallback_used'] is not None]
        fallback_by_arm[arm] = {'n': len(chosen), 'fallback_used': sum(bool(r['fallback_used']) for r in chosen),
                                'fit_failed_or_empty': sum(r['fallback_used'] is None for r in fit_rows if r['arm'] == arm)}
    write_json(out/'fit_status_counts.json', {'overall': fit_status_counts, 'by_arm': fallback_by_arm})

    lines = ['# Main shared-MLE quicklook (Qwen not included; plugin/corrector are the existing offline references)',
            '', f'Source: `{prepared}` / `{references}`', '', '## By arm', '',
            '| arm | method | graphs | MAE2 | ProfileMAE | signed_rho2 |', '|---|---|---|---|---|---|']
    for r in by_arm:
        lines.append(f"| {r['arm']} | {r['method']} | {r['graphs']} | {r['MAE2']:.4f} | {r['ProfileMAE']:.4f} | {r['signed_rho2']:+.4f} |")
    lines += ['', '## By evidence block', '', '| block | method | graphs | MAE2 | ProfileMAE | signed_rho2 |', '|---|---|---|---|---|---|']
    for r in by_block:
        lines.append(f"| {r['stratum']} | {r['method']} | {r['graphs']} | {r['MAE2']:.4f} | {r['ProfileMAE']:.4f} | {r['signed_rho2']:+.4f} |")
    lines += ['', '## Real sources individually', '', '| source | method | MAE2 | ProfileMAE |', '|---|---|---|---|']
    for r in by_source:
        lines.append(f"| {r['source_family']} | {r['method']} | {r['MAE2']:.4f} | {r['ProfileMAE']:.4f} |")
    lines += ['', '## Fit/fallback', '', f'fit_status counts: {fit_status_counts}', '',
             f'fallback by arm: {fallback_by_arm}', '',
             'No model change was made based on these numbers.']
    (out/'MAIN_MLE_QUICKLOOK.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--prepared', type=Path, default=PREPARED)
    p.add_argument('--references', type=Path, default=REFERENCES)
    p.add_argument('--out', type=Path, default=OUT)
    a = p.parse_args()
    main(a.prepared, a.references, a.out)
