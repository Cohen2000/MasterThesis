#!/usr/bin/env python3
"""Analysis-ready snapshot: Main method comparison (plugin / shared_mle /
mechanism-aware reference / Qwen thinking+non-thinking), real-source
breakdown, and shared-MLE fit/adequacy diagnostics.

Reuses the already-computed quicklook/adequacy CSVs (build_shared_mle_main_quicklook.py,
build_qwen_main_quicklook.py, build_shared_mle.py); does not refit anything.
Budget/mixed-budget-ET tables are written only when their source files exist
(BUDGET_METHOD_COMPARISON.csv is skipped, and ANALYSIS_SNAPSHOT.md says so,
until the budget-sensitivity CPU chain and the mixed-budget ET job complete).
"""
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT, write_csv, write_json
from main_experiment.references import mean_and_se, per_graph_mean

QUICKLOOK = ROOT/'results/panel888_shared_mle_main_quicklook'
SHARED_MLE = ROOT/'results/panel888_shared_mle'
OUT = ROOT/'results/panel888_analysis_snapshot'
METHOD_LABEL = {'plugin': 'plugin', 'mechanism_aware_reference': 'mechanism_aware_reference (secondary)',
               'shared_mle': 'shared_mle', 'qwen_thinking': 'qwen_thinking', 'qwen_nonthinking': 'qwen_nonthinking'}


def read_csv(path):
    return list(csv.DictReader(open(path))) if Path(path).exists() else None


def load_rows():
    rows = []
    for r in read_csv(QUICKLOOK/'predictions.csv') or []:
        if r.get('AE2') in (None, ''): continue
        rows.append({'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                     'source_family': r.get('source_family', ''), 'method': r['method'],
                     'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])})
    for r in read_csv(QUICKLOOK/'qwen_predictions.csv') or []:
        if r['valid'] != 'True': continue
        rows.append({'graph_id': r['graph_id'], 'arm': r['arm'], 'stratum': r['stratum'],
                     'source_family': r.get('source_family', ''), 'method': r['config'],
                     'AE2': float(r['AE2']), 'ProfileAE': float(r['ProfileAE']), 'signed_rho2': float(r['signed_rho2'])})
    return rows


def summarize(rows, group_field, methods):
    table = []
    for group in sorted({r[group_field] for r in rows if r[group_field]}):
        for method in methods:
            chosen = [r for r in rows if r[group_field] == group and r['method'] == method]
            if not chosen: continue
            ae = per_graph_mean(chosen, 'AE2'); pe = per_graph_mean(chosen, 'ProfileAE'); se = per_graph_mean(chosen, 'signed_rho2')
            mae2, mae2_se = mean_and_se(ae.values()); pmae, pmae_se = mean_and_se(pe.values())
            signed, signed_se = mean_and_se(se.values())
            table.append({group_field: group, 'method': METHOD_LABEL.get(method, method), 'graphs': len(ae), 'n': len(chosen),
                          'MAE2': mae2, 'MAE2_se': mae2_se, 'ProfileMAE': pmae, 'ProfileMAE_se': pmae_se,
                          'signed_rho2': signed, 'signed_rho2_se': signed_se})
    return table


def qwen_validity(config):
    rows = read_csv(QUICKLOOK/'qwen_predictions.csv') or []
    chosen = [r for r in rows if r['config'] == config]
    return sum(r['valid'] == 'True' for r in chosen)/len(chosen) if chosen else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    methods = ('plugin', 'shared_mle', 'mechanism_aware_reference', 'qwen_thinking', 'qwen_nonthinking')

    by_arm = summarize(rows, 'arm', methods)
    write_csv(OUT/'MAIN_METHOD_COMPARISON.csv', by_arm)

    real_rows = [r for r in rows if r['stratum'] == 'real']
    by_source = summarize(real_rows, 'source_family', methods)
    write_csv(OUT/'MAIN_REAL_BY_SOURCE.csv', by_source)

    surrogate_rows = [r for r in rows if r['stratum'] == 'surrogate']
    write_csv(OUT/'SURROGATE_COMPARISON.csv', summarize(surrogate_rows, 'arm', methods))
    synthetic_rows = [r for r in rows if r['stratum'] == 'synthetic']
    write_csv(OUT/'SYNTHETIC_COMPARISON.csv', summarize(synthetic_rows, 'arm', methods))

    fit_diag = read_csv(QUICKLOOK/'fit_diagnostics.csv')
    if fit_diag: write_csv(OUT/'FIT_DIAGNOSTICS.csv', fit_diag)

    budget_csv = SHARED_MLE/'main_and_budget_observations.csv'
    budget_ready = budget_csv.exists()
    if budget_ready:
        # placeholder for when the budget-sensitivity chain has completed;
        # left for a follow-up run once results/panel888_shared_mle has budget rows.
        pass

    adequacy = read_csv(SHARED_MLE/'adequacy_full_data.csv')
    adequacy_h = read_csv(SHARED_MLE/'adequacy_h_like.csv')
    adequacy_b = read_csv(SHARED_MLE/'adequacy_b_like.csv')

    def adequacy_summary(records, name):
        if not records: return f'{name}: not available'
        ae = [float(r['AE2']) for r in records]; fb = sum(r['fallback_used'] == 'True' for r in records)
        return f'{name}: n={len(records)} MAE2={np.mean(ae):.4f} fallback={fb}/{len(records)}'

    lines = ['# Analysis snapshot (Main only; budget/mixed-budget-ET pending)', '',
            'Methods: plugin, shared_mle, mechanism_aware_reference (secondary same-information corrector),',
            'qwen_thinking, qwen_nonthinking. S-oracle / design_reference intentionally NOT included as a',
            'normal competitor (construct-validity diagnostic only, not produced in this pass).', '',
            '## Main, by arm', '', '| arm | method | graphs | MAE2 | ProfileMAE | signed_rho2 |', '|---|---|---|---|---|---|']
    for r in by_arm:
        lines.append(f"| {r['arm']} | {r['method']} | {r['graphs']} | {r['MAE2']:.4f} | {r['ProfileMAE']:.4f} | {r['signed_rho2']:+.4f} |")
    lines += ['', '## Qwen validity (fraction of formally valid final answers)', '']
    for c in ('qwen_thinking', 'qwen_nonthinking'):
        v = qwen_validity(c)
        lines.append(f'- {c}: {v:.3f}' if v is not None else f'- {c}: not available')
    lines += ['', '## Shared-MLE adequacy (development/training material only; model not tuned on these numbers)', '',
             '- '+adequacy_summary(adequacy, 'full_data_representation'),
             '- '+adequacy_summary(adequacy_h, 'h_like_extrapolation'),
             '- '+adequacy_summary(adequacy_b, 'b_like_observation'), '',
             '## Pending', '',
             '- BUDGET_METHOD_COMPARISON.csv: needs the budget-sensitivity CPU chain to finish (see docs/CURRENT_STATE.md).',
             '- mixed-budget ExtraTrees column: needs the mixed-budget ET job to finish; not in any table yet.']
    (OUT/'ANALYSIS_SNAPSHOT.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
