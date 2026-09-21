#!/usr/bin/env python3
"""Arm H sensitivity to the history fraction h in {.40, .60, .80}.

For every main graph, h and H draw: the panel is recalibrated to T at that h (the
node permutation is common across h, so panels are nested), the observation is
decomposed into node selection, dyad disappearance and within-dyad history loss
(history_diagnostics.decompose), and every reference predicts from it. The
learned references are trained at the primary h = .60, so .40/.80 are cross-h
sensitivities. No LLM call; no primary parameter is chosen from this.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.baselines import h_extrapolator, plugin
from main_experiment.common import (DIAGNOSTICS, H_SENSITIVITY, MAIN_KEYS, REFERENCES, STRATA, draws_for,
                                    fold_for, fresh_directory, graph_stratum, in_stratum, read_json,
                                    write_csv, write_json)
from main_experiment.history_diagnostics import decompose
from main_experiment.observation import features, make
from main_experiment.prepare import prepared_graph
from main_experiment.sampling import draw, h_parameters, history_panel_mask
from main_experiment.training import FOLDS, load_models

PREDICTORS = ('median', 'plugin', 'homogeneous', 'extratrees_pooled', 'extratrees_real_only')


def observation_row(g, key, h, index, budget, models, median):
    counts, _ = draw(g, 'H', index, 'sample', budget)
    panel = history_panel_mask(g, index, 'sample', budget)
    row = {'graph_id': key, 'stratum': graph_stratum(key), 'h': h, 'sample_index': index,
           'n_panel': budget['n_panel_history'], 'node_share': budget['n_panel_history']/g.N,
           'expected_coverage': budget['h_expected_cells']/g.cells, 'realized_coverage': int((counts > 0).sum())/g.cells,
           'saturated': budget['h_saturated'], 'target_unreachable': budget['h_target_unreachable'],
           'budget_matched': budget['h_within_tolerance'], **decompose(g.counts, counts, panel)}
    o = make(g, 'H', budget, counts, None)
    if o['D_obs']:
        fit = h_extrapolator(o)
        row.update(fitted_q=fit['q'], fit_status=fit['status'])
        x = [features(o)]
        predictions = {'median': median, 'plugin': plugin(o), 'homogeneous': fit['prediction'],
                       'extratrees_pooled': models['pooled'].predict(x)[0],
                       'extratrees_real_only': models['real_only'].predict(x)[0]}
    else:
        row.update(fitted_q=None, fit_status='empty_training_median')
        predictions = {name: median for name in PREDICTORS}
    for name in PREDICTORS:
        error = np.abs(np.asarray(predictions[name])-g.truth)
        row[name+'_AE2'] = float(error[0]); row[name+'_ProfileAE'] = float(error.mean())
    return row


def mean_or_none(values):
    return float(np.mean(values)) if values and all(v is not None for v in values) else None


def main():
    out = fresh_directory(DIAGNOSTICS/'history')
    models = {v: load_models(REFERENCES/f'models_{v}') for v in ('pooled', 'real_only')}
    medians = {f: read_json(REFERENCES/'models_pooled'/f/'manifest.json')['median'] for f in FOLDS}
    rows = []
    for key in MAIN_KEYS:
        g, _ = prepared_graph(key)
        fold = fold_for(key)
        for h in H_SENSITIVITY:
            budget = h_parameters(g, .1*g.cells, h)
            for index in range(1, draws_for('H', budget)+1):
                rows.append(observation_row(g, key, h, index, budget,
                                            {v: models[v][fold] for v in models}, medians[fold]))
    write_csv(out/'observations.csv', rows)

    metrics = ['node_share', 'expected_coverage', 'realized_coverage']
    metrics += [f'{part}_{m}' for part in ('node_selection', 'dyad_disappearance', 'within_dyad_history', 'net_history', 'total')
                for m in ('abs_rho2', 'profile_abs')]
    metrics += [f'{p}_{m}' for p in PREDICTORS for m in ('AE2', 'ProfileAE')]
    sources = []
    for h in H_SENSITIVITY:
        for key in MAIN_KEYS:
            draws = [r for r in rows if r['graph_id'] == key and r['h'] == h]
            source = {'h': h, 'graph_id': key, 'stratum': draws[0]['stratum'], 'draws': len(draws),
                      'defined_draws': sum(r['defined'] for r in draws), 'n_panel': draws[0]['n_panel'],
                      'budget_matched': all(r['budget_matched'] for r in draws)}
            for m in metrics: source[m] = mean_or_none([r.get(m) for r in draws])
            for suffix in ('abs_rho2', 'profile_abs'):
                selection, history = source['node_selection_'+suffix], source['net_history_'+suffix]
                source['history_share_'+suffix] = (history/(selection+history)
                                                   if selection is not None and history is not None and selection+history else None)
                # Paired within-draw contrast: net history loss minus node-selection error.
                diffs = [r['net_history_'+suffix]-r['node_selection_'+suffix] for r in draws if r['defined']]
                complete = len(diffs) == len(draws)
                source['history_minus_selection_'+suffix] = float(np.mean(diffs)) if complete else None
                source['history_minus_selection_'+suffix+'_MCSE'] = (
                    float(np.std(diffs, ddof=1)/np.sqrt(len(diffs))) if complete and len(diffs) > 1 else 0. if complete else None)
            sources.append(source)
    write_csv(out/'sources.csv', sources)

    summary = []
    for h in H_SENSITIVITY:
        for block in STRATA:
            group = [s for s in sources if s['h'] == h and in_stratum(s['graph_id'], block)]
            row = {'h': h, 'evidence_block': block, 'sources': len(group),
                   'budget_matched_sources': sum(s['budget_matched'] for s in group),
                   'undefined_sources': sum(s['defined_draws'] != s['draws'] for s in group)}
            for m in metrics: row[m] = mean_or_none([s[m] for s in group])
            for suffix in ('abs_rho2', 'profile_abs'):
                selection, history = row['node_selection_'+suffix], row['net_history_'+suffix]
                row['history_share_'+suffix] = (history/(selection+history)
                                                if selection is not None and history is not None and selection+history else None)
                contrast = 'history_minus_selection_'+suffix
                row[contrast] = mean_or_none([s[contrast] for s in group])
                mcse = [s[contrast+'_MCSE'] for s in group]
                row[contrast+'_MCSE'] = float(np.sqrt(sum(x*x for x in mcse))/len(mcse)) if None not in mcse else None
            summary.append(row)
    write_csv(out/'summary.csv', summary)
    write_json(out/'report.json', {'h_primary': .6, 'h_values': list(H_SENSITIVITY), 'observations': len(rows),
                                   'learned_reference': 'trained at h=.60; h=.40/.80 are cross-h sensitivities',
                                   'diagnosis': 'net history includes dyad disappearance; descriptive fixed-panel comparisons',
                                   'rows': summary})
    print('history sensitivity:', len(rows), 'H observations over', len(MAIN_KEYS), 'graphs x', len(H_SENSITIVITY), 'h')


if __name__ == '__main__':
    main()
