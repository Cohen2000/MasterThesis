#!/usr/bin/env python3
"""Oracle split of the plug-in error into dyad selection and lost history.

For one observation let
  truth  = profile over all dyads of the graph,
  oracle = profile over the observed dyads, computed from their complete histories,
  plugin = profile computable from the observation itself.
Then selection = oracle - truth and history = plugin - oracle add up to the
plug-in error exactly. Complete histories are used for this evaluation only;
nothing here feeds a predictor. Run on the 100 development pool graphs and on
the 24 main graphs.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARMS, BUILD, DIAGNOSTICS, MAIN_KEYS, REFERENCES, STRATA, draws_for, fresh_directory,
                                    graph_stratum, in_stratum, read_json, write_csv, write_json)
from main_experiment.pool import regenerate
from main_experiment.prepare import prepared_graph
from main_experiment.sampling import Walk, draw


def profile(K):
    return np.array([float((K >= k).mean()) for k in range(2, 6)])


def decompose_graph(g, budget, domain, labels, build):
    walk = Walk(g, build)
    rows = []
    truth = np.array(g.truth)
    for arm in ARMS:
        for index in range(1, draws_for(arm, budget, domain)+1):
            counts, _ = draw(g, arm, index, domain, budget, walk)
            seen = counts.sum(1) > 0
            if not seen.any(): continue
            oracle = profile((g.counts[seen] > 0).sum(1))
            plugin = profile((counts[seen] > 0).sum(1))
            rows.append({**labels, 'arm': arm, 'sample_index': index, 'D_full': g.D, 'D_obs': int(seen.sum()),
                         'dyad_coverage': float(seen.mean()), 'truth_rho2': truth[0], 'oracle_rho2': oracle[0],
                         'plugin_rho2': plugin[0], 'selection_rho2': oracle[0]-truth[0],
                         'history_rho2': plugin[0]-oracle[0], 'total_rho2': plugin[0]-truth[0],
                         'selection_profile': float(np.mean(oracle-truth)),
                         'history_profile': float(np.mean(plugin-oracle)),
                         'total_profile': float(np.mean(plugin-truth))})
    return rows


def summarise(rows, group_field, groups, member):
    """Graph-equal means per group and arm; shares of the summed absolute components."""
    summary = []
    for group in groups:
        for arm in ARMS:
            chosen = [r for r in rows if r['arm'] == arm and member(r, group)]
            if not chosen: continue
            per_graph = {}
            for r in chosen: per_graph.setdefault(r['graph_id'], []).append(r)

            def graph_mean(k): return float(np.mean([np.mean([x[k] for x in v]) for v in per_graph.values()]))
            abs_selection = float(np.mean([abs(r['selection_rho2']) for r in chosen]))
            abs_history = float(np.mean([abs(r['history_rho2']) for r in chosen]))
            abs_total = float(np.mean([abs(r['total_rho2']) for r in chosen]))
            summary.append({group_field: group, 'arm': arm, 'graphs': len(per_graph),
                            'dyad_coverage': graph_mean('dyad_coverage'),
                            'selection_rho2': graph_mean('selection_rho2'), 'history_rho2': graph_mean('history_rho2'),
                            'total_rho2': graph_mean('total_rho2'), 'abs_selection': abs_selection,
                            'abs_history': abs_history, 'abs_total': abs_total,
                            # Selection and history can have opposite signs and partly cancel.
                            'history_share_of_abs_components': abs_history/(abs_selection+abs_history)
                            if abs_selection+abs_history else None,
                            'cancellation_ratio': (abs_selection+abs_history)/abs_total if abs_total else None})
    return summary


def main():
    out = fresh_directory(DIAGNOSTICS/'decomposition')
    development = []
    for spec in read_json(REFERENCES/'pool_definition.json')['graphs']:
        if spec['partition'] != 'dev': continue
        budget = read_json(REFERENCES/'pool/observations'/f'{spec["key"]}.json')['budget']
        development += decompose_graph(regenerate(spec), budget, 'pool_dev',
                                       {'graph_id': spec['key'], 'family': spec['family']}, BUILD)
    main_rows = []
    for key in MAIN_KEYS:
        g, budget = prepared_graph(key)
        main_rows += decompose_graph(g, budget, 'sample', {'graph_id': key, 'stratum': graph_stratum(key)}, BUILD)
    write_csv(out/'development.csv', development)
    write_csv(out/'main.csv', main_rows)
    report = {'formula': 'selection = oracle - truth; history = plugin - oracle; total = selection + history',
              'development': summarise(development, 'family', ('all', 'dar', 'ad'),
                                       lambda r, f: f in ('all', r['family'])),
              'main': summarise(main_rows, 'evidence_block', STRATA, lambda r, s: in_stratum(r['graph_id'], s))}
    write_csv(out/'development_summary.csv', report['development'])
    write_csv(out/'main_summary.csv', report['main'])
    write_json(out/'report.json', report)
    print('decomposition:', len(development), 'development and', len(main_rows), 'main observations')


if __name__ == '__main__':
    main()
