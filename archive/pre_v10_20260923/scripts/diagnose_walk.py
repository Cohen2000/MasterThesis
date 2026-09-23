#!/usr/bin/env python3
"""Construct validity of the degree-biased random walk shared by S1 and S2.

For each of the 24 main graphs and two lengths (production L and min(4L, 10^6)),
1000 independent diagnostic walks from uniform starts. Per walk:
  plugin      profile of the distinct traversed dyads (what the models see),
  design      the design reference sum_t I(K>=k)/(d_u d_v) / sum_t 1/(d_u d_v),
  coverage, revisits and traversal/hub concentration.
Per graph, three fixed targets computed exactly from the full archive:
  truth             rho_k over all dyads,
  component target  sum_c (n_c/N) rho_k(c): the design reference's limit for a long
                    walk from a uniform start (edges uniform within a component),
  selection target  sum_c (n_c/N) sum_{e in c} d_u d_v I(K_e>=k) / sum_{e in c} d_u d_v:
                    the profile of the walk's stationary traversal distribution, i.e.
                    the structural degree selection of the observed dyads.
Bias, SD, RMSE and MCSE are reported per graph and length and averaged per evidence
block. Diagnostic only; production lengths, draws and references are unchanged.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARM_ID, BUILD, DIAGNOSTICS, MAIN_KEYS, STRATA, fresh_directory, graph_stratum,
                                    in_stratum, seed, write_csv, write_json)
from main_experiment.prepare import prepared_graph
from main_experiment.sampling import Walk

PATHS = 1000
BATCH = 25                  # walks per kernel call; bounds memory on large graphs


def targets(g, walk):
    """Truth, component-mixture target and degree-selection target (rho_2..rho_5)."""
    component = walk.components[g.ends[:, 0]]
    node_share = np.bincount(walk.components, minlength=walk.n_components)/g.N
    product = (walk.degree[g.ends[:, 0]]*walk.degree[g.ends[:, 1]]).astype(float)
    uniform = np.zeros((walk.n_components, 4)); selection = np.zeros((walk.n_components, 4))
    for c in range(walk.n_components):
        inside = component == c
        for j, k in enumerate(range(2, 6)):
            active = g.K[inside] >= k
            uniform[c, j] = active.mean()
            selection[c, j] = product[inside][active].sum()/product[inside].sum()
    return np.array(g.truth), node_share@uniform, node_share@selection, component


def path_rows(g, key, label, L, walk, component, hubs):
    rows = []
    seeds = [seed('walk_diagnostic', key, ARM_ID['S1'], i) for i in range(1, PATHS+1)]
    product = (walk.degree[g.ends[:, 0]]*walk.degree[g.ends[:, 1]]).astype(float)
    for first in range(0, PATHS, BATCH):
        _, volumes, traversals, executed = walk.run(seeds[first:first+BATCH], int(L), True)
        assert (executed == L).all() and (traversals.sum(1) == L).all()
        for i, r in enumerate(traversals):
            seen = r > 0
            assert len(np.unique(component[seen])) == 1                 # a walk never leaves its component
            weight = r/product
            node_traversals = np.bincount(g.ends.ravel(), weights=np.repeat(r, 2), minlength=g.N)/(2*L)
            row = {'graph_id': key, 'stratum': graph_stratum(key), 'length': label, 'L': int(L), 'path': first+i+1,
                   'component': int(component[seen][0]), 'distinct_dyad_coverage': float(seen.mean()),
                   'cell_coverage': float(volumes[i]/g.cells), 'repeated_step_fraction': float(1-seen.sum()/L),
                   'edge_traversal_hhi': float(((r/L)**2).sum()), 'max_edge_traversal_share': float(r.max()/L),
                   'top_degree_one_percent_node_traversal_share': float(node_traversals[hubs].sum())}
            for k in range(2, 6):
                row[f'plugin_rho{k}'] = float((g.K[seen] >= k).mean())
                row[f'design_rho{k}'] = float(weight[g.K >= k].sum()/weight.sum())
            rows.append(row)
    return rows


def summarise(rows, key, label, L, truth, component_target, selection_target):
    out = {'graph_id': key, 'stratum': graph_stratum(key), 'length': label, 'L': L, 'paths': len(rows)}
    for name in ('distinct_dyad_coverage', 'cell_coverage', 'repeated_step_fraction', 'edge_traversal_hhi',
                 'top_degree_one_percent_node_traversal_share'):
        out[name] = float(np.mean([r[name] for r in rows]))
    for estimator in ('plugin', 'design'):
        for k in range(2, 6):
            values = np.array([r[f'{estimator}_rho{k}'] for r in rows])
            prefix = f'{estimator}_rho{k}'
            out[prefix+'_mean'] = float(values.mean())
            out[prefix+'_sd'] = float(values.std(ddof=1))
            out[prefix+'_bias'] = float(values.mean()-truth[k-2])
            out[prefix+'_bias_MCSE'] = float(values.std(ddof=1)/np.sqrt(len(values)))
            out[prefix+'_rmse'] = float(np.sqrt(np.mean((values-truth[k-2])**2)))
            out[prefix+'_minus_component_target'] = float(values.mean()-component_target[k-2])
        out[f'{estimator}_ProfileAE_mean'] = float(np.mean([np.mean([abs(r[f'{estimator}_rho{k}']-truth[k-2])
                                                                     for k in range(2, 6)]) for r in rows]))
    return out


def main():
    out = fresh_directory(DIAGNOSTICS/'walk')
    graphs = []; paths = []; summary = []
    for key in MAIN_KEYS:
        g, budget = prepared_graph(key)
        walk = Walk(g, BUILD)
        truth, component_target, selection_target, component = targets(g, walk)
        hubs = np.argsort(walk.degree, kind='stable')[-max(1, int(np.ceil(.01*g.N))):]
        node_share = np.bincount(walk.components)/g.N
        cells = np.bincount(component, weights=g.K, minlength=walk.n_components)
        graphs.append({'graph_id': key, 'stratum': graph_stratum(key), 'components': walk.n_components,
                       'largest_node_share': float(node_share.max()), 'largest_cell_share': float(cells.max()/g.cells),
                       'expected_ceiling_cells': float(node_share@cells), 'target_cells': budget['T'],
                       'L': budget['L'], 'validated_coverage': budget['validation_mean']/g.cells,
                       'budget_matched': budget['walk_budget_matched'],
                       'truth': truth.tolist(), 'component_target': component_target.tolist(),
                       'selection_target': selection_target.tolist(),
                       'selection_effect_rho2': float(selection_target[0]-truth[0]),
                       'component_effect_rho2': float(component_target[0]-truth[0])})
        for label, L in (('L', budget['L']), ('4L', min(4*budget['L'], 1_000_000))):
            rows = path_rows(g, key, label, L, walk, component, hubs)
            paths += rows
            summary.append(summarise(rows, key, label, L, truth, component_target, selection_target))
        print(key, 'walks done', flush=True)
    blocks = []
    for label in ('L', '4L'):
        for block in STRATA:
            chosen = [s for s in summary if s['length'] == label and in_stratum(s['graph_id'], block)]
            row = {'length': label, 'evidence_block': block, 'sources': len(chosen)}
            for estimator in ('plugin', 'design'):
                for metric in ('bias', 'rmse', 'sd'):
                    row[f'{estimator}_rho2_{metric}'] = float(np.mean([s[f'{estimator}_rho2_{metric}'] for s in chosen]))
                row[f'{estimator}_rho2_abs_bias'] = float(np.mean([abs(s[f'{estimator}_rho2_bias']) for s in chosen]))
            row['selection_effect_rho2'] = float(np.mean([g['selection_effect_rho2'] for g in graphs
                                                          if in_stratum(g['graph_id'], block)]))
            blocks.append(row)
    write_csv(out/'graphs.csv', graphs)
    write_csv(out/'paths.csv', paths)
    write_csv(out/'summary.csv', summary)
    write_csv(out/'blocks.csv', blocks)
    write_json(out/'report.json', {'paths_per_length': PATHS, 'lengths': 'L and min(4L, 1000000)',
                                   'graph_rows': graphs, 'walk_rows': summary, 'block_rows': blocks,
                                   'interpretation': 'design reference: consistent for the component-mixture target, '
                                                     'not unbiased at finite L; plugin: selects dyads by the walk',
                                   'production_length_changed': False})
    print('walk diagnostics:', len(paths), 'paths')


if __name__ == '__main__':
    main()
