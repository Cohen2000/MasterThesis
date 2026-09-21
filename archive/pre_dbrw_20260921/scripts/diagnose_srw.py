#!/usr/bin/env python3
"""Arm S (simple random walk) diagnostics on the 24 main graphs.

Per graph: connected components and the structural coverage ceiling from a
uniform start, and the stationary component-mixture profile. Per graph and
length (production L and min(4L, 10^6)): 32 independent diagnostic walks with
traversal-frequency error, finite-walk vs. local stationary profile, revisit
rate, dyad/cell coverage and edge/node/hub traversal concentration. Diagnostic
only; production lengths and predictions are unchanged.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import (ARM_ID, BUILD, DIAGNOSTICS, MAIN_KEYS, fresh_directory, graph_stratum, seed,
                                    write_csv, write_json)
from main_experiment.prepare import prepared_graph
from main_experiment.sampling import Walk

PATHS = 32


def component_row(g, key, budget, walk, dyad_component, component_profiles):
    node_share = np.bincount(walk.components)/g.N
    dyad_share = np.bincount(dyad_component, minlength=walk.n_components)/g.D
    cells = np.bincount(dyad_component, weights=g.K, minlength=walk.n_components)
    stationary = node_share@component_profiles     # uniform start picks a component with its node share
    return {'graph_id': key, 'stratum': graph_stratum(key), 'components': walk.n_components,
            'largest_node_share': float(node_share.max()), 'largest_edge_share': float(dyad_share.max()),
            'largest_cell_share': float(cells.max()/g.cells),
            'node_vs_edge_component_total_variation': float(np.abs(node_share-dyad_share).sum()/2),
            'expected_ceiling_cells': float(node_share@cells), 'target_cells': budget['T'],
            'unreachable': bool(node_share@cells < budget['T']), 'L': budget['L'],
            'validated_coverage': budget['validation_mean']/g.cells,
            'validation_mcse_relative_to_target': budget['validation_mcse']/budget['T'],
            'budget_matched': budget['walk_budget_matched'],
            'stationary_component_mixture_profile': stationary.tolist(),
            'stationary_mixture_signed_rho2': float(stationary[0]-g.truth[0]),
            'stationary_mixture_ProfileAE': float(np.mean(np.abs(stationary-g.truth)))}


def path_row(g, key, label, L, number, traversals, volume, dyad_component, component_profiles, hubs):
    truth = np.array(g.truth)
    seen = traversals > 0
    components = np.unique(dyad_component[seen])
    assert len(components) == 1                      # a walk never leaves its component
    frequency = np.array([traversals[g.K >= k].sum()/L for k in range(2, 6)])
    plugin = np.array([(g.K[seen] >= k).mean() for k in range(2, 6)])
    local = component_profiles[components[0]]
    edge_share = traversals/L
    node_share = np.bincount(g.ends.ravel(), weights=np.repeat(traversals, 2), minlength=g.N)/(2*L)
    return {'graph_id': key, 'stratum': graph_stratum(key), 'length': label, 'L': L, 'path': number,
            'component': int(components[0]), 'traversal_rho2': float(frequency[0]),
            'traversal_AE2': float(abs(frequency[0]-truth[0])),
            'traversal_ProfileAE': float(np.mean(np.abs(frequency-truth))),
            'finite_vs_local_stationary_rho2': float(frequency[0]-local[0]),
            'finite_vs_local_stationary_ProfileAE': float(np.mean(np.abs(frequency-local))),
            'plugin_AE2': float(abs(plugin[0]-truth[0])),
            'edge_traversal_hhi': float(edge_share@edge_share), 'max_edge_traversal_share': float(edge_share.max()),
            'node_traversal_hhi': float(node_share@node_share),
            'top_degree_one_percent_node_traversal_share': float(node_share[hubs].sum()),
            'repeated_step_fraction': float(1-seen.sum()/L), 'distinct_dyad_coverage': float(seen.mean()),
            'cell_coverage': float(volume/g.cells)}


def main():
    out = fresh_directory(DIAGNOSTICS/'srw')
    components = []; paths = []; summary = []
    for key in MAIN_KEYS:
        g, budget = prepared_graph(key)
        walk = Walk(g, BUILD)
        dyad_component = walk.components[g.ends[:, 0]]
        component_profiles = np.array([[np.mean(g.K[dyad_component == c] >= k) for k in range(2, 6)]
                                       for c in range(walk.n_components)])
        components.append(component_row(g, key, budget, walk, dyad_component, component_profiles))
        degree = np.bincount(g.ends.ravel(), minlength=g.N)
        hubs = np.argsort(degree, kind='stable')[-max(1, int(np.ceil(.01*g.N))):]
        seeds = [seed('srw_diagnostic', key, ARM_ID['S'], i) for i in range(1, PATHS+1)]
        for label, L in (('primary', budget['L']), ('longer', min(4*budget['L'], 1_000_000))):
            rows = []
            for first in range(0, PATHS, 4):          # small batches bound memory on large graphs
                _, volumes, traversals, executed = walk.run(seeds[first:first+4], int(L), True)
                assert (executed == L).all() and (traversals.sum(1) == L).all()
                for i in range(len(volumes)):
                    rows.append(path_row(g, key, label, L, first+i+1, traversals[i], volumes[i],
                                         dyad_component, component_profiles, hubs))
            paths += rows
            aggregate = {'graph_id': key, 'stratum': graph_stratum(key), 'length': label, 'L': L, 'paths': PATHS}
            for metric in rows[0]:
                if metric in aggregate or metric in ('path', 'component'): continue
                values = [r[metric] for r in rows]
                aggregate[metric] = float(np.mean(values))
                aggregate[metric+'_MCSE'] = float(np.std(values, ddof=1)/np.sqrt(PATHS))
            aggregate['traversal_mean_bias_rho2'] = aggregate['traversal_rho2']-g.truth[0]
            summary.append(aggregate)
    write_csv(out/'components.csv', components)
    write_csv(out/'paths.csv', paths)
    write_csv(out/'summary.csv', summary)
    write_json(out/'report.json', {'paths_per_length': PATHS, 'lengths': 'L and min(4L, 1000000)',
                                   'component_rows': components, 'walk_rows': summary,
                                   'interpretation': 'local selection with complete histories; finite-start/mixing '
                                                     'and component bias remain; no unbiasedness claim',
                                   'production_length_changed': False})
    print('srw diagnostics:', len(paths), 'paths')


if __name__ == '__main__':
    main()
