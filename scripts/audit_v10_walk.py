#!/usr/bin/env python3
"""Offline interaction-walk gate on the sealed v9 graph inputs; no model calls."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import MAIN_KEYS, graph_stratum, seed, write_csv, write_json
from main_experiment.data import load_graph
from main_experiment.walk_v10_audit import AUDIT_ARM_ID, Walk, walk_length, validate_walk_length

PATHS = 1000
BATCH = 16


def audit_graph(g, build):
    walk = Walk(g, build)
    target = .10 * g.cells
    calibration = walk_length(g, walk, target)
    L = calibration['L']
    validation = validate_walk_length(g, walk, L, target)
    if L <= 0:
        raise ValueError(f'{g.key}: zero calibrated steps')
    k = g.K
    truth = np.array(g.truth)
    degree = walk.degree
    weights = {'uniform': np.ones(g.D),
               'degree': degree[g.ends[:, 0]].astype(float) * degree[g.ends[:, 1]],
               'interaction': g.m.astype(float)}
    targets = {name: [(w[k >= j].sum() / w.sum()) for j in range(2, 6)]
               for name, w in weights.items()}
    values = {name: [] for name in ('plugin', 'design_S', 'design_S_obs')}
    distinct = []; revisits = []; ess = []; ratio_ess_values = []; predicted_bias = []
    seen_count = np.zeros(g.D, dtype=np.int32)
    seeds = [seed('v10_walk_audit', g.key, AUDIT_ARM_ID, i) for i in range(1, PATHS + 1)]
    for first in range(0, PATHS, BATCH):
        _, _, counts, executed = walk.run(seeds[first:first+BATCH], L, True)
        if not np.all(executed == L) or not np.all(counts.sum(axis=1) == L):
            raise AssertionError('walk did not execute exactly L steps')
        for c in counts:
            seen = c > 0
            seen_count += seen
            unique = seen.sum()
            if not unique: raise AssertionError('empty walk')
            w_log = c / g.m
            w_obs = seen / g.m
            for name, w in (('plugin', seen.astype(float)), ('design_S', w_log),
                            ('design_S_obs', w_obs)):
                values[name].append([float(w[k >= j].sum() / w.sum()) for j in range(2, 6)])
            distinct.append(int(unique))
            revisits.append(1 - unique / L)
            ess.append(float(w_log.sum() ** 2 / np.square(w_log).sum()))
            q = c / np.square(g.m.astype(float))
            rho_w2 = q[k >= 2].sum() / q.sum()
            ratio_ess = w_log.sum() ** 2 / q.sum()
            ratio_ess_values.append(float(ratio_ess))
            predicted_bias.append(float((truth[0] - rho_w2) / ratio_ess))
    shift = targets['interaction'][0] - truth[0]
    row = {'graph_id': g.key, 'stratum': graph_stratum(g.key), 'L': L,
           'target_cells': target, 'calibration_mean': calibration['calibration_mean'],
           'validation_mean': validation['validation_mean'], 'validation_mcse': validation['validation_mcse'],
           'budget_relative_error': (validation['validation_mean'] - target) / target,
           'stationary_shift_rho2': shift, 'rho_event_weighted': float(g.m[k >= 2].sum() / g.M),
           'distinct_dyads_mean': float(np.mean(distinct)), 'revisit_rate_mean': float(np.mean(revisits)),
           'weight_ess_mean': float(np.mean(ess)),
           'ratio_ess_mean': float(np.mean(ratio_ess_values)),
           'first_order_ratio_bias_mean': float(np.mean(predicted_bias)),
           'near_certain_inclusion_share': float(np.mean(seen_count >= .99 * PATHS))}
    for name, p in targets.items():
        for j, x in enumerate(p, 2): row[f'stationary_{name}_rho{j}'] = x
    for name, v in values.items():
        a = np.array(v)
        for j in range(4):
            row[f'{name}_rho{j+2}_bias'] = float(a[:, j].mean() - truth[j])
            row[f'{name}_rho{j+2}_sd'] = float(a[:, j].std(ddof=1))
            row[f'{name}_rho{j+2}_rmse'] = float(np.sqrt(np.mean((a[:, j] - truth[j]) ** 2)))
    row['design_S_rho2_bias_mcse'] = row['design_S_rho2_sd'] / np.sqrt(PATHS)
    plugin_bias = row['plugin_rho2_bias']
    row['design_bias_share_of_plugin_bias'] = (row['design_S_rho2_bias'] / plugin_bias
                                                if plugin_bias else None)
    row['gate_applicable'] = bool(row['stratum'] in ('real', 'surrogate') and abs(shift) > .05)
    row['gate_pass'] = bool(not row['gate_applicable'] or
                        (abs(row['design_S_rho2_bias']) <= .1 * abs(plugin_bias) and
                         row['design_S_rho2_rmse'] <= .5 * row['plugin_rho2_rmse']))
    row['not_correctable_at_this_budget'] = bool(row['gate_applicable'] and not row['gate_pass'])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in MAIN_KEYS:
        g = load_graph(args.graphs / key)
        row = audit_graph(g, args.build)
        rows.append(row)
        write_csv(args.output / 'walk_gate.csv', rows)
        write_json(args.output / 'walk_gate.json', {'design_version': 'panel888-access-v10-20260923',
                                                   'paths_per_graph': PATHS, 'rows': rows,
                                                   'gate_pass': all(r['gate_pass'] for r in rows)})
        print(key, 'L', row['L'], 'shift', row['stationary_shift_rho2'],
              'bias', row['design_S_rho2_bias'], 'pass', row['gate_pass'], flush=True)
    if not all(r['gate_pass'] for r in rows): raise SystemExit('WALK GATE FAILED')


if __name__ == '__main__': main()
