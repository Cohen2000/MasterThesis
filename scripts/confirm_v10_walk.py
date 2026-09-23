#!/usr/bin/env python3
"""Three-graph confirmation of start-transient and walk-length effects."""
import argparse
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import seed, write_json
from main_experiment.data import load_graph
from main_experiment.walk_v10_audit import AUDIT_ARM_ID, Walk

KEYS = ('sp_hospital', 'sp_hospital__pwt', 'sp_highschool2013__pwt')
PATHS = 1000
BATCH = 16


def condition(g, walk, L, mode):
    truth = g.truth[0]
    seeds = [seed('v10_walk_audit', g.key, AUDIT_ARM_ID, i) for i in range(1, PATHS + 1)]
    starts = None
    if mode == 'strength_start':
        strength = np.bincount(g.ends.ravel(), weights=np.repeat(g.m, 2), minlength=g.N)
        draw = np.random.default_rng(seed('v10_strength_start', g.key, AUDIT_ARM_ID))
        starts = draw.choice(g.N, PATHS, p=strength / strength.sum())
    elif mode != 'uniform_4L': raise ValueError(mode)
    steps = L if mode == 'strength_start' else 4 * L
    values = []
    for first in range(0, PATHS, BATCH):
        _, _, counts, executed = walk.run(seeds[first:first+BATCH], steps, True,
                                          None if starts is None else starts[first:first+BATCH])
        if not np.all(executed == steps): raise AssertionError('incomplete walk')
        for c in counts:
            w = c / g.m
            values.append(float(w[g.K >= 2].sum() / w.sum()))
    a = np.asarray(values)
    return {'mode': mode, 'L': steps, 'paths': PATHS, 'design_S_rho2_bias': float(a.mean() - truth),
            'design_S_rho2_sd': float(a.std(ddof=1)),
            'design_S_rho2_bias_mcse': float(a.std(ddof=1) / np.sqrt(PATHS)),
            'design_S_rho2_rmse': float(np.sqrt(np.mean((a - truth) ** 2)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--graphs', type=Path, required=True)
    ap.add_argument('--old-gate', type=Path, required=True)
    ap.add_argument('--build', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--index', type=int, required=True)
    a = ap.parse_args()
    key = KEYS[a.index]
    old = next(r for r in csv.DictReader(a.old_gate.open()) if r['graph_id'] == key)
    g = load_graph(a.graphs / key)
    walk = Walk(g, a.build)
    result = {'graph_id': key, 'calibrated_L': int(old['L']),
              'uniform_L_bias': float(old['design_S_rho2_bias']),
              'uniform_L_mcse': float(old['design_S_rho2_sd']) / np.sqrt(PATHS),
              'conditions': [condition(g, walk, int(old['L']), mode)
                             for mode in ('strength_start', 'uniform_4L')]}
    a.output.mkdir(parents=True, exist_ok=True)
    write_json(a.output / f'{key}.json', result)
    print(result, flush=True)


if __name__ == '__main__': main()
