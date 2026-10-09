#!/usr/bin/env python3
"""Two checks on the complete networks, without any model.

  walk   arm S: 1000 walks per network. How far is the observed share from the truth, and does
         the re-weighted share (each pair weighted by visits / events) remove that bias?
  time   arm H: what seeing only three of five windows does to the observed share and to the MLE.

Writes results/WALK.csv and results/HISTORY.csv and prints how far they are from the frozen
tables in docs/results/final. Needs results/samples and the raw networks.
"""
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import BUDGET, FINAL, LABEL, OUT, REAL, family, group, read_json, seed, write_csv  # noqa: E402
from study.data import network, twin  # noqa: E402
from study.estimators import mle_counts  # noqa: E402
from study.sample import parse  # noqa: E402
from study.synthetic import test_networks  # noqa: E402
from study.walk import Walk, walk_length  # noqa: E402

WALKS = 1000


def walk_check(g):
    walk = Walk(g)
    L = walk_length(g, walk, BUDGET*g.cells)
    K, truth = g.K, g.truth[0]
    share = {'plugin': [], 'ratio': []}
    pairs, own, ratio_size, weight_size = [], [], [], []
    seeds = [seed('v10_walk_audit', g.key, LABEL['S'], i) for i in range(1, WALKS+1)]
    for first in range(0, WALKS, 16):
        for visits in walk.run(seeds[first:first+16], L, True)[2]:
            seen, w = visits > 0, visits/g.m
            share['plugin'].append(float(seen[K >= 2].sum()/seen.sum()))
            share['ratio'].append(float(w[K >= 2].sum()/w.sum()))
            pairs.append(int(seen.sum())); own.append(1-seen.sum()/L)
            weight_size.append(float(w.sum()**2/np.square(w).sum()))
            ratio_size.append(float(w.sum()**2/(visits/np.square(g.m.astype(float))).sum()))
    stat = {k: (float(np.mean(v)-truth), float(np.std(v, ddof=1)), float(np.sqrt(np.mean((np.array(v)-truth)**2)))) for k, v in share.items()}
    shift = float(g.m[K >= 2].sum()/g.M)-truth       # what the walk's preference for busy pairs does on its own
    needed = bool(group(g.key) != 'synthetic' and abs(shift) > .05)
    return {'graph_id': g.key, 'stratum': group(g.key), 'L': L, 'stationary_shift_rho2': shift,
            'plugin_rho2_bias': stat['plugin'][0], 'plugin_rho2_sd': stat['plugin'][1],
            'design_S_rho2_bias': stat['ratio'][0], 'design_S_rho2_sd': stat['ratio'][1], 'gate_applicable': needed,
            # correctable: the re-weighted share removes at least 90% of the bias and halves the error
            'gate_pass': bool(not needed or (abs(stat['ratio'][0]) <= .1*abs(stat['plugin'][0]) and stat['ratio'][2] <= .5*stat['plugin'][2])),
            'weight_ess_mean': float(np.mean(weight_size)), 'ratio_ess_mean': float(np.mean(ratio_size)),
            'revisit_rate_mean': float(np.mean(own)), 'distinct_dyads_mean': float(np.mean(pairs))}


def time_check(g, h_samples):
    """K = active windows of a pair among all five, J = among three. The observed share J >= k among
    pairs with J >= 1 is rho_k (1 - numerator loss) / (1 - denominator loss)."""
    truth, active = g.truth, g.counts > 0
    K, rows = active.sum(1), []
    name, tail = {'graph_id': g.key}, {'group': group(g.key), 'family': family(g.key)}
    for part, windows in (('last60', (2, 3, 4)), ('first60', (0, 1, 2))):
        J = active[:, windows].sum(1)
        fit = mle_counts(np.bincount(J, minlength=4).astype(float).tolist(), 3)
        for k in range(2, 6):
            lost, gone = 1-(J >= k).sum()/max(1, (K >= k).sum()), 1-(J >= 1).sum()/(K >= 1).sum()
            share = float(np.mean(J[J >= 1] >= k))
            rows.append({**name, 'level': 'population', 'window': part, 'k': k, 'rho_k': truth[k-2],
                         'numerator_loss': float(lost), 'denominator_loss': float(gone), 'loss_gap': float(lost-gone),
                         'plugin': share, 'plugin_error': share-truth[k-2], 'mle': fit[k-2], 'mle_error': fit[k-2]-truth[k-2], **tail})
    for block in h_samples:           # the actual H samples: time cut plus random nodes
        o = parse(block)
        fit = mle_counts([0.]+[sum(r[1] for r in o['table'] if r[0].count('1') == j) for j in (1, 2, 3)], 3)
        for k in range(2, 6):
            share = sum(r[1] for r in o['table'] if r[0].count('1') >= k)/o['D_obs']
            rows.append({**name, 'level': 'H_sampled', 'window': 'last60', 'k': k, 'rho_k': truth[k-2],
                         'plugin': share, 'plugin_error': share-truth[k-2], 'mle': fit[k-2], 'mle_error': fit[k-2]-truth[k-2], **tail})
    return rows


def gap(new, old, keys):
    """Largest difference between two tables with the same rows (numbers only)."""
    worst = 0.
    for a, b in zip(sorted(new, key=keys), sorted(old, key=keys)):
        for k, v in a.items():
            if isinstance(v, float) and b[k] != '': worst = max(worst, abs(v-float(b[k])))
            elif not isinstance(v, float) and str(v) != b[k]: raise SystemExit(f'{k}: {v} differs from {b[k]}')
    return worst


def main():
    blocks = {}
    for p in sorted((OUT/'samples').glob('*__H-*.json')):
        s = read_json(p); blocks.setdefault(s['graph_id'], []).append(s['block'])
    real = [network(key) for key in REAL]
    twins = [twin(g) for g in real]
    walks = sorted((walk_check(g) for g in [*real, *twins, *test_networks()]), key=lambda r: r['graph_id'])
    times = [r for g in [*real, *twins] for r in time_check(g, sorted(blocks[g.key]))]
    write_csv(OUT/'WALK.csv', walks); write_csv(OUT/'HISTORY.csv', times)
    frozen = lambda name: list(csv.DictReader((FINAL/name).open()))
    print(f"largest difference to the frozen tables: WALK.csv {gap(walks, frozen('WALK.csv'), lambda r: r['graph_id']):.1g}, "
          f"HISTORY.csv {gap(times, frozen('HISTORY.csv'), lambda r: (r['graph_id'], r['level'], r['window'], str(r['k']))):.1g}")


if __name__ == '__main__':
    main()
