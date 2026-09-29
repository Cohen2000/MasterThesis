#!/usr/bin/env python3
"""How the true persistence depends on the number of time windows W (truth only, no sampling).

In plain words: the study cuts every network into W = 5 windows. This script rebuilds the
12 real networks and their time-shuffled copies from the raw data, checks that W = 5 gives
exactly TRUTH.json, and recomputes the truth for other W. Two readings are compared with
W = 5: a fixed k (rho_k = share of pairs active in at least k windows) and a fixed share of
the windows (k = ceil(0.4 W), which is k = 2 at W = 5). The question is whether the ordering
of the networks by persistence depends on W. No method is re-run.

usage: python scripts/window_sensitivity.py [--out docs/results/final]   (needs data/raw)
"""
import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('PIPELINE_RAW', str(ROOT / 'data/raw'))
sys.path.insert(0, str(ROOT / 'src'))
from study.common import STAGE1_REAL, write_csv  # noqa: E402
from study.data import prepare_real  # noqa: E402
from study.surrogates import shuffle  # noqa: E402
from pipeline.core import CFG  # noqa: E402
from pipeline.real_networks import checked_graph, load_raw  # noqa: E402

GRID = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)
SHARE = 0.4                                   # fixed-share reading: k = ceil(0.4 W)


def networks():
    """The 24 networks exactly as the study builds them (checked against TRUTH.json)."""
    graphs, tmp = {}, Path(tempfile.mkdtemp())
    for key in (*STAGE1_REAL, 'nr_radoslaw_email'):
        graphs[key] = prepare_real(key, ROOT / 'data/raw', tmp / key)
    for key, spec in CFG['stage2_sources'].items():
        if not spec.get('reuse_stage1_graph'):
            graphs[key] = checked_graph(key, load_raw(key, spec)[0], spec['proximity'])[0]
    for key in list(graphs):
        graphs[key + '__pwt'] = shuffle(graphs[key])
    return graphs


def windows(g, W):
    """Window of every event for W equal windows (same rule as the study: a cut goes to the later window)."""
    cuts = [g.horizon[0] + (g.horizon[1] - g.horizon[0]) * j / W for j in range(1, W)]
    return np.searchsorted(cuts, g.t, side='right')


def profile(g, W):
    """rho_k for k = 1..W, plus the number of empty windows and the smallest window's share of events."""
    w = windows(g, W)
    K = np.bincount(np.unique(g.pair * W + w) // W, minlength=g.D)
    events = np.bincount(w, minlength=W)
    return ({k: float(np.mean(K >= k)) for k in range(1, W + 1)},
            int((events == 0).sum()), float(events.min() / events.sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'docs/results/final')
    a = ap.parse_args()
    truth = json.loads((a.out / 'TRUTH.json').read_text())
    graphs = networks()
    prof = {(s, W): profile(g, W) for s, g in graphs.items() for W in GRID}
    for s in graphs:        # W = 5 must reproduce the study's truth exactly
        if not np.allclose([prof[s, 5][0][k] for k in range(2, 6)], truth[s], atol=1e-12, rtol=0):
            raise AssertionError(f'{s}: W = 5 does not reproduce TRUTH.json')
    blocks = {'real': sorted(s for s in graphs if not s.endswith('__pwt')),
              'time-shuffled copy': sorted(s for s in graphs if s.endswith('__pwt'))}
    per_network, summary = [], []
    for block, sources in blocks.items():
        for W in GRID:
            k_share = math.ceil(SHARE * W)
            for s in sources:
                p, empty, smallest = prof[s, W]
                per_network.append({'network': s, 'group': block, 'W': W, 'rho_2': p[2],
                                    'k_share': k_share, 'rho_share': p[k_share] if k_share >= 2 else None,
                                    'empty_windows': empty, 'smallest_window_event_share': smallest})
            readings = [('fixed k', k) for k in range(2, min(W, 5) + 1)] + \
                       ([('fixed share', k_share)] if k_share >= 2 else [])
            for reading, k in readings:
                k5 = 2 if reading == 'fixed share' else k
                now = np.array([prof[s, W][0][k] for s in sources])
                ref = np.array([prof[s, 5][0][k5] for s in sources])
                summary.append({'group': block, 'W': W, 'reading': reading, 'k': k, 'networks': len(sources),
                                'mean_value': float(now.mean()), 'mean_absolute_change_vs_W5': float(np.abs(now - ref).mean()),
                                'max_absolute_change_vs_W5': float(np.abs(now - ref).max()),
                                'spearman_vs_W5': float(spearmanr(now, ref).statistic),
                                'networks_with_empty_windows': sum(prof[s, W][1] > 0 for s in sources)})
    write_csv(a.out / 'W_SENSITIVITY.csv', summary)
    write_csv(a.out / 'W_SENSITIVITY_NETWORKS.csv', per_network)
    (a.out / 'W_SENSITIVITY.md').write_text(markdown(summary))
    print('W sensitivity written:', len(summary), 'summary rows,', len(per_network), 'network rows')


def markdown(summary):
    lines = ['# Number of time windows', '',
             'How to read this: the study uses W = 5 windows. Here the true persistence of the 12 real networks',
             'and their time-shuffled copies is recomputed for other W (no sampling, no methods). "Fixed k" keeps',
             'k = 2 (active in at least 2 windows); "fixed share" uses k = ceil(0.4 W), i.e. active in at least 40%',
             'of the windows, which equals the study target at W = 5. Spearman is the rank correlation of the',
             'networks with their W = 5 values: 1 means the same ordering. Computed by `scripts/window_sensitivity.py`,',
             'which first checks that W = 5 reproduces `TRUTH.json` exactly.', '']
    for block in ('real', 'time-shuffled copy'):
        lines += [f'## {"Real networks" if block == "real" else "Time-shuffled copies"}', '',
                  '| W | Reading | k | Mean rho | Mean change vs W = 5 | Max change | Spearman vs W = 5 | Networks with empty windows |',
                  '|---:|---|---:|---:|---:|---:|---:|---:|']
        for r in summary:
            if r['group'] == block and (r['reading'] == 'fixed share' or r['k'] == 2):
                lines.append(f"| {r['W']} | {r['reading']} | {r['k']} | {r['mean_value']:.3f} | "
                             f"{r['mean_absolute_change_vs_W5']:.3f} | {r['max_absolute_change_vs_W5']:.3f} | "
                             f"{r['spearman_vs_W5']:.3f} | {r['networks_with_empty_windows']}/{r['networks']} |")
        lines.append('')
    lines += ['All k and every network: `W_SENSITIVITY.csv`, `W_SENSITIVITY_NETWORKS.csv`.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
