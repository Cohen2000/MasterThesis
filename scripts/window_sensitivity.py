#!/usr/bin/env python3
"""How the true persistence depends on the number of time windows W (truth only, no sampling).

The thesis cuts every network into W = 5 windows and calls a pair persistent if it is active in at
least two of them. This script rebuilds the 12 networks and their time-shuffled copies, checks that
W = 5 gives exactly data/truth.json, and recomputes the share of pairs active in at least two of W
windows for W = 2 ... 20. Per W it reports the mean and largest change against W = 5 and the rank
correlation of the networks with their W = 5 values (Sections 3.2 and 5.7 of the thesis).

  python scripts/window_sensitivity.py      writes results/window_sensitivity.csv (needs data/raw)
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import DATA, OUT, REAL, TWIN, read_json, write_csv  # noqa: E402
from study.data import network, twin  # noqa: E402

GRID = (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)


def rho_2(g, W):
    """Share of pairs active in at least two of W equal windows (a cut goes to the later window, as in the study)."""
    cuts = [g.horizon[0]+(g.horizon[1]-g.horizon[0])*j/W for j in range(1, W)]
    w = np.searchsorted(cuts, g.t, side='right')
    K = np.bincount(np.unique(g.pair*W+w)//W, minlength=g.D)
    return float(np.mean(K >= 2))


def main():
    truth = read_json(DATA/'truth.json')
    graphs = {}
    for key in REAL:
        g = network(key)
        graphs[key], graphs[key+TWIN] = g, twin(g)
    value = {(s, W): rho_2(g, W) for s, g in graphs.items() for W in GRID}
    for s in graphs:
        if value[s, 5] != truth[s][0]: raise SystemExit(f'{s}: W = 5 does not reproduce data/truth.json')
    rows = []
    for shuffled in (0, 1):
        sources = [s for s in graphs if s.endswith(TWIN) == bool(shuffled)]
        ref = np.array([value[s, 5] for s in sources])
        for W in GRID:
            now = np.array([value[s, W] for s in sources])
            rows.append({'shuffled': shuffled, 'W': W, 'mean_rho_2': float(now.mean()),
                         'mean_absolute_change_vs_W5': float(np.abs(now-ref).mean()),
                         'max_absolute_change_vs_W5': float(np.abs(now-ref).max()),
                         'spearman_vs_W5': float(spearmanr(now, ref).statistic)})
    write_csv(OUT/'window_sensitivity.csv', rows)
    print('written results/window_sensitivity.csv')


if __name__ == '__main__':
    main()
