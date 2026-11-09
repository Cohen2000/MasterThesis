#!/usr/bin/env python3
"""Facts about the 12 networks and their time-shuffled copies, computed from the raw data.

  python scripts/networks.py            writes results/networks.csv and compares it with data/networks.csv

One row per network: size (Table 3.1 of the thesis), true persistence, window length, share of one-off pairs, and
what time truncation and event sampling hide of each network (Section 5.1 of the thesis). Needs data/raw.
"""
import csv
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import BUDGET, DATA, H_WINDOWS, ONLINE, OUT, REAL, read_json, write_csv  # noqa: E402
from study.data import network, twin  # noqa: E402
from study.sampling import keep_probability  # noqa: E402

NAME = {'reality_mining': ('Reality Mining', 'Bluetooth'), 'nr_radoslaw_email': ('Radoslaw', 'email'),
        'sp_malawi': ('Village', 'face-to-face'), 'snap_email_eu': ('Email EU', 'email'),
        'sp_highschool2013': ('High school', 'face-to-face'), 'copenhagen_bluetooth': ('Copenhagen', 'Bluetooth'),
        'sp_hospital': ('Hospital', 'face-to-face'), 'sp_workplace': ('Workplace', 'face-to-face'),
        'lkml_reply': ('Linux mailing list', 'online'), 'snap_collegemsg': ('CollegeMsg', 'online'),
        'snap_mathoverflow': ('MathOverflow', 'online'), 'nr_digg_reply': ('Digg', 'online')}


def truncation_losses(g):
    """Time truncation sees the last three windows. Share of persistent pairs that no longer look
    persistent, and share of all pairs that are hidden entirely (no activity in the last three)."""
    active = g.counts > 0
    K, J = active.sum(1), active[:, 5-H_WINDOWS:].sum(1)
    return float(np.mean(J[K >= 2] < 2)), float(np.mean(J == 0))


def event_loss(g, p):
    """Event sampling keeps each interaction with probability p. Expected persistence of the pairs that
    are still seen, before and after their quiet windows are lost (ratios of expected counts)."""
    seen_in_window = -np.expm1(g.counts*np.log1p(-p))          # a window keeps at least one interaction
    seen = -np.expm1(g.m*np.log1p(-p))                         # a pair keeps at least one interaction
    gone = 1-seen_in_window
    never = gone.prod(axis=1)
    once = sum(seen_in_window[:, j]*np.delete(gone, j, axis=1).prod(axis=1) for j in range(5))
    return float(np.sum(seen*(g.K >= 2))/seen.sum()), float(np.sum(1-never-once)/seen.sum())


def row(g, key, copy):
    name, kind = NAME[key]
    lost_returns, hidden_pairs = truncation_losses(g)
    p = keep_probability(g, BUDGET*g.cells)
    seen_true, seen_observed = event_loss(g, p)
    return {'network': key, 'shuffled': int(copy), 'name': name, 'kind': kind, 'online': int(key in ONLINE),
            'nodes': g.N, 'pairs': g.D, 'interactions': g.M, 'interactions_per_pair': g.M/g.D,
            **{f'rho_{k}': g.truth[k-2] for k in range(2, 6)},
            'window_days': (g.horizon[1]-g.horizon[0])/5/86400,          # all twelve record time in seconds
            'share_one_interaction': float(np.mean(g.m == 1)),
            'truncation_persistent_pairs_losing_return': lost_returns, 'truncation_pairs_hidden': hidden_pairs,
            'event_keep_probability': p, 'event_rho_2_of_pairs_still_seen': seen_true, 'event_rho_2_seen': seen_observed}


def main():
    rows = []
    for key in REAL:
        g = network(key)
        rows += [row(g, key, False), row(twin(g), key, True)]
        print(key, 'done', flush=True)
    truth = read_json(DATA/'truth.json')
    for r in rows:
        if [r[f'rho_{k}'] for k in range(2, 6)] != truth[r['network']+('__pwt' if r['shuffled'] else '')]:
            raise SystemExit(f"{r['network']}: truth differs from data/truth.json")
    write_csv(OUT/'networks.csv', rows)
    frozen = list(csv.DictReader((DATA/'networks.csv').open())) if (DATA/'networks.csv').exists() else []
    if frozen:
        worst = max(abs(float(a[k])-float(v)) for r, a in zip(rows, frozen) for k, v in r.items()
                    if isinstance(v, (int, float)) and not isinstance(v, bool))
        print(f'largest difference to data/networks.csv: {worst:.1g}')


if __name__ == '__main__':
    main()
