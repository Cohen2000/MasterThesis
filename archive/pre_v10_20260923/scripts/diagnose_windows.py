#!/usr/bin/env python3
"""Window-count and target-definition sensitivity of the persistence profile.

For every main graph and W = 2..20 (inherited census grid, including the
historical W in {4,5,8}): rho_k for all k, relative thresholds K >= ceil(tau W),
mean/median window occupancy and one-step persistence
C = #(active in j and j+1) / #(active in j). Also the W = 4 and W = 8 profiles
compared with W = 5 within each evidence block, and the census-style
descriptors of each main graph (shifted windows, event-rank windows, lifetimes,
inter-event times, burstiness, censoring, event-count feasibility bounds).
"""
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from census import W_GRID
from dataset_census import THRESHOLDS, WINDOWS, count_feasibility, relative_cutoff
from main_experiment.common import DIAGNOSTICS, MAIN_KEYS, STRATA, fresh_directory, graph_stratum, in_stratum, write_csv, write_json
from main_experiment.prepare import prepared_graph


def window_row(g, key, W):
    cuts = np.array([g.horizon[0]+(g.horizon[1]-g.horizon[0])*j/W for j in range(1, W)])
    window = np.searchsorted(cuts, g.t, side='right')
    active = np.bincount(g.pair*W+window, minlength=g.D*W).reshape(-1, W) > 0
    K = active.sum(1)
    active_before = np.sum(active[:, :-1])
    row = {'graph_id': key, 'stratum': graph_stratum(key), 'W': W, 'mean_occupancy': float(K.mean()/W),
           'median_occupancy': float(np.median(K/W)),
           'C_one_step': float(np.sum(active[:, :-1] & active[:, 1:])/active_before) if active_before else None}
    row.update({f'rho_{k}': float(np.mean(K >= k)) for k in range(2, W+1)})
    row.update({f'rho_relative_{t}': float(np.mean(K >= relative_cutoff(t, W))) for t in THRESHOLDS})
    row.update({f'event_upper_bound_{k}': float(np.mean(g.m >= k)) for k in range(2, W+1)})
    return row


def compare_with_five(rows):
    """For W in {4, 8}: change of each rho_k / occupancy / persistence against W = 5, per evidence block."""
    by = {(r['graph_id'], r['W']): r for r in rows}
    out = []
    for block in STRATA:
        keys = [k for k in MAIN_KEYS if in_stratum(k, block)]
        for W in (w for w in W_GRID if w != 5):
            for metric in [*(f'rho_{k}' for k in range(2, min(W, 5)+1)), 'mean_occupancy', 'C_one_step']:
                a = np.array([by[k, 5][metric] for k in keys], float)
                b = np.array([by[k, W][metric] for k in keys], float)
                rank = float(spearmanr(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else None
                out.append({'evidence_block': block, 'W': W, 'metric': metric, 'sources': len(keys),
                            'mean_absolute_change': float(np.abs(a-b).mean()),
                            'max_absolute_change': float(np.abs(a-b).max()), 'spearman': rank})
    return out


def census_row(g, key):
    """Inherited census descriptors on the canonical event multiset (surrogate multiplicity retained)."""
    lo, hi = g.horizon; span = hi-lo
    order = np.lexsort((np.arange(g.M), g.t, g.pair))
    pair = g.pair[order]; t = g.t[order]
    first = np.r_[0, np.flatnonzero(pair[1:] != pair[:-1])+1]
    last = np.r_[first[1:]-1, len(pair)-1]
    lifetime = t[last]-t[first]
    same_dyad = pair[1:] == pair[:-1]
    gaps = np.diff(t)[same_dyad]; gap_dyad = pair[1:][same_dyad]
    mean_gap = float(gaps.mean()) if len(gaps) else 0.
    sd_gap = float(gaps.std()) if len(gaps) else 0.
    n = np.bincount(gap_dyad, minlength=g.D)
    total = np.bincount(gap_dyad, weights=gaps, minlength=g.D)
    squares = np.bincount(gap_dyad, weights=gaps**2, minlength=g.D)
    eligible = n >= 2
    means = total[eligible]/n[eligible]
    sds = np.sqrt(np.maximum(0, squares[eligible]/n[eligible]-means**2))
    denominator = sds+means
    burstiness = (sds[denominator > 0]-means[denominator > 0])/denominator[denominator > 0]
    shifted = np.searchsorted(lo+(np.arange(5)+.5)*span/5, g.t, side='right')     # windows shifted by half a width
    K_shifted = np.bincount(np.unique(g.pair*6+shifted)//6, minlength=g.D)
    # Equal-event windows: stable event-rank rule of census.equal_event_windows.
    rank = np.empty(g.M, dtype=np.int64)
    rank[np.argsort(t, kind='stable')] = (np.arange(g.M)*5)//g.M
    K_rank = np.bincount(np.unique(pair*5+rank)//5, minlength=g.D)
    distinct = np.bincount(pair[np.r_[True, (pair[1:] != pair[:-1]) | (t[1:] != t[:-1])]], minlength=g.D)
    median_gap = float(np.median(gaps)) if len(gaps) else None
    row = {'graph_id': key, 'stratum': graph_stratum(key), 'rho2': g.truth[0],
           'rho_shifted_half_window': float(np.mean(K_shifted >= 2)),
           'rho_equal_event_rank_windows': float(np.mean(K_rank >= 2)),
           'rho_event_weighted': float(g.m[g.K >= 2].sum()/g.M),
           'lifetime_mean': float(lifetime.mean()), 'lifetime_median': float(np.median(lifetime)),
           'lifetime_mean_over_horizon': float(lifetime.mean()/span), 'median_interevent_time': median_gap,
           'burstiness_pooled': (sd_gap-mean_gap)/(sd_gap+mean_gap) if sd_gap+mean_gap else None,
           'burstiness_pair_median': float(np.median(burstiness)) if len(burstiness) else None,
           'censoring_share_last_window': float(np.mean(t[first] >= lo+4*span/5)),
           'window_length_over_median_interevent_time': span/5/median_gap if median_gap else None,
           'share_single_event_dyads': float(np.mean(g.m == 1)), 'repeat_rate': float(np.mean(g.m > 1))}
    return row, count_feasibility(key, g.m, g.truth[0], distinct)


def main():
    out = fresh_directory(DIAGNOSTICS/'windows')
    rows = []; census = []; feasibility = []
    for key in MAIN_KEYS:
        g, _ = prepared_graph(key)
        rows += [window_row(g, key, W) for W in WINDOWS]
        descriptors, bounds = census_row(g, key)
        census.append(descriptors); feasibility.append(bounds)
    comparisons = compare_with_five(rows)
    write_csv(out/'window_sensitivity.csv', rows)
    write_csv(out/'W_4_5_8_comparisons.csv', comparisons)
    write_csv(out/'census_sensitivities.csv', census)
    write_csv(out/'count_feasibility.csv', feasibility)
    write_json(out/'report.json', {'W_grid': list(WINDOWS), 'historical_W': list(W_GRID),
                                   'relative_thresholds': list(THRESHOLDS), 'rows': len(rows),
                                   'comparisons': len(comparisons), 'census_graphs': len(census),
                                   'window_convention': 'fixed archive horizon; equal cut points, side=right'})
    print('window diagnostics:', len(rows), 'graph x W rows')


if __name__ == '__main__':
    main()
