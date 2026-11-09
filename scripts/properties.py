#!/usr/bin/env python3
"""The 99 properties behind "What makes a network hard" (Section 5.6 of the thesis): 80 properties of each network
and its time-shuffled copy, computed from the raw data, and 17 properties of each sample, read from
the sample text. scripts/thesis.py adds two more (the signed and absolute bias of the observed
persistence) and correlates each property with the methods' median error.

  python scripts/properties.py          writes results/network_properties.csv and results/sample_properties.csv
  python scripts/properties.py --samples-only    only the sample properties (no raw data needed)

Both files are compared with their frozen copies in data/. Needs data/raw for the network properties.
"""
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.common import DATA, OUT, REAL, group, read_json  # noqa: E402
from study.data import network, twin  # noqa: E402
from study.estimators import mle_counts  # noqa: E402
from study.sample import parse  # noqa: E402

warnings.filterwarnings('ignore', category=RuntimeWarning)     # pairs with too few gaps give NaN, as intended


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2*np.arange(1, n+1) - n - 1).dot(x)/(n*x.sum())) if x.sum() > 0 else 0.0


def cv(x):
    x = np.asarray(x, float); return float(x.std()/x.mean()) if x.mean() > 0 else 0.0


def skew(x):
    x = np.asarray(x, float); s = x.std(); return float(((x - x.mean())**3).mean()/s**3) if s > 0 else 0.0


def burst(x):
    x = np.asarray(x, float)
    if len(x) < 2: return np.nan
    m, s = x.mean(), x.std(); return float((s - m)/(s + m)) if s + m > 0 else np.nan


def jac(a, b):
    u = np.sum(a | b); return float(np.sum(a & b)/u) if u else np.nan


def static_props(g):
    """Properties of the aggregated network (identical for a network and its time-shuffled copy)."""
    N, D = g.N, g.D; m = g.m.astype(float); a, b = g.ends[:, 0], g.ends[:, 1]
    deg = np.bincount(a, minlength=N) + np.bincount(b, minlength=N)
    stren = np.bincount(a, weights=m, minlength=N) + np.bincount(b, weights=m, minlength=N)
    A = sparse.coo_matrix((np.ones(2*D), (np.r_[a, b], np.r_[b, a])), shape=(N, N)).tocsr()
    Wm = sparse.coo_matrix((np.r_[m, m], (np.r_[a, b], np.r_[b, a])), shape=(N, N)).tocsr()
    # triangles by edge iteration over sorted neighbour lists
    nb = [set(A.indices[A.indptr[i]:A.indptr[i+1]]) for i in range(N)]
    tri_e = np.array([len(nb[u] & nb[v]) if len(nb[u]) < len(nb[v]) else len(nb[v] & nb[u]) for u, v in g.ends])
    triangles = tri_e.sum()/3
    wedges = float(np.sum(deg*(deg - 1))/2)
    tri_node = np.bincount(a, weights=tri_e, minlength=N) + np.bincount(b, weights=tri_e, minlength=N)
    with np.errstate(invalid='ignore', divide='ignore'):
        local = np.where(deg > 1, (tri_node/2)/(deg*(deg - 1)/2), 0.0)
    ncomp, lab = sparse.csgraph.connected_components(A, directed=False)
    big = np.bincount(lab).argmax(); keep = np.flatnonzero(lab == big)

    def gap(M):
        M = M[keep][:, keep]; d = np.asarray(M.sum(axis=1)).ravel()
        Dm = sparse.diags(1/np.sqrt(d)); S = Dm @ M @ Dm
        vals = eigsh(S, k=3, which='LA', return_eigenvectors=False, tol=1e-6, maxiter=20000)
        vals = np.sort(vals)[::-1]; return float(1 - vals[1])
    da, db = deg[a].astype(float), deg[b].astype(float)
    assort = float(np.corrcoef(np.r_[da, db], np.r_[db, da])[0, 1])
    top = np.sort(m)[::-1]
    return {
        'nodes': N, 'pairs': D, 'events': g.M, 'density': 2*D/(N*(N - 1)), 'mean_degree': 2*D/N,
        'median_degree': float(np.median(deg)), 'max_degree': int(deg.max()), 'degree_cv': cv(deg), 'degree_gini': gini(deg),
        'degree_skew': skew(deg), 'max_over_mean_degree': float(deg.max()/deg.mean()), 'share_degree_one': float(np.mean(deg == 1)),
        'degree_assortativity': assort, 'transitivity': float(3*triangles/wedges) if wedges else 0.0,
        'mean_local_clustering': float(local.mean()), 'components': int(ncomp), 'largest_component_share': float(len(keep)/N),
        'events_per_pair': g.M/D, 'median_events_per_pair': float(np.median(m)), 'max_events_per_pair': float(m.max()),
        'events_per_pair_cv': cv(m), 'events_per_pair_gini': gini(m), 'events_per_pair_skew': skew(np.log(m)),
        'effective_pairs': float(1/np.sum((m/m.sum())**2)), 'effective_pair_share': float(1/np.sum((m/m.sum())**2))/D,
        'share_one_event': float(np.mean(m == 1)), 'events_top1pct_pairs': float(top[:max(1, D//100)].sum()/m.sum()),
        'events_top10pct_pairs': float(top[:max(1, D//10)].sum()/m.sum()),
        'events_per_node_cv': cv(stren), 'events_per_node_gini': gini(stren),
        'degree_vs_events_per_node': float(pd.Series(deg).corr(pd.Series(stren), method='spearman')),
        'pair_events_vs_end_degrees': float(pd.Series(m).corr(pd.Series(da*db), method='spearman')),
        'walk_spectral_gap_weighted': gap(Wm), 'walk_spectral_gap_unweighted': gap(A),
    }


def temporal_props(g):
    """Properties that depend on when interactions happen (they differ between a network and its copy)."""
    c = g.counts; on = c > 0; K = g.K; m = g.m.astype(float); D = g.D
    ew = c.sum(axis=0).astype(float); aw = on.sum(axis=0).astype(float)
    first = on.argmax(axis=1); last = 4 - on[:, ::-1].argmax(axis=1)
    new = [float(np.mean(first[on[:, j]] == j)) for j in range(1, 5)]
    # pair lifetimes and inter-event times from the event list
    order = np.lexsort((g.t, g.pair)); p, t = g.pair[order], g.t[order]
    span = g.horizon[1] - g.horizon[0]
    tmin = np.full(D, np.inf); tmax = np.full(D, -np.inf)
    np.minimum.at(tmin, p, t); np.maximum.at(tmax, p, t)
    life = (tmax - tmin)/span
    same = p[1:] == p[:-1]; gaps = (t[1:] - t[:-1])[same]; gp = p[1:][same]
    gaps_pos = gaps[gaps > 0]
    # per-pair burstiness for pairs with at least 3 events
    s1 = np.bincount(gp, weights=gaps, minlength=D); s2 = np.bincount(gp, weights=gaps**2, minlength=D)
    n = np.bincount(gp, minlength=D).astype(float); ok = n >= 2
    mu = s1[ok]/n[ok]; sd = np.sqrt(np.maximum(s2[ok]/n[ok] - mu**2, 0))
    pb = np.where(sd + mu > 0, (sd - mu)/(sd + mu), np.nan)
    # node activity per window
    a, b = g.ends[:, 0], g.ends[:, 1]
    node_on = np.zeros((g.N, 5), bool)
    for j in range(5):
        node_on[a[on[:, j]], j] = True; node_on[b[on[:, j]], j] = True
    nK = node_on.sum(axis=1)
    # topological overlap of a node's partners between consecutive windows (temporal correlation coefficient)
    tc = []
    for j in range(4):
        both = on[:, j] & on[:, j+1]
        num = np.bincount(a[both], minlength=g.N) + np.bincount(b[both], minlength=g.N)
        d1 = np.bincount(a[on[:, j]], minlength=g.N) + np.bincount(b[on[:, j]], minlength=g.N)
        d2 = np.bincount(a[on[:, j+1]], minlength=g.N) + np.bincount(b[on[:, j+1]], minlength=g.N)
        den = np.sqrt(d1*d2.astype(float)); okn = den > 0
        tc.append(float(np.sum(num[okn]/den[okn])/g.N))
    cells = c[on].astype(float)
    pers = K >= 2
    consec = np.array([np.all(np.diff(np.flatnonzero(r)) == 1) for r in on[pers]]) if pers.any() else np.array([])
    late = (c[:, 2:] > 0).sum(axis=1)
    fit = mle_counts([0.] + [float(np.sum(late == j)) for j in (1, 2, 3)], 3)
    out = {
        **{f'rho{k}': float(np.mean(K >= k)) for k in range(2, 6)},
        **{f'share_{k}_windows': float(np.mean(K == k)) for k in range(2, 6)},
        'mean_active_windows': float(K.mean()), 'share_one_window_several': float(np.mean((K == 1) & (m > 1))),
        'only_early_pairs': float(np.mean(c[:, 2:].sum(axis=1) == 0)), 'only_late_pairs': float(np.mean(c[:, :2].sum(axis=1) == 0)),
        **{f'early_late_mismatch_{k}': float(fit[k-2] - np.mean(K >= k)) for k in range(2, 6)},
        'abs_early_late_mismatch_2': abs(float(fit[0] - np.mean(K >= 2))),
        'activity_trend_log_last_first': float(np.log((ew[4] + 1)/(ew[0] + 1))),
        'events_per_window_cv': cv(ew), 'events_share_window1': float(ew[0]/ew.sum()), 'events_share_window5': float(ew[4]/ew.sum()),
        'active_pairs_share_window1': float(aw[0]/D), 'active_pairs_share_window5': float(aw[4]/D), 'active_pairs_per_window_cv': cv(aw),
        'active_pairs_trend_log_last_first': float(np.log((aw[4] + 1)/(aw[0] + 1))),
        'jaccard_consecutive_windows': float(np.nanmean([jac(on[:, j], on[:, j+1]) for j in range(4)])),
        'jaccard_first_last_window': jac(on[:, 0], on[:, 4]), 'new_pair_share_per_window': float(np.mean(new)),
        'mean_pair_lifetime': float(life.mean()), 'median_pair_lifetime': float(np.median(life)),
        'mean_lifetime_persistent_pairs': float(life[pers].mean()) if pers.any() else 0.0,
        'burstiness_pooled_gaps': burst(gaps_pos), 'mean_pair_burstiness': float(np.nanmean(pb)) if len(pb) else np.nan,
        'share_gaps_zero': float(np.mean(gaps == 0)) if len(gaps) else 0.0,
        'events_per_active_window': float(cells.mean()), 'share_active_windows_one_event': float(np.mean(cells == 1)),
        'events_per_active_window_cv': cv(cells), 'node_share_active_2plus_windows': float(np.mean(nK >= 2)),
        'node_share_active_5_windows': float(np.mean(nK == 5)), 'node_mean_active_windows': float(nK.mean()),
        'temporal_correlation_coefficient': float(np.mean(tc)),
        'share_persistent_with_consecutive_windows': float(consec.mean()) if len(consec) else 0.0,
        'event_weighted_rho2': float(m[pers].sum()/m.sum()), 'event_share_pairs_all_windows': float(m[K == 5].sum()/m.sum()),
        'events_vs_active_windows': float(pd.Series(m).corr(pd.Series(K.astype(float)), method='spearman')),
        'mean_gap_between_active_windows': float(np.mean((last - first + 1 - K)[pers])) if pers.any() else 0.0,
    }
    return out


def sample_props(block):
    o = parse(block); tab = o['table']; arm = o['arm']
    acc = [j for j in range(5) if o['Temporal_access'][j]]
    pat = [r[0] for r in tab]; dy = np.array([r[1] for r in tab], float); ev = np.array([r[2] for r in tab], float)
    k = np.array([sum(ch == '1' for ch in s) for s in pat]); D = dy.sum()
    epw = np.array([x for x in o['Events_per_window'] if x is not None], float)
    full = k == len(acc)
    out = {'N_obs': o['N_obs'], 'D_obs': o['D_obs'], 'M_obs': o['M_obs'], 'pairs_per_node_obs': o['D_obs']/o['N_obs'],
           'events_per_pair_obs': o['M_obs']/o['D_obs'], 'observed_rho2': float(dy[k >= 2].sum()/D),
           'observed_rho3': float(dy[k >= 3].sum()/D), 'observed_share_all_accessible': float(dy[full].sum()/D),
           'observed_events_per_window_cv': cv(epw), 'observed_events_share_last_window': float(epw[-1]/epw.sum()),
           'observed_events_per_pair_one_window': float(ev[k == 1].sum()/max(dy[k == 1].sum(), 1)),
           'observed_events_per_pair_all_accessible': float(ev[full].sum()/max(dy[full].sum(), 1)),
           'observed_event_share_all_accessible': float(ev[full].sum()/ev.sum()),
           'observed_event_weighted_rho2': float(ev[k >= 2].sum()/ev.sum())}
    if arm == 'S':
        inv = np.array([r[3] for r in tab]); tr = np.array([r[4] for r in tab], float)
        out['walk_mean_inverse_events'] = float(inv.sum()/D)
        out['walk_traversals_per_pair'] = float(tr.sum()/D)
        out['walk_textbook_rho2'] = float(inv[k >= 2].sum()/inv.sum())
    return out


def compare(new, name, keys):
    """Largest difference to the frozen table in data/ (numbers only)."""
    if not (DATA/name).exists(): return
    old = pd.read_csv(DATA/name).set_index(keys).sort_index()
    new = new.set_index(keys).sort_index()
    if list(new.columns) != list(old.columns) or not new.index.equals(old.index): raise SystemExit(f'{name}: other rows or columns')
    gap = (new.astype(float)-old.astype(float)).abs().max().max()
    print(f'{name}: largest difference to data/{name} {gap:.1g}')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--samples-only' not in sys.argv:
        truth, rows = read_json(DATA/'truth.json'), []
        for key in REAL:
            g = network(key)
            tw = twin(g)
            for h in (g, tw):
                if h.truth != truth[h.key]: raise SystemExit(f'{h.key}: truth differs from data/truth.json')
            static = static_props(g)                 # the same for a network and its copy
            rows.append({'network': key, 'shuffled': 0, **static, **temporal_props(g)})
            rows.append({'network': key, 'shuffled': 1, **static, **temporal_props(tw)})
            print(key, 'done', flush=True)
        net = pd.DataFrame(rows)
        net.to_csv(OUT/'network_properties.csv', index=False)
        compare(net, 'network_properties.csv', ['network', 'shuffled'])
    rows = []
    for path in sorted((DATA/'samples').glob('*.json')):
        s = read_json(path)
        rows.append({'sample_id': s['id'], 'network': s['graph_id'].removesuffix('__pwt'),
                     'shuffled': int(group(s['graph_id']) == 'surrogate'), 'sampler': s['arm'], **sample_props(s['block'])})
    smp = pd.DataFrame(rows)
    smp.to_csv(OUT/'sample_properties.csv', index=False)
    compare(smp, 'sample_properties.csv', ['sample_id', 'network', 'shuffled', 'sampler'])


if __name__ == '__main__':
    main()
