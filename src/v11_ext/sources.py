"""The four additional real test sources: parsing, window check, graph, budget and draws.

Same pipeline as the v11 real sources: undirected, self-events removed, W=5 equal
windows over the observed range, the 10% expected-discovered-active-dyad-window
calibration of all arms, three main draws per arm (one for a saturated H panel),
and the v11 R/H panel release. One timestamped record is one event; proximity
sources drop identical (u, v, t) records exactly as sp_* and copenhagen_bluetooth do.
"""
import gzip
import io
import tempfile
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from main_experiment.common import SAMPLER_DRAWS, digest, draws_for, sha, write_json
from main_experiment.data import canonical, load_graph, save_graph
from main_experiment.sampling import calibrate
from main_experiment.walk_v10_audit import Walk
from .core import ARMS, CFG, NEW_FOLD, RAW, V10, Cache
from .observe import (draw_block, et_row, fold_median, observation_record, offline_predictions,
                      training_truths)


# ---------------------------------------------------------------- raw records
def load_raw(key, spec, raw_dir=None):
    """Timestamped records (u, v, t) of one source plus parser facts; fails on any
    layout that is not one event per record."""
    path = Path(raw_dir or RAW)/spec['file']
    if sha(path) != spec['sha256']: raise ValueError(f'{key}: raw file hash differs from config')
    cols = spec['columns']
    if path.suffix == '.zip':
        with zipfile.ZipFile(path) as z, z.open(spec['member']) as f:
            text = io.TextIOWrapper(f, encoding='utf-8')
            header = text.readline().strip()
            if header != '# source, target, weight, time': raise ValueError(f'{key}: unexpected edge header {header!r}')
            x = pd.read_csv(text, header=None, dtype={cols['u']: str, cols['v']: str})
        if x.shape[1] != 4: raise ValueError(f'{key}: expected four columns')
        weight = x[cols['weight']].to_numpy()
        # The netzschleuder weight column counts multiplicity; every record must be one event.
        if not np.all(weight == 1): raise ValueError(f'{key}: weight column is not 1 throughout; expansion undefined')
        facts = {'weight_values': sorted(map(int, np.unique(weight)))}
    else:
        with gzip.open(path, 'rt') as f:
            header = f.readline().strip()
            if header != spec['header']: raise ValueError(f'{key}: unexpected header {header!r}')
            x = pd.read_csv(f, header=None, dtype={cols['u']: str, cols['v']: str})
        t = x[cols['t']].to_numpy()
        # SocioPatterns 20 s contact records: one row per active 20 s interval, no onset/duration.
        if not np.all(t % 20 == 0): raise ValueError(f'{key}: timestamps are not 20 s records')
        facts = {'record_resolution_seconds': 20, 'onset_duration_expansion': False}
    frame = pd.DataFrame({'u': x[cols['u']].astype(str), 'v': x[cols['v']].astype(str),
                          't': x[cols['t']].astype(float)})
    facts.update(records=int(len(frame)), raw_sha256=spec['sha256'], raw_file=spec['file'],
                 min_timestamp=float(frame.t.min()), max_timestamp=float(frame.t.max()))
    return frame, facts


# ---------------------------------------------------------------- window shares
def window_shares(g):
    return (g.counts.sum(0)/g.M).tolist()


def end_outliers(t, max_share):
    """(low, high) bounds that drop isolated end-of-range timestamps: at most
    max_share of the records at either end, each dropped group separated from the
    rest by a gap longer than one window (a fifth) of the remaining range."""
    s = np.sort(np.asarray(t, float)); n = len(s); limit = int(max_share*n)
    lead = trail = 0
    for _ in range(2):          # alternate once so each end sees the other's trimmed range
        hi = s[n-1-trail]
        lead = max([k for k in range(1, limit+1) if s[k]-s[k-1] > (hi-s[k])/5], default=0)
        lo = s[lead]
        trail = max([k for k in range(1, limit+1) if s[n-k]-s[n-k-1] > (s[n-k-1]-lo)/5], default=0)
    return float(s[lead]), float(s[n-1-trail]), lead, trail


def checked_graph(key, frame, proximity, windows=None):
    """Canonical graph after the documented window-share rule; returns (g, frame, cleaning, report)."""
    windows = windows or CFG['windows']
    g, clean, cleaning = canonical(key, frame, proximity=proximity)
    shares = window_shares(g)
    report = {'shares': shares, 'min_share': min(shares), 'threshold': windows['min_share'],
              'trimmed_records': 0, 'flagged': False, 'rule': 'none needed'}
    if min(shares) >= windows['min_share']:
        return g, clean, cleaning, report
    lo, hi, lead, trail = end_outliers(clean.t, windows['max_trim_share'])
    if lead or trail:
        kept = frame[(frame.t >= lo) & (frame.t <= hi)]
        g2, clean2, cleaning2 = canonical(key, kept, proximity=proximity)
        if min(window_shares(g2)) >= windows['min_share']:
            report.update(shares_before=shares, shares=window_shares(g2), min_share=min(window_shares(g2)),
                          trimmed_records=int(len(frame)-len(kept)), trimmed_leading=lead, trimmed_trailing=trail,
                          rule='isolated end-of-range timestamp outliers trimmed', kept_range=[lo, hi])
            return g2, clean2, cleaning2, report
    report.update(flagged=True, rule='window below threshold not caused by isolated end outliers; kept unchanged')
    return g, clean, cleaning, report


# ---------------------------------------------------------------- source stage
def source_stage(task, out, _inputs):
    key = task.params['source']
    spec = CFG['new_sources'][key]
    build = Path(tempfile.mkdtemp(prefix='v11ext_walk_'))
    if spec.get('reuse_v10_graph'):
        g = load_graph(V10/'prepared/graphs'/key)
        from main_experiment.common import read_json
        budget = read_json(V10/'prepared/calibration'/f'{key}.json')
        walk = Walk(g, build)
        windows = {'shares': window_shares(g), 'min_share': min(window_shares(g)),
                   'threshold': CFG['windows']['min_share'], 'trimmed_records': 0,
                   'flagged': min(window_shares(g)) < CFG['windows']['min_share'],
                   'rule': 'v10 graph reused unchanged'}
        facts = {'reused_v10_graph': True, 'graph_sha256': sha(V10/'prepared/graphs'/key/'graph.npz')}
    else:
        frame, facts = load_raw(key, spec)
        g, _, cleaning, windows = checked_graph(key, frame, spec['proximity'])
        save_graph(out/'graph', g, {'source_family': key, 'provider': spec['url'], 'cleaning': cleaning,
                                    'parser': facts, 'timestamp_unit': spec['timestamp_unit'],
                                    'window_check': windows})
        budget, walk, validation = calibrate(g, build)
        write_json(out/'validation_volumes.json', validation)
    print(key, 'window shares', ' '.join(f'{s:.4f}' for s in windows['shares']),
          'FLAGGED' if windows['flagged'] else '', flush=True)
    print(f'{key}: N={g.N} D={g.D} M={g.M} cells={g.cells} T={budget["T"]:.1f} n={budget["n_panel"]} '
          f'L={budget["L"]} n_H={budget["n_panel_history"]} p={budget["p"]:.5f} '
          f'H_saturated={budget["h_saturated"]} matched={budget["budget_matched"]}', flush=True)
    write_json(out/'calibration.json', budget)
    fold = NEW_FOLD[key]
    median = fold_median(training_truths(V10), fold)
    features, mles = Cache('features'), Cache('mle')
    offline, rows, X = [], [], {}
    for arm in ARMS:
        for index in range(1, draws_for(arm, budget, 'sample')+1):
            hidden, block, counts = draw_block(g, arm, index, 'sample', budget, walk)
            record = observation_record(g, arm, index, 'sample', budget, hidden, block, counts, 'real')
            record.update(group='new4', fold=fold)
            if record['empty']: raise ValueError(f'{record["id"]}: empty main observation')
            write_json(out/'observations'/f'{record["id"]}.json', record)
            predictions, fit = offline_predictions(block, median, mles)
            for method, p in predictions.items():
                offline.append({'id': record['id'], 'method': method, 'prediction': p,
                                **({'mle_fit': fit} if method == 'mle' else {})})
            row, x = et_row(block, record['id'], key, arm, 'main', 'real', fold, g.truth, features)
            rows.append(row); X.setdefault(arm, []).append(x)
    if len(rows) != sum(draws_for(a, budget, 'sample') for a in ARMS) or max(
            draws_for(a, budget, 'sample') for a in ARMS) != SAMPLER_DRAWS:
        raise AssertionError('incomplete draw grid')
    write_json(out/'offline.json', offline)
    write_json(out/'et_rows.json', rows)
    for arm, xs in X.items(): np.save(out/f'X_{arm}.npy', np.asarray(xs, float))
    write_json(out/'summary.json', {'source': key, 'label': spec['label'], 'fold': fold, 'N': g.N, 'D': g.D,
                                    'M': g.M, 'active_dyad_windows': g.cells, 'truth': g.truth,
                                    'events_per_window': g.counts.sum(0).tolist(), 'window_check': windows,
                                    'parser': facts, 'budget_matched_by_arm': budget['budget_matched_by_arm'],
                                    'unmatched_reasons': budget['unmatched_reasons'],
                                    'h_saturated': budget['h_saturated'], 'n_panel': budget['n_panel'],
                                    'n_panel_history': budget['n_panel_history'], 'L': budget['L'],
                                    'p': budget['p'], 'observations': len(rows), 'median_profile': median,
                                    'graph_digest': digest(g.counts.tolist()),
                                    'cache': {'features_hits': features.hits, 'mle_hits': mles.hits}})
