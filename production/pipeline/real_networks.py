"""Stage-2 real test sources: parsing, window check, graph, budget and draws.

In plain words: loads each raw dataset, checks that no time window is nearly empty, builds
the graph and draws its test observations, exactly like the stage-1 sources: undirected, self-events removed, W=5 equal
windows over the observed range, the 10% expected-discovered-active-dyad-window
calibration of all arms, three main draws per arm (one for a saturated H panel),
and the R/H panel release. One timestamped record is one event; proximity
sources drop identical (u, v, t) records exactly as sp_* and copenhagen_bluetooth do.
The file readers and the window rule (load_raw, checked_graph) are in src/study/data.py.
"""
import tempfile
from pathlib import Path
import numpy as np
from study.common import SAMPLER_DRAWS, digest, draws_for, sha, write_json
from study.data import checked_graph, load_graph, load_raw, save_graph, window_shares
from study.sampling import calibrate
from study.walk import Walk
from .core import ARMS, CFG, STAGE2_FOLD, RAW, STAGE1, Cache
from .observe import (draw_block, et_row, fold_median, observation_record, offline_predictions,
                      training_truths)


# ---------------------------------------------------------------- source stage
# Stage 'source': one real source from raw file to prepared graph, budget, test draws,
# offline predictions and a summary of facts (sizes, cleaning, window shares).
def source_stage(task, out, _inputs):
    key = task.params['source']
    spec = CFG['stage2_sources'][key]
    build = Path(tempfile.mkdtemp(prefix='v11ext_walk_'))
    if spec.get('reuse_stage1_graph'):
        g = load_graph(STAGE1/'prepared/graphs'/key)
        from study.common import read_json
        budget = read_json(STAGE1/'prepared/calibration'/f'{key}.json')
        walk = Walk(g, build)
        windows = {'shares': window_shares(g), 'min_share': min(window_shares(g)),
                   'threshold': CFG['windows']['min_share'], 'trimmed_records': 0,
                   'flagged': min(window_shares(g)) < CFG['windows']['min_share'],
                   'rule': 'stage-1 graph reused unchanged'}
        facts = {'reused_v10_graph': True, 'graph_sha256': sha(STAGE1/'prepared/graphs'/key/'graph.npz')}
    else:
        frame, facts = load_raw(key, spec, RAW)
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
    fold = STAGE2_FOLD[key]
    median = fold_median(training_truths(STAGE1), fold)
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
