#!/usr/bin/env python3
"""Position of each productive P[w,t] surrogate in its parent's null distribution.

For every real parent, 99 further independent timestamp shuffles (indices 1..99)
are drawn offline and audited like the productive one. They are never selected,
trained on or sent to a model; only the rank of the productive surrogate's
profile among them is reported.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import DIAGNOSTICS, REAL_TEST, fresh_directory, seed, write_csv, write_json
from main_experiment.prepare import prepared_graph
from main_experiment.surrogates import NULL_SHUFFLES, audit, shuffle


def main():
    out = fresh_directory(DIAGNOSTICS/'null_model')
    shuffles = []; summary = []
    for parent in REAL_TEST:
        g, _ = prepared_graph(parent)
        productive, _ = prepared_graph(parent+'__pwt')
        null = []
        for index in range(1, NULL_SHUFFLES+1):
            s = shuffle(g, index)
            check = audit(g, s)
            null.append({'parent': parent, 'shuffle': index, 'seed': seed('pwt_null_diagnostic', parent, '', index),
                         'rho': s.truth, 'delta_rho': (np.array(s.truth)-g.truth).tolist(),
                         'collisions': check['surrogate_collisions'], 'invariants_passed': check['passed']})
        shuffles += null
        values = np.array([r['rho'] for r in null])
        summary.append({'parent': parent, 'parent_rho': g.truth, 'productive_rho': productive.truth,
                        'productive_delta_rho': (np.array(productive.truth)-g.truth).tolist(),
                        'null_mean': values.mean(0).tolist(), 'null_min': values.min(0).tolist(),
                        'null_max': values.max(0).tolist(),
                        # (1 + #null <= productive) / (1 + 99), per component k = 2..5
                        'productive_lower_rank_fraction': ((1+(values <= productive.truth).sum(0))/(NULL_SHUFFLES+1)).tolist(),
                        'productive_upper_rank_fraction': ((1+(values >= productive.truth).sum(0))/(NULL_SHUFFLES+1)).tolist(),
                        'null_collisions_median': float(np.median([r['collisions'] for r in null]))})
        print(parent, 'null complete', flush=True)
    write_csv(out/'shuffles.csv', shuffles)
    write_json(out/'report.json', {'null_shuffles': len(shuffles), 'per_parent': NULL_SHUFFLES,
                                   'all_invariants_passed': all(r['invariants_passed'] for r in shuffles),
                                   'summary': summary, 'diagnostic_only': True, 'productive_selection_changed': False})


if __name__ == '__main__':
    main()
