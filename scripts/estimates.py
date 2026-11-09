#!/usr/bin/env python3
"""Observed persistence, statistical model and learned reference (ExtraTrees) for the 288 samples.

  python scripts/estimates.py              observed persistence and statistical model (about 1 minute)
  python scripts/estimates.py --learned    also the learned reference (about 1 hour; needs all 16 training
                                           networks of config/networks.yaml in data/raw)

Writes results/estimates.csv and prints the largest difference to data/predictions.csv.
"""
import argparse
import csv
import os
import sys
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')     # the parallel workers must not compete for the cores
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.answers import valid_profile  # noqa: E402
from study.common import ARMS, DATA, OUT, TRAIN, TRAIN_DRAWS, fold, read_json, write_csv  # noqa: E402
from study.data import network  # noqa: E402
from study.estimators import mle, plugin  # noqa: E402
from study.extratrees import predict, row, select  # noqa: E402
from study.sample import parse, text  # noqa: E402
from study.sampling import draw, draws, sizes  # noqa: E402
from study.synthetic import training_network, training_specs  # noqa: E402
from study.walk import Walk  # noqa: E402


def training_rows(g, kind):
    """The training rows of one network for the learned reference, per sampler."""
    walk = Walk(g)
    size = sizes(g, walk)
    rows = {}
    for arm in ARMS:
        for index in range(1, draws(g, arm, size, TRAIN_DRAWS)+1):
            r = row(parse(text(g, arm, size, *draw(g, arm, index, 'training' if kind == 'real' else 'pool_train', size, walk))),
                    g.key, kind, g.truth)
            if r: rows.setdefault(arm, []).append(r)
    return rows


def synthetic_rows(spec):
    return training_rows(training_network(spec), spec['family'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--learned', action='store_true', help='also fit the learned reference')
    learned = ap.parse_args().learned
    samples = [read_json(p) for p in sorted((DATA/'samples').glob('*.json'))]
    out, test, train = [], {}, {}
    for s in samples:
        o = parse(s['block'])
        for method, p in (('observed', plugin(o)), ('statistical_model', mle(o))):
            out.append({'sample_id': s['id'], 'method': method, 'prediction': p})
        test.setdefault(s['arm'], []).append((s, row(o, s['graph_id'], None, [0.]*4)))
    if learned:
        for key in TRAIN:                               # the 16 real training networks
            for arm, rows in training_rows(network(key), 'real').items(): train.setdefault(arm, []).extend(rows)
        with Pool() as pool:                            # the 400 synthetic training networks
            for part in pool.imap(synthetic_rows, training_specs(), chunksize=4):
                for arm, rows in part.items(): train.setdefault(arm, []).extend(rows)
        for arm in ARMS:
            for f in sorted({fold(s['graph_id']) for s in samples}):
                rows = [(s, r) for s, r in test[arm] if fold(s['graph_id']) == f]
                choice = select(train[arm], arm, f, 'v10_et_nested')
                fitted = predict(train[arm], [r for _, r in rows], arm, f, choice, 'v10_et_final')
                out += [{'sample_id': s['id'], 'method': 'learned_reference', 'prediction': valid_profile(p)}
                        for (s, _), p in zip(rows, fitted)]
            print(f'learned reference, sampler {arm} done', flush=True)
    write_csv(OUT/'estimates.csv', out)

    frozen = {(r['sample_id'], r['method']): [float(r[f'rho_{k}']) for k in range(2, 6)]
              for r in csv.DictReader((DATA/'predictions.csv').open()) if r['method'] in ('observed', 'statistical_model', 'learned_reference')}
    gap = {}
    for r in out:
        gap[r['method']] = max(gap.get(r['method'], 0.), float(np.max(np.abs(np.subtract(r['prediction'], frozen[r['sample_id'], r['method']])))))
    print('largest difference to data/predictions.csv:', ', '.join(f'{k} {v:.1g}' for k, v in gap.items()))


if __name__ == '__main__':
    main()
