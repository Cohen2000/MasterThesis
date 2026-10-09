#!/usr/bin/env python3
"""Observed share, training median, MLE and ExtraTrees for the 384 samples.

  python scripts/estimates.py              observed share, training median and MLE (2 minutes)
  python scripts/estimates.py --fits 1     also ExtraTrees, the reported fit (about 1 hour)
  python scripts/estimates.py --fits 11    also the ten further fits on newly drawn training samples

Writes results/estimates.csv and prints how far each method is from the frozen PREDICTIONS.csv.
Needs results/samples (scripts/samples.py) and the raw networks.
"""
import argparse
import csv
import json
import os
import sys
from functools import partial
from multiprocessing import Pool
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS', '1')     # the parallel workers must not compete for the cores
import numpy as np  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from study.answers import valid_profile  # noqa: E402
from study.common import ARMS, FINAL, OUT, TRAIN, TRAIN_DRAWS, fold, read_json, write_csv  # noqa: E402
from study.data import network  # noqa: E402
from study.estimators import mle, plugin  # noqa: E402
from study.extratrees import predict, row, select  # noqa: E402
from study.sample import parse, text  # noqa: E402
from study.sampling import draw, draws, sizes  # noqa: E402
from study.synthetic import training_network, training_specs  # noqa: E402
from study.walk import Walk  # noqa: E402


def training_rows(g, kind, fits):
    """The ExtraTrees training rows of one network, per fit and arm. Fit 0 is the reported one;
    every further fit draws its training samples anew (the labels are fixed parts of the seeds)."""
    walk = Walk(g)
    size = sizes(g, walk)
    rows = {}
    for k in range(fits):
        domain = ('training' if kind == 'real' else 'pool_train')+(f'_r{k}' if k else '')
        for arm in ARMS:
            for index in range(1, draws(g, arm, size, TRAIN_DRAWS)+1):
                r = row(parse(text(g, arm, size, *draw(g, arm, index, domain, size, walk))), g.key, kind, g.truth)
                if r: rows.setdefault((k, arm), []).append(r)
    return rows


def synthetic_rows(spec, fits):
    return training_rows(training_network(spec), spec['family'], fits)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--fits', type=int, default=0, help='number of ExtraTrees fits (0 to 11)')
    fits = ap.parse_args().fits
    samples = [read_json(p) for p in sorted((OUT/'samples').glob('*.json'))]
    train, truth = {}, {}
    for key in TRAIN:                               # the 16 real training networks
        g = network(key)
        truth[key] = g.truth
        if fits:
            for cell, rows in training_rows(g, 'real', fits).items(): train.setdefault(cell, []).extend(rows)
    if fits:                                        # the 400 synthetic training networks
        with Pool() as pool:
            for part in pool.imap(partial(synthetic_rows, fits=fits), training_specs(), chunksize=4):
                for cell, rows in part.items(): train.setdefault(cell, []).extend(rows)

    out, test = [], {}
    for s in samples:
        o = parse(s['block'])
        # Training median: the median truth of the training networks that the sample's model may see.
        median = np.median([truth[n] for n in TRAIN if n != fold(s['graph_id'])], axis=0).tolist()
        for method, p in (('plugin', plugin(o)), ('median', median), ('mle', mle(o))):
            out.append({'observation_id': s['id'], 'method': method, 'replicate': '', 'prediction': p})
        test.setdefault(s['arm'], []).append((s, row(o, s['graph_id'], None, [0.]*4)))
    for k in range(fits):
        tail = f'_r{k}' if k else ''
        for arm in ARMS:
            for f in sorted({fold(s['graph_id']) for s in samples}):
                rows = [(s, r) for s, r in test[arm] if fold(s['graph_id']) == f]
                choice = select(train[k, arm], arm, f, 'v10_et_nested'+tail)
                fitted = predict(train[k, arm], [r for _, r in rows], arm, f, choice, 'v10_et_final'+tail)
                out += [{'observation_id': s['id'], 'method': 'et', 'replicate': k, 'prediction': valid_profile(p)}
                        for (s, _), p in zip(rows, fitted)]
            print(f'ExtraTrees fit {k}, arm {arm} done', flush=True)
    write_csv(OUT/'estimates.csv', out)

    frozen = {(r['observation_id'], r['method'], r['replicate']): json.loads(r['prediction'])
              for r in csv.DictReader((FINAL/'PREDICTIONS.csv').open()) if r['method'] in ('plugin', 'median', 'mle', 'et')}
    gap = {}
    for r in out:
        key = r['method']+(f" fit {r['replicate']}" if r['method'] == 'et' else '')
        gap[key] = max(gap.get(key, 0.), float(np.max(np.abs(np.subtract(r['prediction'], frozen[r['observation_id'], r['method'], str(r['replicate'])])))))
    print('largest difference to PREDICTIONS.csv:', ', '.join(f'{k} {v:.1g}' for k, v in gap.items()))


if __name__ == '__main__':
    main()
