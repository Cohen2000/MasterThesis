#!/usr/bin/env python3
"""Predictions and errors of the mixed-budget ExtraTrees models
(build_mixed_budget_et.py) on Main + every budget-sensitivity level.

Does not retrain anything; only scores the already-fitted, budget-pooled,
arm-specific forests (anchor + predicted residual) against ground truth,
exactly like baselines.all_references does for the per-fold ET reference.
"""
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.baselines import anchor_profile
from main_experiment.common import ARMS, BUDGET_GRID, COVERAGE_FRACTION, PREPARED, ROOT, BUDGET_SENSITIVITY, fold_for, write_csv
from main_experiment.observation import features, parse
from main_experiment.training import load_models

ET_ROOT = ROOT/'results/panel888_et_mixed'
OUT = ROOT/'results/panel888_et_mixed_evaluation.csv'


def budget_dir(fraction):
    return PREPARED if fraction == COVERAGE_FRACTION else BUDGET_SENSITIVITY/f'b{round(fraction*1000):03d}'


def error_record(prediction, truth):
    e = np.asarray(prediction, float)-np.asarray(truth, float)
    return {'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0])}


def main():
    models = {arm: load_models(ET_ROOT/'models'/arm) for arm in ARMS}
    rows = []
    for fraction in BUDGET_GRID:
        for path in sorted((budget_dir(fraction)/'observations/sample').glob('*.json')):
            row = json.loads(path.read_text())
            o = parse(row['block'])
            if o['D_obs'] == 0: continue
            fold = fold_for(row['graph_id'])
            x = features(o).reshape(1, -1)
            prediction = np.asarray(anchor_profile(o))+models[row['arm']][fold].predict(x)[0]
            rows.append({'budget': fraction, 'graph_id': row['graph_id'], 'arm': row['arm'], 'stratum': row['stratum'],
                        'source_family': row['source_family'], **error_record(prediction, row['truth'])})
    write_csv(OUT, rows)
    print(f'wrote {len(rows)} rows to {OUT}')


if __name__ == '__main__':
    main()
