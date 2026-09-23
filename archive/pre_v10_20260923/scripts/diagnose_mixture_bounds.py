#!/usr/bin/env python3
"""Arm B mixture reference: sensitivity of each main fit to its parameter box.

Every non-empty main B observation is refitted with all bounds widened by two
decades; the change of the objective and of the predicted profile is reported.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.baselines import mixture_start
from main_experiment.common import DIAGNOSTICS, PREPARED, fresh_directory, read_json, write_csv, write_json
from main_experiment.mixtures import bound_sensitivity
from main_experiment.observation import parse


def main():
    out = fresh_directory(DIAGNOSTICS/'mixture_bounds')
    rows = []
    for path in sorted((PREPARED/'observations/sample').glob('*.json')):
        row = read_json(path)
        if row['arm'] != 'B' or row['empty']: continue
        o = parse(row['block'])
        result = bound_sensitivity(o, *mixture_start(o), decades=2.)
        rows.append({'observation': row['id'], 'graph_id': row['graph_id'],
                     **{k: None if isinstance(v, float) and not np.isfinite(v) else v for k, v in result.items()}})
    write_csv(out/'mixture_bound_sensitivity.csv', rows)
    shifts = [r['max_profile_shift'] for r in rows]
    write_json(out/'report.json', {'observations': len(rows), 'decades': 2,
                                   'max_profile_shift': max(shifts), 'median_profile_shift': float(np.median(shifts))})
    print('mixture bounds:', len(rows), 'B observations')


if __name__ == '__main__':
    main()
