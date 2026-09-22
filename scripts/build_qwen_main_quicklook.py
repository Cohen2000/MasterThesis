#!/usr/bin/env python3
"""Qwen main (v9) validity/MAE2 by arm x config, from an already-collected
responses.jsonl (scripts/collect_qwen_answers.py) and its paired observations
-- no re-inference, no baseline re-fit; only the released final answers.

usage: build_qwen_main_quicklook.py --responses R.jsonl --observations DIR --out OUT.csv
"""
import argparse
import json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import write_csv
from main_experiment.evaluation import errors, parse_final


def main(responses, observations_dir, out):
    obs = {}
    for p in Path(observations_dir).glob('*.json'):
        d = json.loads(p.read_text())
        obs[d['id']] = d
    rows = []
    for line in Path(responses).read_text().splitlines():
        r = json.loads(line)
        oid, config = r['id'].rsplit('__', 3)[:2]
        o = obs[oid]
        if r.get('refusal'): values, reason = None, 'refusal'
        elif r.get('technical_error'): values, reason = None, 'technical_error'
        else: values, reason = parse_final(r.get('final_text', ''))
        e = errors(values, o['truth'])
        profile_ae = None if values is None else float(np.mean(np.abs(np.array(values)-np.array(o['truth']))))
        rows.append({'budget': o.get('budget'), 'graph_id': o['graph_id'], 'arm': o['arm'], 'stratum': o['stratum'], 'source_family': o['source_family'],
                    'config': config, 'valid': values is not None, 'validation_reason': reason,
                    'AE2': e['AE2'], 'ProfileAE': profile_ae, 'signed_rho2': e['signed_rho2']})
    write_csv(out, rows)
    print(f'wrote {len(rows)} rows to {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--responses', required=True)
    p.add_argument('--observations', required=True)
    p.add_argument('--out', required=True, type=Path)
    a = p.parse_args()
    main(a.responses, a.observations, a.out)
