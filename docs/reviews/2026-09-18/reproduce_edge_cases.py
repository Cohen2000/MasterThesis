"""Reproduce the H counterexample and all-invalid evaluation case, offline only."""
from pathlib import Path
import csv
import itertools
import json
import sys
import tempfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from main_experiment.data import canonical
from main_experiment.sampling import budget_parameters, draw
from main_experiment.observation import make, serialize
from evaluate_main_responses import evaluate


def h_counterexample():
    pairs = list(itertools.combinations(range(20), 2))[:100]
    rows1, rows2 = [], []
    for i, (u, v) in enumerate(pairs):
        late = [(u, v, t) for t in [.81, .83, .85, .87, .89]]
        rows1 += late + [(u, v, .1)]
        rows2 += late + ([(u, v, .1), (u, v, .3)] if i < 50 else [])
    graphs = [canonical('H_counterexample', pd.DataFrame(rows, columns=['u', 'v', 't']),
                        horizon=(0, 1))[0] for rows in (rows1, rows2)]
    budgets = [budget_parameters(g) for g in graphs]
    blocks = []
    for g, b in zip(graphs, budgets):
        counts, traversals = draw(g, 'H', 1, 'audit', b)
        blocks.append(serialize(make(g, 'H', b, counts, traversals)))
    assert blocks[0] == blocks[1]
    assert [b['n_dyads'] for b in budgets] == [20, 20]
    result = {
        'same_H_observation': True,
        'truth_1': graphs[0].truth, 'truth_2': graphs[1].truth,
        'D_full': [g.D for g in graphs], 'M_full': [g.M for g in graphs],
        'active_cells': [g.cells for g in graphs],
        'n_dyads': [b['n_dyads'] for b in budgets], 'block': blocks[0],
        'argument': 'Identical topology and last five events for every dyad; uniform subset '
                    'has the same law for both archives. Both calibrate to d=20. Thus the '
                    'entire H input distribution, not only one sample, coincides.',
    }
    (OUT / 'h_identifiability_counterexample.json').write_text(json.dumps(result, indent=2) + '\n')


def all_invalid():
    run = ROOT / 'results/main_experiment/cells10_20260917'
    baseline = ROOT / 'results/baseline_revision_cells10_20260917/primary_baselines.json'
    requests = [json.loads(line) for line in (run / 'requests.jsonl').read_text().splitlines()]
    with tempfile.TemporaryDirectory(prefix='masterarbeit_audit_mock_') as tmp:
        folder = Path(tmp)
        source = folder / 'responses.jsonl'
        with source.open('w') as f:
            for r in requests:
                record = {
                    'id': r['id'], 'mock': True, 'started': True, 'terminal': True,
                    'prompt_sha256': r['prompt_sha256'],
                    'final_text': 'INVALID' if r['config_id'] == 'sol' else
                                  '{"rho_2":0.4,"rho_3":0.3,"rho_4":0.2,"rho_5":0.1}',
                }
                f.write(json.dumps(record) + '\n')
        evaluate(run, source, folder / 'evaluation', mock=True, baselines=baseline)
        report = json.loads((folder / 'evaluation/report.json').read_text())
        with (folder / 'evaluation/summary.csv').open() as f:
            rows = [{k: r[k] for k in ['arm', 'fallback_only', 'complete', 'AE2',
                                      'plugin_AE2', 'valid_fraction']}
                    for r in csv.DictReader(f) if r['stratum'] == 'real' and r['config_id'] == 'sol']
        result = {
            'mock_only': True, 'inference_calls': 0,
            'complete_main_result': report['complete_main_result'], 'summary': rows,
            'construction': 'All 3360 planned requests marked started+terminal+mock; '
                            'Sol final_text=INVALID, all other configs valid constant JSON. '
                            'evaluate_main_responses called with --mock and the current primary_baselines.',
        }
        (OUT / 'all_invalid_mock.json').write_text(json.dumps(result, indent=2) + '\n')
        # Pins the reviewed behavior; this assertion should fail after a proper fix.
        assert report['complete_main_result'] and all(r['AE2'] == '' for r in rows)


if __name__ == '__main__':
    h_counterexample()
    all_invalid()
    print('Both audit edge cases reproduced; no model calls.')
