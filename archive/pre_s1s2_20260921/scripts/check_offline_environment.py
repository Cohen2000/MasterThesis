#!/usr/bin/env python3
"""Check that another machine reproduces the main study's offline artifacts exactly.

Recalibrates and redraws three main graphs at 0.10 and refits one real-only
ExtraTrees fold; blocks, prompts and predictions must equal the local ones.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ARMS, BUILD, PREPARED, REFERENCES, draws_for, read_json
from main_experiment.observation import features, parse
from main_experiment.prepare import observation_row, prepared_graph
from main_experiment.references import real_training_rows
from main_experiment.sampling import calibrate, draw
from main_experiment.training import fit_folds

checked = 0
for key in ('sp_hospital', 'snap_email_eu__pwt', 'dar_a08_r2'):
    g, stored = prepared_graph(key)
    budget, walk, _ = calibrate(g, BUILD)
    assert all(budget[k] == stored[k] for k in ('n_panel', 'L', 'n_panel_history', 'p', 'validation_mean')), key
    for arm in ARMS:
        for index in range(1, draws_for(arm, budget)+1):
            row = observation_row(g, arm, index, 'sample', budget, *draw(g, arm, index, 'sample', budget, walk))
            assert row['block'] == read_json(PREPARED/'observations/sample'/f'{row["id"]}.json')['block']
            checked += 1
rows, truth = real_training_rows()
models, _ = fit_folds(rows, truth, BUILD/'environment_check_models')
references = read_json(REFERENCES/'primary_baselines.json')['observations']
compared = 0
for oid, entry in references.items():
    if entry['fold'] != 'sp_hospital' or entry['extratrees_real_only']['status'] != 'ok': continue
    o = parse(read_json(PREPARED/'observations/sample'/f'{oid}.json')['block'])
    assert list(map(float, models['sp_hospital'].predict([features(o)])[0])) == entry['extratrees_real_only']['prediction'], oid
    compared += 1
print('ENVIRONMENT_EQUIVALENT', checked, 'observations,', compared, 'ExtraTrees predictions identical')
