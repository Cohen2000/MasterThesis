"""Quick checks that need neither raw data nor a model: python -m pytest -q"""
import csv
from collections import Counter

import pandas as pd
import pytest

from study.answers import parse_final, valid_profile
from study.common import DATA, ROOT, digest, read_json, sha
from study.data import graph, twin
from study.estimators import mle, mle_counts, plugin, profile
from study.sample import parse, prompt


def test_only_exact_answers_count():
    assert parse_final('{"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1, "rho_5": 0}') == ([.5, .2, .1, 0], 'valid')
    assert parse_final('```json\n{"rho_2":1,"rho_3":1,"rho_4":1,"rho_5":1}\n```')[1] == 'valid_after_fence'
    for bad, reason in (('{"rho_2": 0.1, "rho_3": 0.2, "rho_4": 0, "rho_5": 0}', 'monotonicity'),
                        ('{"rho_2": 1.2, "rho_3": 0.2, "rho_4": 0, "rho_5": 0}', 'range'),
                        ('{"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1}', 'keys_or_object'),
                        ('The answer is {"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1, "rho_5": 0}', 'invalid_json')):
        assert parse_final(bad)[0] is None and parse_final(bad)[1].startswith(reason)
    assert valid_profile([1.2, .5, .6, -.1]) == [1., .5, .5, 0.]


def test_samples_are_the_frozen_ones():
    files = sorted((DATA/'samples').glob('*.json'))
    assert {p.name: sha(p) for p in files} == read_json(DATA/'checksums.json')
    assert Counter(read_json(p)['arm'] for p in files) == {'R': 72, 'S': 72, 'H': 72, 'B': 72}


def test_prompts_are_the_ones_the_models_received():
    """The prompt files in config/prompts rebuild every prompt exactly."""
    for path in sorted((DATA/'samples').glob('*.json')):
        s = read_json(path)
        assert prompt(s['block']) == s['messages'] and digest(s['messages']) == s['prompt_sha256']


def test_windows_truth_and_copy():
    events = pd.DataFrame({'u': [1, 1, 1, 2, 3], 'v': [2, 2, 3, 3, 4], 't': [0., .9, .1, .5, 1.]})
    g = graph('toy', events, horizon=(0, 1))
    assert g.D == 4 and list(g.K) == [2, 1, 1, 1] and g.truth[0] == pytest.approx(1/4)
    c = twin(g)
    assert c.D == g.D and list(c.m) == list(g.m)          # same pairs and interaction counts, new times


def test_mle_finds_the_persistence_of_its_own_model():
    """If all pairs are alike, the model's profile is reproduced from exact counts."""
    from math import comb
    q = .3
    pk = [comb(5, j)*q**j*(1-q)**(5-j) for j in range(6)]
    counts = [0.]+[1e6*p for p in pk[1:]]
    assert mle_counts(counts, 5) == pytest.approx(profile(q), abs=1e-4)


def test_estimators_reproduce_the_frozen_predictions():
    frozen = {(r['sample_id'], r['method']): [float(r[f'rho_{k}']) for k in range(2, 6)]
              for r in csv.DictReader((DATA/'predictions.csv').open()) if r['method'] in ('observed', 'statistical_model')}
    for path in sorted((DATA/'samples').glob('*.json'))[::12]:
        s = read_json(path)
        o = parse(s['block'])
        assert plugin(o) == frozen[s['id'], 'observed']
        assert mle(o) == pytest.approx(frozen[s['id'], 'statistical_model'], abs=5e-4)


def test_predictions_are_complete():
    rows = list(csv.DictReader((DATA/'predictions.csv').open()))
    n = Counter(r['method'] for r in rows)
    assert n == {'observed': 288, 'statistical_model': 288, 'learned_reference': 288, 'gpt_6_sol': 864,
                 'deepseek_v4_1_flash': 864, 'qwen3_6': 864, 'gpt_6_sol_python': 864, 'gpt_6_sol_no_walk_log': 216}
    assert sum(r['valid'] == 'False' for r in rows) == 2


def test_every_number_of_the_thesis():
    import thesis
    assert thesis.main() == []
    assert (ROOT/'thesis/numbers.md').exists()
    for name in ('networks', 'results', 'walk', 'history', 'windows', 'thresholds', 'copies', 'per_network'):
        assert (ROOT/'thesis/tables'/f'{name}.csv').exists()


@pytest.mark.parametrize('script', ['samples', 'estimates', 'api', 'ablation_walk_log', 'score', 'networks',
                                    'properties', 'window_sensitivity', 'thesis', 'figures'])
def test_every_script_imports(script):
    __import__(script)
