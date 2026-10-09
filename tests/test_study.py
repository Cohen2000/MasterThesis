"""The essentials: the answer check, networks, sampling arms, the sample text, estimators and the tables."""
import csv
import hashlib
import importlib
import math
import re
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from study.answers import parse_final, valid_profile
from study.common import ARMS, ROOT
from study.data import graph, twin
from study.estimators import cell_probs, mle, mle_counts, plugin
from study.sample import parse, prompt, text
from study.sampling import draw, sizes
from study.synthetic import DAR, dar_latents, dar_rows
from study.tables import signflip, summary
from study.walk import Walk
from study.common import rng


def test_only_exact_answers_count():
    assert parse_final('{"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1, "rho_5": 0}') == ([.5, .2, .1, 0], 'valid')
    assert parse_final('```json\n{"rho_2":1,"rho_3":1,"rho_4":1,"rho_5":1}\n```')[1] == 'valid_after_fence'
    for bad, reason in (('{"rho_2": 0.1, "rho_3": 0.2, "rho_4": 0, "rho_5": 0}', 'monotonicity'),
                        ('{"rho_2": 1.2, "rho_3": 0.2, "rho_4": 0, "rho_5": 0}', 'range'),
                        ('{"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1}', 'keys_or_object'),
                        ('The answer is {"rho_2": 0.5, "rho_3": 0.2, "rho_4": 0.1, "rho_5": 0}', 'invalid_json')):
        assert parse_final(bad)[0] is None and parse_final(bad)[1].startswith(reason)
    assert valid_profile([1.2, .5, .6, -.1]) == [1., .5, .5, 0.]


def small_network():
    """A DAR network with 60 nodes, generated like the synthetic test networks."""
    rows = dar_rows(dar_latents(rng('test', 'small'), **{**DAR, 'N': 60, 'E': 400}), .8)[0]
    return graph('small', pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))


def test_windows_truth_and_twin():
    events = pd.DataFrame([('a', 'b', 0.), ('b', 'a', .2), ('a', 'b', .4), ('a', 'c', .6), ('a', 'c', .8), ('b', 'c', 1.), ('c', 'c', .5)],
                          columns=['u', 'v', 't'])
    g = graph('tiny', events, horizon=(0, 1))
    assert (g.N, g.D, g.M) == (3, 3, 6)                          # the event of c with itself is dropped
    assert g.counts.sum(0).tolist() == [1, 1, 1, 1, 2]           # a time on a cut belongs to the later window
    assert g.truth == [2/3, 1/3, 0., 0.]                         # a-b: 3 windows, a-c: 2, b-c: 1
    t = twin(small_network())
    assert np.array_equal(t.m, small_network().m) and sorted(t.t) == sorted(small_network().t)


def test_every_arm_gives_a_consistent_sample_text():
    g = small_network()
    walk = Walk(g)
    size = sizes(g, walk)
    for arm in ARMS:
        counts, visits = draw(g, arm, 1, 'test', size, walk)
        assert (counts <= g.counts).all()                         # a sample never shows more than the network
        block = text(g, arm, size, counts, visits)
        o = parse(block)
        assert (sum(r[1] for r in o['table']), sum(r[2] for r in o['table'])) == (o['D_obs'], o['M_obs']) == (int((counts.sum(1) > 0).sum()), int(counts.sum()))
        assert block in prompt(block)[1]['content']
        assert len(plugin(o)) == len(mle(o)) == 4
    assert parse(text(g, 'H', size, *draw(g, 'H', 1, 'test', size)))['Temporal_access'] == [0, 0, 1, 1, 1]
    assert parse(text(g, 'S', size, *draw(g, 'S', 1, 'test', size, walk)))['parameter'] == size['S']    # the visits add up to the steps


def test_prompts_are_the_ones_the_models_received():
    sealed = {'system.txt': '402ac97f8c3b0cf6b48d18ebf0a20eb1b39dcc314042801883e48a0a973a867d',
              'task.txt': '0f9a7e95378a126a643108bd5fabdc350ebad02e2fbc983e73eecf4a936558f7',
              'R.txt': '5c03504920525783cdb387e70d7fc5d906fd37d723d01a644ba45b37f6e512bd', 'S.txt': '4d749e42b05c95a406f4bd5c70d61ca2734ede4a6090f7a878d0255d4ae1ed65', 'H.txt': '2f6fe5a03d682fb758218230e01f2180548ed0647d286fa2db5a811ef546c9c5', 'B.txt': '06cd8addbc248768d6685e0bcd5f1b3baabf382fa583a09df611543ec23e5057'}
    for name, digest in sealed.items():
        assert hashlib.sha256((ROOT/'config/prompts'/name).read_bytes()).hexdigest() == digest


def test_model_probabilities_are_exact():
    """cell_probs against the same formula in exact fractions, also where probabilities are tiny."""
    def exact(a, b, d, n=5):
        fa, fb, fd = Fraction(a), Fraction(b), Fraction(d)
        def moment(r, s):
            v = Fraction(1)
            for t in range(r): v *= fa+t
            for t in range(s): v *= fb+t
            for t in range(r+s): v /= fa+fb+t
            return v
        return [math.comb(n, j)*fd**j*sum(math.comb(n-j, s)*(1-fd)**(n-j-s)*moment(n-s, s) for s in range(n-j+1)) for j in range(n+1)]
    for a in (1e-3, .5, 3., 1e5):
        for b in (1e-3, .5, 3., 1e5):
            for d in (1e-6, .3, 1.):
                for got, want in zip(cell_probs(a, b, d, 5), exact(a, b, d)):
                    assert float(want) < 1e-300 or abs(got-float(want)) <= 1e-13*float(want)


def test_mle_finds_the_persistence_of_its_own_model():
    r = np.random.default_rng(1)
    K = r.binomial(5, r.beta(.6, 1.4, 200_000))
    counts = [0.]+[float((K == j).sum()) for j in range(1, 6)]
    truth = [float(np.mean(K[K > 0] >= k)) for k in range(2, 6)]
    assert np.allclose(mle_counts(counts, 5), truth, atol=.005)
    late = r.binomial(3, r.beta(.6, 1.4, 200_000))                 # only three windows visible, as in arm H
    assert np.allclose(mle_counts([0.]+[float((late == j).sum()) for j in (1, 2, 3)], 3), truth, atol=.01)


def test_tables_average_per_network_and_wait_for_complete_methods():
    rows = [{'group': 'real', 'arm': 'R', 'method': m, 'source': s, 'valid': True, 'AE2': e, 'ProfileAE': e, 'signed_rho2': e}
            for s, n in (('a', 1), ('b', 9)) for m in ('plugin', 'gpt_6_sol') for e in [.1 if s == 'a' else .3]*n
            if not (m == 'gpt_6_sol' and s == 'b')]
    table = {r['method']: r for r in summary(rows) if r['group'] == 'real' and r['arm'] == 'R'}
    assert table['plugin']['MAE_2'] == pytest.approx(.2)          # each network counts once, whatever its number of rows
    assert table['gpt_6_sol']['MAE_2'] is None and table['gpt_6_sol']['status'].startswith('pending')
    assert signflip([.1, .1, .1]) == .25                          # all three differences positive: 2 of 8 sign patterns


@pytest.mark.parametrize('script', ['samples', 'estimates', 'checks', 'evaluate', 'window_sensitivity', 'analysis_figures', 'api', 'qwen'])
def test_every_script_imports(script):
    importlib.import_module(script)


def test_readme_shows_the_final_errors():
    """The main table of the README repeats SUMMARY.csv (real networks, MAE_2, three decimals)."""
    order = ('plugin', 'median', 'mle', 'et', 'qwen_thinking', 'qwen_nonthinking', 'deepseek_flash', 'gpt_6_sol', 'gpt_6_sol_tools')
    final = {(r['arm'], r['method']): f"{float(r['MAE_2']):.3f}"
             for r in csv.DictReader((ROOT/'docs/results/final/SUMMARY.csv').open()) if r['group'] == 'real'}
    readme = (ROOT/'README.md').read_text()
    for arm in ARMS:
        cells = re.findall(rf'^\| {arm} \|(.*)\|$', readme, re.M)[-1]          # the row of the result table
        assert [c.strip() for c in cells.split('|')] == [final[arm, m] for m in order]
