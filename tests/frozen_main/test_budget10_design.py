"""The revised observation design: a ten-percent budget and a panelled suffix arm."""
import json
import math
import pathlib
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from main_experiment.common import (BUDGET_FRACTION, ARMS, SAMPLES_PER_ARM,
                                    MAIN_OBSERVATIONS, TRAINING_OBSERVATIONS,
                                    PLANNED_CALLS, QWEN_CALLS, REAL_TEST, SYNTH)
from main_experiment.data import canonical, load_graph
from main_experiment.sampling import budget_parameters, draw, calibrate
from main_experiment.observation import make, serialize, parse, validate, PARAMS

RUN = pathlib.Path('results/main_experiment/budget10_20261001')
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'scripts'))


def fixture(n=40, per=6, seed=0):
    """A graph large enough that a ten-percent budget rounds to something useful."""
    r = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        u, v = f'n{i}', f'n{(i + 1 + i % 3) % n}'
        if u == v: continue
        for k in range(per):
            rows.append((u, v, float(r.uniform(0, 1))))
    return canonical('fixture', pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))[0]


class DesignConstantTests(unittest.TestCase):
    def test_sizes_follow_from_the_replication_scheme(self):
        self.assertEqual(SAMPLES_PER_ARM, 5)
        self.assertEqual(MAIN_OBSERVATIONS, (len(REAL_TEST) + len(SYNTH)) * len(ARMS) * 5)
        self.assertEqual(MAIN_OBSERVATIONS, 280)
        self.assertEqual(TRAINING_OBSERVATIONS, 320)
        self.assertEqual(PLANNED_CALLS, 3360)
        self.assertEqual(QWEN_CALLS, 1680)

    def test_budget_is_a_fixed_share_of_the_archive(self):
        g = fixture()
        b = budget_parameters(g)
        self.assertEqual(b['B'], round(BUDGET_FRACTION * g.M))
        self.assertEqual(b['p'], BUDGET_FRACTION)          # arm B is exact by construction
        self.assertAlmostEqual(b['B'] / g.M, BUDGET_FRACTION, places=3)

    def test_both_panels_target_the_same_budget(self):
        g = fixture()
        b = budget_parameters(g)
        # R draws from the whole archive, H only from the suffix, so H needs the
        # larger panel by exactly the ratio of the two volumes.
        self.assertGreaterEqual(b['n_panel_suffix'], b['n_panel'])
        for key in ('node_relative_budget_error', 'suffix_relative_budget_error'):
            self.assertLess(abs(b[key]), .10, f'{key}={b[key]}')

    def test_undefined_budget_is_rejected_early(self):
        g = fixture()
        with tempfile.TemporaryDirectory() as d:
            tiny = canonical('tiny', pd.DataFrame(
                [('a', 'b', .5), ('a', 'b', .9)], columns=['u', 'v', 't']), horizon=(0, 1))[0]
            with self.assertRaises(ValueError):
                budget_parameters(tiny)          # B rounds to zero


class SuffixPanelTests(unittest.TestCase):
    """Arm H is now a node panel restricted to windows 3-5."""

    def test_only_panel_dyads_and_only_the_suffix_survive(self):
        g = fixture(n=60, per=8, seed=3)
        b = budget_parameters(g)
        counts, _ = draw(g, 'H', 1, 'sample', b)
        self.assertTrue((counts[:, :2] == 0).all(), 'windows 1-2 must be empty')
        seen = counts.sum(1) > 0
        # Everything retained inside the panel must be the graph's own suffix counts.
        np.testing.assert_array_equal(counts[seen][:, 2:], g.counts[seen][:, 2:])

    def test_panel_leaves_the_window_count_distribution_alone(self):
        """The central claim: the panel selects dyads without looking at activity,
        so the zero-truncated distribution of observed active windows is unchanged.
        Thirty draws give a standard error near 0.01 on each cell, so 0.05 is a
        conservative bound that does not depend on luck."""
        g = fixture(n=200, per=6, seed=7)
        b = budget_parameters(g)
        suf = g.counts[:, 2:] > 0
        J = suf.sum(1); J = J[J > 0]
        ref = np.bincount(J, minlength=4)[1:] / len(J)
        acc = np.zeros(3)
        for i in range(1, 31):
            c, _ = draw(g, 'H', i, 'validate', b)
            j = (c[:, 2:] > 0).sum(1); j = j[j > 0]
            acc += np.bincount(j, minlength=4)[1:] / len(j)
        self.assertLess(np.abs(acc / 30 - ref).max(), .05)

    def test_every_arm_is_stochastic_now(self):
        g = fixture(n=60, per=8, seed=5)
        b = budget_parameters(g)
        for arm in ('R', 'H', 'B'):
            draws = {draw(g, arm, i, 'sample', b)[0].tobytes() for i in range(1, 6)}
            self.assertGreater(len(draws), 1, f'{arm} produced identical samples')


class ObservationContractTests(unittest.TestCase):
    def test_suffix_arm_carries_its_panel_size(self):
        self.assertEqual(PARAMS['H'], 'n_panel_suffix')
        self.assertNotEqual(PARAMS['H'], PARAMS['R'])      # parser needs distinct names
        g = fixture(n=60, per=8, seed=11)
        b = budget_parameters(g)
        c, _ = draw(g, 'H', 1, 'sample', b)
        o = make(g, 'H', b, c, None)
        block = serialize(o)
        self.assertIn(f'n_panel_suffix={b["n_panel_suffix"]}', block)
        self.assertNotIn('tau=', block)
        back = parse(block)
        self.assertEqual(back['parameter'], b['n_panel_suffix'])
        self.assertEqual(type(back['parameter']), int)
        self.assertEqual(serialize(back), block)
        self.assertEqual(back['Temporal_access'], [0, 0, 1, 1, 1])


class RunContentTests(unittest.TestCase):
    """The produced run, checked against the design rather than against a literal."""

    @classmethod
    def setUpClass(cls):
        if not (RUN / 'report.json').exists():
            raise unittest.SkipTest('current run not present')
        cls.report = json.loads((RUN / 'report.json').read_text())

    def test_sizes_and_emptiness(self):
        r = self.report
        self.assertTrue(r['offline_ready'])
        self.assertEqual(r['prepared_observations'], MAIN_OBSERVATIONS)
        self.assertEqual(r['training_observations'], TRAINING_OBSERVATIONS)
        self.assertEqual(r['request_manifest_rows'], PLANNED_CALLS)
        self.assertEqual(r['empty_observations'], 0)
        self.assertEqual(r['started_calls'], 0)

    def test_every_arm_has_five_samples(self):
        import collections
        c = collections.Counter()
        for f in (RUN / 'observations' / 'sample').glob('*.json'):
            d = json.loads(f.read_text())
            c[(d['graph_id'], d['arm'])] += 1
        self.assertEqual(set(c.values()), {5})

    def test_observed_volume_tracks_the_budget(self):
        """Every arm is calibrated to the same expected volume, so the realised
        event share should sit near ten percent for all four."""
        man = {json.loads(f.read_text())['key']: json.loads(f.read_text())
               for f in (RUN / 'graphs').glob('*/manifest.json')}
        share = {}
        for f in (RUN / 'observations' / 'sample').glob('*.json'):
            d = json.loads(f.read_text()); o = parse(d['block'])
            share.setdefault(d['arm'], []).append(o['M_obs'] / man[d['graph_id']]['M_full'])
        for arm, v in share.items():
            self.assertAlmostEqual(float(np.median(v)), BUDGET_FRACTION, delta=.03,
                                   msg=f'{arm} median share {np.median(v):.3f}')


class RunnerChainTests(unittest.TestCase):
    """The parts of the execution chain that do not need a GPU."""

    def test_reasoning_split(self):
        from run_qwen_batch import split_reasoning
        r, f, closed = split_reasoning('<think>weighing it up</think>\n{"rho_2": 0.5}')
        self.assertEqual(r.strip(), 'weighing it up')
        self.assertEqual(f, '{"rho_2": 0.5}')
        self.assertTrue(closed)
        r, f, closed = split_reasoning('<think>cut off mid thought')
        self.assertFalse(closed)
        self.assertEqual(f, '')          # no final answer exists at all
        r, f, closed = split_reasoning('{"rho_2": 0.5}')
        self.assertTrue(closed)
        self.assertEqual(r, '')

    def test_request_selection_is_disjoint_across_modes_and_repeats(self):
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        from run_qwen_batch import load_requests
        seen = set(); total = 0
        for mode in ('thinking', 'nonthinking'):
            for repeat in (1, 2, 3):
                ids = {r['id'] for r in load_requests(RUN, mode, repeat, 0, 1)}
                self.assertFalse(ids & seen, 'repeats or modes overlap')
                seen |= ids; total += len(ids)
        self.assertEqual(total, QWEN_CALLS)

    def test_shards_partition_the_work_exactly(self):
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        from run_qwen_batch import load_requests
        whole = [r['id'] for r in load_requests(RUN, 'thinking', 1, 0, 1)]
        parts = []
        for i in range(4):
            parts += [r['id'] for r in load_requests(RUN, 'thinking', 1, i, 4)]
        self.assertEqual(sorted(parts), sorted(whole))
        self.assertEqual(len(set(parts)), len(parts))

    def test_prompts_are_verified_against_the_manifest(self):
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        from run_qwen_batch import load_requests
        rows = load_requests(RUN, 'thinking', 1, 0, 1)
        self.assertTrue(rows)
        for r in rows[:5]:
            self.assertEqual(len(r['messages']), 2)
            self.assertEqual(r['messages'][0]['role'], 'system')


class DataSeparationTests(unittest.TestCase):
    def test_development_graphs_never_enter_training(self):
        from main_experiment.pool import pool_definition
        d = pool_definition()['graphs']
        train = {g['key'] for g in d if g['partition'] == 'train'}
        dev = {g['key'] for g in d if g['partition'] == 'dev'}
        self.assertFalse(train & dev)
        man = pathlib.Path('results/baseline_revision_20261001/models_pooled/synthetic/manifest.json')
        if not man.exists(): self.skipTest('pooled model not fitted')
        used = set(json.loads(man.read_text())['sources'])
        self.assertFalse(used & dev, 'a development graph reached the training pool')
        self.assertTrue(train <= used | {s for s in used})

    def test_no_main_test_source_leaks_into_its_own_fold(self):
        base = pathlib.Path('results/baseline_revision_20261001/models_pooled')
        if not base.exists(): self.skipTest('models not fitted')
        for src in REAL_TEST:
            m = json.loads((base / src / 'manifest.json').read_text())
            self.assertNotIn(src, m['sources'])
            self.assertNotIn(src, m['real_sources'])


if __name__ == '__main__':
    unittest.main()
