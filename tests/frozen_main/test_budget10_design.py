"""The cells10 design (arms matched on active dyad-windows), its recent-cap arm H and the legacy suffix-panel variant."""
import json
import math
import pathlib
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from main_experiment.common import (BUDGET_FRACTION, COVERAGE_FRACTION, ARMS, SAMPLER_DRAWS, LEGACY_H, TRAIN,
                                    REAL_TEST, SYNTH, planned_sizes, draws_for,
                                    observations_per_graph, CURRENT_RUN, CURRENT_REVISION)
from main_experiment.data import canonical, load_graph
from main_experiment.sampling import budget_parameters, draw, calibrate
from main_experiment.observation import make, serialize, parse, validate, PARAMS

RUN = pathlib.Path(CURRENT_RUN)
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
    def test_sizes_follow_from_the_calibrated_design(self):
        """No fixed count: a saturated H graph contributes one H observation."""
        self.assertEqual(SAMPLER_DRAWS, 5)
        free = {k: {'h_saturated': False} for k in set(TRAIN) | set(SYNTH)}
        s = planned_sizes(free, [*REAL_TEST, *SYNTH], TRAIN)
        self.assertEqual((s['main_observations'], s['training_observations']), (280, 320))
        self.assertEqual((s['planned_calls'], s['qwen_calls']), (3360, 1680))
        one = dict(free); one['sp_highschool2013'] = {'h_saturated': True}
        s = planned_sizes(one, [*REAL_TEST, *SYNTH], TRAIN)
        self.assertEqual((s['main_observations'], s['training_observations']), (276, 316))
        self.assertEqual((s['planned_calls'], s['qwen_calls']), (3312, 1656))
        self.assertEqual(draws_for('H', one['sp_highschool2013']), 1)
        self.assertEqual(draws_for('R', one['sp_highschool2013']), 5)

    def test_every_arm_targets_the_same_active_dyad_windows(self):
        g = fixture(n=200, per=6, seed=1)   # large enough for integer rounding
        b = budget_parameters(g)
        self.assertEqual(b['active_dyad_windows'], int((g.counts > 0).sum()))
        self.assertAlmostEqual(b['T'], COVERAGE_FRACTION * b['active_dyad_windows'])
        self.assertAlmostEqual(b['bernoulli_expected_cells'], b['T'], places=6)   # B solved exactly
        for key in ('node_relative_budget_error', 'h_relative_budget_error'):
            self.assertLess(abs(b[key]), .10, key)
        # the superseded event budget survives only as a descriptor
        self.assertEqual(b['B'], round(BUDGET_FRACTION * g.M))

    def test_expected_cells_are_exact_for_the_analytic_arms(self):
        """Monte Carlo over 400 draws: realised active dyad-windows track T for R, H, B.
        The sampling SD of one draw is below 0.35*T on this fixture, so 0.1*T is
        more than three standard errors of the mean."""
        g = fixture(n=80, per=6, seed=2)
        b = budget_parameters(g)
        for arm in ('R', 'H', 'B'):
            v = [int((draw(g, arm, i, 'cells_test', b)[0] > 0).sum()) for i in range(1, 401)]
            want = {'R': b['node_expected_cells'], 'H': b['h_expected_cells'], 'B': b['bernoulli_expected_cells']}[arm]
            self.assertLess(abs(np.mean(v) - want), .1 * b['T'], arm)

    def test_both_panels_target_the_same_budget(self):
        g = fixture(n=200, per=6, seed=1)
        b = budget_parameters(g)
        # Legacy variant: R draws from the whole archive, the suffix panel only
        # from windows 3-5, so it needs the larger panel.
        legacy = b['legacy_suffix_panel']
        self.assertGreaterEqual(legacy['n_panel_suffix'], b['n_panel'])
        self.assertLess(abs(b['node_relative_budget_error']), .10)
        self.assertLess(abs(legacy['suffix_relative_budget_error']), .10)
        self.assertLess(abs(b['h_relative_budget_error']), .10)

    def test_undefined_budget_is_rejected_early(self):
        g = fixture()
        with tempfile.TemporaryDirectory() as d:
            tiny = canonical('tiny', pd.DataFrame(
                [('a', 'b', .5), ('a', 'b', .9)], columns=['u', 'v', 't']), horizon=(0, 1))[0]
            with self.assertRaises(ValueError):
                budget_parameters(tiny)          # B rounds to zero


class SuffixPanelTests(unittest.TestCase):
    """The legacy variant: a node panel restricted to windows 3-5."""

    def test_only_panel_dyads_and_only_the_suffix_survive(self):
        g = fixture(n=60, per=8, seed=3)
        b = budget_parameters(g)
        counts, _ = draw(g, LEGACY_H, 1, 'sample', b)
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
            c, _ = draw(g, LEGACY_H, i, 'validate', b)
            j = (c[:, 2:] > 0).sum(1); j = j[j > 0]
            acc += np.bincount(j, minlength=4)[1:] / len(j)
        self.assertLess(np.abs(acc / 30 - ref).max(), .05)

    def test_every_arm_is_stochastic_now(self):
        g = fixture(n=60, per=8, seed=5)
        b = budget_parameters(g)
        self.assertFalse(b['h_saturated'])
        for arm in ('R', 'H', 'B', LEGACY_H):
            draws = {draw(g, arm, i, 'sample', b)[0].tobytes() for i in range(1, 6)}
            self.assertGreater(len(draws), 1, f'{arm} produced identical samples')


class ObservationContractTests(unittest.TestCase):
    def test_suffix_arm_carries_its_panel_size(self):
        self.assertEqual(PARAMS[LEGACY_H], 'n_panel_suffix')
        self.assertEqual(PARAMS['H'], 'n_dyads')
        self.assertEqual(len(set(PARAMS.values())), len(PARAMS))   # parser needs distinct names
        g = fixture(n=60, per=8, seed=11)
        b = budget_parameters(g)
        c, _ = draw(g, LEGACY_H, 1, 'sample', b)
        o = make(g, LEGACY_H, b, c, None)
        block = serialize(o)
        n = b['legacy_suffix_panel']['n_panel_suffix']
        self.assertIn(f'n_panel_suffix={n}', block)
        self.assertNotIn('tau=', block)
        back = parse(block)
        self.assertEqual(back['arm'], LEGACY_H)
        self.assertEqual(back['parameter'], n)
        self.assertEqual(type(back['parameter']), int)
        self.assertEqual(serialize(back), block)
        self.assertEqual(back['Temporal_access'], [0, 0, 1, 1, 1])

    def test_recent_arm_carries_its_sample_size(self):
        g = fixture(n=60, per=8, seed=11)
        b = budget_parameters(g)
        c, _ = draw(g, 'H', 1, 'sample', b)
        block = serialize(make(g, 'H', b, c, None))
        self.assertIn(f'n_dyads={b["n_dyads"]}', block)
        back = parse(block)
        self.assertEqual((back['arm'], back['parameter'], back['D_obs']), ('H', b['n_dyads'], b['n_dyads']))
        self.assertEqual(back['Temporal_access'], [1] * 5)


class RunContentTests(unittest.TestCase):
    """The produced run, checked against the design rather than against a literal."""

    @classmethod
    def setUpClass(cls):
        if not (RUN / 'report.json').exists():
            raise unittest.SkipTest('current run not present')
        cls.report = json.loads((RUN / 'report.json').read_text())

    def _budgets(self):
        return {k: json.loads((RUN / 'calibration' / k / 'budget.json').read_text())
                for k in set(TRAIN) | set(SYNTH)}

    def test_sizes_and_emptiness(self):
        r = self.report
        s = planned_sizes(self._budgets(), [*REAL_TEST, *SYNTH], TRAIN)
        self.assertTrue(r['offline_ready'])
        self.assertEqual(r['prepared_observations'], s['main_observations'])
        self.assertEqual(r['planned_observations'], s['main_observations'])
        self.assertEqual(r['training_observations'], s['training_observations'])
        self.assertEqual(r['request_manifest_rows'], s['planned_calls'])
        self.assertEqual(r['planned_logical_calls'], s['planned_calls'])
        self.assertEqual(r['empty_observations'], 0)
        self.assertEqual(r['started_calls'], 0)

    def test_every_arm_has_its_design_draw_count(self):
        import collections
        budgets = self._budgets()
        for domain in ('sample', 'training'):
            c = collections.Counter()
            for f in (RUN / 'observations' / domain).glob('*.json'):
                d = json.loads(f.read_text())
                c[(d['graph_id'], d['arm'])] += 1
            for (g, arm), n in c.items():
                self.assertEqual(n, draws_for(arm, budgets[g]), (domain, g, arm))
        # A deterministic H draw is carried once, and the status table agrees.
        det = [k for k, b in budgets.items() if b['h_saturated']]
        self.assertEqual(sorted(det), self.report['deterministic_h_graphs'])
        import csv
        with open(RUN / 'observation_status.csv') as f:
            status = list(csv.DictReader(f))
        self.assertEqual(len(status), self.report['planned_observations'])
        self.assertTrue(all(int(r['planned_logical_calls']) == 12 for r in status))

    def test_observed_cells_track_the_target(self):
        """Every arm is calibrated to the same expected number of observed active
        dyad-windows, so the realised share should sit near ten percent for all four."""
        share = {}
        for f in (RUN / 'observations' / 'sample').glob('*.json'):
            d = json.loads(f.read_text())
            share.setdefault(d['arm'], []).append(d['internal_evaluation']['observed_cell_fraction'])
        self.assertEqual(set(share), set(ARMS))
        for arm, v in share.items():
            self.assertAlmostEqual(float(np.median(v)), COVERAGE_FRACTION, delta=.02,
                                   msg=f'{arm} median share {np.median(v):.3f}')


class RunnerChainTests(unittest.TestCase):
    """The parts of the execution chain that do not need a GPU."""

    def report_calls(self):
        return json.loads((RUN / 'report.json').read_text())['planned_logical_calls']

    def test_reasoning_split_matches_the_chat_template(self):
        """The template opens the block in the prompt, so the output carries only
        the closing tag in thinking mode and no tag at all otherwise."""
        from run_qwen_batch import split_reasoning
        # thinking mode: reasoning, then the closing tag, then the answer
        r, f, closed = split_reasoning('weighing it up\n</think>\n\n{"rho_2": 0.5}', True)
        self.assertEqual(r, 'weighing it up')
        self.assertEqual(f, '{"rho_2": 0.5}')
        self.assertTrue(closed)
        # thinking mode, stopped inside the reasoning: no final answer exists
        r, f, closed = split_reasoning('cut off mid thought', True)
        self.assertFalse(closed)
        self.assertEqual(f, '')
        self.assertEqual(r, 'cut off mid thought')
        # non-thinking mode: the block is closed in the prompt, output is the answer
        r, f, closed = split_reasoning('{"rho_2": 0.5}', False)
        self.assertTrue(closed)
        self.assertEqual(r, '')
        self.assertEqual(f, '{"rho_2": 0.5}')
        # a stray opening tag inside the reasoning must not confuse the split
        r, f, closed = split_reasoning('a <think> b\n</think>\n{"x":1}', True)
        self.assertEqual(r, 'b')
        self.assertEqual(f, '{"x":1}')

    def test_request_selection_is_disjoint_across_modes_and_repeats(self):
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        from run_qwen_batch import load_requests
        seen = set(); total = 0
        for mode in ('thinking', 'nonthinking'):
            for repeat in (1, 2, 3):
                ids = {r['id'] for r in load_requests(RUN, mode, repeat, 0, 1)}
                self.assertFalse(ids & seen, 'repeats or modes overlap')
                seen |= ids; total += len(ids)
        self.assertEqual(total, self.report_calls() // 2)

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


class FinalAnswerParsingTests(unittest.TestCase):
    """A markdown fence is a wrapper; everything inside stays strictly validated."""

    def setUp(self):
        from main_experiment.evaluation import parse_final
        self.parse = parse_final
        self.good = '{"rho_2": 0.1, "rho_3": 0.05, "rho_4": 0.01, "rho_5": 0.0}'

    def test_bare_and_fenced_agree_on_the_values(self):
        for wrapper in ('%s', '```json\n%s\n```', '```\n%s\n```'):
            v, r = self.parse(wrapper % self.good)
            self.assertEqual(v, [0.1, 0.05, 0.01, 0.0], wrapper)
            self.assertIn(r, ('valid', 'valid_after_fence'))
        self.assertEqual(self.parse(self.good)[1], 'valid')
        self.assertEqual(self.parse('```json\n%s\n```' % self.good)[1], 'valid_after_fence')

    def test_a_fence_with_prose_around_it_is_still_invalid(self):
        v, r = self.parse('Here is my answer:\n```json\n%s\n```' % self.good)
        self.assertIsNone(v)

    def test_every_content_check_still_applies_inside_a_fence(self):
        bad = {
            'monotonicity': '{"rho_2": 0.1, "rho_3": 0.5, "rho_4": 0.01, "rho_5": 0.0}',
            'range':        '{"rho_2": 1.5, "rho_3": 0.5, "rho_4": 0.01, "rho_5": 0.0}',
            'keys_or_object': '{"rho_2": 0.1, "rho_3": 0.05, "rho_4": 0.01}',
        }
        for reason, body in bad.items():
            v, r = self.parse('```json\n%s\n```' % body)
            self.assertIsNone(v, reason)
            self.assertEqual(r, reason)

    def test_duplicate_keys_are_rejected_inside_a_fence(self):
        v, r = self.parse('```json\n{"rho_2": 0.1, "rho_2": 0.2, "rho_3": 0.05,'
                          ' "rho_4": 0.01, "rho_5": 0.0}\n```')
        self.assertIsNone(v)

    def test_only_one_enclosing_fence_is_removed(self):
        from main_experiment.evaluation import strip_fence
        text, fenced = strip_fence('```json\n```json\n%s\n```\n```' % self.good)
        self.assertTrue(fenced)
        self.assertIn('```json', text)      # the inner one survives and then fails


class DataSeparationTests(unittest.TestCase):
    def test_development_graphs_never_enter_training(self):
        from main_experiment.pool import pool_definition
        d = pool_definition()['graphs']
        train = {g['key'] for g in d if g['partition'] == 'train'}
        dev = {g['key'] for g in d if g['partition'] == 'dev'}
        self.assertFalse(train & dev)
        man = pathlib.Path(CURRENT_REVISION) / 'models_pooled/synthetic/manifest.json'
        if not man.exists(): self.skipTest('pooled model not fitted')
        used = set(json.loads(man.read_text())['sources'])
        self.assertFalse(used & dev, 'a development graph reached the training pool')
        self.assertTrue(train <= used | {s for s in used})

    def test_no_main_test_source_leaks_into_its_own_fold(self):
        base = pathlib.Path(CURRENT_REVISION) / 'models_pooled'
        if not base.exists(): self.skipTest('models not fitted')
        for src in REAL_TEST:
            m = json.loads((base / src / 'manifest.json').read_text())
            self.assertNotIn(src, m['sources'])
            self.assertNotIn(src, m['real_sources'])


if __name__ == '__main__':
    unittest.main()
