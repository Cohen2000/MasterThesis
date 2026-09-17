"""Arm H of budget10-hrecent5-20260917: uniform dyads, five most recent events each.

Every Monte-Carlo tolerance is stated with the sampling error it comes from, and
every simulation has a fixed seed.
"""
import json
import math
import pathlib
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

from main_experiment.common import (H_CAP, LEGACY_H, ARM_ID, REAL_TEST, SYNTH, SAMPLER_DRAWS,
                                    observation_id, rng, draws_for)
from main_experiment.data import canonical, load_graph
from main_experiment.sampling import (budget_parameters, draw, recent_counts, reservoir_counts,
                                      h_parameters, _dyads_for, sample_dyads)
from main_experiment.observation import (make, serialize, parse, validate, features, messages,
                                         FEATURE_NAMES, RULE_FILES)
from main_experiment.baselines import plugin, corrector, h_bounds, h_midpoint
from main_experiment.evaluation import paired_summary, cell_variance

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUN = ROOT / 'results/main_experiment/hrecent5_20260917'
OLD = ROOT / 'results/main_experiment/budget10_20261001'
sys.path.insert(0, str(ROOT / 'scripts'))


def random_graph(seed, n=30, dyads=80, max_events=12, ties=True):
    r = np.random.default_rng(seed)
    rows = []
    for k in range(dyads):
        u, v = r.choice(n, 2, replace=False)
        m = int(r.integers(1, max_events + 1))
        t = r.uniform(0, 1, m)
        if ties:
            # exact window cut points and repeated timestamps
            t[: m // 3] = r.choice([0., .2, .4, .6, .8, 1.], m // 3)
            if m > 3: t[-1] = t[-2]
        rows += [(f'n{u}', f'n{v}', float(x)) for x in t]
    return canonical(f'rg{seed}', pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))[0]


def by_timestamp(g, cap=H_CAP):
    """Reference: sort each dyad's events by time and keep the last `cap`."""
    out = np.zeros_like(g.counts)
    order = np.lexsort((g.t, g.pair))
    pair, w = g.pair[order], g.w[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pair)) + 1, len(pair)]
    for a, b in zip(starts[:-1], starts[1:]):
        for j in w[max(a, b - cap):b]:
            out[pair[a], j] += 1
    return out


class RecentCountTests(unittest.TestCase):
    def test_backward_fill_equals_timestamp_selection(self):
        for seed in range(12):
            g = random_graph(seed)
            np.testing.assert_array_equal(recent_counts(g.counts), by_timestamp(g))
            for cap in (1, 2, 3, 7):
                np.testing.assert_array_equal(recent_counts(g.counts, cap), by_timestamp(g, cap))

    def test_equivalence_on_real_sources(self):
        """The two real sources with the most and the fewest events per dyad."""
        if not (RUN / 'graphs').exists(): self.skipTest('current run not present')
        for key in ('sp_hospital', 'snap_collegemsg'):
            g = load_graph(RUN / 'graphs' / key)
            np.testing.assert_array_equal(recent_counts(g.counts), by_timestamp(g))

    def test_basic_properties(self):
        g = random_graph(3)
        c = recent_counts(g.counts)
        np.testing.assert_array_equal(c.sum(1), np.minimum(g.m, H_CAP))
        self.assertTrue((c <= g.counts).all())
        # Everything after the earliest retained window is retained completely.
        for row, full in zip(c, g.counts):
            b = int(np.argmax(row > 0))
            np.testing.assert_array_equal(row[b + 1:], full[b + 1:])

    def test_reservoir_keeps_the_same_volume_and_is_uniform(self):
        """Hypergeometric mean m_j*k/m per window; 4000 draws give a standard error
        below 0.02 events per cell, so 0.08 is a four-sigma bound."""
        counts = np.array([[7, 0, 3, 1, 9], [1, 1, 1, 1, 1], [2, 0, 0, 0, 0], [0, 20, 0, 0, 1]])
        r = np.random.Generator(np.random.PCG64(5))
        acc = np.zeros(counts.shape)
        for _ in range(4000):
            x = reservoir_counts(counts, H_CAP, r)
            np.testing.assert_array_equal(x.sum(1), np.minimum(counts.sum(1), H_CAP))
            self.assertTrue((x <= counts).all())
            acc += x
        m = counts.sum(1, keepdims=True)
        want = counts * np.minimum(m, H_CAP) / m
        self.assertLess(np.abs(acc / 4000 - want).max(), .08)


class BudgetTests(unittest.TestCase):
    def test_closest_integer_with_ties_to_the_smaller(self):
        self.assertEqual(_dyads_for(10, 5, 10), (5, 5.0))
        # d*C/D = 2.5 d; B=5 -> d=2 exactly; B=6.25 is a tie between 2 and 3
        self.assertEqual(_dyads_for(25, 6.25, 10)[0], 2)
        self.assertEqual(_dyads_for(25, 1, 10)[0], 1)     # never zero

    def test_unreachable_saturated_and_tolerance_are_separate(self):
        rows = []
        for i in range(40):
            rows += [(f'a{i}', f'b{i}', t) for t in np.linspace(.01, .99, 40)]   # m_e = 40
        g = canonical('dense', pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))[0]
        h = h_parameters(g, g.B)
        # C = 200 < B = 160? no: B = 0.1*1600 = 160, C = 5*40 = 200
        self.assertEqual((h['C_cap'], g.B), (200, 160))
        self.assertFalse(h['h_target_unreachable'])
        self.assertEqual(h['n_dyads'], 32)
        h2 = h_parameters(g, 199)       # d=40 gives 200, d=39 gives 195: closest is d=D
        self.assertTrue(h2['h_saturated']); self.assertFalse(h2['h_target_unreachable'])
        h3 = h_parameters(g, 205)       # not reachable, but 200 is within 5 %
        self.assertTrue(h3['h_saturated'] and h3['h_target_unreachable'] and h3['h_within_tolerance'])
        h4 = h_parameters(g, 400)       # not reachable and far outside tolerance
        self.assertTrue(h4['h_saturated'] and h4['h_target_unreachable'])
        self.assertFalse(h4['h_within_tolerance'])

    def test_expected_volume_is_exact_over_all_subsets(self):
        """Exhaustive: the mean retained volume over all d-subsets equals d*C/D."""
        import itertools
        g = random_graph(1, n=8, dyads=7, max_events=9)
        cap = np.minimum(g.m, H_CAP)
        for d in range(1, g.D + 1):
            vols = [cap[list(s)].sum() for s in itertools.combinations(range(g.D), d)]
            self.assertAlmostEqual(np.mean(vols), d * cap.sum() / g.D, places=9)

    def test_saturated_sample_is_deterministic_and_single(self):
        g = random_graph(2)
        b = budget_parameters(g) | {'n_dyads': g.D, 'h_saturated': True}
        c, _ = draw(g, 'H', 1, 'sample', b)
        np.testing.assert_array_equal(c, recent_counts(g.counts))
        with self.assertRaises(ValueError): draw(g, 'H', 2, 'sample', b)
        free = budget_parameters(g)
        self.assertFalse(free['h_saturated'])
        self.assertEqual(draws_for('H', free), SAMPLER_DRAWS)
        self.assertEqual(draws_for('H', b), 1)

    def test_dyad_inclusion_is_uniform(self):
        """Inclusion frequency d/D per dyad; with 3000 draws the binomial standard
        error is at most 0.0092, so 0.04 is a four-sigma bound for every dyad."""
        g = random_graph(4, dyads=60)
        b = budget_parameters(g)
        d = b['n_dyads']
        hits = np.zeros(g.D)
        for i in range(3000):
            r = rng('uniformity_test', g.key, 'H', i)
            hits[np.sort(r.choice(g.D, d, replace=False))] += 1
        self.assertLess(np.abs(hits / 3000 - d / g.D).max(), .04)
        # production draws use the same generator call on their own stream
        s = sample_dyads(g, 1, 'sample', b)
        self.assertEqual(len(s), d); self.assertEqual(len(set(s)), d)

    def test_streams_are_separate_from_the_legacy_arm(self):
        self.assertEqual(ARM_ID['H'], 'H-recent5')
        self.assertEqual(ARM_ID[LEGACY_H], 'H')
        self.assertEqual({ARM_ID[a] for a in 'RSB'}, set('RSB'))
        self.assertEqual(observation_id('g', 'H', 1), 'g__H-recent5__s1')
        self.assertEqual(observation_id('g', 'R', 1), 'g__R__s1')


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.g = random_graph(6, dyads=120)
        self.b = budget_parameters(self.g)
        c, _ = draw(self.g, 'H', 1, 'sample', self.b)
        self.c = c
        self.o = make(self.g, 'H', self.b, c, None)

    def test_round_trip_and_header(self):
        block = serialize(self.o)
        self.assertEqual(block.splitlines()[8], 'pattern,dyads,events,at_cap_dyads')
        back = parse(block)
        self.assertEqual(serialize(back), block)
        np.testing.assert_array_equal(features(back), features(self.o))
        self.assertEqual(len(back['table']), 31)
        self.assertEqual(back['D_obs'], self.b['n_dyads'])

    def test_at_cap_is_the_observed_count_not_true_truncation(self):
        seen = self.c.sum(1) > 0
        self.assertEqual(sum(r[3] for r in self.o['table']), int((self.c[seen].sum(1) == H_CAP).sum()))
        # A dyad with exactly five events is at cap but not truncated.
        rows = [('a', 'b', t) for t in (.1, .3, .5, .7, .9)] + [('c', 'd', t) for t in np.linspace(0, 1, 8)]
        rows += [(f'x{i}', f'y{i}', .5) for i in range(40)]
        g = canonical('cap', pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))[0]
        b = budget_parameters(g) | {'n_dyads': g.D, 'h_saturated': True}
        o = make(g, 'H', b, recent_counts(g.counts), None)
        table = {r[0]: r for r in o['table']}
        self.assertEqual(table['11111'][3], 1)          # m=5: exactly five, not truncated
        self.assertEqual(sum(r[3] for r in o['table']), 2)

    def test_block_does_not_depend_on_unretained_history(self):
        """Two archives that differ only before the retained events give the same block."""
        base = [('a', 'b', t) for t in (.65, .7, .75, .85, .9)] + [(f'x{i}', f'y{i}', .5) for i in range(30)]
        more = base + [('a', 'b', t) for t in (.05, .1, .3, .45)]
        blocks = []
        for rows in (base, more):
            g = canonical('same', pd.DataFrame(rows, columns=['u', 'v', 't']), horizon=(0, 1))[0]
            b = budget_parameters(g) | {'n_dyads': g.D, 'h_saturated': True}
            blocks.append(serialize(make(g, 'H', b, recent_counts(g.counts), None)))
        self.assertEqual(blocks[0], blocks[1])

    def test_validation_rejects_impossible_cap_rows(self):
        o = json.loads(json.dumps(self.o))
        o['table'] = [tuple(r) for r in o['table']]
        for i, r in enumerate(o['table']):
            if r[1] > 0 and r[3] < r[1]:
                bad = dict(o); t = list(o['table']); t[i] = (r[0], r[1], r[2], r[1] + 1); bad['table'] = t
                with self.assertRaises(ValueError): validate(bad)
                break
        full = dict(o); t = list(o['table'])
        t[-1] = ('11111', 1, 5, 0)        # five active windows need five events: must be at cap
        full['table'] = t; full['D_obs'] += 1; full['M_obs'] += 5
        full['Events_per_window'] = [x + 1 for x in o['Events_per_window']]
        with self.assertRaises(ValueError): validate(full)
        wrong = dict(o); wrong['parameter'] = o['D_obs'] + 1
        with self.assertRaises(ValueError): validate(wrong)
        header = serialize(self.o).replace('pattern,dyads,events,at_cap_dyads', 'pattern,dyads,events')
        with self.assertRaises(ValueError): parse(header)

    def test_rule_text_is_versioned_and_the_others_are_unchanged(self):
        self.assertEqual(RULE_FILES['H'], 'rule_H_recent5_v2.txt')
        self.assertEqual(RULE_FILES[LEGACY_H], 'rule_H_suffix_panel_v1.txt')
        text = messages(serialize(self.o))[1]['content']
        self.assertIn('most recent min(5, m_e) event records', text)
        self.assertIn('at_cap_dyads', text)
        self.assertNotIn('windows 3,4,5', text)
        for name in ('N_full', 'D_full', 'M_full', 'truth', 'truncat'):
            self.assertNotIn(name, serialize(self.o))

    def test_old_rsb_prompts_are_byte_identical(self):
        if not (RUN / 'observations').exists() or not (OLD / 'observations').exists():
            self.skipTest('runs not present')
        n = 0
        for f in sorted((RUN / 'observations' / 'sample').glob('*.json')):
            new = json.loads(f.read_text())
            if new['arm'] == 'H': continue
            old = json.loads((OLD / 'observations' / 'sample' / f.name).read_text())
            self.assertEqual((new['block'], new['prompt_sha256']), (old['block'], old['prompt_sha256']))
            n += 1
        self.assertEqual(n, 14 * 3 * 5)

    def test_legacy_variant_reproduces_the_old_h_blocks(self):
        if not (OLD / 'observations').exists(): self.skipTest('previous run not present')
        for key in ('sp_hospital', 'dar_a08_r1'):
            g = load_graph(OLD / 'graphs' / key)
            b = budget_parameters(g)
            for ix in (1, 5):
                c, _ = draw(g, LEGACY_H, ix, 'sample', b)
                old = json.loads((OLD / 'observations/sample' / f'{key}__H__s{ix}.json').read_text())
                self.assertEqual(serialize(make(g, LEGACY_H, b, c, None)), old['block'])
                self.assertEqual(messages(old['block']), old['messages'])


class BoundTests(unittest.TestCase):
    def test_hand_computed_example(self):
        table = {f'{p:05b}': (0, 0, 0) for p in range(1, 32)}
        table['00011'] = (4, 18, 3)    # J=2, b=4: capped dyads K in [2,5]
        table['01000'] = (2, 6, 1)     # J=1, b=2: capped dyad K in [1,2]
        table['10000'] = (1, 5, 1)     # J=1, b=1: capped but exact
        rows = [(p, *v) for p, v in table.items()]
        o = {'arm': 'H', 'N_obs': 8, 'D_obs': 7, 'M_obs': 29, 'Temporal_access': [1] * 5,
             'Events_per_window': [5, 6, 0, 9, 9], 'Walk_A': None, 'parameter': 7, 'table': rows}
        validate(o)
        lo, hi = h_bounds(o)
        self.assertEqual(lo, [4 / 7, 0, 0, 0])
        self.assertEqual(hi, [5 / 7, 3 / 7, 3 / 7, 3 / 7])
        self.assertEqual(lo, plugin(o))
        self.assertEqual(h_midpoint(o), [(a + b) / 2 for a, b in zip(lo, hi)])
        self.assertEqual(corrector(o), h_midpoint(o))

    def test_bounds_always_contain_the_sample_profile(self):
        """The intervals are statements about the sampled dyads, and they hold for
        every draw, not on average."""
        for seed in range(8):
            g = random_graph(100 + seed, dyads=150)
            b = budget_parameters(g)
            for ix in range(1, SAMPLER_DRAWS + 1):
                c, _ = draw(g, 'H', ix, 'sample', b)
                o = parse(serialize(make(g, 'H', b, c, None)))
                seen = c.sum(1) > 0
                K = (g.counts[seen] > 0).sum(1)
                oracle = [float(np.mean(K >= k)) for k in range(2, 6)]
                lo, hi = h_bounds(o)
                for l, x, h in zip(lo, oracle, hi):
                    self.assertLessEqual(l, x + 1e-12); self.assertLessEqual(x, h + 1e-12)
                mid = h_midpoint(o)
                self.assertTrue(all(0 <= x <= 1 for x in mid))
                self.assertTrue(all(x >= y for x, y in zip(mid, mid[1:])))

    def test_no_legacy_correction_in_the_current_features(self):
        self.assertNotIn('n_panel_suffix', FEATURE_NAMES)
        g = random_graph(9, dyads=150)
        b = budget_parameters(g)
        c, _ = draw(g, 'H', 1, 'sample', b)
        o = make(g, 'H', b, c, None)
        v = features(o)
        self.assertEqual(list(v[-4:]), h_midpoint(o))
        with self.assertRaises(ValueError): h_bounds(make(g, 'R', b, draw(g, 'R', 1, 'sample', b)[0], None))


class ReplicationTests(unittest.TestCase):
    def test_deterministic_draw_has_no_sampler_variance(self):
        a = np.array([[.1, .2, .3]])
        v = cell_variance(a)
        self.assertEqual(v['sampler'], 0.)
        self.assertAlmostEqual(v['total'], np.var([.1, .2, .3], ddof=1) / 3)
        self.assertTrue(v['deterministic_draw'])
        s = paired_summary({'x': a, 'y': np.tile([[.1], [.3], [.2], [.4], [.5]], (1, 3))}, {'x': 1, 'y': 5})
        self.assertAlmostEqual(s['mean'], (0.2 + 0.3) / 2)
        self.assertAlmostEqual(s['sources']['y']['mcse_model_repeats'], 0.)
        with self.assertRaises(ValueError): paired_summary({'x': a})          # expects 5 draws
        with self.assertRaises(ValueError): paired_summary({'x': np.ones((5, 3))}, {'x': 1})

    def test_components_match_the_total_for_several_draws(self):
        r = np.random.default_rng(0)
        a = r.normal(size=(5, 3)) + r.normal(size=(5, 1))
        v = cell_variance(a)
        self.assertAlmostEqual(v['total'], np.var(a.mean(1), ddof=1) / 5)
        self.assertGreaterEqual(v['sampler'], 0.)
        self.assertAlmostEqual(v['model'], np.mean(np.var(a, axis=1, ddof=1)) / 15)


class EngineRunnerTests(unittest.TestCase):
    """The GPU-free parts of run_qwen_engine.py."""

    def test_selection_sharding_and_layout(self):
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        from run_qwen_engine import load_requests, parse_passes, result_path, GENERATION_CONFIG, MODES
        import run_qwen_batch as batch
        passes = parse_passes('thinking:1,thinking:2,thinking:3,nonthinking:1,nonthinking:2,nonthinking:3')
        whole = load_requests(RUN, passes, {'H'}, 0, 1)
        report = json.loads((RUN / 'report.json').read_text())
        n_h = sum(1 for f in (RUN / 'observations/sample').glob('*__H-recent5__*.json'))
        self.assertEqual(len(whole), n_h * 6)
        self.assertEqual({r['arm'] for r in whole}, {'H'})
        parts = []
        for i in range(4): parts += [r['id'] for r in load_requests(RUN, passes, {'H'}, i, 4)]
        self.assertEqual(sorted(parts), sorted(r['id'] for r in whole))
        self.assertEqual(len(set(parts)), len(parts))
        r = whole[0]
        path = result_path(pathlib.Path('/x'), r)
        self.assertEqual(path.parent.name, f"{r['mode']}_r{r['repeat_index']}")
        self.assertEqual(MODES, batch.MODES)
        self.assertEqual((GENERATION_CONFIG['top_k'], GENERATION_CONFIG['presence_penalty']),
                         (batch.TOP_K, batch.PRESENCE_PENALTY))
        with tempfile.TemporaryDirectory() as d:
            from run_qwen_engine import write_result
            write_result(pathlib.Path(d), r, {'status': 'completed'})
            self.assertTrue(result_path(pathlib.Path(d), r).exists())
            self.assertFalse(list(pathlib.Path(d).rglob('*.tmp')))


if __name__ == '__main__':
    unittest.main()


class EvaluatorTests(unittest.TestCase):
    """The response evaluator on the real run with explicitly marked mock answers."""

    @classmethod
    def setUpClass(cls):
        prim = ROOT / 'results/baseline_revision_hrecent5_20260917/primary_baselines.json'
        if not (RUN / 'requests.jsonl').exists() or not prim.exists():
            raise unittest.SkipTest('run or primary baselines not present')
        from evaluate_main_responses import evaluate
        import csv
        requests = [json.loads(x) for x in (RUN / 'requests.jsonl').read_text().splitlines()]
        records = []
        for r in requests:
            if r['config_id'] == 'qwen_thinking':
                v = {1: .5, 2: .4, 3: .3}[r['repeat_index']]
                text = '```json\n{"rho_2":%s,"rho_3":0.2,"rho_4":0.1,"rho_5":0.0}\n```' % v
            elif r['config_id'] == 'qwen_nonthinking':
                text = 'Let me derive this.\n{"rho_2":0.5,"rho_3":0.2,"rho_4":0.1,"rho_5":0.0}'
            else:
                continue
            records.append({'id': r['id'], 'mock': True, 'started': True, 'terminal': True,
                            'final_text': text, 'prompt_sha256': r['prompt_sha256']})
        cls.tmp = tempfile.TemporaryDirectory(suffix='_mock')
        d = pathlib.Path(cls.tmp.name)
        src = d / 'mock.jsonl'
        src.write_text(''.join(json.dumps(x) + '\n' for x in records))
        evaluate(RUN, src, d / 'eval_mock', True, prim)
        def read(name):
            with open(d / 'eval_mock' / name) as f:
                return list(csv.DictReader(f))
        cls.summary, cls.sources, cls.secondary = read('summary.csv'), read('source_results.csv'), read('secondary.csv')
        cls.report = json.loads((d / 'eval_mock' / 'report.json').read_text())
        cls.errors = read('answer_errors.csv')

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_fallback_only_cells_carry_no_model_number(self):
        rows = [r for r in self.summary if r['config_id'] == 'qwen_nonthinking']
        self.assertTrue(rows)
        for r in rows:
            self.assertEqual(r['fallback_only'], 'True')
            self.assertEqual(r['numeric_model_estimate'], 'False')
            self.assertEqual(r['AE2'], '')
            self.assertEqual(r['shrink_plugin_AE2'], '')
            self.assertNotEqual(r['plugin_AE2'], '')          # references stay visible
            self.assertNotEqual(r['median_AE2'], '')
        self.assertTrue(all(r['value'] == '' for r in self.sources
                            if r['config_id'] == 'qwen_nonthinking' and r['metric'] == 'AE2'))
        self.assertIn('real/H/qwen_nonthinking', self.report['fallback_only_cells'])

    def test_valid_cells_and_the_deterministic_source(self):
        real_h = [r for r in self.summary if r['stratum'] == 'real' and r['arm'] == 'H'
                  and r['config_id'] == 'qwen_thinking'][0]
        self.assertEqual(real_h['numeric_model_estimate'], 'True')
        self.assertIn('sp_highschool2013', real_h['deterministic_draw_sources'])
        src = [r for r in self.sources if r['source'] == 'sp_highschool2013' and r['arm'] == 'H'
               and r['config_id'] == 'qwen_thinking' and r['metric'] == 'AE2'][0]
        self.assertEqual(src['draws'], '1')
        self.assertEqual(float(src['MCSE_sampler']), 0.)
        self.assertGreater(float(src['MCSE_model_repeats']), 0.)
        h_rows = [r for r in self.errors if r['arm'] == 'H' and r['config_id'] == 'qwen_thinking']
        self.assertTrue(all(r['reference_name'] == 'corrector' for r in h_rows))
        self.assertTrue(all(r['rho2_inside_h_bounds'] in ('True', 'False') for r in h_rows))

    def test_secondary_metrics(self):
        sec = [r for r in self.secondary if r['config_id'] == 'qwen_thinking' and r['stratum'] == 'real'
               and r['arm'] == 'R'][0]
        # repeats answer 0.5, 0.4, 0.3: the median answer is 0.4 and the spread sd is 0.1
        self.assertAlmostEqual(float(sec['spread_rho2_sd_valid']), .1, places=12)
        med = [r for r in self.errors if r['config_id'] == 'qwen_thinking' and r['repeat_index'] == '2'
               and r['stratum'] == 'real' and r['arm'] == 'R']
        per_source = {}
        for r in med: per_source.setdefault(r['graph_id'], []).append(float(r['AE2']))
        self.assertAlmostEqual(float(sec['median3_MAE2']), np.mean([np.mean(v) for v in per_source.values()]), places=12)
        row = [r for r in self.errors if r['config_id'] == 'qwen_thinking'][0]
        pred = json.loads(row['prediction_json'])
        self.assertEqual(float(row['validation_reason'] == 'valid_after_fence'), 1.)
        self.assertEqual(len(pred), 4)


class ResumeTests(unittest.TestCase):
    def test_pipeline_binding_rejects_changed_dependencies(self):
        from main_experiment.pipeline import binding
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'inputs.json'
            binding(path, {'design_version': 'budget10-hrecent5-20260917', 'x': 1})
            binding(path, {'design_version': 'budget10-hrecent5-20260917', 'x': 1})   # same: accepted
            with self.assertRaises(ValueError):
                binding(path, {'design_version': 'budget10-20261001', 'x': 1})

    def test_engine_resume_skips_exactly_the_written_results(self):
        from run_qwen_engine import pending, write_result
        reqs = [{'id': f'g__H-recent5__s1__qwen_thinking__r{i}', 'mode': 'thinking', 'repeat_index': i,
                 'observation_id': 'g__H-recent5__s1', 'graph_id': 'g', 'arm': 'H', 'sample_index': 1,
                 'config_id': 'qwen_thinking', 'seed': i, 'prompt_sha256': 'x'} for i in (1, 2, 3)]
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            done, todo = pending(out, reqs)
            self.assertEqual((len(done), len(todo)), (0, 3))
            write_result(out, reqs[1], {'status': 'completed'})
            # an interrupted write leaves only a temporary file, which never counts
            (out / 'thinking_r3').mkdir()
            (out / 'thinking_r3' / f"{reqs[2]['id']}.tmp").write_text('{')
            done, todo = pending(out, reqs)
            self.assertEqual(done, {reqs[1]['id']})
            self.assertEqual([r['id'] for r in todo], [reqs[0]['id'], reqs[2]['id']])

    def test_engine_main_reaches_model_loading_without_a_gpu(self):
        """main() up to the vLLM import: argument handling, selection and resume."""
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            cmd = [sys.executable, str(ROOT / 'scripts/run_qwen_engine.py'), '--run', str(RUN), '--out', d,
                   '--model', 'none', '--arms', 'H', '--shard-index', '0', '--shard-count', '4', '--limit', '2']
            r = subprocess.run(cmd, capture_output=True, text=True)
            self.assertIn('to run', r.stdout)
            # Without vLLM installed the run stops at the import, not before it.
            self.assertTrue(r.returncode == 0 or 'vllm' in r.stderr, r.stderr[-500:])
            self.assertNotIn('UnboundLocalError', r.stderr)

    def test_collector_refuses_an_incomplete_set_and_keeps_prompt_hashes(self):
        if not (RUN / 'requests.jsonl').exists(): self.skipTest('run not present')
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            req = next(json.loads(x) for x in (RUN / 'requests.jsonl').read_text().splitlines()
                       if x and json.loads(x)['config_id'] == 'qwen_thinking')
            (d / 'answers' / 'thinking_r1').mkdir(parents=True)
            rec = {'id': req['id'], 'status': 'completed', 'final_text': '{}', 'prompt_sha256': req['prompt_sha256'],
                   'end_state': 'model_end', 'reasoning_closed': True}
            (d / 'answers' / 'thinking_r1' / f"{req['id']}.json").write_text(json.dumps(rec))
            cmd = [sys.executable, str(ROOT / 'scripts/collect_qwen_answers.py'), '--run', str(RUN),
                   '--answers', str(d / 'answers'), '--out', str(d / 'out.jsonl')]
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertTrue((d / 'out.jsonl.missing.txt').exists())
            subprocess.run(cmd + ['--allow-incomplete'], check=True, capture_output=True)
            line = json.loads((d / 'out.jsonl').read_text().splitlines()[0])
            self.assertEqual(line['prompt_sha256'], req['prompt_sha256'])
