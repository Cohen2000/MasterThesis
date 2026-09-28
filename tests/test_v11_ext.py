"""Invariants of the additive v11 extension (new sources, ET replicates, orchestration)."""
import gzip
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
from helpers import ring
from main_experiment.common import ARM_ID, observation_id, seed
from main_experiment.observation import parse
from main_experiment.requests import validate_request
from main_experiment.sampling import analytic_parameters
from v11_ext import core, observe, qwen, sources

V10_DOMAINS = {'sample', 'training', 'pool_train', 'pool_dev', 'pool', 'pool_definition', 'graph',
               'llm', 'v10_et_nested', 'v10_et_final'}


def budget(g):
    b = analytic_parameters(g)
    b['budget_matched_by_arm'] = {a: True for a in core.ARMS}
    b['budget_matched'] = True
    return b


class StreamDomainTests(unittest.TestCase):
    def test_replicate_zero_is_production_and_others_are_new(self):
        for kind in ('training', 'pool_train', 'et_nested', 'et_final'):
            self.assertIn(core.stream_domain(kind, 0), V10_DOMAINS)
            new = {core.stream_domain(kind, k) for k in range(1, 11)}
            self.assertEqual(len(new), 10)
            self.assertFalse(new & V10_DOMAINS)

    def test_replicate_seeds_differ_from_production(self):
        a = seed(core.stream_domain('training', 0), 'sp_hospital', ARM_ID['R'], 1)
        b = seed(core.stream_domain('training', 1), 'sp_hospital', ARM_ID['R'], 1)
        self.assertNotEqual(a, b)


class ReleaseTests(unittest.TestCase):
    def test_panel_release_adds_only_n_panel(self):
        g = ring()
        b = budget(g)
        for arm in ('R', 'H'):
            hidden, block, counts = observe.draw_block(g, arm, 1, 'sample', b, None)
            self.assertNotIn('n_panel', hidden)
            self.assertEqual(block.replace(f'n_panel={b[observe.PANEL_FIELD[arm]]}\n', ''), hidden)
            record = observe.observation_record(g, arm, 1, 'sample', b, hidden, block, counts, 'real')
            self.assertEqual(record['paired_hidden_id'], observation_id(g.key, arm, 1))
            self.assertTrue(record['id'].endswith('-panel-release__s1'))
        _, block, _ = observe.draw_block(g, 'B', 1, 'sample', b, None)
        self.assertNotIn('n_panel', block)
        with self.assertRaises(ValueError): observe.release(block, 5)

    def test_release_is_deterministic_and_domain_keyed(self):
        g = ring()
        b = budget(g)
        one = observe.draw_block(g, 'B', 1, 'training', b, None)[1]
        self.assertEqual(one, observe.draw_block(g, 'B', 1, 'training', b, None)[1])
        self.assertNotEqual(one, observe.draw_block(g, 'B', 1, core.stream_domain('training', 1), b, None)[1])


class WindowRuleTests(unittest.TestCase):
    def test_isolated_leading_outlier_is_trimmed(self):
        t = np.r_[0., np.linspace(1000., 2000., 5000)]
        lo, hi, lead, trail = sources.end_outliers(t, .001)
        self.assertEqual((lo, hi, lead, trail), (1000., 2000., 1, 0))
        frame = pd.DataFrame({'u': [str(i % 7) for i in range(len(t))], 'v': [str(i % 7+1) for i in range(len(t))], 't': t})
        g, _, _, report = sources.checked_graph('x', frame, False, {'min_share': .01, 'max_trim_share': .001})
        self.assertEqual(report['trimmed_records'], 1)
        self.assertFalse(report['flagged'])
        self.assertGreaterEqual(min(sources.window_shares(g)), .01)

    def test_genuine_sparse_window_is_flagged_not_trimmed(self):
        t = np.r_[np.linspace(0., 100., 3000), np.linspace(900., 1000., 3000), [450.]]
        frame = pd.DataFrame({'u': ['a']*len(t), 'v': ['b']*len(t), 't': t})
        g, _, _, report = sources.checked_graph('x', frame, False, {'min_share': .01, 'max_trim_share': .001})
        self.assertTrue(report['flagged'])
        self.assertEqual(report['trimmed_records'], 0)
        self.assertEqual(g.M, len(t))


class RawParserTests(unittest.TestCase):
    def write_zip(self, folder, weight):
        text = '# source, target, weight, time\n'+''.join(f'{i},{i+1},{weight},{100+i}\n' for i in range(20))
        path = folder/'x.csv.zip'
        with zipfile.ZipFile(path, 'w') as z: z.writestr('edges.csv', text)
        return path

    def spec(self, path, **extra):
        return {'file': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **extra}

    def test_netzschleuder_weight_must_be_one(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            ok = self.write_zip(d, 1)
            frame, facts = sources.load_raw('x', self.spec(ok, member='edges.csv', columns={'u': 0, 'v': 1, 'weight': 2, 't': 3}), d)
            self.assertEqual(len(frame), 20)
            self.assertEqual(facts['weight_values'], [1])
            bad = self.write_zip(d, 2)
            with self.assertRaises(ValueError):
                sources.load_raw('x', self.spec(bad, member='edges.csv', columns={'u': 0, 'v': 1, 'weight': 2, 't': 3}), d)
            spec = self.spec(ok, member='edges.csv', columns={'u': 0, 'v': 1, 'weight': 2, 't': 3})
            spec['sha256'] = '0'*64
            with self.assertRaises(ValueError): sources.load_raw('x', spec, d)

    def test_sociopatterns_records_must_be_20s(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            path = d/'m.csv.gz'
            with gzip.open(path, 'wt') as f: f.write(',contact_time,day,id1,id2\n0,0,22,1,2\n1,20,22,2,3\n')
            spec = self.spec(path, header=',contact_time,day,id1,id2', columns={'u': 3, 'v': 4, 't': 1})
            frame, _ = sources.load_raw('m', spec, d)
            self.assertEqual(frame.t.tolist(), [0., 20.])
            with gzip.open(path, 'wt') as f: f.write(',contact_time,day,id1,id2\n0,7,22,1,2\n')
            with self.assertRaises(ValueError): sources.load_raw('m', self.spec(path, header=spec['header'], columns=spec['columns']), d)


class QwenRequestTests(unittest.TestCase):
    def test_released_rh_requests_use_hidden_sampler_seed(self):
        g = ring()
        b = budget(g)
        rows = []
        for arm in ('R', 'B'):
            hidden, block, counts = observe.draw_block(g, arm, 1, 'sample', b, None)
            rows.append({**observe.observation_record(g, arm, 1, 'sample', b, hidden, block, counts, 'real')})
        requests = qwen.requests_for(rows)
        self.assertEqual(len(requests), 2*2*3)
        for r in requests:
            validate_request(r)
            if r['arm'] == 'R':
                self.assertEqual(r['seed'], seed('llm', g.key, ARM_ID['R'], 1, r['repeat_index'],
                                                 r['config_id']+':panel888-access-v9-20260922'))
                self.assertIn('-panel-release__s1__', r['id'])


class TaskTests(unittest.TestCase):
    def test_keys_propagate_and_runs_are_idempotent(self):
        with tempfile.TemporaryDirectory() as d, mock.patch.object(core, 'WORK', Path(d)):
            frozen = {'x': 1}
            a = core.Task('testset', 'a', {'p': 1}).finalize(frozen)
            b = core.Task('testset', 'b', {}, [a]).finalize(frozen)
            a2 = core.Task('testset', 'a', {'p': 2}).finalize(frozen)
            b2 = core.Task('testset', 'b', {}, [a2]).finalize(frozen)
            self.assertNotEqual(b.key, b2.key)
            calls = []

            def fn(task, out, inputs):
                calls.append(task.name); (out/'x.json').write_text(json.dumps(sorted(inputs)))
            with self.assertRaises(RuntimeError): core.run(b, fn)
            core.run(a, fn); core.run(a, fn); core.run(b, fn)
            self.assertEqual(calls, ['a', 'b'])
            self.assertTrue(a.done and b.done)
            self.assertEqual(json.loads((b.out/'x.json').read_text()), ['a'])

    def test_cache_computes_once(self):
        with tempfile.TemporaryDirectory() as d:
            cache = core.Cache('features', d)
            calls = []
            for _ in range(3): self.assertEqual(cache.get('ab'*32, lambda: calls.append(1) or [1.5]), [1.5])
            self.assertEqual((len(calls), cache.hits), (1, 2))


class DagTests(unittest.TestCase):
    def test_task_graph_shape(self):
        from v11_ext import dag
        with tempfile.TemporaryDirectory() as d, mock.patch.object(dag, 'frozen_inputs', lambda: {}), \
                mock.patch.object(core, 'WORK', Path(d)):
            tasks = dag.build(2)
        count = lambda stage: sum(t.stage == stage for t in tasks.values())
        self.assertEqual(count('source'), 4)
        self.assertEqual(count('draw_real'), 2*16*5)
        self.assertEqual(count('draw_pool'), 2*20)
        self.assertEqual(count('select'), 5+2*50)        # replicate 0: only the new radoslaw fold
        self.assertEqual(count('train'), 3*50)
        rep0 = tasks['train:0:H:sp_hospital']
        self.assertEqual([t.name for t in rep0.deps], ['testset'])
        self.assertIn('select:0:H:nr_radoslaw_email', [t.name for t in tasks['train:0:H:nr_radoslaw_email'].deps])
        order = ['source', 'testset', 'qwen_bundle', 'draw_real', 'draw_pool', 'select', 'train', 'report']
        for t in tasks.values():
            for dep in t.deps: self.assertLess(order.index(dep.stage), order.index(t.stage)+(t.stage == dep.stage))


if __name__ == '__main__':
    unittest.main()
