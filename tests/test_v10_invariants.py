"""Scientific invariants of the v10 interaction-walk study."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor

from main_experiment.common import digest, seed
from main_experiment.data import Graph, load_graph, prepare_real
from main_experiment.observation import features, make, messages, parse, serialize, validate
from main_experiment.baselines import design_estimate, plugin
from main_experiment.requests import payload, protocol_version
from main_experiment.sampling import calibrate, draw
from main_experiment.shared_mle import fit
from main_experiment.walk_v10_audit import Walk


def toy():
    ends = np.array([[0, 1], [0, 2], [1, 2]], dtype=np.int64)
    counts = np.array([[1, 0, 0, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 0, 0]], dtype=np.int64)
    pair = np.repeat(np.arange(3), counts.sum(1))
    u = ends[pair, 0]
    v = ends[pair, 1]
    w = np.concatenate([np.repeat(np.arange(5), c) for c in counts])
    return Graph('toy', u, v, w.astype(float), w, pair, ends, counts, (0., 4.))


class V10Invariants(unittest.TestCase):
    def test_weighted_kernel_and_design_limit(self):
        g = toy()
        with tempfile.TemporaryDirectory() as d:
            walk = Walk(g, d)
            _, _, c, executed = walk.run(np.arange(1, 501, dtype=np.uint64), 3000, True)
        self.assertTrue(np.all(executed == 3000))
        frequency = c.sum(0) / c.sum()
        np.testing.assert_allclose(frequency, g.m / g.m.sum(), atol=.015)
        design = np.array([(x[g.K >= 2] / g.m[g.K >= 2]).sum() / (x / g.m).sum() for x in c])
        self.assertAlmostEqual(float(design.mean()), g.truth[0], delta=.015)

    def test_s_blocks_round_trip_and_validate_columns(self):
        g = toy()
        counts = g.counts
        traversal = np.array([2, 1, 3])
        for arm in ('S', 'S_obs'):
            o = make(g, arm, {'L': 6}, counts, traversal)
            p = parse(serialize(o))
            self.assertEqual(serialize(p), serialize(o))
            self.assertAlmostEqual(design_estimate(p)[0], design_estimate(o)[0])
            broken = {**p, 'table': list(p['table'])}
            index = next(i for i, row in enumerate(broken['table']) if row[1] > 0)
            row = list(broken['table'][index])
            row[3] = 0.
            broken['table'][index] = tuple(row)
            with self.assertRaises(ValueError): validate(broken)

    def test_weighted_mle_unit_weights(self):
        g = toy()
        ordinary = parse(serialize(make(g, 'R', {'n_panel': g.N}, g.counts, None)))
        # c_e = m_e makes every released S pseudo-weight c_e/m_e equal one.
        weighted = parse(serialize(make(g, 'S', {'L': g.M}, g.counts, g.m)))
        a = fit(ordinary)
        b = fit(weighted)
        np.testing.assert_allclose(a.rho, b.rho, atol=1e-10)
        self.assertEqual(a.fallback_used, b.fallback_used)

    def test_non_llm_estimators_use_serialized_block(self):
        g = toy()
        o = make(g, 'S', {'L': 6}, g.counts, np.array([2, 1, 3]))
        p = parse(serialize(o))
        np.testing.assert_allclose(plugin(o), plugin(p))
        np.testing.assert_allclose(design_estimate(o), design_estimate(p))
        np.testing.assert_allclose(fit(o).rho, fit(p).rho)
        np.testing.assert_allclose(features(o), features(p))
        x = np.vstack([features(p), features(p) * 1.01, features(p) * .99])
        forest = ExtraTreesRegressor(n_estimators=5, random_state=7).fit(x, np.array([g.truth] * 3))
        np.testing.assert_allclose(forest.predict(features(o)[None])[0],
                                   forest.predict(features(p)[None])[0])

    def test_rhb_matches_v9_cpu_preparation_fixture(self):
        raw = Path(__file__).resolve().parents[1] / 'data/raw'
        if not raw.exists(): self.skipTest('raw source archive unavailable')
        expected = {
            'R': ('2692451c80d3937c8485effbf5867408ca9dd1b301ac6f952b6b5d934ffd2626',
                  '3a99d2ab1b8da3f26cb75244a3adb9f545ae1d38732612075c427d8e6e901563',
                  'f0f06e9d64fb0e77c7eb628920b3421a9c0cbff55c44cac143593ec86f5f696f', 736186645164153965),
            'H': ('a61c0b5321d0c740b357fe13ed1887b578a405c6acba2bec1cd228c36e9348b1',
                  'c2fa10c1694373c81d5bd7af1909475bcf5c951c5070eccbc26ac97c811cd5f9',
                  'cb9c7db8a93af9f1b238841d5b22df98b11a33c76b524d679ce6db0699c6f1d0', 221502688853208556),
            'B': ('80f8c1ae9102b01475c35ccb6cb49c5d8d2ed7ac50e172a93d5e89f27a7ede67',
                  'e0184f22e0d175c5f9e1e3cb9bb02f16d7513e268c44f5e743d06f2a28cb753a',
                  '362d42ea501adf6c899e781ac8ca9413e3cfa82263eddd93b5b0b50f378bf746', 7154907299515610023)}
        with tempfile.TemporaryDirectory() as d:
            g = prepare_real('sp_hospital', raw, Path(d) / 'graph')
            budget, walk, _ = calibrate(g, Path(d) / 'build')
            for arm, (block_hash, prompt_hash, payload_hash, expected_seed) in expected.items():
                counts, traversal = draw(g, arm, 1, 'sample', budget, walk)
                block = serialize(make(g, arm, budget, counts, traversal))
                msg = messages(block)
                self.assertEqual(digest(block), block_hash)
                self.assertEqual(digest(msg), prompt_hash)
                sampler = f'{arm}-p888-access-v9-20260922'
                request_seed = seed('llm', 'sp_hospital', sampler, 1, 1,
                                    'qwen_thinking:' + protocol_version(arm))
                self.assertEqual(request_seed, expected_seed)
                self.assertEqual(digest(payload('qwen_thinking', msg, request_seed,
                                                protocol_version(arm))), payload_hash)


if __name__ == '__main__': unittest.main()
