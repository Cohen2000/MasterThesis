"""Canonical graphs, synthetic generators, P[w,t] surrogates and the study config."""
import unittest
import numpy as np
import yaml
from helpers import graph, tiny
from main_experiment.common import (ARM_ID, DESIGN_VERSION, MAIN_KEYS, REAL_TEST, ROOT, SURROGATES, SYNTH, TRAIN,
                                    fold_for, parent_source, seed)
from main_experiment.observation import FEATURE_NAMES, FEATURE_VERSION
from main_experiment.surrogates import audit, collisions, shuffle
from main_experiment.synthetic import generate_pair


class CanonicalGraphTests(unittest.TestCase):
    def test_window_boundaries_and_duplicates(self):
        self.assertEqual(tiny().counts.sum(0).tolist(), [1, 1, 1, 1, 2])
        # No epsilon: just below a cut stays in the earlier window; on the cut goes later.
        rows = [('a', 'b', 0.), ('a', 'b', np.nextafter(.4, 0)), ('b', 'a', .4), ('a', 'b', .4), ('a', 'b', 1.)]
        plain, proximity = graph(rows), graph(rows, proximity=True)
        self.assertEqual((plain.M, proximity.M), (5, 4))       # duplicates removed only for proximity data
        self.assertEqual(plain.counts.sum(0).tolist(), [1, 1, 2, 0, 1])
        self.assertEqual((proximity.D, proximity.horizon), (1, (0., 1.)))

    def test_truth_is_the_dyad_equal_window_profile(self):
        g = graph([('a', 'b', 0.), ('a', 'b', .5), ('a', 'c', .1), ('a', 'c', .15), ('b', 'c', .9)])
        self.assertEqual(g.K.tolist(), [2, 1, 1])
        self.assertEqual(g.truth, [1/3, 0., 0., 0.])
        self.assertEqual(g.cells, 4)


class GeneratorTests(unittest.TestCase):
    def test_dar_pair_shares_latents_and_has_stationary_activity(self):
        pair = list(generate_pair('dar', 1))
        self.assertEqual(pair[0][2]['shared_latents'], pair[1][2]['shared_latents'])
        for g, _, _ in pair:
            self.assertEqual(g.horizon, (0., 1.))
            active = (g.counts > 0).sum(0)                # each window ~ Bin(5000, .2)
            self.assertTrue(np.all(abs(active-1000) < 6*np.sqrt(5000*.2*.8)))
        q = .2
        expected = (1-(1-q)**5-5*q*(1-q)**4)/(1-(1-q)**5)  # alpha = 0: independent windows
        independent = pair[0][0]
        self.assertLess(abs(independent.truth[0]-expected), 6*np.sqrt(expected*(1-expected)/independent.D))
        np.testing.assert_array_equal(independent.t, next(generate_pair('dar', 1))[0].t)

    def test_ad_pair_has_synchronous_unique_contacts(self):
        pair = list(generate_pair('ad', 1))
        self.assertEqual(pair[0][2]['shared_latents'], pair[1][2]['shared_latents'])
        for g, frame, _ in pair:
            self.assertEqual(len(frame), len(frame.drop_duplicates(['u', 'v', 't'])))
            rounds = np.rint(g.t*1000-.5).astype(int)
            np.testing.assert_array_equal(g.w, rounds//200)
        self.assertEqual(pair[0][0].M+pair[0][2]['mutual_initiations_collapsed'],
                         pair[1][0].M+pair[1][2]['mutual_initiations_collapsed'])


class SurrogateTests(unittest.TestCase):
    def parent(self):
        return graph([(str(a), str((a+1) % 30), t) for a in range(30) for t in (0., .2, .4, .6, .8, 1.)],
                     'sp_hospital', proximity=True)

    def test_invariants_hold_and_collisions_are_kept(self):
        g = self.parent(); s = shuffle(g)
        self.assertTrue(audit(g, s)['passed'])
        self.assertEqual(s.M, g.M)                             # no re-deduplication after the shuffle
        self.assertGreater(collisions(s), collisions(g))
        np.testing.assert_array_equal(s.t, shuffle(g).t)       # deterministic productive shuffle
        self.assertFalse(np.array_equal(s.t, shuffle(g, 1).t))  # null shuffles are different streams

    def test_audit_rejects_a_changed_multiset(self):
        g = self.parent(); s = shuffle(g)
        s.t[0] = .123
        with self.assertRaises(AssertionError): audit(g, s)

    def test_surrogate_uses_the_parent_fold_and_streams(self):
        for key in SURROGATES:
            parent = key.removesuffix('__pwt')
            self.assertEqual((parent_source(key), fold_for(key)), (parent, parent))
        self.assertEqual(fold_for('dar_a0_r1'), 'synthetic')


class ConfigurationTests(unittest.TestCase):
    def test_panel_and_study_yaml_match_the_code(self):
        self.assertEqual((len(REAL_TEST), len(SURROGATES), len(SYNTH), len(MAIN_KEYS)), (8, 8, 8, 24))
        self.assertTrue(set(REAL_TEST) <= set(TRAIN) and len(TRAIN) == 16)
        spec = yaml.safe_load((ROOT/'config/study.yaml').read_text())
        self.assertEqual(spec['design_version'], DESIGN_VERSION)
        self.assertEqual(spec['panel']['real'], list(REAL_TEST))
        self.assertEqual(spec['panel']['synthetic'], list(SYNTH))
        self.assertEqual(spec['training']['feature_version'], FEATURE_VERSION)
        self.assertEqual(spec['training']['features'], len(FEATURE_NAMES))
        self.assertEqual(spec['training']['random_state'], seed('extratrees', 'all_folds') % 2**32)
        self.assertEqual(spec['arm_H']['history_fraction'], .6)
        self.assertFalse(spec['evaluation']['llm_imputation'])
        self.assertEqual((ARM_ID['S1'], ARM_ID['S2']), ('S-dbrw-p888-20260921', 'S2-dbrw-p888-20260921'))


if __name__ == '__main__':
    unittest.main()
