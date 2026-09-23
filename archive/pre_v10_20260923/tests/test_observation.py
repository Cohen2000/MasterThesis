"""Observation blocks, prompts and the 80 learned-reference features."""
import tempfile
import unittest
import numpy as np
from helpers import complete6, tiny
from main_experiment.common import ARMS, H_SENSITIVITY
from main_experiment.observation import (BASE_FEATURE_NAMES, DERIVED_FEATURE_NAMES, FEATURE_NAMES, features, make,
                                         messages, parse, serialize, validate)
from main_experiment.sampling import Walk, analytic_parameters, draw, h_parameters


class SerializationTests(unittest.TestCase):
    def test_every_arm_round_trips_and_hides_the_source(self):
        g = tiny(); b = analytic_parameters(g) | {'L': 7}
        with tempfile.TemporaryDirectory() as d:
            walk = Walk(g, d)
            for arm in ARMS:
                counts, traversals = draw(g, arm, 1, 'sample', b, walk)
                o = make(g, arm, b, counts, traversals)
                block = serialize(o); back = parse(block)
                self.assertEqual((serialize(back), back['arm']), (block, arm))
                self.assertEqual(len(o['table']), 7 if arm == 'H' else 31)
                self.assertNotIn('9999999', block)                     # floats have 12 significant digits
                np.testing.assert_array_equal(features(o), features(back))
                text = messages(block)[1]['content']
                self.assertNotIn('fixture', text); self.assertNotIn('budget_matched', text)
                self.assertEqual('traversals,inverse_degree_weight' in block, arm == 'S2')   # only S2 shows walker data

    def test_h_blocks_mark_inaccessible_windows(self):
        g = complete6()
        for h in H_SENSITIVITY:
            b = h_parameters(g, .1*g.cells, h)
            o = make(g, 'H', b, draw(g, 'H', 1, 'sample', b)[0])
            n = round(5*h)
            self.assertEqual(len(o['table']), 2**n-1)
            self.assertEqual(o['Events_per_window'][:5-n], [None]*(5-n))
            block = serialize(o)
            self.assertNotIn('n_panel_history', block)          # n_panel is never released, not even for H
            with self.assertRaises(ValueError): parse(block.replace('Temporal_access=', 'unexpected='))

    def test_inconsistent_observations_are_rejected(self):
        g = tiny(); b = analytic_parameters(g)
        o = make(g, 'R', b | {'n_panel': g.N}, g.counts)
        o['Events_per_window'][0] += 1
        with self.assertRaises(ValueError): validate(o)
        c = draw(g, 'H', 1, 'sample', b | {'n_panel_history': g.N, 'h_saturated': True})[0]
        c[:, 0] = 1
        with self.assertRaises(ValueError): make(g, 'H', b, c)      # events in a non-retrievable window


class FeatureTests(unittest.TestCase):
    def test_feature_blocks(self):
        # features-v9-access-contract-20260922: a flat released-evidence-only
        # vector; BASE_FEATURE_NAMES/DERIVED_FEATURE_NAMES are compatibility
        # aliases only (see observation.py), so both equal FEATURE_NAMES/[].
        self.assertEqual((len(BASE_FEATURE_NAMES), len(DERIVED_FEATURE_NAMES), len(FEATURE_NAMES)), (80, 0, 80))
        self.assertIs(BASE_FEATURE_NAMES, FEATURE_NAMES)
        self.assertEqual(FEATURE_NAMES[-4:], ['log1p_N_obs', 'log1p_D_obs', 'log1p_M_obs', 'p'])
        # S2 (walker traversal/degree information) is not an active arm.
        self.assertEqual([n for n in FEATURE_NAMES if 'traversal' in n or 'degree' in n], [])
        banned = ('N_full', 'D_full', 'M_full', 'truth', 'coverage', 'source', 'family', 'alpha', 'chi', 'generator',
                  'Walk', 'A_', 'n_panel', 'n_panel_history', 'history_fraction', ' L')
        for name in FEATURE_NAMES:
            for word in banned: self.assertNotIn(word, name)

    def test_derived_features_are_shares_and_reference_profiles(self):
        from main_experiment.baselines import anchor_profile
        from main_experiment.observation import ALL_PATTERNS
        g = complete6(); b = h_parameters(g, .1*g.cells, .6)
        o = make(g, 'H', b, draw(g, 'H', 1, 'sample', b)[0])
        v = dict(zip(FEATURE_NAMES, features(o)))
        self.assertAlmostEqual(sum(v[f'share_{p}_dyads'] for p in ALL_PATTERNS), 1., places=12)
        self.assertAlmostEqual(sum(v[f'share_window_{i}'] for i in range(1, 6)), 1., places=12)
        self.assertEqual([v[f'anchor_rho_{k}'] for k in range(2, 6)], list(anchor_profile(o)))
        self.assertEqual(v['events_per_dyad'], o['M_obs']/o['D_obs'])

    def test_empty_observation_has_zero_derived_features(self):
        o = {'arm': 'B', 'N_obs': 0, 'D_obs': 0, 'M_obs': 0, 'Temporal_access': [1]*5, 'Events_per_window': [0]*5,
             'parameter': .5, 'table': [(f'{p:05b}', 0, 0) for p in range(1, 32)]}
        v = features(o)
        self.assertTrue(np.isfinite(v).all() and (v[len(BASE_FEATURE_NAMES):] == 0).all())


if __name__ == '__main__':
    unittest.main()
