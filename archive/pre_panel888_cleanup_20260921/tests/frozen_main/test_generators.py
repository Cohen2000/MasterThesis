import unittest
import numpy as np
from main_experiment.synthetic import generate_pair

class GeneratorTests(unittest.TestCase):
    def test_dar_paired_stationarity_and_fixed_horizon(self):
        pair=list(generate_pair('dar',1))
        self.assertEqual(len(pair),2)
        self.assertEqual(pair[0][2]['shared_latents'],pair[1][2]['shared_latents'])
        for g,x,m in pair:
            self.assertEqual(g.horizon,(0.,1.))
            self.assertLess(g.D,5000)
            self.assertTrue(np.all(g.counts[g.counts>0]>=1))
            # Each marginal state count has Bin(5000,.2) law. Six SD is conservative;
            # dependence across windows does not invalidate the individual bounds.
            active=(g.counts>0).sum(0)
            self.assertTrue(np.all(abs(active-1000)<6*np.sqrt(5000*.2*.8)))
        independent=pair[0][0]
        q=.2; den=1-(1-q)**5
        expected=(1-(1-q)**5-5*q*(1-q)**4)/den
        # Conditional on D, observed persistent edge fraction is binomial.
        se=np.sqrt(expected*(1-expected)/independent.D)
        self.assertLess(abs(independent.truth[0]-expected),6*se)
        repeated=list(generate_pair('dar',1))
        np.testing.assert_array_equal(pair[0][0].t,repeated[0][0].t)

    def test_ad_pair_synchronous_contacts(self):
        pair=list(generate_pair('ad',1))
        self.assertEqual(pair[0][2]['shared_latents'],pair[1][2]['shared_latents'])
        for g,x,m in pair:
            self.assertEqual(g.horizon,(0.,1.))
            self.assertEqual(len(x),len(x.drop_duplicates(['u','v','t'])))
            rounds=np.rint(g.t*1000-.5).astype(int)
            self.assertTrue(np.all((rounds>=0)&(rounds<1000)))
            np.testing.assert_array_equal(g.w,rounds//200)
            self.assertTrue(np.all(g.u!=g.v))
        # Initiation counts coincide because activities and activations are shared.
        self.assertEqual(pair[0][0].M+pair[0][2]['mutual_initiations_collapsed'],
                         pair[1][0].M+pair[1][2]['mutual_initiations_collapsed'])
