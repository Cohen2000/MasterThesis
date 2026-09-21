import tempfile
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from main_experiment.data import canonical
from main_experiment.surrogates import shuffle,audit,collisions
from main_experiment.common import seed,parent_source,fold_for,REAL_TEST,TRAIN,draws_for
from main_experiment.sampling import Walk,draw,budget_parameters,history_panel_mask

class SurrogateTests(unittest.TestCase):
    def graph(self):
        rows=[(str(a),str((a+1)%30),t) for a in range(30) for t in (0.,.2,.4,.6,.8,1.)]
        return canonical('sp_hospital',pd.DataFrame(rows,columns=['u','v','t']),proximity=True,horizon=(0,1))[0]

    def test_multiset_invariants_and_collision_retention(self):
        g=self.graph(); s=shuffle(g)
        self.assertTrue(audit(g,s)['passed'])
        self.assertGreater(collisions(s),collisions(g))
        self.assertEqual(s.M,g.M)
        np.testing.assert_array_equal(s.t,shuffle(g).t)
        self.assertFalse(np.array_equal(s.t,shuffle(g,1).t))

    def test_crn_node_panels_walks_and_bernoulli_records(self):
        g=self.graph(); s=shuffle(g); b=budget_parameters(g)|{'L':101}
        bs=budget_parameters(s)|{'L':101}
        # R budget node size agrees because budget is the same relative share.
        for arm in ('R','H'):
            pa=history_panel_mask(g,1,'sample',b) if arm=='H' else None
            pb=history_panel_mask(s,1,'sample',b) if arm=='H' else None
            if arm=='H':np.testing.assert_array_equal(pa,pb)
        with tempfile.TemporaryDirectory() as d:
            wg=Walk(g,d); ws=Walk(s,d)
            rg=draw(g,'S',1,'sample',b,wg)[1];rs=draw(s,'S',1,'sample',bs,ws)[1]
            np.testing.assert_array_equal(rg,rs)
        # Same probability implies same retained record identities; times differ.
        bs['p']=b['p']
        cg,_=draw(g,'B',1,'sample',b);cs,_=draw(s,'B',1,'sample',bs)
        np.testing.assert_array_equal(cg.sum(1),cs.sum(1))
        self.assertEqual(fold_for(s.key),g.key)
        self.assertEqual(draws_for('R',b),3)
        self.assertEqual(draws_for('R',b,'training'),5)
