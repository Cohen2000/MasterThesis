import itertools
import tempfile
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
import yaml
from main_experiment.common import ROOT,DESIGN_VERSION,ARM_ID,seed
from main_experiment.data import canonical
from main_experiment.sampling import Walk,budget_parameters,draw
from main_experiment.observation import make,features,FEATURE_VERSION,FEATURE_NAMES,messages,serialize
from main_experiment.baselines import corrector

def fixture():
    pairs=[(0,1),(0,2),(0,3),(1,2),(2,3),(3,4)]
    rows=[(str(a),str(b),t) for i,(a,b) in enumerate(pairs) for t in np.linspace(0,1,i+2)]
    return canonical('srw_fixture',pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))[0]

class SRWTests(unittest.TestCase):
    def test_event_multiplicities_do_not_change_paths(self):
        g=fixture(); counts=g.counts.copy(); counts[0]*=1000
        other=replace(g,counts=counts)
        with tempfile.TemporaryDirectory() as d:
            a=Walk(g,d); b=Walk(other,d)
            sa=a.run(list(range(100)),31,True)[2]; sb=b.run(list(range(100)),31,True)[2]
            np.testing.assert_array_equal(sa,sb)
            self.assertTrue((sa.sum(1)==31).all())

    def test_exact_uniform_neighbor_edge_stationarity(self):
        g=fixture(); directed=[(a,b) for a,b in g.ends]+[(b,a) for a,b in g.ends]
        P=np.zeros((len(directed),len(directed)))
        for i,(a,b) in enumerate(directed):
            dest=[j for j,(c,d) in enumerate(directed) if c==b]
            P[i,dest]=1/len(dest)
        stationary=np.full(len(directed),1/len(directed))
        np.testing.assert_allclose(stationary@P,stationary)
        np.testing.assert_allclose(P.sum(1),1)

    def test_uniform_start_first_step_probabilities(self):
        g=fixture(); degree=np.bincount(g.ends.ravel(),minlength=g.N)
        expected=np.array([(1/degree[a]+1/degree[b])/g.N for a,b in g.ends])
        with tempfile.TemporaryDirectory() as d:
            walk=Walk(g,d); rr=walk.run([seed('srw_test',sample_index=i) for i in range(30000)],1,True)[2]
            got=rr.mean(0)
        np.testing.assert_array_less(np.abs(got-expected),6*np.sqrt(expected*(1-expected)/30000))

    def test_components_limit_coverage_and_all_transitions_count(self):
        rows=[('a','b',0.),('a','b',1.),('c','d',0.),('d','e',.5),('e','c',1.)]
        g=canonical('disconnected',pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))[0]
        with tempfile.TemporaryDirectory() as d:
            w=Walk(g,d,volume='cells'); _,v,r,executed=w.run(list(range(100)),100,True)
            self.assertEqual(w.n_components,2)
            self.assertAlmostEqual(np.mean(w.component_volume),.4*2+.6*3)
            self.assertTrue((executed==100).all()); self.assertTrue((r.sum(1)==100).all())
            for row in r:
                comps={w.components[g.ends[i,0]] for i in np.flatnonzero(row)}
                self.assertEqual(len(comps),1)

    def test_reference_uses_raw_counts_not_event_inverse(self):
        g=fixture(); b=budget_parameters(g)|{'L':51}
        with tempfile.TemporaryDirectory() as d:
            w=Walk(g,d); c,r=draw(g,'S',1,'test',b,w)
            o=make(g,'S',b,c,r)
        self.assertEqual(sum(o['Walk_A']),51)
        expected=[float(r[g.K>=k].sum()/r.sum()) for k in range(2,6)]
        np.testing.assert_allclose(corrector(o),expected)
        np.testing.assert_allclose(features(o)[-4:],expected)
        text=messages(serialize(o))[1]['content']
        self.assertNotIn('r_e/m_e',text); self.assertIn('uniformly among',text)
        self.assertIn('sums of r_e',text)

    def test_yaml_matches_executable_current_state(self):
        spec=yaml.safe_load((ROOT/'config/study.yaml').read_text())
        self.assertEqual(spec['design_version'],DESIGN_VERSION)
        self.assertEqual(spec['training']['feature_version'],FEATURE_VERSION)
        self.assertEqual(spec['training']['features'],len(FEATURE_NAMES))
        self.assertEqual(spec['sampling']['S'],'simple_random_walk_full_history')
        self.assertEqual(spec['arm_H']['history_fraction'],.6)
        self.assertFalse(spec['evaluation']['llm_imputation'])
        self.assertEqual(ARM_ID['S'],'S-p888-20260921')
