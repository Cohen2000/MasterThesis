import itertools
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.stats import binom
from main_experiment.common import H_SENSITIVITY,PREVIOUS_RUN,CURRENT_RUN,DESIGN_VERSION,ROOT,digest,read_json
from main_experiment.data import canonical
from main_experiment.sampling import history_counts,history_start,h_parameters,history_panel_mask,draw
from main_experiment.observation import make,serialize,parse,features,messages
from main_experiment.baselines import h_extrapolator,profile
from main_experiment.history_diagnostics import decompose
from main_experiment.requests import planned


def graph():
    rows=[(str(a),str(b),t) for a,b in itertools.combinations(range(6),2)
          for t in [0.,.199,.2,.399,.4,.47,.6,.8,1.]]
    return canonical('fixture_h',pd.DataFrame(rows,columns=['u','v','t']),horizon=(0.,1.))[0]


class TimeHistoryTests(unittest.TestCase):
    def test_cutoff_is_elapsed_time_not_window_or_event_cap(self):
        g=graph()
        for h in (*H_SENSITIVITY,.53):
            c=history_counts(g,h); start=history_start(g,h)
            direct=np.zeros_like(c)
            for e,w,t in zip(g.pair,g.w,g.t):
                if t>=start: direct[e,w]+=1
            np.testing.assert_array_equal(c,direct)
        self.assertTrue((history_counts(g,.8).sum(1)>5).all())
        self.assertEqual(history_start(g,.6),.4)
        self.assertEqual(int(history_counts(g,.6).sum()),5*g.D)

    def test_time_translation_scaling_preserves_access(self):
        g=graph(); frame=pd.DataFrame({'u':g.u,'v':g.v,'t':100+20*g.t})
        shifted=canonical('shifted',frame,horizon=(100,120))[0]
        for h in H_SENSITIVITY: np.testing.assert_array_equal(history_counts(g,h),history_counts(shifted,h))

    def test_integer_calibration_and_exact_panel_expectation(self):
        g=graph(); T=.1*g.cells
        for h in H_SENSITIVITY:
            b=h_parameters(g,T,h); c=history_counts(g,h); n=b['n_panel_history']
            expectations=[m*(m-1)/(g.N*(g.N-1))*(c>0).sum() for m in range(g.N+1)]
            self.assertEqual(n,min(range(g.N+1),key=lambda m:abs(expectations[m]-T)))
            volumes=[]
            for nodes in itertools.combinations(range(g.N),n):
                mask=np.isin(g.ends[:,0],nodes)&np.isin(g.ends[:,1],nodes)
                volumes.append(int((c[mask]>0).sum()))
            self.assertAlmostEqual(np.mean(volumes),b['h_expected_cells'])

    def test_paired_panels_nested_and_saturated_once(self):
        g=graph(); small=h_parameters(g,.1*g.cells,.8); large=h_parameters(g,.1*g.cells,.4)
        for i in range(1,20):
            a=history_panel_mask(g,i,'test',small); b=history_panel_mask(g,i,'test',large)
            self.assertFalse((a&~b).any())
        saturated=h_parameters(g,10*g.cells,.6)
        self.assertTrue(saturated['h_saturated']); self.assertTrue(saturated['h_target_unreachable'])
        with self.assertRaises(ValueError): draw(g,'H',2,'test',saturated)

    def test_masks_roundtrip_features_and_unobserved_dyads(self):
        g=graph()
        for h in H_SENSITIVITY:
            b=h_parameters(g,.1*g.cells,h); c,_=draw(g,'H',1,'sample',b)
            o=make(g,'H',b,c,None); block=serialize(o); back=parse(block)
            self.assertEqual(block,serialize(back))
            np.testing.assert_array_equal(features(o),features(back))
            n=round(5*h)
            self.assertEqual(len(o['table']),2**n-1)
            self.assertEqual(o['Events_per_window'][:5-n],[None]*(5-n))
            self.assertNotIn('fixture_h',messages(block)[1]['content'])
            with self.assertRaises(ValueError): parse(block.replace('History_fraction=','unexpected='))

    def test_likelihood_fit_against_independent_optimization(self):
        g=graph()
        for h in H_SENSITIVITY:
            n=round(5*h); rng=np.random.default_rng(92)
            counts=np.zeros_like(g.counts)
            counts[:,5-n:]=(rng.random((g.D,n))<.55)
            b=h_parameters(g,g.cells,h); o=make(g,'H',b,counts,None)
            fit=h_extrapolator(o); j=(counts>0).sum(1); j=j[j>0]
            loss=lambda q: -np.sum(binom.logpmf(j,n,q)-math.log1p(-(1-q)**n))
            opt=minimize_scalar(loss,bounds=(1e-8,1-1e-8),method='bounded')
            self.assertAlmostEqual(fit['q'],opt.x,places=5)
            np.testing.assert_allclose(fit['prediction'],binom.sf(np.arange(1,5),5,fit['q'])/(1-(1-fit['q'])**5))

    def test_binomial_boundary_and_empty(self):
        g=graph(); b=h_parameters(g,g.cells,.6)
        for cols,expected in [([4],[0.]*4),([2,3,4],[1.]*4)]:
            c=np.zeros_like(g.counts); c[:,cols]=1
            self.assertEqual(h_extrapolator(make(g,'H',b,c,None))['prediction'],expected)
        with self.assertRaises(ValueError): h_extrapolator(make(g,'H',b,np.zeros_like(g.counts),None))

    def test_disappearance_is_history_not_node_selection(self):
        full=np.array([[1,1,0,0,0],[1,0,1,1,1],[0,0,0,0,1]])
        c=full.copy(); c[:,:2]=0
        d=decompose(full,c,np.ones(3,dtype=bool))
        self.assertEqual(d['node_selection'],[0.]*4)
        self.assertEqual(d['lost_panel_dyads'],1)
        self.assertNotEqual(d['dyad_disappearance'],[0.]*4)
        np.testing.assert_allclose(np.array(d['dyad_disappearance'])+d['within_dyad_history'],d['net_history'])
        self.assertFalse(decompose(full,np.zeros_like(full),np.ones(3,dtype=bool))['defined'])

    def test_final_generation_contract_creates_new_identities_for_all_arms(self):
        import json
        run=ROOT/CURRENT_RUN
        current_requests=[json.loads(line) for line in (run/'requests.jsonl').read_text().splitlines()]
        obs=[read_json(p) for p in (run/'observations/sample').glob('*.json')]
        self.assertEqual(len(obs),280)
        for o in obs:
            self.assertEqual(messages(o['block']),o['messages'])
            if o['arm']!='S':
                self.assertNotIn('Walk_A',o['block'])
                self.assertNotIn('Auxiliary statistics:',o['messages'][1]['content'])
            else:
                self.assertIn('Walk_A=',o['block'])
                self.assertIn('Auxiliary statistics:',o['messages'][1]['content'])
        self.assertEqual(len(current_requests),3360)
        self.assertTrue(all(r['design_version']==DESIGN_VERSION for r in current_requests))

    def test_new_h_shards_and_attempt_resume(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
        from run_qwen_engine import pending,write_result,result_path
        r={'id':'new_H','mode':'thinking','repeat_index':1,'observation_id':'o','graph_id':'g',
           'arm':'H','sample_index':1,'config_id':'qwen_thinking','seed':4,'prompt_sha256':'p','payload_sha256':'b'}
        with tempfile.TemporaryDirectory() as td:
            out=Path(td); self.assertEqual(len(pending(out,[r])[1]),1)
            write_result(out,r,{'status':'completed'})
            self.assertEqual(len(pending(out,[r])[1]),0)
            self.assertTrue(result_path(out,r).exists())
