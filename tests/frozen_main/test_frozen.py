import itertools
import json
import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from main_experiment.common import seed,TRAIN,REAL_TEST,LEGACY_H
from main_experiment.data import canonical
from main_experiment.sampling import Walk,budget_parameters,draw,calibrate
from main_experiment.observation import FEATURE_NAMES, make,serialize,parse,features,messages,validate
from main_experiment.baselines import activity,profile,corrector,plugin,all_baselines
from main_experiment.evaluation import parse_final,resolve,errors,paired_summary
from main_experiment.training import fold_rows
from main_experiment.requests import retry_decision,reserve_allowed,payload,watchdog


def fixture(rows=None,proximity=False):
    return canonical('fixture',pd.DataFrame(rows or [
        ('a','b',0.),('b','a',.2),('a','b',.4),('a','c',.6),('a','c',.8),('b','c',1.)],
        columns=['u','v','t']),proximity,horizon=(0,1))[0]


def reference(w,s,L):
    mask=(1<<64)-1
    def next_int():
        nonlocal s
        s=(s+0x9e3779b97f4a7c15)&mask; z=s
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z=((z^(z>>27))*0x94d049bb133111eb)&mask
        return z^(z>>31)
    def bounded(n):
        threshold=((1<<64)-n)%n
        while True:
            x=next_int()
            if x>=threshold: return x%n
    node=bounded(w.g.N); re=np.zeros(w.g.D,dtype=np.int64); volumes=[0]
    for _ in range(L):
        a,b=w.ptr[node:node+2]; ticket=bounded(int(w.cum[b-1])); j=a
        while w.cum[j]<=ticket: j+=1
        e=w.edges[j]; re[e]+=1; node=w.neighbors[j]
        volumes.append(int(w.g.m[re>0].sum()))
    return re,volumes

class FrozenTests(unittest.TestCase):
    def test_boundaries_and_duplicates(self):
        g=fixture(); self.assertEqual(g.counts.sum(0).tolist(),[1,1,1,1,2])
        # No epsilon: just below a boundary must stay in previous window.
        rows=[('a','b',0.),('a','b',np.nextafter(.4,0)),('b','a',.4),('a','b',.4),('a','b',1.)]
        a=fixture(rows); b=fixture(rows,True)
        self.assertEqual(a.M,5); self.assertEqual(b.M,4)
        self.assertEqual(a.counts.sum(0).tolist(),[1,1,2,0,1])
        self.assertEqual(b.D,1)
        self.assertEqual(b.horizon,(0.,1.))
        with self.assertRaises(ValueError): fixture([('a','b',.1)])

    def test_invalid_B(self):
        with self.assertRaises(ValueError): fixture([('a','b',0.),('a','b',.1)])

    def test_walk_optimized_vs_reference(self):
        g=fixture()
        with tempfile.TemporaryDirectory() as d:
            w=Walk(g,d)
            for s in [0,1,42,2**63-1]:
                re,volumes=reference(w,s,100)
                prefix,v,r,executed=w.run([s],100,True)
                np.testing.assert_array_equal(prefix,volumes)
                np.testing.assert_array_equal(r[0],re)
                self.assertEqual(executed[0],100)
                p,v,_,e=w.run([s],100,False)
                np.testing.assert_array_equal(p,volumes)
                self.assertLess(e[0],100)
            # Full histories are deduplicated, all repeated traversals enter A.
            b=budget_parameters(g)|{'L':100}
            counts,re=draw(g,'S',1,'sample',b,w); obs=make(g,'S',b,counts,re)
            self.assertEqual(re.sum(),100); self.assertEqual(obs['M_obs'],g.M)
            A=obs['Walk_A']; expected=[sum(A[k-1:])/sum(A) for k in range(2,6)]
            np.testing.assert_array_equal(corrector(obs),expected)

    def test_component_cap_and_validation_freeze(self):
        # Twelve disjoint dyads of three events each: M=36, so the ten-percent
        # budget is 4, while no start component can ever yield more than 3.
        rows=[]
        for i in range(12):
            a,b=f'n{2*i}',f'n{2*i+1}'
            rows += [(a,b,.5),(a,b,.7),(a,b,.9)]
        g=fixture(rows)
        with tempfile.TemporaryDirectory() as d:
            b,w=calibrate(g,Path(d)/'cal',Path(d)/'build')
            self.assertEqual(b['L'],b['C']); self.assertFalse(b['budget_matched'])
            self.assertIn('calibration_cap',b['walk_unmatched_reasons'])
            self.assertIn('S:calibration_cap',b['unmatched_reasons'])
            self.assertFalse(b['walk_budget_matched']); self.assertFalse(b['budget_matched_by_arm']['S'])
            b2,_=calibrate(g,Path(d)/'cal',Path(d)/'build')
            self.assertEqual(b2['L'],b['L'])
            self.assertEqual(b2['validation_mean'],3.)

    def test_sampling_exact_expectations(self):
        g=fixture(); b=budget_parameters(g); n=b['n_panel']
        cells=[]; events=[]
        for panel in itertools.combinations(range(g.N),n):
            selected=np.isin(g.ends[:,0],panel)&np.isin(g.ends[:,1],panel)
            cells.append(int((g.counts[selected]>0).sum())); events.append(g.m[selected].sum())
        self.assertAlmostEqual(np.mean(cells),b['node_expected_cells'])
        self.assertAlmostEqual(np.mean(events),b['node_expected_events'])
        # Exhaustive Bernoulli outcomes of the six-event fixture: expected observed
        # active dyad-windows at the solved p equal the target exactly.
        p=b['p']; expected=0.
        for bits in itertools.product([0,1],repeat=g.M):
            kept=np.array(bits,bool)
            c=np.bincount(g.pair[kept]*5+g.w[kept],minlength=g.D*5)
            expected+=int((c>0).sum())*p**kept.sum()*(1-p)**(g.M-kept.sum())
        self.assertAlmostEqual(expected,b['T'])
        # Fixed seeds, conservative six-standard-error test of production draws.
        volumes=[draw(g,'B',i,'sample',b)[0].sum() for i in range(1,4001)]
        se=math.sqrt(g.M*p*(1-p)/4000)
        self.assertLess(abs(np.mean(volumes)-p*g.M),6*se)

    def test_serialization_features_and_masks(self):
        g=fixture(); b=budget_parameters(g)|{'L':7}
        with tempfile.TemporaryDirectory() as d:
            w=Walk(g,d)
            for arm in ['R','S','H','B',LEGACY_H]:
                c,r=draw(g,arm,1,'sample',b,w); o=make(g,arm,b,c,r)
                block=serialize(o); restored=parse(block)
                self.assertEqual(serialize(restored),block)
                self.assertEqual(restored['arm'],arm)
                self.assertEqual(len(o['table']),7 if arm==LEGACY_H else 31)
                self.assertEqual({len(row) for row in o['table']},{4} if arm=='H' else {3})
                text=messages(block)[1]['content']
                self.assertNotIn('fixture',text); self.assertNotIn('budget_matched',text)
                if arm==LEGACY_H:
                    # Development variant only: no features in the current design.
                    with self.assertRaises(ValueError): features(o)
                    self.assertEqual(restored['Events_per_window'][:2],[None,None])
                    o['Events_per_window'][0]=0
                    with self.assertRaises(ValueError): validate(o)
                    continue
                np.testing.assert_array_equal(features(o),features(restored))
                self.assertEqual(len(features(o)),len(FEATURE_NAMES))
                if arm=='H':
                    self.assertIn('pattern,dyads,events,at_cap_dyads',block)
                    self.assertEqual(restored['Temporal_access'],[1]*5)
                    self.assertNotIn(None,restored['Events_per_window'])

    def test_corrector_edges(self):
        self.assertEqual(profile(0),[0.]*4); self.assertEqual(profile(1),[1.]*4)
        for n in [3,5]:
            self.assertEqual(activity(1,n),0); self.assertEqual(activity(n,n),1)
        g=fixture(); b=budget_parameters(g)
        for arm in ('H',LEGACY_H):
            c,r=draw(g,arm,1,'sample',b); o=make(g,arm,b,c,r)
            self.assertTrue(all(0<=x<=1 for x in corrector(o)))
        # q=1 boundary: visible all five windows but only one event per window.
        g=fixture([('a','b',t) for t in [0,.2,.4,.6,1.]])
        o=make(g,'B',{'p':.1},g.counts,None)
        self.assertEqual(corrector(o),[1.]*4)
        # r=0,d=p; theta=0, hence q=0.
        c=np.zeros_like(g.counts); c[0,0]=1
        o=make(g,'B',{'p':.2},c,None)
        self.assertEqual(corrector(o),[0.]*4)

    def test_hurdle_against_direct_optimization(self):
        # Compare profile values against constrained 2D conditional likelihood,
        # not against a second implementation of the scalar equations.
        g=fixture(); b=budget_parameters(g)
        for D,S,M,p in [(20,37,80,.4),(20,90,95,.2),(50,75,250,.8),(30,50,52,.7)]:
            base=[1]*D
            for j in range(S-D): base[j%D]+=1
            counts=np.zeros((D,5),dtype=np.int64)
            for i,k in enumerate(base): counts[i,:k]=1
            counts[0,0]+=M-S
            from main_experiment.observation import validate
            table=[]
            for pattern in range(1,32):
                matched=(counts>0)@np.array([16,8,4,2,1])==pattern
                table.append((f'{pattern:05b}',int(matched.sum()),int(counts[matched].sum())))
            o={'arm':'B','N_obs':D+1,'D_obs':D,'M_obs':M,'Temporal_access':[1]*5,
               'Events_per_window':counts.sum(0).tolist(),'Walk_A':None,'parameter':p,'table':table}
            answer=corrector(o)
            def nll(z):
                q,r=z; d=-np.expm1(-r)/-np.expm1(-r/p); th=q*d
                if not 0<th<1: return 1e100
                return -(S*np.log(th)+(5*D-S)*np.log1p(-th)-D*np.log(-np.expm1(5*np.log1p(-th)))
                         +M*np.log(r)-S*np.log(np.expm1(r)))
            fits=[minimize(nll,[q,r],bounds=[(1e-8,1.),(1e-8,20)],method='Nelder-Mead',
                           options={'xatol':1e-10,'fatol':1e-10,'maxiter':5000})
                  for q,r in [(.2,.5),(.8,2.),(.99,.1)]]
            best=min(fits,key=lambda x:x.fun)
            np.testing.assert_allclose(answer,profile(best.x[0]),atol=2e-6)

    def test_parser_and_replacements(self):
        good=' {"rho_5":0,"rho_2":0.5,"rho_4":0.1,"rho_3":0.2}\n'
        self.assertEqual(parse_final(good)[0],[.5,.2,.1,0])
        # A whole-answer markdown fence is a transport wrapper and is accepted
        # since the smoke test; it is reported separately as valid_after_fence so
        # the share can be stated. Everything inside stays strictly validated.
        self.assertEqual(parse_final('```json\n'+good+'```')[0],[.5,.2,.1,0])
        self.assertEqual(parse_final('```json\n'+good+'```')[1],'valid_after_fence')
        invalid=['{}','[]','prose\n```json\n'+good+'```',good+' {}',
                 '{"rho_2":.2}',good.replace('0.5','true'),good.replace('0.5','"0.5"'),
                 good.replace('0.5','null'),good.replace('0.5','NaN'),good.replace('0.5','Infinity'),
                 good.replace('0.5','1.1'),good.replace('0.5','-0.1'),good.replace('0.5','0.01'),
                 good.replace('"rho_5":0','"rho_5":0,"rho_5":0'),good[:-3]]
        for text in invalid: self.assertIsNone(parse_final(text)[0],text)
        g=fixture(); b=budget_parameters(g); c,r=draw(g,'H',1,'sample',b); o=make(g,'H',b,c,r)
        self.assertIsNone(resolve(o,[.4]*4)['prediction'])
        result=resolve(o,[.4]*4,{'started':True,'terminal':True,'limit_hit':True,'final_text':good})
        self.assertTrue(result['valid']); self.assertTrue(result['limit_hit'])
        bad=resolve(o,[.4]*4,{'started':True,'terminal':True,'final_text':'oops'})
        self.assertEqual(bad['prediction'],plugin(o))
        empty=make(g,'B',{'p':.5},np.zeros_like(g.counts),None)
        self.assertEqual(resolve(empty,[.4]*4)['prediction'],[.4]*4)
        self.assertTrue(all(x['prediction']==[.4]*4 for x in all_baselines(empty,[.4]*4,[0]*4).values()))

    def test_folds_weights_and_mcse(self):
        rows=[{'source_family':s,'arm':a} for s in TRAIN for a in 'RSHB' for i in range(1 if a=='H' else 5)]
        for source in REAL_TEST:
            selected,w=fold_rows(rows,source)
            self.assertEqual(len(selected),240); self.assertAlmostEqual(sum(w),1)
            self.assertNotIn(source,{r['source_family'] for r in selected})
            for s in set(TRAIN)-{source}:
                for a in 'RSHB': self.assertAlmostEqual(sum(v for r,v in zip(selected,w) if r['source_family']==s and r['arm']==a),1/60)
        cells={str(i):np.tile(np.arange(5)[:,None]/10,(1,3)) for i in range(6)}
        x=paired_summary(cells)
        self.assertAlmostEqual(x['mean'],.2)
        self.assertAlmostEqual(x['mcse'],math.sqrt(np.var(np.arange(5)/10,ddof=1)/5/6))
        self.assertEqual(errors([0]*4,[.5]*4)['AE2'],.5)
        # Error first differs from ensembling predictions.
        self.assertEqual(np.mean([errors([0]*4,[.5]*4)['AE2'],errors([1]*4,[.5]*4)['AE2']]),.5)

    def test_transport_policy_mocks_only(self):
        self.assertEqual(retry_decision(1,0,429,True)['delay'],5)
        self.assertEqual(retry_decision(2,0,503,True,retry_after=40)['delay'],40)
        self.assertEqual(retry_decision(3,0,503,True)['action'],'terminal_no_retry')
        self.assertEqual(retry_decision(1,1,503,True)['action'],'terminal_no_retry')
        self.assertEqual(retry_decision(1,0,503,True,True)['action'],'reconcile')
        self.assertEqual(retry_decision(1,0,401)['action'],'stop_configuration')
        self.assertFalse(reserve_allowed('sol',179,0,1))
        self.assertTrue(reserve_allowed('sol',179,0,1,True))
        self.assertFalse(reserve_allowed('deepseek',49.9,0,1))
        self.assertEqual(watchdog(1800,1800,0,0),'first_token_deadline')
        for config in ['sol','deepseek','qwen_thinking','qwen_nonthinking']:
            p=payload(config,[],123)
            self.assertNotIn('tools',p)
            if config in ['sol','deepseek']: self.assertNotIn('seed',p)

if __name__=='__main__': unittest.main()
