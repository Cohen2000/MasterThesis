"""Pool definition, partition independence, the 133-feature block and the
50/25/25 training weights."""
import json
import math
import pathlib
import unittest
import numpy as np
from main_experiment import pool as poolmod
from main_experiment.common import TRAIN, REAL_TEST, seed
from main_experiment.observation import (FEATURE_NAMES, BASE_FEATURE_NAMES,
                                         DERIVED_FEATURE_NAMES, features, parse, validate)
from main_experiment.synthetic import generate_pair
from main_experiment.training import fold_rows, BLOCK_WEIGHTS

FROZEN=pathlib.Path('results/main_experiment/frozen_20260916')
# Observation blocks follow the current design; the superseded suffix design is
# kept under frozen_20260916 and is not read for contract checks any more.
RUN=pathlib.Path('results/main_experiment/hrecent5_20260917')


class PoolDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.d=poolmod.pool_definition()

    def test_counts_and_partitions(self):
        c={}
        for g in self.d['graphs']: c[(g['family'],g['partition'])]=c.get((g['family'],g['partition']),0)+1
        self.assertEqual(c,{('dar','train'):200,('dar','dev'):50,
                            ('ad','train'):200,('ad','dev'):50})
        self.assertEqual(self.d['n_graphs'],500)

    def test_keys_and_seeds_unique(self):
        keys=[g['key'] for g in self.d['graphs']]
        self.assertEqual(len(set(keys)),len(keys))
        seeds=[g['seed'] for g in self.d['graphs']]
        self.assertEqual(len(set(seeds)),len(seeds))

    def test_train_and_dev_streams_are_disjoint(self):
        tr={g['seed'] for g in self.d['graphs'] if g['partition']=='train'}
        dv={g['seed'] for g in self.d['graphs'] if g['partition']=='dev'}
        self.assertFalse(tr&dv)

    def test_pool_streams_disjoint_from_main_tests(self):
        """No pool graph may share a stream with a synthetic main-test pair."""
        main={seed('graph',f'{f}_pair_r{r}') for f in ('dar','ad') for r in (1,2)}
        pool={g['seed'] for g in self.d['graphs']}
        self.assertFalse(main&pool)

    def test_definition_is_deterministic(self):
        again=poolmod.pool_definition()
        self.assertEqual(json.dumps(self.d['graphs'],sort_keys=True),
                         json.dumps(again['graphs'],sort_keys=True))

    def test_parameters_stay_inside_the_declared_ranges(self):
        v=poolmod.VARIED
        for g in self.d['graphs']:
            p=g['parameters']; spec=v[g['family']]
            for name,s in spec.items():
                if name not in p or p[name] is None: continue
                if 'range' in s: self.assertTrue(s['range'][0]-1e-12<=p[name]<=s['range'][1]+1e-12,(g['key'],name))
                if 'grid' in s: self.assertIn(p[name],s['grid'],(g['key'],name))
                if 'values' in s: self.assertIn(p[name],s['values'])
            if g['family']=='dar':
                self.assertLessEqual(p['E'],p['N']*(p['N']-1)//2)

    def test_every_stratum_cell_is_filled(self):
        cells={}
        for g in self.d['graphs']:
            cells.setdefault((g['family'],json.dumps(g['stratum'],sort_keys=True)),[]).append(g['partition'])
        self.assertEqual(len([k for k in cells if k[0]=='dar']),25)
        self.assertEqual(len([k for k in cells if k[0]=='ad']),10)
        for k,v in cells.items():
            self.assertGreater(v.count('train'),0,k)
            self.assertGreater(v.count('dev'),0,k)


class FrozenGeneratorTests(unittest.TestCase):
    def test_main_synthetic_instances_are_unchanged(self):
        """Parameterising the generators must not move the eight frozen instances."""
        if not (FROZEN/'graphs').exists(): self.skipTest('frozen run not present')
        n=0
        for fam in ('dar','ad'):
            for rep in (1,2):
                for g,x,meta in generate_pair(fam,rep):
                    m=json.loads((FROZEN/'graphs'/g.key/'manifest.json').read_text())
                    self.assertEqual((m['N_full'],m['D_full'],m['M_full']),(g.N,g.D,g.M),g.key)
                    # m['B'] is the superseded suffix budget of the previous
                    # design; it must still equal this graph's suffix volume.
                    self.assertEqual(m['B'],g.M_suffix,g.key)
                    self.assertEqual(m['truth'],list(g.truth),g.key)
                    self.assertEqual(m.get('shared_latents'),meta.get('shared_latents'),g.key)
                    self.assertEqual(m.get('states_sha256'),meta.get('states_sha256'),g.key)
                    n+=1
        self.assertEqual(n,8)


class FeatureTests(unittest.TestCase):
    def test_counts(self):
        # 88 original base entries plus 31 at-cap counts; 45 original derived
        # entries plus 31 at-cap shares.
        self.assertEqual(len(BASE_FEATURE_NAMES),119)
        self.assertEqual(len(DERIVED_FEATURE_NAMES),76)
        self.assertEqual(len(FEATURE_NAMES),195)
        self.assertEqual(len(set(FEATURE_NAMES)),195)

    def test_base_block_structure(self):
        self.assertEqual(BASE_FEATURE_NAMES[:3],['N_obs','D_obs','M_obs'])
        # One parameter slot per arm; H carries its dyad-sample size.
        self.assertEqual(BASE_FEATURE_NAMES[-4:],['n_panel','L','n_dyads','p'])
        self.assertNotIn('n_panel_suffix',FEATURE_NAMES)

    def test_derived_block_is_scale_free_and_consistent(self):
        if not (RUN/'observations').exists(): self.skipTest('current run not present')
        f=sorted((RUN/'observations'/'sample').glob('*__B__*.json'))[0]
        o=parse(json.loads(f.read_text())['block'])
        v=features(o); names=list(FEATURE_NAMES)
        share=[v[names.index(f'share_{p:05b}_dyads')] for p in range(1,32)]
        self.assertAlmostEqual(sum(share),1.,places=12)
        win=[v[names.index(f'share_window_{i}')] for i in range(1,6)]
        self.assertAlmostEqual(sum(win),1.,places=12)
        self.assertAlmostEqual(v[names.index('events_per_dyad')],o['M_obs']/o['D_obs'],places=9)
        from main_experiment.baselines import plugin
        for k,x in zip(range(2,6),plugin(o)):
            self.assertAlmostEqual(v[names.index(f'plugin_rho_{k}')],x,places=12)

    def test_empty_sample_codes_derived_features_as_zero(self):
        o={'arm':'B','N_obs':0,'D_obs':0,'M_obs':0,'Temporal_access':[1]*5,
           'Events_per_window':[0]*5,'Walk_A':None,'parameter':.5,
           'table':[(f'{p:05b}',0,0) for p in range(1,32)]}
        validate(o)
        v=features(o)
        self.assertEqual(len(v),195)
        self.assertTrue(np.isfinite(v).all())
        self.assertTrue((v[len(BASE_FEATURE_NAMES):]==0).all())

    def test_h_features_carry_the_cap_column_and_no_suffix_correction(self):
        """The at-cap column enters as counts and shares, and the corrector slots of
        arm H hold the bound midpoint, not the three-to-five-window extrapolation."""
        if not (RUN/'observations').exists(): self.skipTest('current run not present')
        from main_experiment.baselines import h_midpoint, activity, profile
        f=sorted((RUN/'observations'/'sample').glob('*__H-recent5__*.json'))[0]
        o=parse(json.loads(f.read_text())['block'])
        v=features(o); names=list(FEATURE_NAMES)
        caps=[r[3] for r in o['table']]
        self.assertEqual([v[names.index(f'{p:05b}_at_cap')] for p in range(1,32)],caps)
        for p,c in zip(range(1,32),caps):
            self.assertAlmostEqual(v[names.index(f'share_{p:05b}_at_cap')],c/o['D_obs'],places=12)
        self.assertEqual([v[names.index(f'corrector_rho_{k}')] for k in range(2,6)],h_midpoint(o))
        self.assertEqual([v[names.index(f'access_{i}')] for i in range(1,6)],[1.]*5)
        g=sorted((RUN/'observations'/'sample').glob('*__B__*.json'))[0]
        b=features(parse(json.loads(g.read_text())['block']))
        self.assertTrue(all(b[names.index(f'{p:05b}_at_cap')]==0 for p in range(1,32)))

    def test_no_forbidden_quantity_is_named(self):
        banned=('N_full','D_full','M_full','truth','rho_true','coverage','source','family',
                'alpha','chi','eta','gamma','generator')
        for n in FEATURE_NAMES:
            for b in banned:
                self.assertNotIn(b,n,f'{n} looks like a forbidden feature')


class WeightingTests(unittest.TestCase):
    def _rows(self):
        real=[{'id':f'{s}__{a}__s{i}','source_family':s,'arm':a,'sample_index':i,'block':''}
              for s in TRAIN for a,n in (('R',5),('S',5),('H',5),('B',5)) for i in range(1,n+1)]
        pool=[{'id':f'{g}__{a}__s{i}','source_family':g,'arm':a,'sample_index':i,'block':'',
               'block_group':'dar' if g.startswith('d') else 'ad'}
              for g in [f'd{k}' for k in range(200)]+[f'a{k}' for k in range(200)]
              for a,n in (('R',5),('S',5),('H',5),('B',5)) for i in range(1,n+1)]
        return real,pool

    def test_block_shares_are_exact(self):
        real,pool=self._rows()
        sel,w=fold_rows(real,'sp_hospital',pool)
        got={}
        for x,r in zip(w,sel): got[r.get('block_group','real')]=got.get(r.get('block_group','real'),0)+x
        for k,v in BLOCK_WEIGHTS.items(): self.assertAlmostEqual(got[k],v,places=12)
        self.assertAlmostEqual(w.sum(),1.,places=12)

    def test_held_out_source_and_all_its_rows_are_removed(self):
        real,pool=self._rows()
        for test in REAL_TEST:
            sel,_=fold_rows(real,test,pool)
            self.assertFalse(any(r['source_family']==test for r in sel),test)

    def test_equal_weight_per_graph_arm_and_observation(self):
        real,pool=self._rows()
        sel,w=fold_rows(real,'sp_hospital',pool)
        per={}
        for x,r in zip(w,sel): per[r['source_family']]=per.get(r['source_family'],0)+x
        reals=[v for k,v in per.items() if k in TRAIN]
        dars=[v for k,v in per.items() if k.startswith('d')]
        self.assertAlmostEqual(min(reals),max(reals),places=12)
        self.assertAlmostEqual(min(dars),max(dars),places=12)
        self.assertAlmostEqual(reals[0],BLOCK_WEIGHTS['real']/15,places=12)
        arms={}
        for x,r in zip(w,sel):
            if r['source_family']=='sp_workplace': arms[r['arm']]=arms.get(r['arm'],0)+x
        self.assertAlmostEqual(min(arms.values()),max(arms.values()),places=12)

    def test_real_only_fold_keeps_the_original_scale(self):
        real,_=self._rows()
        sel,w=fold_rows(real,'sp_hospital')
        self.assertAlmostEqual(w.sum(),1.,places=12)
        self.assertAlmostEqual(float(w[0]),1/(15*4*5),places=12)

    def test_synthetic_pool_cannot_outvote_the_real_block_by_volume(self):
        real,pool=self._rows()
        sel,w=fold_rows(real,'sp_hospital',pool)
        n_real=sum(1 for r in sel if r.get('block_group','real')=='real')
        self.assertLess(n_real,len(sel)/10)          # far fewer rows
        share=sum(x for x,r in zip(w,sel) if r.get('block_group','real')=='real')
        self.assertAlmostEqual(share,.5,places=12)   # but half the weight


if __name__=='__main__': unittest.main()
