"""Synthetic training pool definition and the leave-one-source-out weighting."""
import json
import unittest
from collections import Counter
from main_experiment import pool
from main_experiment.common import REAL_TEST, SURROGATES, TRAIN, seed
from main_experiment.training import BLOCK_WEIGHTS, FOLDS, FOREST_SEED, fold_rows


class PoolDefinitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.specs = pool.pool_definition()['graphs']

    def test_counts_keys_and_seeds(self):
        counts = Counter((g['family'], g['partition']) for g in self.specs)
        self.assertEqual(counts, {('dar', 'train'): 200, ('dar', 'dev'): 50, ('ad', 'train'): 200, ('ad', 'dev'): 50})
        self.assertEqual(len({g['key'] for g in self.specs}), 500)
        self.assertEqual(len({g['seed'] for g in self.specs}), 500)
        self.assertTrue(all('pool-v3-panel888-20260921' in g['key'] for g in self.specs))

    def test_streams_are_disjoint_from_the_main_instances(self):
        main = {seed('graph', f'{f}_pair_r{r}') for f in ('dar', 'ad') for r in (1, 2)}
        self.assertFalse(main & {g['seed'] for g in self.specs})

    def test_definition_is_deterministic(self):
        self.assertEqual(json.dumps(self.specs, sort_keys=True), json.dumps(pool.pool_definition()['graphs'], sort_keys=True))

    def test_parameters_stay_in_their_declared_ranges(self):
        for g in self.specs:
            for name, spec in pool.VARIED[g['family']].items():
                value = g['parameters'].get(name)
                if value is None: continue
                if 'range' in spec: self.assertTrue(spec['range'][0]-1e-12 <= value <= spec['range'][1]+1e-12)
                if 'grid' in spec: self.assertIn(value, spec['grid'])
                if 'values' in spec: self.assertIn(value, spec['values'])

    def test_strata_and_size_grids_are_balanced(self):
        for family in ('dar', 'ad'):
            for partition, n in (('train', 200), ('dev', 50)):
                chosen = [g['parameters'] for g in self.specs if g['family'] == family and g['partition'] == partition]
                self.assertEqual(set(Counter(p['N'] for p in chosen).values()), {n//5})
                if family == 'ad':
                    self.assertEqual(set(Counter((p['N'], p['rounds']) for p in chosen).values()), {n//25})
        cells = Counter((g['family'], json.dumps(g['stratum'], sort_keys=True), g['partition']) for g in self.specs)
        self.assertEqual(len(cells), 2*(25+10))            # every stratum cell in both partitions


class FoldTests(unittest.TestCase):
    def rows(self):
        real = [{'id': f'{s}__{a}__s{i}', 'source_family': s, 'arm': a, 'block_group': 'real'}
                for s in TRAIN for a in 'RSHB' for i in range(1, 6)]
        synthetic = [{'id': f'{family}{k}__{a}__s{i}', 'source_family': f'{family}{k}', 'arm': a, 'block_group': family}
                     for family in ('dar', 'ad') for k in range(200) for a in 'RSHB' for i in range(1, 6)]
        return real, synthetic

    def test_the_held_out_source_is_removed_completely(self):
        real, synthetic = self.rows()
        for fold in REAL_TEST:
            selected, _ = fold_rows(real, fold, synthetic)
            self.assertNotIn(fold, {r['source_family'] for r in selected})
        self.assertEqual({r['source_family'] for r in fold_rows(real, 'synthetic')[0]}, set(TRAIN))
        self.assertEqual(FOLDS, (*REAL_TEST, 'synthetic'))
        self.assertFalse(set(SURROGATES) & set(TRAIN))

    def test_block_graph_arm_and_observation_weights(self):
        real, synthetic = self.rows()
        selected, w = fold_rows(real, 'sp_hospital', synthetic)
        per_block = Counter(); per_graph = Counter(); per_arm = Counter()
        for r, x in zip(selected, w):
            per_block[r['block_group']] += x; per_graph[r['source_family']] += x
            if r['source_family'] == 'sp_workplace': per_arm[r['arm']] += x
        for block, share in BLOCK_WEIGHTS.items(): self.assertAlmostEqual(per_block[block], share, places=12)
        self.assertAlmostEqual(per_graph['sp_workplace'], .5/15, places=12)
        self.assertAlmostEqual(min(per_arm.values()), max(per_arm.values()), places=12)
        # Few real rows, but half of the weight.
        self.assertLess(sum(r['block_group'] == 'real' for r in selected), len(selected)/10)

    def test_real_only_fold_and_saturated_h(self):
        real, _ = self.rows()
        real = [r for r in real if not (r['source_family'] == 'sp_workplace' and r['arm'] == 'H' and r['id'][-1] != '1')]
        selected, w = fold_rows(real, 'sp_hospital')
        self.assertAlmostEqual(w.sum(), 1., places=12)
        h = [x for r, x in zip(selected, w) if r['source_family'] == 'sp_workplace' and r['arm'] == 'H']
        self.assertEqual(len(h), 1)
        self.assertAlmostEqual(h[0], 1/(15*4), places=12)     # one deterministic draw carries the whole H weight

    def test_versioned_forest_seed(self):
        self.assertEqual(FOREST_SEED, 1858608657)


if __name__ == '__main__':
    unittest.main()
