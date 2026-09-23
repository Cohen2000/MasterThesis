"""Small offline checks for the frozen API inputs and execution guard."""
import hashlib
import os
from pathlib import Path
import unittest

from main_experiment.common import MAIN_KEYS
from main_experiment.evaluation import parse_final
from scripts.api_runner import API_MAIN_ARMS, guard, manifest, observations, payload

ROOT = Path(__file__).resolve().parents[1]


class APIPreparation(unittest.TestCase):
    def test_sealed_scientific_files(self):
        expected = {
            'docs/results/panel888_v10_main_20260923/REPORT.json': 'b6ee576e2790a9904ba389d2ff12e90afc6d08a34305f56e00b5e54d90a30a73',
            'docs/results/panel888_v10_walk_gate_20260923/WALK_GATE.md': '803b0c522a736c22388734420f6a001268a1e917646546801ed3bf63ab486dad',
            'docs/results/panel888_v10_main_20260923/QWEN_ARCHIVE_VERIFICATION.json': '86ae5625471d4177e22a59b4f6d2b4c6799d10879ac6c210e680156bfcd53a2b',
            'config/main_experiment/system.txt': '402ac97f8c3b0cf6b48d18ebf0a20eb1b39dcc314042801883e48a0a973a867d',
            'config/main_experiment/user_prefix.txt': '0f9a7e95378a126a643108bd5fabdc350ebad02e2fbc983e73eecf4a936558f7',
        }
        for name, digest in expected.items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_plan_counts_scope_and_guard(self):
        rows = [{'id': f'{g}__{arm}__s{i}', 'graph_id': g,
                 'stratum': 'surrogate' if g.endswith('__pwt') else 'synthetic' if g.startswith(('dar_', 'ad_')) else 'real',
                 'arm': arm, 'sample_index': i, 'prompt_sha256': 'frozen',
                 'messages': [{'role': 'system', 'content': 'system'}, {'role': 'user', 'content': 'user'}]}
                for g in MAIN_KEYS for arm in (*API_MAIN_ARMS, 'S_obs') for i in (1, 2, 3)]
        for provider, count in (('deepseek', 288), ('openai', 864)):
            planned = manifest(rows, provider)
            self.assertEqual(len(planned), count)
            self.assertNotIn('S_obs', {r['arm'] for r in planned})
            self.assertEqual(len({r['id'] for r in planned}), count)
            self.assertTrue(all('secret' not in str(payload(provider, r)).lower() for r in planned))
            with self.assertRaisesRegex(ValueError, 'budget'):
                guard(planned, provider, None, False)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                guard(planned[:-1] + planned[:1], provider, 1000, False)

    def test_parser_is_strict(self):
        valid = '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'
        self.assertEqual(parse_final(valid)[0], [0.8, 0.6, 0.4, 0.2])
        for bad in (valid.replace('0.6', '0.9'), valid.replace('0.8', '1.1'),
                    valid.replace('0.4', 'NaN'), valid.replace('"rho_5":0.2', '"extra":0.2'),
                    valid.replace('"rho_2":0.8', '"rho_2":0.8,"rho_2":0.8')):
            self.assertIsNone(parse_final(bad)[0])

    def test_actual_frozen_prompts_when_available(self):
        path = os.environ.get('V10_OBSERVATIONS')
        if not path:
            self.skipTest('set V10_OBSERVATIONS to the sealed sample directory')
        rows = observations(Path(path))
        self.assertEqual(len(manifest(rows, 'deepseek')), 288)
        self.assertEqual(len(manifest(rows, 'openai')), 864)


if __name__ == '__main__':
    unittest.main()
