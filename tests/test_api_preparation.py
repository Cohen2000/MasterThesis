"""Small offline checks for the frozen API inputs and execution guard."""
import hashlib
from contextlib import redirect_stdout
from datetime import datetime, timezone
from io import StringIO
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from main_experiment.common import MAIN_KEYS
from main_experiment.evaluation import parse_final
from scripts.api_runner import (API_MAIN_ARMS, deepseek_progress, deepseek_window,
                                execute_deepseek, guard, main, manifest, observations,
                                openai_record, payload, require_deepseek_offpeak,
                                smoke_observation)

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
        self.assertEqual(guard(manifest(rows, 'deepseek'), 'deepseek', 10, False)['safety_margin'], 1.25)
        self.assertEqual(guard(manifest(rows, 'openai'), 'openai', 200, False)['safety_margin'], 1.25)
        with self.assertRaisesRegex(ValueError, 'approved USD 200'):
            guard(manifest(rows, 'openai'), 'openai', 201, False)
        self.assertEqual(payload('openai', manifest(rows, 'openai')[0])['reasoning']['effort'], 'high')
        self.assertEqual(payload('openai', manifest(rows, 'openai')[0])['reasoning']['summary'], 'auto')
        self.assertEqual(payload('openai', manifest(rows, 'openai')[0])['model'], 'gpt-6-sol')
        self.assertEqual(payload('deepseek', manifest(rows, 'deepseek')[0])['reasoning_effort'], 'high')
        self.assertEqual(payload('deepseek', manifest(rows, 'deepseek')[0])['model'], 'deepseek-flash')

    def test_utc_offpeak_windows_and_buffer(self):
        def window(day, hour, minute=0):
            return deepseek_window(datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc))
        # 2026-09-28 is Monday; 2026-09-26 is Saturday.
        for hour, expected in ((0, True), (2, False), (5, True), (8, False), (12, True)):
            self.assertEqual(window(28, hour, 30 if hour == 0 else 0)[0], expected)
        self.assertTrue(window(26, 8)[0])
        self.assertTrue(window(28, 0, 49)[0])
        self.assertFalse(window(28, 0, 50)[0])
        self.assertFalse(window(28, 0, 55)[0])
        with self.assertRaisesRegex(ValueError, 'peak-price window'):
            require_deepseek_offpeak(datetime(2026, 9, 28, 2, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, 'pre-peak buffer'):
            require_deepseek_offpeak(datetime(2026, 9, 28, 0, 55, tzinfo=timezone.utc))
        with self.assertRaisesRegex(ValueError, 'timezone-aware'):
            deepseek_window(datetime(2026, 9, 28, 12))

    def test_resumption_and_duplicate_ids(self):
        rows = [{'id': x, 'prompt_sha256': x, 'messages': [{'content': 'small'}]} for x in ('one', 'two')]
        record = {'id': 'one', 'prompt_sha256': 'one', 'usage': {'prompt_tokens': 12, 'completion_tokens': 4}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'responses.jsonl').write_text(json.dumps(record) + '\n')
            (path / 'attempts.jsonl').write_text('{"id":"one"}\n')
            completed, spent, remaining, smoke = deepseek_progress(rows, path)
            self.assertEqual(set(completed), {'one'})
            self.assertEqual([r['id'] for r in remaining], ['two'])
            self.assertGreater(spent, 0)
            self.assertEqual(smoke, [])
            (path / 'responses.jsonl').write_text(json.dumps(record) + '\n' + json.dumps(record) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate completed'):
                deepseek_progress(rows, path)
            (path / 'responses.jsonl').write_text(json.dumps(record) + '\n')
            (path / 'attempts.jsonl').write_text('{"id":"one"}\n{"id":"two"}\n')
            with self.assertRaisesRegex(ValueError, 'uncertain outcomes'):
                deepseek_progress(rows, path)

    def test_deepseek_smoke_and_production_share_durable_budget(self):
        rows = [{'id': x, 'prompt_sha256': x, 'messages': [{'role': 'user', 'content': 'small'}]}
                for x in ('one', 'two')]
        smoke = {'id': 'training__deepseek__smoke', 'prompt_sha256': 'training',
                 'messages': [{'role': 'user', 'content': 'small'}], 'kind': 'smoke'}
        answer = {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 12, 'completion_tokens': 4},
                  'choices': [{'message': {'content': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}',
                                           'reasoning_content': 'Provider raw reasoning.'},
                               'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'run'
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.require_deepseek_offpeak'), \
                 patch('scripts.api_runner.request', return_value=answer) as provider, \
                 redirect_stdout(StringIO()):
                with self.assertRaisesRegex(ValueError, 'completed training smoke'):
                    execute_deepseek(rows, path, 2)
                execute_deepseek(rows, path, 1, smoke)
                execute_deepseek(rows, path, 2)
                execute_deepseek(rows, path, 2)
            self.assertEqual(provider.call_count, 3)
            records = [json.loads(line) for line in (path / 'responses.jsonl').read_text().splitlines()]
            self.assertEqual({r['id'] for r in records}, {'one', 'two', smoke['id']})
            self.assertTrue(all(r['reasoning_content'] == 'Provider raw reasoning.' for r in records))
            self.assertTrue(all(r['raw_response'] == answer for r in records))
            self.assertEqual(len((path / 'attempts.jsonl').read_text().splitlines()), 3)
            self.assertGreater(deepseek_progress(rows, path)[1], 0)  # includes smoke usage

    def test_smoke_is_training_only(self):
        path = os.environ.get('V10_SMOKE_OBSERVATION')
        if not path:
            self.skipTest('set V10_SMOKE_OBSERVATION to one sealed training row')
        row = smoke_observation(Path(path), 'deepseek')
        self.assertTrue(row['id'].endswith('__deepseek__smoke'))
        main_path = next(Path(os.environ['V10_OBSERVATIONS']).glob('*.json'))
        with self.assertRaisesRegex(ValueError, 'training/dev/pool'):
            smoke_observation(main_path, 'deepseek')

    def test_submit_without_execute_never_calls_provider(self):
        directory = os.environ.get('V10_OBSERVATIONS')
        if not directory:
            self.skipTest('set V10_OBSERVATIONS to the sealed sample directory')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'run'
            argv = ['api_runner.py', 'submit', '--provider', 'deepseek',
                    '--observations', directory, '--budget-usd', '10', '--output', str(output)]
            with patch.object(sys, 'argv', argv), patch('scripts.api_runner.request', side_effect=AssertionError('provider called')):
                with redirect_stdout(StringIO()):
                    main()
            self.assertFalse(output.exists())

    def test_parser_is_strict(self):
        valid = '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'
        self.assertEqual(parse_final(valid)[0], [0.8, 0.6, 0.4, 0.2])
        for bad in (valid.replace('0.6', '0.9'), valid.replace('0.8', '1.1'),
                    valid.replace('0.4', 'NaN'), valid.replace('"rho_5":0.2', '"extra":0.2'),
                    valid.replace('"rho_2":0.8', '"rho_2":0.8,"rho_2":0.8')):
            self.assertIsNone(parse_final(bad)[0])

    def test_openai_summary_and_final_are_preserved_separately(self):
        summary = {'type': 'summary_text', 'text': 'Provider summary.'}
        body = {'model': 'gpt-6-sol', 'usage': {'output_tokens_details': {'reasoning_tokens': 31}},
                'output': [{'type': 'reasoning', 'summary': [summary]},
                           {'type': 'message', 'content': [{'type': 'output_text',
                            'text': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'}]}]}
        record = openai_record(body, 'request', 'prompt-hash')
        self.assertEqual(record['reasoning_summary'], [summary])
        self.assertEqual(record['reasoning_exposure'], 'provider_reasoning_summary')
        self.assertEqual(record['reasoning_tokens'], 31)
        self.assertEqual(record['prediction'], [0.8, 0.6, 0.4, 0.2])
        self.assertEqual(record['raw_response'], body)
        without_summary = {**body, 'output': body['output'][1:]}
        self.assertEqual(openai_record(without_summary, 'request')['reasoning_summary'], [])

    def test_actual_frozen_prompts_when_available(self):
        path = os.environ.get('V10_OBSERVATIONS')
        if not path:
            self.skipTest('set V10_OBSERVATIONS to the sealed sample directory')
        rows = observations(Path(path))
        self.assertEqual(len(manifest(rows, 'deepseek')), 288)
        self.assertEqual(len(manifest(rows, 'openai')), 864)


if __name__ == '__main__':
    unittest.main()
