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

from main_experiment.common import MAIN_KEYS, digest
from main_experiment.evaluation import parse_final
from scripts.api_runner import (API_MAIN_ARMS, GENERATION_CAP, ProviderRejected, collect_openai,
                                deepseek_progress, deepseek_window, estimate,
                                execute_deepseek, execute_openai, execute_openai_technical, guard, main,
                                manifest, observations, openai_progress, openai_record, payload, provider_id,
                                pilot_manifest, require_deepseek_offpeak,
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
        for provider, count in (('deepseek', 288), ('openai', 288)):
            planned = manifest(rows, provider)
            self.assertEqual(len(planned), count)
            self.assertNotIn('S_obs', {r['arm'] for r in planned})
            self.assertEqual(len({r['id'] for r in planned}), count)
            self.assertTrue(all('secret' not in str(payload(provider, r)).lower() for r in planned))
            self.assertEqual(len({r['provider_id'] for r in planned}), count)
            self.assertTrue(all(r['provider_id'] == provider_id(r['id']) and
                                r['provider_id'].startswith('req_') and
                                r['graph_id'] not in r['provider_id'] and
                                r['arm'] not in r['provider_id'] for r in planned))
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
        self.assertEqual(payload('openai', manifest(rows, 'openai')[0])['max_output_tokens'], GENERATION_CAP)
        self.assertEqual(payload('deepseek', manifest(rows, 'deepseek')[0])['reasoning_effort'], 'high')
        self.assertEqual(payload('deepseek', manifest(rows, 'deepseek')[0])['model'], 'deepseek-flash')
        self.assertEqual(payload('deepseek', manifest(rows, 'deepseek')[0])['max_tokens'], 393216)
        for provider in ('deepseek', 'openai'):
            self.assertEqual([len(manifest(rows, provider, repeats=n)) for n in (1, 2, 3)],
                             [288, 576, 864])
        plan = estimate(manifest(rows, 'deepseek'), 'deepseek')
        self.assertIsNone(plan['conservative_projected_total_usd'])
        self.assertGreater(plan['theoretical_full_main_worst_case_usd'], 10)

    def test_utc_offpeak_windows_and_buffer(self):
        def window(day, hour, minute=0):
            return deepseek_window(datetime(2026, 9, day, hour, minute, tzinfo=timezone.utc))
        # 2026-09-28 is Monday; 2026-09-26 is Saturday.
        # Launches stop 60 minutes before each peak (billing time is undocumented).
        for hour, minute, expected in ((0, 30, False), (2, 0, False), (4, 30, True), (5, 0, False),
                                       (8, 0, False), (12, 0, True)):
            self.assertEqual(window(28, hour, minute)[0], expected)
        self.assertTrue(window(26, 8)[0])
        self.assertTrue(window(27, 23, 59)[0])
        self.assertFalse(window(28, 0, 0)[0])
        self.assertFalse(window(28, 0, 1)[0])
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
        pilots = [{'id': f'pilot-{arm}-{i}__deepseek__pilot', 'prompt_sha256': f'{arm}-{i}',
                   'messages': [{'role': 'user', 'content': 'small'}], 'kind': 'pilot', 'arm': arm}
                  for arm in API_MAIN_ARMS for i in (1, 2)]
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
                execute_deepseek(rows, path, 2, pilot=pilots)
                execute_deepseek(rows, path, 2)
                execute_deepseek(rows, path, 2)
            self.assertEqual(provider.call_count, 11)
            records = [json.loads(line) for line in (path / 'responses.jsonl').read_text().splitlines()]
            self.assertEqual({r['id'] for r in records}, {'one', 'two', smoke['id']} | {r['id'] for r in pilots})
            self.assertTrue(all(r['reasoning_content'] == 'Provider raw reasoning.' for r in records))
            self.assertTrue(all(r['raw_response'] == answer for r in records))
            self.assertEqual(len((path / 'attempts.jsonl').read_text().splitlines()), 11)
            self.assertGreater(deepseek_progress(rows, path)[1], 0)  # includes smoke usage

    def test_deepseek_length_is_flagged_without_parsing(self):
        smoke = {'id': 'training__deepseek__smoke', 'prompt_sha256': 'training',
                 'messages': [{'role': 'user', 'content': 'small'}], 'kind': 'smoke'}
        answer = {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 12, 'completion_tokens': 128000},
                  'choices': [{'message': {'content': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'},
                               'finish_reason': 'length'}]}
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.require_deepseek_offpeak'), \
                 patch('scripts.api_runner.request', return_value=answer), \
                 redirect_stdout(StringIO()):
                execute_deepseek([], Path(directory), 1, smoke)
            record = json.loads((Path(directory) / 'responses.jsonl').read_text())
            self.assertTrue(record['limit_hit'])
            self.assertIsNone(record['prediction'])
            self.assertEqual(record['final_text'], answer['choices'][0]['message']['content'])

    def test_gpt_batch_chunks_wait_for_collection(self):
        messages = [{'role': 'user', 'content': 'small'}]
        rows = [{'id': f'main-{i}__openai__r1', 'prompt_sha256': digest(messages), 'messages': messages}
                for i in range(3)]
        technical = [{'id': 'smoke', 'kind': 'smoke', 'usage': {'input_tokens': 12, 'output_tokens': 8}}]
        technical += [{'id': f'pilot-{i}', 'kind': 'pilot', 'arm': API_MAIN_ARMS[i % 4],
                       'usage': {'input_tokens': 12, 'output_tokens': 100}, 'reasoning_tokens': 40}
                      for i in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'technical_responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in technical))
            def provider(url, *_args):
                return {'id': 'file-1'} if url.endswith('/files') else {'id': 'batch-1'}
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=provider) as network, \
                 redirect_stdout(StringIO()):
                execute_openai(rows, path, 2)
                with self.assertRaisesRegex(ValueError, 'collect the previous'):
                    execute_openai(rows, path, 2)
            self.assertEqual(network.call_count, 2)  # one upload, one Batch creation
            first = [json.loads(line) for line in (path / 'chunk_001.input.jsonl').read_text().splitlines()]
            self.assertEqual([r['custom_id'] for r in first], [provider_id(r['id']) for r in rows[:2]])
            mapping = json.loads((path / 'chunk_001.mapping.json').read_text())
            self.assertEqual(mapping, {provider_id(r['id']): r['id'] for r in rows[:2]})
            result = {'model': 'gpt-6-sol', 'status': 'completed',
                      'usage': {'input_tokens': 12, 'output_tokens': 100},
                      'output': [{'type': 'message', 'content': [{'type': 'output_text',
                                 'text': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'}]}]}
            raw = ''.join(json.dumps({'custom_id': r['custom_id'], 'response': {'body': result}}) + '\n'
                          for r in first).encode()
            with patch('scripts.api_runner.download_openai_file', return_value=raw), redirect_stdout(StringIO()):
                collect_openai(path, {'id': 'batch-1', 'status': 'completed', 'output_file_id': 'output-1'}, 'fake')
                collect_openai(path, {'id': 'batch-1', 'status': 'completed', 'output_file_id': 'output-1'}, 'fake')
            self.assertEqual(len((path / 'responses.jsonl').read_text().splitlines()), 2)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=AssertionError('provider called')):
                with self.assertRaisesRegex(ValueError, 'user budget'):
                    execute_openai(rows, path, 2, budget=.001)
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=provider), \
                 redirect_stdout(StringIO()):
                execute_openai(rows, path, 2)
            second = [json.loads(line) for line in (path / 'chunk_002.input.jsonl').read_text().splitlines()]
            self.assertEqual([r['custom_id'] for r in second], [provider_id(rows[2]['id'])])

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
            for provider, budget in (('deepseek', '10'), ('openai', '200')):
                argv = ['api_runner.py', 'submit', '--provider', provider,
                        '--observations', directory, '--repeats', '1',
                        '--budget-usd', budget, '--output', str(output)]
                with patch.object(sys, 'argv', argv), \
                     patch('scripts.api_runner.request', side_effect=AssertionError('provider called')):
                    with redirect_stdout(StringIO()):
                        main()
            self.assertFalse(output.exists())
            for command in ('status', 'collect'):
                argv = ['api_runner.py', command, '--provider', 'openai', '--output', str(output)]
                with patch.object(sys, 'argv', argv), \
                     patch('scripts.api_runner.request', side_effect=AssertionError('provider called')):
                    with redirect_stdout(StringIO()):
                        main()

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
        limited = {**body, 'status': 'incomplete',
                   'incomplete_details': {'reason': 'max_output_tokens'}}
        flagged = openai_record(limited, 'request')
        self.assertTrue(flagged['limit_hit'])
        self.assertIsNone(flagged['prediction'])
        self.assertEqual(flagged['final_text'], record['final_text'])

    def test_actual_frozen_prompts_when_available(self):
        path = os.environ.get('V10_OBSERVATIONS')
        if not path:
            self.skipTest('set V10_OBSERVATIONS to the sealed sample directory')
        rows = observations(Path(path))
        self.assertEqual(len(rows), 288)
        self.assertEqual({a: sum(r['arm'] == a for r in rows) for a in API_MAIN_ARMS},
                          {'R': 72, 'S': 72, 'H': 72, 'B': 72})
        self.assertNotIn('S_obs', {r['arm'] for r in rows})
        self.assertTrue(all('truth' not in r for r in rows))
        for provider, count, budget in (('deepseek', 288, 10), ('openai', 288, 200)):
            planned = manifest(rows, provider)
            self.assertEqual(len(planned), count)
            def records(output):
                usage = {'prompt_tokens': 100, 'completion_tokens': output} if provider == 'deepseek' else {
                    'input_tokens': 100, 'output_tokens': output}
                return [{'id': f'pilot-{i}', 'kind': 'pilot', 'arm': API_MAIN_ARMS[i % 4],
                         'usage': usage, 'reasoning_tokens': output // 2} for i in range(8)]
            plan = estimate(planned, provider, records(10_000))
            self.assertEqual(plan['pilot_usage']['generated_tokens']['n'], 8)
            self.assertEqual(plan['pilot_usage']['reasoning_tokens']['n'], 8)
            self.assertEqual(plan['pilot_usage']['conservative_tokens_per_request'], 12_500)
            self.assertLess(plan['conservative_projected_total_usd'], budget)
            self.assertGreater(estimate(planned, provider, records(GENERATION_CAP))['conservative_projected_total_usd'], budget)
            self.assertEqual(len(manifest(rows, provider, repeats=3)), 864)
        pilot_path = os.environ.get('V10_PILOT_OBSERVATIONS')
        if pilot_path:
            pilot = pilot_manifest(Path(pilot_path), 'deepseek')
            self.assertEqual(len(pilot), 12)
            self.assertEqual({r['arm'] for r in pilot}, set(API_MAIN_ARMS))

    def test_repeat_extension_skips_completed_r1(self):
        path = os.environ.get('V10_OBSERVATIONS')
        if not path:
            self.skipTest('set V10_OBSERVATIONS to the v11 API freeze')
        samples = observations(Path(path))
        for provider in ('deepseek', 'openai'):
            r1 = manifest(samples, provider, repeats=1)
            r3 = manifest(samples, provider, repeats=3)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                records = [{'id': r['id'], 'kind': 'main', 'prompt_sha256': r['prompt_sha256'],
                            'usage': {'prompt_tokens': 100, 'completion_tokens': 100}
                            if provider == 'deepseek' else {'input_tokens': 100, 'output_tokens': 100}}
                           for r in r1]
                (root / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
                if provider == 'deepseek':
                    completed, _, pending, _ = deepseek_progress(r3, root)
                    self.assertEqual(len(completed), 288)
                    self.assertEqual(len(pending), 576)
                else:
                    _, completed = openai_progress(r3, root)
                    self.assertEqual(len(completed), 288)
                    self.assertEqual(len({r['id'] for r in r3} - {r['id'] for r in completed}), 576)
                (root / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records + records[:1]))
                with self.assertRaisesRegex(ValueError, 'duplicate'):
                    (deepseek_progress if provider == 'deepseek' else openai_progress)(r3, root)

    def test_v11_composite_provenance_when_available(self):
        api_path = os.environ.get('V10_OBSERVATIONS')
        released_path = os.environ.get('V11_RELEASED_RUN')
        if not api_path or not released_path:
            self.skipTest('set v11 API and completed released R/H paths')
        api = {(r['graph_id'], r['arm'], r['sample_index']): r
               for r in observations(Path(api_path))}
        old = {(r['graph_id'], r['arm'], r['sample_index']): r
               for p in (Path.home() / '.local/share/masterthesis/v10_observations').glob('*.json')
               for r in [json.loads(p.read_text())]}
        released = {(r['graph_id'], r['arm'], r['sample_index']): r
                    for p in (Path(released_path) / 'observations/sample').glob('*.json')
                    for r in [json.loads(p.read_text())]}
        requests = [json.loads(line) for line in (Path(released_path) / 'requests.jsonl').read_text().splitlines()]
        prompt_hashes = {r['prompt_sha256'] for r in requests}
        for key, row in api.items():
            source = released[key] if row['arm'] in ('R', 'H') else old[key]
            self.assertEqual(row['block'], source['block'])
            self.assertEqual(row['messages'], source['messages'])
            self.assertEqual(row['prompt_sha256'], source['prompt_sha256'])
            if row['arm'] in ('R', 'H'):
                self.assertIn(row['prompt_sha256'], prompt_hashes)
                hidden = old[key]
                self.assertEqual(row['block'].replace(f"n_panel={source['n_panel']}\n", ''), hidden['block'])
        with (ROOT / 'docs/results/panel888_v10_main_20260923/PREDICTIONS.csv').open() as stream:
            import csv
            old_predictions = {(r['id'], r['method']): r for r in csv.DictReader(stream)
                               if r['arm'] in ('S', 'S_obs', 'B')}
        with (ROOT / 'docs/results/panel888_v11_main_20260923/PREDICTIONS.csv').open() as stream:
            final = list(csv.DictReader(stream))
        self.assertEqual(old_predictions, {(r['id'], r['method']): r for r in final
                                            if r['arm'] in ('S', 'S_obs', 'B')})
        qwen = [r for r in final if r['method'].startswith('qwen')]
        self.assertEqual((len(qwen), sum(r['valid'] == 'True' for r in qwen)), (2160, 2159))
        freeze = json.loads((ROOT / 'docs/results/panel888_v11_main_20260923/API_FREEZE.json').read_text())
        release_verification = json.loads((ROOT / 'docs/results/panel888_v10_RH_panel_release/VERIFICATION.json').read_text())
        self.assertEqual(freeze['qwen_archive_checksums_sha256']['released_RH'],
                         release_verification['archive']['checksums_sha256'])

    def test_high_pilot_usage_blocks_full_production(self):
        path = os.environ.get('V10_OBSERVATIONS')
        if not path:
            self.skipTest('set V10_OBSERVATIONS to the sealed sample directory')
        samples = observations(Path(path))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            deep = root / 'deep'
            deep.mkdir()
            deep_records = [{'id': 'training__deepseek__smoke', 'kind': 'smoke',
                             'usage': {'prompt_tokens': 100, 'completion_tokens': 10}}]
            deep_records += [{'id': f'pilot-{i}__deepseek__pilot', 'kind': 'pilot',
                              'arm': API_MAIN_ARMS[i % 4],
                              'usage': {'prompt_tokens': 100, 'completion_tokens': GENERATION_CAP}}
                             for i in range(8)]
            (deep / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in deep_records))
            output = StringIO()
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.require_deepseek_offpeak'), \
                 patch('scripts.api_runner.request', side_effect=AssertionError('provider called')), \
                 redirect_stdout(output):
                # Pilot spend ~USD 0.61; one more 384k worst case (~USD 0.24) does not fit USD 0.7.
                execute_deepseek(manifest(samples, 'deepseek'), deep, 8, budget=0.7)
            self.assertIn('budget reached', output.getvalue())
            gpt = root / 'gpt'
            gpt.mkdir()
            gpt_records = [{'id': 'smoke', 'kind': 'smoke',
                            'usage': {'input_tokens': 100, 'output_tokens': 10}}]
            gpt_records += [{'id': f'pilot-{i}', 'kind': 'pilot', 'arm': API_MAIN_ARMS[i % 4],
                             'usage': {'input_tokens': 100, 'output_tokens': GENERATION_CAP}}
                            for i in range(8)]
            (gpt / 'technical_responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in gpt_records))
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=AssertionError('provider called')):
                with self.assertRaisesRegex(ValueError, 'conservative projected total'):
                    execute_openai(manifest(samples, 'openai'), gpt, 96)


    def test_unbilled_rejection_allows_one_relaunch(self):
        smoke = {'id': 'training__deepseek__smoke', 'prompt_sha256': 'training',
                 'messages': [{'role': 'user', 'content': 'small'}], 'kind': 'smoke'}
        answer = {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 12, 'completion_tokens': 4},
                  'choices': [{'message': {'content': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'},
                               'finish_reason': 'stop'}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.require_deepseek_offpeak'), \
                 patch('scripts.api_runner.request', side_effect=ProviderRejected(400, 'bad parameter')), \
                 redirect_stdout(StringIO()):
                with self.assertRaises(RuntimeError):
                    execute_deepseek([], path, 1, smoke)
            self.assertEqual(deepseek_progress([], path)[0], {})
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.require_deepseek_offpeak'), \
                 patch('scripts.api_runner.request', return_value=answer), \
                 redirect_stdout(StringIO()):
                execute_deepseek([], path, 1, smoke)
            self.assertEqual(set(deepseek_progress([], path)[0]), {smoke['id']})
            # A launch without rejection or answer stays uncertain.
            with (path / 'attempts.jsonl').open('a') as handle:
                handle.write('{"id":"other"}\n')
            with self.assertRaisesRegex(ValueError, 'uncertain'):
                deepseek_progress([], path)

    def test_openai_technical_background_is_resumed_not_relaunched(self):
        smoke = {'id': 'training__openai__smoke', 'prompt_sha256': 'training', 'arm': 'R',
                 'messages': [{'role': 'user', 'content': 'small'}], 'kind': 'smoke'}
        done = {'id': 'resp_1', 'model': 'gpt-6-sol', 'status': 'completed',
                'usage': {'input_tokens': 12, 'output_tokens': 100},
                'output': [{'type': 'reasoning', 'summary': [], 'content': None},
                           {'type': 'message', 'content': [{'type': 'output_text',
                            'text': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'}]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            calls = []
            def crash_while_polling(url, key, body=None, method='GET'):
                calls.append((url, method, json.loads(body) if body else None))
                if method == 'POST':
                    return {'id': 'resp_1', 'status': 'queued'}
                raise ConnectionError('laptop offline')
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=crash_while_polling), \
                 redirect_stdout(StringIO()):
                with self.assertRaises(ConnectionError):
                    execute_openai_technical([smoke], path)
            self.assertTrue(calls[0][2]['background'])
            self.assertEqual(openai_progress([], path, technical_only=True)[2], {smoke['id']: 'resp_1'})
            with self.assertRaisesRegex(ValueError, 'still open'):
                openai_progress([], path)
            calls.clear()
            def poll(url, key, body=None, method='GET'):
                calls.append((url, method))
                return done
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=poll), \
                 redirect_stdout(StringIO()):
                execute_openai_technical([smoke], path)
            self.assertEqual(calls, [('https://api.openai.com/v1/responses/resp_1', 'GET')])
            technical = openai_progress([], path)[0]
            self.assertEqual(technical[0]['prediction'], [0.8, 0.6, 0.4, 0.2])
            self.assertEqual(technical[0]['response_id'], 'resp_1')

    def test_unbilled_batch_failures_stay_pending(self):
        messages = [{'role': 'user', 'content': 'small'}]
        rows = [{'id': f'main-{i}__openai__r1', 'prompt_sha256': digest(messages), 'messages': messages}
                for i in range(2)]
        technical = [{'id': 'smoke', 'kind': 'smoke', 'usage': {'input_tokens': 12, 'output_tokens': 8}}]
        technical += [{'id': f'pilot-{i}', 'kind': 'pilot', 'arm': API_MAIN_ARMS[i % 4],
                       'usage': {'input_tokens': 12, 'output_tokens': 100}} for i in range(8)]
        result = {'model': 'gpt-6-sol', 'status': 'completed',
                  'usage': {'input_tokens': 12, 'output_tokens': 100},
                  'output': [{'type': 'message', 'content': [{'type': 'output_text',
                             'text': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'}]}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'technical_responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in technical))
            provider = lambda url, *_args: {'id': 'file-1'} if url.endswith('/files') else {'id': 'batch-1'}
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=provider), redirect_stdout(StringIO()):
                execute_openai(rows, path, 2)
            first = [json.loads(line)['custom_id'] for line in (path / 'chunk_001.input.jsonl').read_text().splitlines()]
            output = json.dumps({'custom_id': first[0], 'response': {'status_code': 200, 'body': result}}) + '\n'
            errors = json.dumps({'custom_id': first[1], 'response': None,
                                 'error': {'code': 'batch_expired', 'message': 'expired'}}) + '\n'
            files = {'output-1': output.encode(), 'error-1': errors.encode()}
            with patch('scripts.api_runner.download_openai_file', side_effect=lambda file_id, key: files[file_id]), \
                 redirect_stdout(StringIO()):
                collect_openai(path, {'id': 'batch-1', 'status': 'expired', 'output_file_id': 'output-1',
                                      'error_file_id': 'error-1'}, 'fake')
            technical_records, collected = openai_progress(rows, path)
            self.assertEqual([r['id'] for r in collected], [rows[0]['id']])
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.request', side_effect=provider), redirect_stdout(StringIO()):
                execute_openai(rows, path, 2)
            second = [json.loads(line)['custom_id'] for line in (path / 'chunk_002.input.jsonl').read_text().splitlines()]
            self.assertEqual(second, [first[1]])


    def test_rolling_pool_never_exceeds_budget_with_open_requests(self):
        rows = [{'id': f'main-{i}', 'prompt_sha256': f'p{i}', 'sample_index': i % 3 + 1,
                 'messages': [{'role': 'user', 'content': 'small'}]} for i in range(6)]
        records = [{'id': 'training__deepseek__smoke', 'kind': 'smoke',
                    'usage': {'prompt_tokens': 10, 'completion_tokens': 10}}]
        records += [{'id': f'pilot-{i}__deepseek__pilot', 'kind': 'pilot', 'arm': API_MAIN_ARMS[i % 4],
                     'usage': {'prompt_tokens': 10, 'completion_tokens': 10}} for i in range(8)]
        answer = {'model': 'deepseek-flash', 'usage': {'prompt_tokens': 12, 'completion_tokens': 1000},
                  'choices': [{'message': {'content': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'},
                               'finish_reason': 'stop'}]}
        state = {'open': 0, 'peak': 0}
        guard_lock = __import__('threading').Lock()
        def provider(*_args, **_kwargs):
            with guard_lock:
                state['open'] += 1
                state['peak'] = max(state['peak'], state['open'])
            __import__('time').sleep(.05)
            with guard_lock:
                state['open'] -= 1
            return answer
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / 'responses.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
            (path / 'attempts.jsonl').write_text(''.join(json.dumps({'id': r['id']}) + '\n' for r in records))
            with patch.dict(os.environ, {'DEEPSEEK_API_KEY': 'fake'}), \
                 patch('scripts.api_runner.require_deepseek_offpeak'), \
                 patch('scripts.api_runner.request', side_effect=provider), \
                 redirect_stdout(StringIO()):
                # Room for two 384k worst cases (~USD 0.236 each), not three.
                execute_deepseek(rows, path, 16, budget=0.5)
            self.assertEqual(state['peak'], 2)
            done = [json.loads(line)['id'] for line in (path / 'responses.jsonl').read_text().splitlines()]
            self.assertEqual(set(done[9:]), {r['id'] for r in rows})


    def test_tool_variant_payload_record_and_shared_budget(self):
        from scripts.api_runner import CONTAINER_USD, actual_usd, shared_spend
        row = {'id': 'x__openai_tools__r1', 'messages': [{'role': 'user', 'content': 'small'}], 'tools': True}
        body = payload('openai', row)
        self.assertEqual(body['tools'], [{'type': 'code_interpreter', 'container': {'type': 'auto'}}])
        self.assertNotIn('tools', payload('openai', {**row, 'tools': False}))
        response = {'model': 'gpt-6-sol', 'status': 'completed',
                    'usage': {'input_tokens': 1000, 'output_tokens': 100},
                    'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Let me compute.'}]},
                               {'type': 'code_interpreter_call', 'container_id': 'cntr_1', 'code': 'print(1)'},
                               {'type': 'message', 'content': [{'type': 'output_text',
                                'text': '{"rho_2":0.8,"rho_3":0.6,"rho_4":0.4,"rho_5":0.2}'}]}]}
        record = openai_record(response, row['id'])
        self.assertEqual(record['prediction'], [0.8, 0.6, 0.4, 0.2])
        self.assertEqual((record['tool_calls'], record['containers']), (1, ['cntr_1']))
        self.assertAlmostEqual(actual_usd(record, 'openai'), (1000 * 1.25 + 100 * 5) / 1e6 + CONTAINER_USD)
        with tempfile.TemporaryDirectory() as directory:
            other = Path(directory)
            (other / 'responses.jsonl').write_text(json.dumps({**record, 'id': 'other'}) + '\n')
            self.assertAlmostEqual(shared_spend([other]), actual_usd(record, 'openai'))
            (other / 'chunk_001.batch.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'uncollected'):
                shared_spend([other])


if __name__ == '__main__':
    unittest.main()
