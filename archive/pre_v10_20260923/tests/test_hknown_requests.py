"""H-known ablation request builder: identity with its H-unknown twin, protocol validity."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from main_experiment.common import DESIGN_VERSION, ROOT, digest, read_json, write_json
from main_experiment.requests import payload, validate_request
sys.path.insert(0, str(ROOT/'scripts'))
from build_hknown_requests import main as build_hknown   # noqa: E402


def h_request(rid_suffix, sample_index=1, repeat=1):
    messages = [{'role': 'system', 'content': 'sys'},
                {'role': 'user', 'content': 'rules...\nTemporal_access=0,0,0,1,1\nN_obs=3'}]
    seed = 12345
    body = payload('qwen_thinking', messages, seed)
    return {'id': f'graphX__H-p888-access-v9-20260922__s{sample_index}__qwen_thinking__r{repeat}__{DESIGN_VERSION}',
            'observation_id': f'graphX__H-p888-access-v9-20260922__s{sample_index}', 'graph_id': 'graphX', 'arm': 'H',
            'sample_index': sample_index, 'repeat_index': repeat, 'config_id': 'qwen_thinking', 'seed': seed,
            'status': 'not_started', 'started': False, 'mock': False, 'payload': body,
            'payload_sha256': digest(body), 'prompt_sha256': digest(messages), 'design_version': DESIGN_VERSION,
            'production_dispatch_enabled': True, 'requires_technical_release': False}


def other_request(arm='R'):
    r = h_request('x'); r['arm'] = arm; r['id'] = 'other__'+r['id']
    return r


class HKnownRequestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.src = Path(self.tmp.name)/'src'; self.dst = Path(self.tmp.name)/'dst'
        (self.src/'observations/sample').mkdir(parents=True)
        h1 = h_request('a'); h2 = h_request('b', sample_index=2)
        skipped = h_request('c', sample_index=3); skipped['status'] = 'skipped_empty'
        non_h = other_request('R')
        self.requests = [h1, h2, skipped, non_h]
        (self.src/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in self.requests))
        for oid in (h1['observation_id'], h2['observation_id']):
            write_json(self.src/'observations/sample'/(oid+'.json'), {'id': oid, 'arm': 'H'})
        build_hknown(str(self.src), str(self.dst))
        self.built = [json.loads(l) for l in (self.dst/'requests.jsonl').read_text().splitlines()]

    def test_only_active_h_requests_are_paired(self):
        self.assertEqual(len(self.built), 2)                    # not the skipped_empty one, not the R request
        self.assertTrue(all(r['arm'] == 'H' for r in self.built))

    def test_every_h_known_request_validates_under_the_deployed_contract(self):
        for r in self.built:
            self.assertEqual(r['design_version'], DESIGN_VERSION)   # no separate design_version invented
            validate_request(r)                                     # must not raise request protocol mismatch

    def test_id_is_unique_but_the_case_is_otherwise_identical_to_its_h_unknown_twin(self):
        base_by_obs = {r['observation_id']: r for r in self.requests if r['arm'] == 'H' and r['status'] != 'skipped_empty'}
        for r in self.built:
            base = base_by_obs[r['observation_id']]
            self.assertNotEqual(r['id'], base['id'])
            self.assertTrue(r['id'].startswith(base['id']))
            for field in ('observation_id', 'graph_id', 'sample_index', 'repeat_index', 'config_id', 'seed'):
                self.assertEqual(r[field], base[field], field)

    def test_the_only_prompt_difference_is_the_explicit_h_sentence(self):
        base_by_obs = {r['observation_id']: r for r in self.requests if r['arm'] == 'H' and r['status'] != 'skipped_empty'}
        for r in self.built:
            base = base_by_obs[r['observation_id']]
            old_messages = base['payload']['messages']; new_messages = r['payload']['messages']
            self.assertEqual(old_messages[0], new_messages[0])                 # system message untouched
            self.assertEqual(new_messages[1]['content'],
                             old_messages[1]['content'].replace(
                               'Temporal_access=', 'The most recent fraction h=0.60 of the archive is observable.\n'
                               'Temporal_access=', 1))
            # nothing else about decoding/model config changed
            for key in ('model', 'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'seed'):
                self.assertEqual(r['payload'][key], base['payload'][key], key)

    def test_paired_observation_carries_the_h_known_messages(self):
        # run_qwen_engine.load_requests recomputes digest(obs['messages']) and compares it
        # to the request's prompt_sha256; they must agree or every shard fails at load time.
        for r in self.built:
            obs = read_json(self.dst/'observations/sample'/(r['observation_id']+'.json'))
            self.assertEqual(obs['id'], r['observation_id'])
            self.assertEqual(digest(obs['messages']), r['prompt_sha256'])
            self.assertEqual(obs['messages'], r['payload']['messages'])


if __name__ == '__main__':
    unittest.main()
