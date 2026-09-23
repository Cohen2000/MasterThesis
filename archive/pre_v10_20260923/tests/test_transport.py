"""Transports: the Qwen production runner (no GPU needed here) and the disabled
Sol/DeepSeek ledger. All transports are fakes; no test can issue a request."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from main_experiment.common import DESIGN_VERSION, PREPARED, ROOT, digest, read_json
from main_experiment.execution import (RESERVE, Ledger, collect_batch, parse_deepseek, run_deepseek, submit_batch,
                                       validate_release)
from main_experiment.requests import payload, validate_request
sys.path.insert(0, str(ROOT/'scripts'))
from run_qwen_engine import load_requests, pending, result_path, split_reasoning, write_result   # noqa: E402


def request(rid='r1', provider='deepseek'):
    messages = [{'role': 'user', 'content': 'Return JSON.'}]
    body = payload(provider, messages, 17)
    return {'id': rid, 'config_id': provider, 'seed': 17, 'payload': body, 'payload_sha256': digest(body),
            'prompt_sha256': digest(messages), 'design_version': DESIGN_VERSION}


class RequestTests(unittest.TestCase):
    def test_payloads_are_frozen_and_bound(self):
        for config in ('sol', 'deepseek', 'qwen_thinking', 'qwen_nonthinking'):
            body = payload(config, [], 123)
            self.assertNotIn('tools', body)
            self.assertEqual('seed' in body, config.startswith('qwen'))
        r = request(provider='qwen_thinking'); validate_request(r)
        for field, value in (('payload_sha256', 'x'), ('prompt_sha256', 'x'), ('design_version', 'old')):
            with self.assertRaises(ValueError): validate_request({**r, field: value})
        changed = dict(r); changed['payload'] = {**r['payload'], 'temperature': 0.}
        changed['payload_sha256'] = digest(changed['payload'])
        with self.assertRaises(ValueError): validate_request(changed)       # hash-consistent but not the frozen payload


class QwenRunnerTests(unittest.TestCase):
    def test_reasoning_split_matches_the_chat_template(self):
        self.assertEqual(split_reasoning('weighing it up\n</think>\n\n{"rho_2": 0.5}', True),
                         ('weighing it up', '{"rho_2": 0.5}', True))
        self.assertEqual(split_reasoning('cut off mid thought', True), ('cut off mid thought', '', False))
        self.assertEqual(split_reasoning('{"rho_2": 0.5}', False), ('', '{"rho_2": 0.5}', True))
        self.assertEqual(split_reasoning('a <think> b\n</think>\n{"x":1}', True), ('b', '{"x":1}', True))

    def test_admitted_requests_are_never_run_again(self):
        r = {'id': 'x', 'mode': 'thinking', 'repeat_index': 1, 'observation_id': 'o', 'graph_id': 'g', 'arm': 'H',
             'sample_index': 1, 'config_id': 'qwen_thinking', 'seed': 4, 'prompt_sha256': 'p', 'payload_sha256': 'b'}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            self.assertEqual(len(pending(out, [r])[1]), 1)
            (result_path(out, r).parent).mkdir(parents=True)
            result_path(out, r).with_suffix('.attempt').write_text(json.dumps({'payload_sha256': 'b'}))
            done, todo = pending(out, [r])                 # an interrupted admission becomes a technical failure
            self.assertEqual((len(done), len(todo)), (1, 0))
            self.assertEqual(read_json(result_path(out, r))['end_state'], 'process_interrupted')
            write_result(out, {**r, 'id': 'y'}, {'status': 'completed'})
            with self.assertRaises(ValueError): pending(out, [{**r, 'id': 'y', 'seed': 5}])   # identity mismatch

    def test_modes_repeats_and_shards_partition_the_requests(self):
        if not (PREPARED/'requests.jsonl').exists(): self.skipTest('prepared study absent')
        seen = set(); total = 0
        for mode in ('thinking', 'nonthinking'):
            for repeat in (1, 2, 3):
                ids = {r['id'] for r in load_requests(PREPARED, [(mode, repeat)], set(), 0, 1)}
                self.assertFalse(ids & seen); seen |= ids; total += len(ids)
        planned = read_json(PREPARED/'report.json')['planned_qwen_calls']
        self.assertEqual(total, planned)
        whole = [r['id'] for r in load_requests(PREPARED, [('thinking', 1)], set(), 0, 1)]
        parts = [r['id'] for i in range(6) for r in load_requests(PREPARED, [('thinking', 1)], set(), i, 6)]
        self.assertEqual(sorted(parts), sorted(whole)); self.assertEqual(len(set(parts)), len(parts))


class TransportRegression(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out=Path(self.tmp.name)

    def ledger(self,requests): return Ledger(self.out,requests,{'test':'mock only'})

    def test_release_blocks_before_network(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            validate_release({'authorized':False},self.out,self.out/'none')

    def test_status_is_readable_while_writer_lock_is_held(self):
        import subprocess
        from main_experiment.integrity import exclusive
        ledger=self.ledger([request()]); ledger.admit(['r1'])
        script=ROOT/'scripts/run_api.py'
        with exclusive(self.out/'dispatch.lock'):
            result=subprocess.run([sys.executable,str(script),'status','--run','unused',
                                   '--baselines','unused','--out',str(self.out)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(json.loads(result.stdout)['states'],{'dispatching':1})

    def test_no_repeated_admission_even_after_restart(self):
        req=request(); ledger=self.ledger([req]); ledger.admit(['r1'])
        restarted=self.ledger([req])
        with self.assertRaises(ValueError): restarted.admit(['r1'])
        changed=dict(req); changed['payload_sha256']='wrong'
        with self.assertRaises(ValueError): self.ledger([changed])
        self.assertEqual(restarted.totals('deepseek'),(0,RESERVE['deepseek']))

    def test_unknown_billing_keeps_reserve_and_blocks_further_calls(self):
        ledger=self.ledger([request(),request('r2')]); ledger.admit(['r1'])
        ledger.finish('r1',{'final_text':'','technical_error':True})
        self.assertEqual(ledger.totals('deepseek'),(0,RESERVE['deepseek']))
        with self.assertRaises(ValueError): ledger.admit(['r2'])
        self.assertEqual(ledger.export(self.out/'export.jsonl'),1)

    def test_cost_cap_reserves_every_request_before_admission(self):
        ledger=self.ledger([request(str(i)) for i in range(105)])
        with self.assertRaises(ValueError): ledger.admit([str(i) for i in range(105)])
        self.assertEqual(ledger.state['requests'],{})

    def test_stream_reasoning_never_becomes_final_and_truncation_is_failure(self):
        chunks=[{'id':'d1','model':'deepseek-flash','choices':[{'index':0,'delta':{'reasoning_content':'{"rho_2":1}'},'finish_reason':None}]},
                {'choices':[{'index':0,'delta':{'content':'{}'},'finish_reason':'stop'}]},
                {'choices':[],'usage':{'prompt_tokens':10,'completion_tokens':20}}]
        lines=['data: '+json.dumps(c) for c in chunks]
        r,u=parse_deepseek(lines+['data: [DONE]'])
        self.assertEqual(r['final_text'],'{}'); self.assertIn('rho_2',r['reasoning_text'])
        self.assertFalse(r['technical_error']); self.assertEqual(u['completion_tokens'],20)
        self.assertTrue(parse_deepseek(lines)[0]['technical_error'])

    def test_deepseek_transport_persists_raw_usage_and_never_retries(self):
        lines=[b'data: '+json.dumps({'id':'d1','model':'deepseek-flash',
            'choices':[{'index':0,'delta':{'content':'{}'},'finish_reason':'stop'}]}).encode()+b'\n',
            b'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":20}}\n',b'data: [DONE]\n']
        class Conn:
            sock=None
            def settimeout(self,n): pass
            def close(self): pass
        conn=Conn(); conn.sock=conn
        class Response(io.BytesIO):
            status=200
            def getheaders(self): return [('x-request-id','mock')]
        class FakeHTTP:
            calls=0
            def open(self,*args): self.calls+=1; return conn,Response(b''.join(lines))
        http=FakeHTTP(); ledger=self.ledger([request()])
        run_deepseek(ledger,'r1',http,['deepseek-flash'])
        self.assertEqual(http.calls,1)
        self.assertEqual(ledger.totals('deepseek'),(27,0))
        self.assertEqual((self.out/'raw/r1.sse').read_bytes(),b''.join(lines))
        self.assertFalse(read_json(self.out/'responses/r1.json')['technical_error'])
        with self.assertRaises(ValueError): run_deepseek(ledger,'r1',http,['deepseek-flash'])
        self.assertEqual(http.calls,1)

    def test_ambiguous_batch_create_reconciles_without_second_post(self):
        ledger=self.ledger([request('a','sol'),request('b','sol')])
        class FakeHTTP:
            posts=0
            def upload(self,data): self.input=data; return {'id':'file-1'}
            def json(self,method,path,body=None):
                if method=='POST':
                    self.posts+=1; self.bid=body['metadata']['study_batch']; raise TimeoutError('lost acknowledgement')
                if path.startswith('/v1/batches?'):
                    return {'data':[{'id':'batch-1','metadata':{'study_batch':self.bid}}],'has_more':False}
                return {'status':'completed','output_file_id':'file-out','error_file_id':None}
            def request(self,*args):
                return b'\n'.join(json.dumps({'custom_id':rid,'response':{'status_code':200,'body':{
                    'model':'gpt-5.6-sol','status':'completed','output':[{'type':'message','content':[{'type':'output_text','text':'{}'}]}],
                    'usage':{'input_tokens':10,'output_tokens':20}}}}).encode() for rid in ('b','a'))
        http=FakeHTTP(); bid=submit_batch(ledger,['a','b'],http)
        self.assertEqual(ledger.state['batches'][bid]['status'],'reconcile')
        restarted=self.ledger([request('a','sol'),request('b','sol')])
        self.assertEqual(collect_batch(restarted,bid,http,['gpt-5.6-sol']),'collected')
        self.assertEqual(http.posts,1); self.assertEqual(restarted.totals('sol'),(440,0))
        self.assertEqual(restarted.export(self.out/'export.jsonl'),2)
        collect_batch(restarted,bid,http,['gpt-5.6-sol'])
        self.assertEqual(restarted.totals('sol'),(440,0))


if __name__ == '__main__':
    unittest.main()
