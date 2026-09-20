"""Regression tests for the scientific revision and crash-safe single attempts.
All transports are fakes; these tests cannot issue paid requests.
"""
import csv
import io
import json
import pathlib
import sys
import tempfile
import unittest
import numpy as np
from unittest.mock import patch
from main_experiment.common import digest,write_json,read_json,DESIGN_VERSION
from main_experiment.requests import payload
from main_experiment.evaluation import conditional_summary,paired_summary,resolve
from main_experiment.pool import draw_pool
from main_experiment.execution import (Ledger,RESERVE,run_deepseek,submit_batch,collect_batch,
                                       response_from_batch,parse_deepseek,validate_release)
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[2]/'scripts'))


def request(rid='r1',provider='deepseek'):
    messages=[{'role':'user','content':'Return JSON.'}]
    body=payload(provider,messages,17)
    return {'id':rid,'config_id':provider,'seed':17,'payload':body,
            'payload_sha256':digest(body),'prompt_sha256':digest(messages),'design_version':DESIGN_VERSION}


class ScientificRevision(unittest.TestCase):
    def test_baseline_entrypoint_imports_without_running(self):
        import run_baseline_revision
        from main_experiment.common import CURRENT_RUN
        self.assertEqual(run_baseline_revision.FROZEN,pathlib.Path(CURRENT_RUN))

    def test_pool_balances_each_partition_and_ad_joint_grid(self):
        from collections import Counter
        specs=draw_pool()
        for family in ('dar','ad'):
            for part,n in (('train',200),('dev',50)):
                sub=[r for r in specs if r['family']==family and r['partition']==part]
                self.assertEqual(set(Counter(r['parameters']['N'] for r in sub).values()),{n//5})
                self.assertEqual(len({r['parameters']['N'] for r in sub}),5)
                if family=='ad':
                    joint=Counter((r['parameters']['N'],r['parameters']['rounds']) for r in sub)
                    self.assertEqual(len(joint),25); self.assertEqual(set(joint.values()),{n//25})
        self.assertTrue(all('pool-v2-balanced-20260918' in r['key'] for r in specs))

    def test_mcse_matches_direct_complete_case_formula(self):
        a=np.random.default_rng(13).normal(size=(5,3))
        result=conditional_summary({'x':a,'y':a+2})
        self.assertAlmostEqual(result['mean'],a.mean()+1)
        self.assertAlmostEqual(result['mcse']**2,np.var(a.mean(1),ddof=1)/5/2)
        self.assertEqual(result['mean'],paired_summary({'x':a,'y':a+2})['mean'])
        self.assertNotIn('mcse_model_repeats',result)

    def test_conditional_weights_and_missing_source_are_explicit(self):
        a=np.array([[.2,.4,np.nan],[.9,np.nan,np.nan],[np.nan]*3,[.1,.3,.5],[.6]*3])
        est=conditional_summary({'x':a})
        self.assertAlmostEqual(est['mean'],np.nanmean(a))
        n=np.isfinite(a).sum(1); x=np.nansum(a,axis=1); m=np.nanmean(a)
        self.assertAlmostEqual(est['mcse']**2,5/4*np.sum((x-m*n)**2)/n.sum()**2)
        missing=conditional_summary({'x':a,'y':np.full((5,3),np.nan)})
        self.assertIsNone(missing['mean']); self.assertIsNone(missing['mcse'])
        self.assertEqual(missing['sources_with_valid_answers'],1)

    def test_failure_never_becomes_plugin_or_median(self):
        good='{"rho_2":0.4,"rho_3":0.3,"rho_4":0.2,"rho_5":0.1}'
        for record in ({'started':True,'terminal':True,'final_text':'oops'},
                       {'started':True,'terminal':True,'final_text':good,'technical_error':True},
                       {'started':True,'terminal':True,'final_text':good,'refusal':True}):
            out=resolve({'D_obs':5},[.9]*4,record)
            self.assertIsNone(out['prediction']); self.assertIsNone(out['replacement'])
            self.assertTrue(out['terminal']); self.assertFalse(out['valid'])
        self.assertIsNone(resolve({'D_obs':0},[.9]*4)['prediction'])


class TransportRegression(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.out=pathlib.Path(self.tmp.name)

    def ledger(self,requests): return Ledger(self.out,requests,{'test':'mock only'})

    def test_release_blocks_before_network(self):
        with self.assertRaises((ValueError,FileNotFoundError)):
            validate_release({'authorized':False},self.out,self.out/'none')

    def test_status_is_readable_while_writer_lock_is_held(self):
        import subprocess
        from main_experiment.integrity import exclusive
        ledger=self.ledger([request()]); ledger.admit(['r1'])
        script=pathlib.Path(__file__).resolve().parents[2]/'scripts/run_main_api.py'
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


class EvaluationIntegration(unittest.TestCase):
    def test_all_invalid_incomplete_and_mixed_validity_without_imputation(self):
        from evaluate_main_responses import evaluate
        from main_experiment.requests import planned
        from main_experiment.observation import messages
        from main_experiment.common import CURRENT_RUN, CURRENT_REVISION
        root=pathlib.Path(__file__).resolve().parents[2]
        current=root/CURRENT_RUN
        prim=root/CURRENT_REVISION/'primary_baselines.json'
        if not prim.exists(): self.skipTest('current offline fixture absent')
        with tempfile.TemporaryDirectory(suffix='_mock') as td:
            run=pathlib.Path(td)/'run'; run.mkdir()
            bs=read_json(prim); selected={}; obs=[]
            for path in sorted((current/'observations/sample').glob('dar_a0_r*__R-c10__*.json')):
                row=read_json(path); obs.append(row)
                write_json(run/'observations/sample'/path.name,row)
                selected[row['id']]={**bs['observations'][row['id']],'block_sha256':row['block_sha256']}
            write_json(run/'report.json',{'design_version':DESIGN_VERSION})
            write_json(run/'baselines.json',{'design_version':DESIGN_VERSION,'observations':selected})
            requests=planned(obs)
            (run/'requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in requests))
            records=[{'id':r['id'],'prompt_sha256':r['prompt_sha256'],'payload_sha256':r['payload_sha256'],
                      'started':True,'terminal':True,'mock':True,'final_text':'invalid'} for r in requests]
            response=run/'mock.jsonl'; response.write_text(''.join(json.dumps(r)+'\n' for r in records))
            out=run/'eval_mock'; evaluate(run,response,out,True,run/'baselines.json')
            self.assertTrue(read_json(out/'report.json')['complete_main_result'])
            with open(out/'summary.csv') as f: rows=list(csv.DictReader(f))
            self.assertTrue(all(float(r['valid_fraction'])==0 and r['AE2']=='' for r in rows))
            with open(out/'answer_errors.csv') as f: answers=list(csv.DictReader(f))
            self.assertTrue(all(r['prediction_json']=='' and r['replacement']=='' for r in answers))
            records[0]['terminal']=False
            response.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError): evaluate(run,response,out,True,run/'baselines.json')
            evaluate(run,response,run/'pending_mock',True,run/'baselines.json')
            self.assertFalse(read_json(run/'pending_mock/report.json')['complete_main_result'])
            # Good outputs have matched baseline values; invalid ones never do.
            for i,r in enumerate(records):
                r['terminal']=True
                if i%3: r['final_text']='{"rho_2":0.4,"rho_3":0.3,"rho_4":0.2,"rho_5":0.1}'
            response.write_text(''.join(json.dumps(r)+'\n' for r in records))
            evaluate(run,response,run/'mixed_mock',True,run/'baselines.json')
            with open(run/'mixed_mock/answer_errors.csv') as f: answers=list(csv.DictReader(f))
            for r in answers:
                self.assertEqual(r['baseline_AE2_matched']=='',r['valid']=='False')
            # Missing hashes are rejected, not accepted as unknown provenance.
            records[0].pop('payload_sha256')
            response.write_text(''.join(json.dumps(r)+'\n' for r in records))
            with self.assertRaises(ValueError): evaluate(run,response,run/'bad_mock',True,run/'baselines.json')
