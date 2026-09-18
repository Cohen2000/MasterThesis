"""Single-attempt paid transports with durable state, raw data and cost reserves.

No network work at import. Callers must validate an explicit release before
constructing HTTP. One writer holds the dispatch-directory lock for its lifetime.
Interrupted POSTs are never repeated. OpenAI batch retrieval is safely repeatable.
"""
import json
import math
import os
import time
import http.client
from pathlib import Path
from datetime import datetime,timezone
from .common import read_json,write_json,sha,digest,DESIGN_VERSION
from .integrity import bind,validate_request,code_binding

CAP={'sol':180_000_000,'deepseek':50_000_000}  # micro-USD; no retry budget in v2
RESERVE={'sol':1_300_000,'deepseek':480_000}
# USD / million tokens == micro-USD / token; uncached conservative rates.
PRICES={'sol':{'input':2.,'output':10.},'deepseek':{'input':.30,'output':1.20}}


def utc(): return datetime.now(timezone.utc).isoformat()


def release_template(run,baselines):
    return {'authorized':False,'design_version':DESIGN_VERSION,
            'requests_sha256':sha(Path(run)/'requests.jsonl'),'baselines_sha256':sha(baselines),
            'prices_usd_per_million':PRICES,'price_checked_utc':None,'expires_utc':None,
            'accepted_returned_models':{'sol':[],'deepseek':[]},
            'technical_smoke_evidence':None}


def validate_release(release,run,baselines):
    expected=release_template(run,baselines)
    if release.get('authorized') is not True: raise ValueError('API dispatch not authorized')
    for k in ('design_version','requests_sha256','baselines_sha256','prices_usd_per_million'):
        if release.get(k)!=expected[k]: raise ValueError(f'release mismatch: {k}')
    now=datetime.now(timezone.utc)
    for field in ('price_checked_utc','expires_utc'):
        stamp=datetime.fromisoformat(release[field])
        if stamp.tzinfo is None: raise ValueError('release timestamps need timezone')
        age=(now-stamp).total_seconds()
        if (field=='price_checked_utc' and not 0<=age<=86400) or (field=='expires_utc' and age>=0):
            raise ValueError('release expired or price check stale')
    evidence=release.get('technical_smoke_evidence')
    if not evidence or not Path(evidence['path']).is_file() or sha(evidence['path'])!=evidence['sha256']:
        raise ValueError('verified technical smoke evidence required')
    for provider in CAP:
        if not release.get('accepted_returned_models',{}).get(provider): raise ValueError('returned model IDs not released')


class Ledger:
    def __init__(self,out,requests,inputs):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.requests={r['id']:r for r in requests}
        if len(self.requests)!=len(requests): raise ValueError('duplicate requests')
        for r in requests: validate_request(r)
        bind(self.out/'inputs.json',{'requests':digest(requests),'inputs':inputs,'code':code_binding()})
        self.path=self.out/'ledger.json'
        self.state=read_json(self.path) if self.path.exists() else {'requests':{},'batches':{},'halted':{}}
        self.save()

    def save(self): write_json(self.path,self.state)

    def totals(self,provider):
        rows=[x for x in self.state['requests'].values() if x['provider']==provider]
        return sum(x['charged_upper_micro_usd'] for x in rows),sum(x['reserved_micro_usd'] for x in rows)

    def admit(self,ids,batch=None):
        if not ids or len(ids)!=len(set(ids)): raise ValueError('empty or duplicate admission')
        providers={self.requests[i]['config_id'] for i in ids}
        if len(providers)!=1: raise ValueError('mixed providers')
        provider=providers.pop()
        if provider not in CAP: raise ValueError('paid provider required')
        if provider in self.state['halted']: raise ValueError('configuration halted: '+self.state['halted'][provider])
        if any(i in self.state['requests'] for i in ids): raise ValueError('request already admitted; no retry')
        if any(x['provider']==provider and x['status'] in ('reconcile','dispatching') for x in self.state['requests'].values()):
            raise ValueError('ambiguous earlier request; reconcile before dispatch')
        paid,reserved=self.totals(provider)
        if paid+reserved+len(ids)*RESERVE[provider]>CAP[provider]: raise ValueError('cost cap; run remains incomplete')
        for rid in ids:
            self.state['requests'][rid]={'provider':provider,'status':'dispatching','batch':batch,
                'admitted_utc':utc(),'reserved_micro_usd':RESERVE[provider],'charged_upper_micro_usd':0}
        if batch: self.state['batches'][batch]={'ids':ids,'status':'preparing','provider_id':None}
        self.save()  # durable BEFORE upload / generation POST

    def finish(self,rid,record,usage=None,no_charge=False):
        row=self.state['requests'][rid]; r=self.requests[rid]
        result={'id':rid,'started':True,'terminal':True,'mock':False,
                'prompt_sha256':r['prompt_sha256'],'payload_sha256':r['payload_sha256'],
                'utc':utc(),**record,'usage':usage}
        cost=None
        if usage:
            p=row['provider']; ik='input_tokens' if p=='sol' else 'prompt_tokens'
            ok='output_tokens' if p=='sol' else 'completion_tokens'
            if ik in usage and ok in usage:
                if any(type(usage[k]) is not int or usage[k]<0 for k in (ik,ok)): raise ValueError('invalid usage')
                cost=math.ceil(usage[ik]*PRICES[p]['input']+usage[ok]*PRICES[p]['output'])
        if no_charge: cost=0
        write_json(self.out/'responses'/f'{rid}.json',result)
        row['response_sha256']=sha(self.out/'responses'/f'{rid}.json')
        row['status']='terminal' if cost is not None else 'reconcile'
        if cost is not None:
            row['charged_upper_micro_usd']=cost; row['reserved_micro_usd']=0
            if cost>RESERVE[row['provider']]: self.state['halted'][row['provider']]='usage exceeded reserved upper bound'
        self.save()

    def export(self,path):
        records=[]
        for rid,row in self.state['requests'].items():
            p=self.out/'responses'/f'{rid}.json'
            if p.exists():
                if row.get('response_sha256')!=sha(p): raise ValueError('response checksum mismatch')
                records.append(read_json(p))
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_suffix('.tmp')
        with open(tmp,'w') as f:
            for record in records: f.write(json.dumps(record,sort_keys=True)+'\n')
            f.flush(); os.fsync(f.fileno())
        tmp.replace(target)
        return len(records)


class HTTP:
    """No SDK retries or redirects; keys are only read after release validation."""
    def __init__(self,provider):
        self.host={'sol':'api.openai.com','deepseek':'api.deepseek.com'}[provider]
        self.key=os.environ[{'sol':'OPENAI_API_KEY','deepseek':'DEEPSEEK_API_KEY'}[provider]]

    def open(self,method,path,body=None,content_type='application/json'):
        conn=http.client.HTTPSConnection(self.host,timeout=30)
        headers={'Authorization':'Bearer '+self.key,'Content-Type':content_type}
        if isinstance(body,dict): body=json.dumps(body,allow_nan=False).encode()
        conn.connect()
        # Response headers/first byte can be delayed by provider queuing.
        conn.sock.settimeout(1800)
        conn.request(method,path,body,headers)
        return conn,conn.getresponse()

    def request(self,method,path,body=None,content_type='application/json'):
        conn,response=self.open(method,path,body,content_type)
        try:
            data=response.read()
            if not 200<=response.status<300:
                raise RuntimeError(f'provider HTTP {response.status}: {data[:500].decode(errors="replace")}')
            return data
        finally: conn.close()

    def json(self,method,path,body=None): return json.loads(self.request(method,path,body))

    def upload(self,data):
        boundary='study-'+digest(data.decode())[:24]
        body=(f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\nbatch\r\n'
              f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="requests.jsonl"\r\n'
              'Content-Type: application/jsonl\r\n\r\n').encode()+data+f'\r\n--{boundary}--\r\n'.encode()
        return json.loads(self.request('POST','/v1/files',body,'multipart/form-data; boundary='+boundary))


def parse_deepseek(lines):
    """Reconstruct persisted SSE. Reasoning is preserved, never scored as final text."""
    final=[]; reasoning=[]; usage=None; finish=None; model=None; provider_id=None; fingerprint=None; done=False
    for line in lines:
        text=line.decode() if isinstance(line,bytes) else line
        if not text.startswith('data:'): continue
        data=text[5:].strip()
        if data=='[DONE]': done=True; break
        if not data: continue
        chunk=json.loads(data)
        if chunk.get('error'): raise ValueError('provider stream error')
        model=chunk.get('model',model); provider_id=chunk.get('id',provider_id)
        fingerprint=chunk.get('system_fingerprint',fingerprint)
        if chunk.get('usage'): usage=chunk['usage']
        for choice in chunk.get('choices',[]):
            if choice.get('index',0)!=0: raise ValueError('unexpected extra choice')
            delta=choice.get('delta',{})
            final.append(delta.get('content') or '')
            reasoning.append(delta.get('reasoning_content') or '')
            finish=choice.get('finish_reason') or finish
    return {'final_text':''.join(final),'reasoning_text':''.join(reasoning),'returned_model':model,
            'provider_id':provider_id,'system_fingerprint':fingerprint,'finish_reason':finish,
            'limit_hit':finish=='length','technical_error':not(done and finish in ('stop','length'))},usage


def run_deepseek(ledger,rid,http,accepted):
    ledger.admit([rid]); r=ledger.requests[rid]
    raw=ledger.out/'raw'/f'{rid}.sse'; raw.parent.mkdir(parents=True,exist_ok=True)
    conn=None; error=None
    try:
        start=time.monotonic(); last=None
        conn,response=http.open('POST','/chat/completions',r['payload'])
        write_json(raw.with_suffix('.headers.json'),{'status':response.status,'headers':dict(response.getheaders())})
        if response.status!=200:
            raw.write_bytes(response.read())
            ledger.finish(rid,{'final_text':'','technical_error':True,'http_status':response.status},
                          no_charge=response.status in (400,401,403,404,422,429))
            if response.status in (400,401,403,404,422):
                ledger.state['halted']['deepseek']=f'HTTP {response.status}'; ledger.save()
            return
        with open(raw,'xb') as f:
            while True:
                now=time.monotonic()
                remaining=min(86400-(now-start),1800-(now-start) if last is None else 3600-(now-last))
                if remaining<=0: raise TimeoutError('model progress deadline')
                # HTTPResponse may own the socket after Connection: close.
                sock=conn.sock or response.fp.raw._sock
                sock.settimeout(remaining)
                line=response.readline()
                if not line: break
                f.write(line); f.flush(); os.fsync(f.fileno())
                if line.startswith(b'data:'):
                    data=line[5:].strip()
                    if data==b'[DONE]': break
                    if data:
                        chunk=json.loads(data)
                        if any(c.get('delta',{}).get(k) for c in chunk.get('choices',[]) for k in ('content','reasoning_content')):
                            last=time.monotonic()
    except Exception as exc:
        error=type(exc).__name__+': '+str(exc)
    finally:
        if conn: conn.close()
    try: record,usage=parse_deepseek(raw.read_bytes().splitlines()) if raw.exists() else ({'final_text':'','technical_error':True},None)
    except (ValueError,UnicodeError): record,usage={'final_text':'','technical_error':True},None
    if error: record.update(technical_error=True,transport_error=error)
    if record.get('returned_model') not in accepted:
        record['technical_error']=True
        ledger.state['halted']['deepseek']='unreleased returned model'
    record['raw_sha256']=sha(raw) if raw.exists() else None
    ledger.finish(rid,record,usage)


def submit_batch(ledger,ids,http):
    bid=digest([ledger.requests[i]['payload_sha256'] for i in ids])
    ledger.admit(ids,bid)
    batch=ledger.state['batches'][bid]
    data=''.join(json.dumps({'custom_id':i,'method':'POST','url':'/v1/responses',
                            'body':ledger.requests[i]['payload']})+'\n' for i in ids).encode()
    raw=ledger.out/'raw'/bid; raw.mkdir(parents=True,exist_ok=True)
    (raw/'input.jsonl').write_bytes(data)
    try:
        uploaded=http.upload(data); write_json(raw/'upload.json',uploaded)
        batch.update(input_file_id=uploaded['id'],status='creating'); ledger.save()
        result=http.json('POST','/v1/batches',{'input_file_id':uploaded['id'],'endpoint':'/v1/responses',
                         'completion_window':'24h','metadata':{'study_batch':bid}})
        write_json(raw/'create.json',result)
        batch.update(provider_id=result['id'],status='submitted')
        for rid in ids: ledger.state['requests'][rid]['status']='submitted'
    except Exception as exc:
        batch.update(status='reconcile',error=type(exc).__name__+': '+str(exc))
        for rid in ids: ledger.state['requests'][rid]['status']='reconcile'
    ledger.save()
    return bid


def response_from_batch(line):
    response=line.get('response') or {}; body=response.get('body') or {}
    texts=[]; refusals=[]
    for item in body.get('output',[]):
        if item.get('type')!='message': continue
        for content in item.get('content',[]):
            if content.get('type')=='output_text': texts.append(content.get('text',''))
            if content.get('type')=='refusal': refusals.append(content.get('refusal',''))
    return {'final_text':''.join(texts),'refusal':bool(refusals),
            'technical_error':response.get('status_code')!=200 or not (body.get('status')=='completed' or (body.get('status')=='incomplete' and (body.get('incomplete_details') or {}).get('reason')=='max_output_tokens')),
            'returned_model':body.get('model'),'provider_id':body.get('id'),
            'provider_request_id':response.get('request_id'),'system_fingerprint':body.get('system_fingerprint'),
            'finish_reason':body.get('status'),'limit_hit':(body.get('incomplete_details') or {}).get('reason')=='max_output_tokens',
            'provider_error':line.get('error') or body.get('error')},body.get('usage')


def collect_batch(ledger,bid,http,accepted):
    batch=ledger.state['batches'][bid]
    raw=ledger.out/'raw'/bid
    if not batch.get('provider_id'):
        # Reconcile an ambiguous create by its persisted metadata, never another POST.
        matches=[]; after=''
        while True:
            page=http.json('GET','/v1/batches?limit=100'+('&after='+after if after else ''))
            matches.extend(x for x in page['data'] if (x.get('metadata') or {}).get('study_batch')==bid)
            if not page.get('has_more'): break
            after=page['data'][-1]['id']
        if len(matches)!=1: raise ValueError('batch create unresolved; no resubmission permitted')
        batch['provider_id']=matches[0]['id']; ledger.save()
    status=http.json('GET','/v1/batches/'+batch['provider_id']); write_json(raw/'status.json',status)
    if status['status'] not in ('completed','expired','cancelled','failed'): return status['status']
    found={}
    for field in ('output_file_id','error_file_id'):
        if not status.get(field): continue
        content=http.request('GET','/v1/files/'+status[field]+'/content')
        (raw/(field+'.jsonl')).write_bytes(content)
        for line in content.splitlines():
            obj=json.loads(line); rid=obj['custom_id']
            if rid not in batch['ids'] or rid in found: raise ValueError('unexpected/duplicate batch result')
            found[rid]=obj
    for rid in batch['ids']:
        if ledger.state['requests'][rid]['status']=='terminal': continue
        line=found.get(rid)
        if line is None:
            ledger.finish(rid,{'final_text':'','technical_error':True,'provider_error':'missing batch result'},
                          no_charge=status['status']=='failed')
            continue
        record,usage=response_from_batch(line)
        if record.get('returned_model') and record['returned_model'] not in accepted:
            record['technical_error']=True; ledger.state['halted']['sol']='unreleased returned model'
        code=(line.get('error') or {}).get('code')
        record['raw_line_sha256']=digest(line)
        ledger.finish(rid,record,usage,no_charge=code in ('batch_expired','batch_cancelled'))
    batch['status']='collected'; ledger.save()
    return batch['status']
