#!/usr/bin/env python3
"""Sol / DeepSeek request ledger. `prepare` only writes the disabled ledger and a
release template; dispatch requires an explicit, validated technical release
(execution.validate_release) and is not part of this study's current run."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import PREPARED,REFERENCES,RESULTS,read_json,write_json,sha
from main_experiment.integrity import exclusive
from main_experiment.execution import (Ledger,HTTP,release_template,validate_release,
                                      run_deepseek,submit_batch,collect_batch,parse_deepseek)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare','status','export','dispatch','collect'])
    p.add_argument('--run',default=str(PREPARED)); p.add_argument('--baselines',default=str(REFERENCES/'primary_baselines.json'))
    p.add_argument('--out',default=str(RESULTS/'api')); p.add_argument('--provider',choices=['sol','deepseek'])
    p.add_argument('--release'); p.add_argument('--execute',action='store_true')
    p.add_argument('--limit',type=int,default=1,help='requests to admit; sol at most 64')
    a=p.parse_args(); run=Path(a.run); out=Path(a.out)
    if a.action=='status':
        # Atomic ledger snapshots remain readable while a long request holds the writer lock.
        from collections import Counter
        state=read_json(out/'ledger.json')
        totals={provider:{'charged_upper_micro_usd':sum(r['charged_upper_micro_usd'] for r in state['requests'].values() if r['provider']==provider),
                          'reserved_micro_usd':sum(r['reserved_micro_usd'] for r in state['requests'].values() if r['provider']==provider)}
                for provider in ('sol','deepseek')}
        print(json.dumps({'states':dict(Counter(r['status'] for r in state['requests'].values())),
                          'totals':totals,'halted':state['halted'],'batches':state['batches']},indent=2))
        return
    requests=[json.loads(x) for x in (run/'requests.jsonl').read_text().splitlines()
              if json.loads(x)['config_id'] in ('sol','deepseek')]
    with exclusive(out/'dispatch.lock'):
        ledger=Ledger(out,requests,{'requests_sha256':sha(run/'requests.jsonl'),
                                  'baselines_sha256':sha(a.baselines),'cli_sha256':sha(__file__)})
        if a.action=='prepare':
            target=out/'release.template.json'
            if not target.exists(): write_json(target,release_template(run,a.baselines))
            print('PREPARED_OFFLINE; API dispatch disabled; no network access'); return
        if a.action=='export':
            print('exported',ledger.export(out/'responses.jsonl')); return
        if not a.execute or not a.release or not a.provider:
            raise ValueError('network actions require --execute, --release and --provider')
        release=read_json(a.release); validate_release(release,run,a.baselines)
        from main_experiment.integrity import bind
        # Model allowlist and price identity cannot change within a dispatch directory.
        bind(out/'release_identity.json',{k:release[k] for k in ('design_version','requests_sha256',
             'baselines_sha256','prices_usd_per_million','accepted_returned_models','technical_smoke_evidence')})
        accepted=release['accepted_returned_models'][a.provider]
        if a.provider=='deepseek':
            for rid,row in list(ledger.state['requests'].items()):
                if row['provider']!='deepseek' or row['status']!='dispatching': continue
                raw=out/'raw'/f'{rid}.sse'
                try: record,usage=parse_deepseek(raw.read_bytes().splitlines())
                except (OSError,ValueError,UnicodeError): record,usage={'final_text':'','technical_error':True},None
                if record.get('returned_model') not in accepted: record['technical_error']=True
                record['recovered_after_interruption']=True
                ledger.finish(rid,record,usage)
        http=HTTP(a.provider)  # first access to key; no construction before release
        if a.action=='collect':
            if a.provider!='sol': raise ValueError('DeepSeek streams are collected during dispatch')
            for bid,b in ledger.state['batches'].items():
                if b['status']!='collected': print(bid,collect_batch(ledger,bid,http,accepted))
        else:
            if not 1<=a.limit<=64: raise ValueError('limit must be 1..64')
            todo=[r['id'] for r in requests if r['config_id']==a.provider and r['status']!='skipped_empty'
                  and r['id'] not in ledger.state['requests']][:a.limit]
            if a.provider=='sol':
                if todo: print('batch',submit_batch(ledger,todo,http))
            else:
                for rid in todo:
                    validate_release(release,run,a.baselines)
                    run_deepseek(ledger,rid,http,accepted)
        ledger.export(out/'responses.jsonl')
        print('totals_micro_usd',ledger.totals(a.provider))


if __name__=='__main__': main()
