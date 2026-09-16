"""Request construction and pure transport policies. No network client is exposed."""
from .common import CONFIGS, seed, digest

QWEN='Qwen/Qwen3.6-35B-A3B'
REVISION='995ad96eacd98c81ed38be0c5b274b04031597b0'

def payload(config,messages,request_seed):
    if config=='sol':
        return {'model':'gpt-5.6-sol','input':messages,'reasoning':{'effort':'high'},
                'text':{'format':{'type':'json_object'}},'max_output_tokens':128000}
    if config=='deepseek':
        return {'model':'deepseek-flash','messages':messages,'thinking':{'type':'enabled'},
                'reasoning_effort':'high','top_p':1.,'max_tokens':393216,
                'response_format':{'type':'json_object'},'stream':True,
                'stream_options':{'include_usage':True}}
    if config not in CONFIGS: raise ValueError(config)
    thinking=config=='qwen_thinking'
    return {'model':QWEN,'messages':messages,'max_tokens':258048,
            'temperature':1. if thinking else .7,'top_p':.95 if thinking else .80,
            'top_k':20,'min_p':0.,'presence_penalty':1.5,'repetition_penalty':1.,
            'chat_template_kwargs':{'enable_thinking':thinking},'seed':request_seed,
            'response_format':{'type':'json_object'},'stream':True,
            'stream_options':{'include_usage':True}}


def planned(observations):
    # Real block first, then synthetic; cyclic graph x arm cells, then sample/repeat.
    records=[]
    for stratum in ['real','synthetic']:
        cells={}
        for o in observations:
            if o['stratum']==stratum: cells.setdefault((o['graph_id'],o['arm']),[]).append(o)
        queues=[]
        for cell in sorted(cells):
            queue=[]
            for obs in sorted(cells[cell],key=lambda x:x['sample_index']):
                for repeat in range(1,4): queue.append((obs,repeat))
            queues.append(queue)
        for turn in range(max(map(len,queues),default=0)):
            for queue in queues:
                if turn>=len(queue): continue
                obs,repeat=queue[turn]
                for config in CONFIGS:
                    s=seed('llm',obs['graph_id'],obs['arm'],obs['sample_index'],repeat,config)
                    rid=f'{obs["id"]}__{config}__r{repeat}'
                    records.append({'id':rid,'observation_id':obs['id'],'graph_id':obs['graph_id'],
                        'arm':obs['arm'],'sample_index':obs['sample_index'],'repeat_index':repeat,
                        'config_id':config,'stratum':stratum,'seed':s,
                        'status':'skipped_empty' if obs['empty'] else 'not_started',
                        'started':False,'mock':False,'prompt_sha256':obs['prompt_sha256'],
                        'payload':payload(config,obs['messages'],s),
                        'production_dispatch_enabled':False,
                        'requires_technical_release':True})
    return records


def retry_decision(attempt,model_tokens,status,transient=False,ambiguous=False,retry_after=0):
    if ambiguous: return {'action':'reconcile','delay':None}
    if status in (400,401) or status in ('ignored_parameter','repeated_oom'):
        return {'action':'stop_configuration','delay':None}
    if model_tokens>0: return {'action':'terminal_no_retry','delay':None}
    if transient and attempt<3:
        return {'action':'retry','delay':max((5,20)[attempt-1],retry_after)}
    return {'action':'terminal_no_retry','delay':None}


def reserve_allowed(provider,paid,reserved,new_count,retry=False):
    if min(paid,reserved,new_count)<0: raise ValueError('negative ledger')
    if provider=='sol': cap=200 if retry else 180; unit=1.30
    elif provider=='deepseek': cap=50; unit=.48
    else: raise ValueError('not a paid provider')
    return paid+reserved+unit*new_count<=cap


def watchdog(active_seconds,first_token_seconds,last_token_age,model_tokens):
    if active_seconds>=86400: return 'active_deadline'
    if not model_tokens and first_token_seconds>=1800: return 'first_token_deadline'
    if model_tokens and last_token_age>=3600: return 'model_progress_deadline'
    return None

EXECUTION_POLICY={
 'dispatch_enabled':False,'inference_authorized':False,
 'connect_timeout_seconds':30,'first_model_token_seconds':1800,
 'no_model_progress_seconds':3600,'active_request_seconds':86400,
 'sdk_retries':0,'http_read_timeout':None,'max_attempts':3,
 'retry_delays_seconds':[5,20],'honor_retry_after':True,
 'retry_only_transient_before_model_output':True,'reconcile_ambiguous_before_retry':True,
 'no_retry_after_model_output':True,'no_continuation_after_output_limit':True,
 'persist_each_attempt_and_stream_fragment':True,
 'sol':{'transport':'Responses Batch','batch_max_requests':64,'batch_window':'24h',
        'regular_cap_usd':180,'total_cap_usd':200,'per_open_request_reserve_usd':1.30,
        'price_recheck_before_dispatch':True},
 'deepseek':{'max_active_requests':4,'total_cap_usd':50,'per_open_request_reserve_usd':.48,
             'price_recheck_before_dispatch':True},
 'qwen':{'max_active_requests_across_both_modes':4,'revision':REVISION,
         'tokenizer_revision':REVISION,'vllm_version':'0.20.1','dtype':'bfloat16',
         'tensor_parallel_size':2,'required_gpus':'2 x H100 80GB','max_model_len':262144,
         'max_num_seqs':4,'max_num_batched_tokens':8192,'enable_chunked_prefill':True,
         'gpu_memory_utilization':.90,'reasoning_parser':'qwen3','server_seed':20260916,
         'language_model_only':True,'generation_config':'vllm','yarn':False},
 'order':'real block, synthetic block; cycle graph x arm cells; sample index then repeat',
 'stop_configuration':['400','401','ignored_required_parameter','repeated_oom','confirmed_model_change'],
 'record_provider_metadata':['UTC','returned_model','system_fingerprint','usage','finish_reason','reasoning'],
 'smoke_tests_executed':0,
}
