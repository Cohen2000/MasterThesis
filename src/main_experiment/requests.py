"""Frozen request construction; transport lives in execution.py."""
from .common import CONFIGS, LLM_REPEATS, ARM_ID, DESIGN_VERSION, PREVIOUS_DESIGN_VERSION, seed, digest

QWEN='Qwen/Qwen3.6-35B-A3B'
REVISION='995ad96eacd98c81ed38be0c5b274b04031597b0'

def generation_version(arm):
    return DESIGN_VERSION if arm=='S' else PREVIOUS_DESIGN_VERSION if arm=='H' else 'cells10-json-20260918'


def payload(config,messages,request_seed,version=DESIGN_VERSION):
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
    # Generic JSON only, matching the API configurations. No task-specific grammar.
    return {'model':QWEN,'messages':messages,'max_tokens':258048,
            'temperature':1. if thinking else .7,'top_p':.95 if thinking else .80,
            'top_k':20,'min_p':0.,'presence_penalty':1.5,'repetition_penalty':1.,
            'chat_template_kwargs':{'enable_thinking':thinking},'seed':request_seed,
            'structured_output':{'json_object':True,'reasoning_parser':'qwen3',
                                 'applies':'after reasoning end'},
            'design_version':version,
            'executed_transport':'vllm offline engine (LLM.enqueue + LLMEngine.step)',
            'executed_streaming':False}


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
                for repeat in range(1,LLM_REPEATS+1): queue.append((obs,repeat))
            queues.append(queue)
        for turn in range(max(map(len,queues),default=0)):
            for queue in queues:
                if turn>=len(queue): continue
                obs,repeat=queue[turn]
                for config in CONFIGS:
                    # R, S and B keep their letters, so their seeds are those of the
                    # previous design; the new H has its own identifier.
                    version=generation_version(obs['arm'])
                    s=seed('llm',obs['graph_id'],ARM_ID[obs['arm']],obs['sample_index'],repeat,config+':'+version)
                    rid=f'{obs["id"]}__{config}__r{repeat}__{version}'
                    records.append({'id':rid,'observation_id':obs['id'],'graph_id':obs['graph_id'],
                        'arm':obs['arm'],'sample_index':obs['sample_index'],'repeat_index':repeat,
                        'config_id':config,'stratum':stratum,'seed':s,
                        'status':'skipped_empty' if obs['empty'] else 'not_started',
                        'started':False,'mock':False,'prompt_sha256':obs['prompt_sha256'],
                        'payload':payload(config,obs['messages'],s,version),
                        # Qwen is authorised for this study; the paid providers
                        # are planned but not released.
                        'production_dispatch_enabled':config.startswith('qwen'),
                        'requires_technical_release':not config.startswith('qwen')})
    for record in records:
        record['design_version']=DESIGN_VERSION
        record['payload_sha256']=digest(record['payload'])
    return records


def retry_decision(attempt,model_tokens,status,transient=False,ambiguous=False,retry_after=0):
    # One generation attempt. Ambiguous delivery must be reconciled, never resent.
    if ambiguous: return {'action':'reconcile','delay':None}
    if status in (400,401,403) or status in ('ignored_parameter','repeated_oom'):
        return {'action':'stop_configuration','delay':None}
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
 'dispatch_enabled':True,'inference_authorized':['qwen_thinking','qwen_nonthinking'],
 'paid_providers_authorized':False,
 'connect_timeout_seconds':30,'first_model_token_seconds':1800,
 'no_model_progress_seconds':3600,'active_request_seconds':86400,
 'sdk_retries':0,'http_read_timeout':None,'max_attempts':1,
 'retry_delays_seconds':[],'honor_retry_after':True,
 'automatic_generation_retry':False,'reconcile_ambiguous_before_retry':True,
 'no_retry_after_model_output':True,'no_continuation_after_output_limit':True,
 'persist_each_attempt_and_stream_fragment':True,
 'sol':{'transport':'Responses Batch','batch_max_requests':64,'batch_window':'24h',
        'regular_cap_usd':180,'total_cap_usd':200,'per_open_request_reserve_usd':1.30,
        'price_recheck_before_dispatch':True},
 'deepseek':{'max_active_requests':1,'total_cap_usd':50,'per_open_request_reserve_usd':.48,
             'price_recheck_before_dispatch':True},
 # Verified on the cluster, not assumed: vLLM 0.11 (the newest release resolvable
 # against Python 3.9) does not know this architecture; 0.29.0 does, and the model
 # card asks for >= 0.19.0. The environment is pinned in
 # $WS/mainexp/requirements.pinned.txt.
 'qwen':{'revision':REVISION,'tokenizer_revision':REVISION,
         'architecture':'Qwen3_5MoeForConditionalGeneration',
         'vllm_version':'0.29.0','transformers_version':'5.17.0',
         'torch_version':'2.13.0+cu130','dtype':'bfloat16',
         'serving':'vllm offline engine (no HTTP server); generic JSON object after reasoning (reasoning_parser qwen3)',
         'required_gpus':'1 x H100 94GB','max_model_len':262144,
         'max_output_tokens':258048,'gpu_memory_utilization':.90,
         'reasoning_split':'<think>...</think> markers of the pinned chat template',
         'server_seed':20260916,'language_model_only':True,'yarn':False,
         'sampling_source':'official model card, Qwen3.6-35B-A3B',
         'thinking':{'temperature':1.0,'top_p':.95,'top_k':20,'min_p':0.,
                     'presence_penalty':1.5,'repetition_penalty':1.0},
         'nonthinking':{'temperature':.7,'top_p':.80,'top_k':20,'min_p':0.,
                        'presence_penalty':1.5,'repetition_penalty':1.0}},
 'order':'real block, synthetic block; cycle graph x arm cells; sample index then repeat',
 'stop_configuration':['400','401','ignored_required_parameter','repeated_oom','confirmed_model_change'],
 'record_provider_metadata':['UTC','returned_model','system_fingerprint','usage','finish_reason','reasoning'],
 'smoke_tests_executed':0,
}
