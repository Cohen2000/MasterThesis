from pathlib import Path
import os
from .common import sha,read_json

# Input sizes checked for every main prompt; no prompt may exceed 4096 tokens.
TOKEN_COUNTS=('qwen_thinking','qwen_nonthinking','deepseek_message_texts','sol_o200k_message_proxy')

class TokenCounters:
    def __init__(self,folder):
        from transformers import AutoTokenizer
        from tokenizers import Tokenizer
        self.folder=Path(folder)
        for name in ['qwen','deepseek']:
            meta=read_json(self.folder/name/'provenance.json')
            for f,h in meta['files'].items():
                if sha(self.folder/name/f)!=h: raise ValueError('tokenizer checksum')
        self.qwen=AutoTokenizer.from_pretrained(self.folder/'qwen',local_files_only=True,trust_remote_code=False)
        self.deep=Tokenizer.from_file(str(self.folder/'deepseek/tokenizer.json'))
        os.environ['TIKTOKEN_CACHE_DIR']=str(self.folder/'tiktoken')
        import tiktoken
        self.sol=tiktoken.get_encoding('o200k_base')

    def count(self,messages):
        counts={}
        for thinking in [True,False]:
            name='qwen_thinking' if thinking else 'qwen_nonthinking'
            ids=self.qwen.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=thinking)
            token_ids=ids['input_ids'] if hasattr(ids,'keys') else ids
            if token_ids and isinstance(token_ids[0],list): raise ValueError('unexpected batched tokenization')
            rendered=self.qwen.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=thinking)
            reference=self.qwen.encode(rendered,add_special_tokens=False)
            if list(token_ids)!=reference: raise ValueError('chat-template tokenization mismatch')
            counts[name]=len(token_ids)
        counts['deepseek_message_texts']=sum(len(self.deep.encode(m['content']).ids) for m in messages)
        counts['sol_o200k_message_proxy']=sum(len(self.sol.encode(m['content'])) for m in messages)
        if max(counts[k] for k in TOKEN_COUNTS)>4096: raise ValueError('input exceeds 4096; no truncation allowed')
        return {**counts,'qwen_exact_template_checked':True,
                'deepseek_provider_template_verified':False,'sol_provider_template_verified':False,
                'api_template_note':'DeepSeek message text count; Sol tokenizer proxy. Provider framing requires later technical release.'}
