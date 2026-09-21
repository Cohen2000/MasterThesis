#!/usr/bin/env python3
"""Download tokenizer artifacts only; never model weights or inference."""
from pathlib import Path
import sys
import os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT, sha, write_json
from huggingface_hub import snapshot_download

out=ROOT/'data/tokenizers'; out.mkdir(parents=True,exist_ok=True)
for name,repo,revision in [
 ('qwen','Qwen/Qwen3.6-35B-A3B','995ad96eacd98c81ed38be0c5b274b04031597b0'),
 ('deepseek','deepseek-ai/DeepSeek-V4.1-Flash','dba1be0a40aa45a94ad051997016db3960a90277')]:
    dest=out/name
    snapshot_download(repo,revision=revision,local_dir=dest,
                      allow_patterns=['tokenizer.json','tokenizer_config.json','special_tokens_map.json',
                                      'chat_template.jinja','vocab.json','merges.txt','config.json'])
    write_json(dest/'provenance.json',{'repo':repo,'revision':revision,
               'files':{p.name:sha(p) for p in dest.iterdir() if p.is_file() and p.name!='provenance.json'}})
os.environ['TIKTOKEN_CACHE_DIR']=str(out/'tiktoken')
import tiktoken
tiktoken.get_encoding('o200k_base')
