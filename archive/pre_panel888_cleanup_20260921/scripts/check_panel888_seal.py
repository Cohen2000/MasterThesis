#!/usr/bin/env python3
"""Fail closed before bundling if any sealed source or artifact has changed."""
import hashlib,json,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,sha,read_json
freeze=ROOT/'docs/results/panel888_20260921'
for name,h in read_json(freeze/'FREEZE.json')['evidence_sha256'].items():assert sha(freeze/name)==h,name
sources=read_json(freeze/'SOURCE_CHECKSUMS.json')
for name,h in sources.items():
    assert sha(ROOT/name)==h,name
    committed=subprocess.check_output(['git','show','HEAD:'+name],cwd=ROOT)
    assert hashlib.sha256(committed).hexdigest()==h,('uncommitted source',name)
for name,h in read_json(freeze/'ARTIFACT_CHECKSUMS.json').items():assert sha(ROOT/name)==h,name
print('SEALED_SOURCE_AND_ARTIFACTS_VERIFIED',subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
