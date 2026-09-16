from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REAL_TEST = ('sp_hospital','sp_highschool2013','copenhagen_bluetooth',
             'snap_email_eu','snap_collegemsg','snap_mathoverflow')
TRAIN = ('sp_hospital','sp_primaryschool','sp_highschool2013','sp_workplace',
         'sp_hypertext2009','snap_collegemsg','snap_email_eu','snap_mathoverflow',
         'snap_bitcoin_otc','nr_radoslaw_email','nr_digg_reply','jodie_wikipedia',
         'jodie_reddit','jodie_lastfm','jodie_mooc','copenhagen_bluetooth')
ARMS = ('R','S','H','B')
CONFIGS = ('sol','deepseek','qwen_thinking','qwen_nonthinking')
SYNTH = tuple(f'{family}_{mode}_r{r}' for family,modes in
              [('dar',('a0','a08')),('ad',('memoryless','memory'))]
              for r in (1,2) for mode in modes)
SEEDS = {}

def seed(domain, graph_id='', arm_id='', sample_index=0, repeat_index=0, config_id=''):
    fields=[20260916,domain,graph_id,arm_id,sample_index,repeat_index,config_id]
    s=int.from_bytes(hashlib.sha256(json.dumps(fields,separators=(',',':'),ensure_ascii=False).encode()).digest()[:8],'big') % 2**63
    key=json.dumps(fields,separators=(',',':'))
    if s in SEEDS and SEEDS[s]!=key:
        raise RuntimeError('seed collision')
    SEEDS[s]=key
    return s

def rng(*args):
    return np.random.Generator(np.random.PCG64(seed(*args)))

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def write_json(path,x):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'w') as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def read_json(path):
    return json.loads(Path(path).read_text())

def atomic_npz(path,**arrays):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    with open(tmp,'wb') as f:
        np.savez_compressed(f,**arrays); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)

def verify_immutable_checkpoints(out):
    """Reject altered completed inputs while allowing mutable timing/status reports.

    Newly committed checkpoints after an interrupted run may not yet be listed;
    stage files themselves are atomically written. Full audit checks every file.
    """
    out=Path(out)
    if not (out/'checksums.json').exists(): return
    for name,expected in read_json(out/'checksums.json').items():
        p=Path(name)
        immutable=p.parts[0] in ('graphs','observations','models') or (
            p.parts[0]=='calibration' and (p.name.startswith(('prefix_','validation_')) or p.name=='calibrated.json'))
        if immutable and sha(out/p)!=expected:
            raise ValueError(f'completed checkpoint checksum mismatch: {name}')
