"""Configuration, paths, content-addressed task outputs and the per-observation cache.

Every task output lives in WORK/<stage>/<name>/<key16>, where the key hashes the
stage code, the relevant configuration, the task parameters, the frozen v10/v11
input manifests and the keys of all upstream tasks. A directory with DONE.json is
complete and is never recomputed; a changed input yields a new key and directory.
"""
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import shutil
import socket
import time
import yaml
from main_experiment.common import ROOT, digest, read_json, sha, write_json

CFG = yaml.safe_load((ROOT/'config/v11_ext.yaml').read_text())
VERSION = CFG['version']
NEW_SOURCES = tuple(CFG['new_sources'])                     # the four additional real test sources
NEW_FOLD = {k: v['fold'] for k, v in CFG['new_sources'].items()}
RADOSLAW = 'nr_radoslaw_email'
ARMS = ('R', 'S', 'S_obs', 'H', 'B')
RH_ARMS = ('R', 'H')


def cluster_path(name):
    ws = Path(os.environ.get('V11EXT_WS', CFG['cluster']['workspace']))
    return ws/CFG['cluster'][name] if name != 'workspace' else ws


WS = cluster_path('workspace')
V10 = cluster_path('v10')                                  # sealed v10 preparation, pool and ET
RH = cluster_path('rh')                                    # completed v11 R/H panel-release run
EXT = cluster_path('ext')
WORK = Path(os.environ.get('V11EXT_WORK', EXT/'work'))
RAW = Path(os.environ.get('V11EXT_RAW', EXT/'raw'))


def stream_domain(kind, k):
    """Random-stream domain of ET replicate k (replicate 0 keeps the v10/v11 domain)."""
    base = {'training': 'training', 'pool_train': 'pool_train',
            'et_nested': 'v10_et_nested', 'et_final': 'v10_et_final'}[kind]
    return base if k == 0 else CFG['stream_domains'][kind].format(k=k)


# ---------------------------------------------------------------- hashing
STAGE_CODE = {
    'source': ['v11_ext/core.py', 'v11_ext/observe.py', 'v11_ext/sources.py'],
    'testset': ['v11_ext/core.py', 'v11_ext/observe.py', 'v11_ext/replicates.py'],
    'draw_real': ['v11_ext/core.py', 'v11_ext/observe.py', 'v11_ext/replicates.py'],
    'draw_pool': ['v11_ext/core.py', 'v11_ext/observe.py', 'v11_ext/replicates.py'],
    'select': ['v11_ext/core.py', 'v11_ext/observe.py', 'v11_ext/replicates.py', 'v11_ext/et.py', '../scripts/build_v10_et.py'],
    'train': ['v11_ext/core.py', 'v11_ext/observe.py', 'v11_ext/replicates.py', 'v11_ext/et.py', '../scripts/build_v10_et.py'],
    'qwen_bundle': ['v11_ext/core.py', 'v11_ext/qwen.py', '../scripts/run_qwen_engine.py'],
    'report': ['v11_ext/core.py', 'v11_ext/qwen.py', 'v11_ext/report.py'],
}
CONFIG_KEYS = ('version', 'stream_domains', 'pool_partition', 'pool_chunk', 'windows', 'new_sources')


def code_hash(stage):
    src = ROOT/'src'
    files = sorted((src/'main_experiment').glob('*.py')) + sorted((src/'main_experiment').glob('*.cpp'))
    files += sorted((ROOT/'config/main_experiment').glob('*.txt'))
    files += [src/p for p in STAGE_CODE[stage]]
    return digest({str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in
                   (f.resolve() for f in files)})


def frozen_inputs():
    """Identity of the sealed v10/v11 artifacts every stage reads (not their full bytes)."""
    return {'v10_checksums': sha(V10/'prepared/checksums.json'),
            'v10_et_rows': sha(V10/'et/rows.json'),
            'rh_bundle': sha(RH/'BUNDLE_SHA256SUMS'),
            'rh_et_rows': sha(RH/'et_run/et/rows.json')}


@dataclass
class Task:
    stage: str
    name: str
    params: dict
    deps: list = field(default_factory=list)
    cpus: int = 1
    mem_gb: int = 8
    minutes: int = 30          # requested wall time
    estimate: float = 5.       # expected minutes, for the dry run
    key: str = ''

    def finalize(self, frozen):
        self.key = digest({'stage': self.stage, 'name': self.name, 'params': self.params,
                           'code': code_hash(self.stage), 'config': {k: CFG[k] for k in CONFIG_KEYS},
                           'frozen': frozen, 'deps': sorted(d.key for d in self.deps)})
        return self

    @property
    def out(self):
        return WORK/self.stage/self.name.replace(':', '__')/self.key[:16]

    @property
    def done(self):
        return (self.out/'DONE.json').exists()


def run(task, fn):
    """Run one task idempotently: skip if done, otherwise compute into a scratch
    directory and publish it atomically with its DONE marker."""
    if task.done:
        print('SKIP', task.name, task.key[:16], flush=True)
        return
    for dep in task.deps:
        if not dep.done: raise RuntimeError(f'{task.name}: upstream {dep.name} is incomplete')
    scratch = task.out.with_name(task.out.name+f'.tmp{os.environ.get("SLURM_JOB_ID", os.getpid())}')
    if scratch.exists(): shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    started = time.time()
    fn(task, scratch, {d.name: d.out for d in task.deps})
    write_json(scratch/'DONE.json', {'key': task.key, 'stage': task.stage, 'name': task.name,
                                     'params': task.params, 'seconds': time.time()-started,
                                     'host': socket.gethostname(), 'slurm_job': os.environ.get('SLURM_JOB_ID'),
                                     'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())})
    if task.out.exists():
        raise RuntimeError(f'{task.out} appeared concurrently; refusing to overwrite')
    scratch.rename(task.out)
    print('DONE', task.name, task.key[:16], f'{time.time()-started:.1f}s', flush=True)


# ---------------------------------------------------------------- per-observation cache
class Cache:
    """Content-addressed JSON cache keyed by (kind, observation block hash).

    Values are deterministic functions of the block (features, MLE fit), so a
    concurrent writer can only write identical bytes; writes are atomic.
    """

    def __init__(self, kind, root=None):
        self.root = Path(root or WORK/'cache')/kind
        self.hits = self.misses = 0

    def get(self, block_sha, compute):
        path = self.root/block_sha[:2]/f'{block_sha}.json'
        if path.exists():
            self.hits += 1
            return read_json(path)['value']
        self.misses += 1
        value = compute()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps({'block_sha256': block_sha, 'value': value}, allow_nan=False))
        tmp.replace(path)
        return value
