"""Configuration, paths, content-addressed task outputs and the per-observation cache.

In plain words: the bookkeeping of stage 2. Every task writes into a folder whose name
is a fingerprint of everything it depends on, so a finished task is recognised and never
recomputed, and any change of code or input automatically leads to a fresh folder.

Every task output lives in WORK/<stage>/<name>/<key16>, where the key hashes the
stage code, the relevant configuration, the task parameters, the frozen stage-1
input manifests and the keys of all upstream tasks. A directory with DONE.json is
complete and is never recomputed; a changed input yields a new key and directory.
"""
from dataclasses import dataclass, field
from functools import lru_cache
import json
import os
from pathlib import Path
import shutil
import socket
import time
import yaml
from study.common import ROOT, digest, read_json, sha, write_json

CFG = yaml.safe_load((ROOT/'config/pipeline.yaml').read_text())
STAGE2_SOURCES = tuple(CFG['stage2_sources'])                  # real test sources prepared in stage 2
STAGE2_FOLD = {k: v['fold'] for k, v in CFG['stage2_sources'].items()}
RADOSLAW = 'nr_radoslaw_email'
ARMS = ('R', 'S', 'S_obs', 'H', 'B')
PANEL_ARMS = ('R', 'H')
MLE_ANCHOR_ARMS = tuple(CFG['et_anchor_mle_arms'])   # four-estimator rule


# Folder on the cluster workspace named in config/pipeline.yaml (overridable for tests).
def cluster_path(name):
    ws = Path(os.environ.get('PIPELINE_WS', CFG['cluster']['workspace']))
    return ws/CFG['cluster'][name] if name != 'workspace' else ws


STAGE1 = cluster_path('stage1')                               # frozen stage-1 preparation, pool and ET
PANEL_RUN = cluster_path('panel_run')                         # completed R/H panel-release run
PIPELINE_DIR = cluster_path('pipeline')                      # stage-2 workspace
WORK = Path(os.environ.get('PIPELINE_WORK', PIPELINE_DIR/'work'))
RAW = Path(os.environ.get('PIPELINE_RAW', PIPELINE_DIR/'raw'))


def stream_domain(kind, k):
    """Random-stream domain of ET replicate k (replicate 0 keeps the stage-1 domain).

    The domain strings are fixed seed labels: changing their text would change the forests."""
    base = {'training': 'training', 'pool_train': 'pool_train',
            'et_nested': 'v10_et_nested', 'et_final': 'v10_et_final'}[kind]
    return base if k == 0 else CFG['stream_domains'][kind].format(k=k)


# ---------------------------------------------------------------- hashing
STAGE_CODE = {
    'source': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/real_networks.py'],
    'testset': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/training_draws.py'],
    'draw_real': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/training_draws.py'],
    'draw_pool': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/training_draws.py'],
    'anchor0': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/training_draws.py'],
    'select': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/training_draws.py', 'pipeline/extratrees_fits.py', 'extratrees.py'],
    'train': ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/training_draws.py', 'pipeline/extratrees_fits.py', 'extratrees.py'],
    'qwen_bundle': ['pipeline/core.py', 'pipeline/qwen.py', 'run_qwen_engine.py'],
    'report': ['pipeline/core.py', 'pipeline/qwen.py', 'pipeline/report.py'],
}
CONFIG_KEYS = ('version', 'et_anchor_mle_arms', 'stream_domains', 'pool_partition', 'pool_chunk', 'windows', 'stage2_sources')


@lru_cache(maxsize=None)
# Fingerprint of the code a stage runs: the core modules plus the stage's own files.
def code_hash(stage):
    study = ROOT/'src/study'
    files = sorted(study.glob('*.py')) + sorted(study.glob('*.cpp'))
    files += sorted((ROOT/'config/prompts').glob('*.txt'))
    files += [ROOT/'production'/p for p in STAGE_CODE[stage]]
    return digest({str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): sha(p) for p in
                   (f.resolve() for f in files)})


@lru_cache(maxsize=None)
def frozen_inputs():
    """Identity of the frozen stage-1 artifacts every stage reads (not their full bytes).

    The dictionary keys are part of every task key and are kept as they are."""
    return {'v10_checksums': sha(STAGE1/'prepared/checksums.json'),
            'v10_et_rows': sha(STAGE1/'et/rows.json'),
            'rh_bundle': sha(PANEL_RUN/'BUNDLE_SHA256SUMS'),
            'rh_et_rows': sha(PANEL_RUN/'et_run/et/rows.json')}


@dataclass
# One unit of work: stage name, parameters, upstream tasks and the SLURM resources it asks for.
# finalize() computes its key; the output folder is WORK/<stage>/<name>/<first 16 characters of the key>.
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


# Execute one task (see docstring); every stage function receives (task, output folder, input folders).
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
# Small on-disk cache so the same observation block is never featurised or fitted twice.
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
