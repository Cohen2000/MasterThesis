"""Small immutable bindings shared by offline stages and inference runners."""
from pathlib import Path
from contextlib import contextmanager
import fcntl
from .common import read_json,write_json,sha,digest,ROOT,DESIGN_VERSION


def bind(path,value):
    path=Path(path)
    if path.exists():
        if read_json(path)!=value: raise ValueError(f'changed inputs: {path}; use a new directory')
    else: write_json(path,value)


def files(paths,base=None):
    return {str(Path(p).relative_to(base) if base else p):sha(p) for p in sorted(paths)}


def code_binding():
    return {'design_version':DESIGN_VERSION,'code':files(
        list((ROOT/'src/main_experiment').glob('*.py'))+
        list((ROOT/'src/main_experiment').glob('*.cpp')),ROOT)}


@contextmanager
def exclusive(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield


def validate_request(r):
    if r.get('design_version')!=DESIGN_VERSION: raise ValueError('request protocol mismatch')
    if r.get('payload_sha256')!=digest(r['payload']): raise ValueError('request payload mismatch')
    messages=r['payload'].get('messages',r['payload'].get('input'))
    if r.get('prompt_sha256')!=digest(messages): raise ValueError('request prompt mismatch')

    from .requests import payload,generation_version
    if r['payload']!=payload(r['config_id'],messages,r['seed'],generation_version(r.get('arm','H'))):
        raise ValueError('payload differs from frozen configuration')
