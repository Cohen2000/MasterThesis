"""Resume guards of the long-running model runs (Qwen rounds, API ledger).

In plain words: a long run may be interrupted and restarted. These helpers make sure a
restart uses exactly the same inputs and that two processes never write at the same time.
"""
from contextlib import contextmanager
import fcntl
from pathlib import Path
from study.common import read_json, write_json


def bind(path, value):
    """Record the inputs of a resumable run once; refuse to resume with different inputs."""
    path = Path(path)
    if path.exists():
        if read_json(path) != value: raise ValueError(f'changed inputs: {path}; use a new directory')
    else:
        write_json(path, value)


@contextmanager
def exclusive(path):
    """Non-blocking exclusive lock: a second writer fails instead of waiting."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield
