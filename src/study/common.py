"""Shared settings: networks, sampling arms, seeds and small file helpers."""
from pathlib import Path
import csv
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT/'data/raw'                               # raw networks (not in the repository)
EXTERNAL = Path.home()/'.local/share/masterthesis'  # answers of the language models (not in the repository)
OUT = ROOT/'results'                                # what the scripts recompute (not in the repository)
FINAL = ROOT/'docs/results/final'                   # the result tables

W = 5                # time windows
BUDGET = 0.10        # every arm sees, on average, 10% of the active (pair, window) cells
H_WINDOWS = 3        # arm H sees the last 60% of the time span: the last three windows
DRAWS = 3            # samples per test network and arm
TRAIN_DRAWS = 5      # samples per training network and arm
REPEATS = 3          # answers of a language model per sample
ARMS = ('R', 'S', 'H', 'B')

# The 16 real training networks of ExtraTrees. Nine of them are also test networks.
TRAIN = ('sp_hospital', 'sp_primaryschool', 'sp_highschool2013', 'sp_workplace', 'sp_hypertext2009', 'snap_collegemsg',
         'snap_email_eu', 'snap_mathoverflow', 'snap_bitcoin_otc', 'nr_radoslaw_email', 'nr_digg_reply', 'jodie_wikipedia',
         'jodie_reddit', 'jodie_lastfm', 'jodie_mooc', 'copenhagen_bluetooth')
# The 12 real test networks. Each also has a time-shuffled twin, named with the suffix __pwt.
REAL = ('sp_hospital', 'sp_highschool2013', 'copenhagen_bluetooth', 'sp_workplace', 'snap_email_eu', 'snap_collegemsg',
        'snap_mathoverflow', 'nr_digg_reply', 'reality_mining', 'lkml_reply', 'sp_malawi', 'nr_radoslaw_email')
TWIN = '__pwt'

# Fixed text labels. They are part of every random seed and of every sample ID, so they must
# never change: a different text would draw different samples.
MASTER_SEED = 20260921
LABEL = {'R': 'R-p888-access-v9-20260922', 'S': 'S-interaction-p888-access-v10-20260923',
         'H': 'H-p888-access-v9-20260922', 'B': 'B-p888-access-v9-20260922'}
ID_LABEL = {**LABEL, 'R': LABEL['R']+'-panel-release', 'H': LABEL['H']+'-panel-release'}


def family(key):
    """A twin belongs to the family of its original network."""
    return key.removesuffix(TWIN)


def group(key):
    return 'surrogate' if key.endswith(TWIN) else 'real' if key in REAL else 'synthetic'


def fold(key):
    """Which ExtraTrees model predicts a network: a training network and its twin get the model
    trained without that network; every other network gets the model trained on all 16."""
    return family(key) if family(key) in TRAIN else 'synthetic'


def sample_id(key, arm, index):
    return f'{key}__{ID_LABEL[arm]}__s{index}'


def seed(domain, graph='', arm='', index=0, repeat=0, config=''):
    """One 63-bit seed from text labels; every random number of the study starts from such a seed."""
    fields = json.dumps([MASTER_SEED, domain, graph, arm, index, repeat, config], separators=(',', ':'), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(fields.encode()).digest()[:8], 'big') % 2**63


def rng(*labels):
    return np.random.Generator(np.random.PCG64(seed(*labels)))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def digest(value):
    """SHA-256 of a value written as compact JSON (fingerprint of a sample text or prompt)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def write_csv(path, rows):
    """CSV with the union of the row keys; lists and dicts are written as JSON."""
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
