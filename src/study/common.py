"""Stage-1 panel, design constants, deterministic seeds and I/O helpers."""
# Shared settings of the whole study, in plain words:
# - which graphs are tested (real sources, their surrogates, synthetic graphs) and which
#   real graphs are used for training;
# - the sampling design (5 time windows, the sampling arms, the 10% budget, draws, repeats);
# - one function (seed) that derives every random number stream from fixed labels, so the
#   whole study is exactly reproducible;
# - small file helpers (hashing, JSON/CSV writing) used everywhere.
from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# Fixed identity stamp written into prepared files and model answers and checked against
# them later. It is data, not a description: changing the text would break those checks.
DESIGN_VERSION = 'panel888-access-v10-20260923'
# The single root of all randomness in the study.
MASTER_SEED = 20260921

# Local output tree of stage 1 (not committed; the folder name matches the existing
# workspaces). Every stage writes into its own subfolder.
RESULTS = ROOT/'results/panel888_v10'
PREPARED = RESULTS/'prepared'         # graphs, calibration, observations, requests
REFERENCES = RESULTS/'references'     # training pool, ExtraTrees folds, baseline predictions
QWEN = RESULTS/'qwen'                 # collected Qwen answers and their evaluation
BUILD = RESULTS/'build'               # compiled walk kernel (not an artifact)

# ---------------------------------------------------------------- panel
# Real test graphs of the base panel (the other real test sources are listed in config/).
STAGE1_REAL = ('sp_hospital', 'sp_highschool2013', 'copenhagen_bluetooth', 'sp_workplace',
             'snap_email_eu', 'snap_collegemsg', 'snap_mathoverflow', 'nr_digg_reply')
# Each real test graph has one surrogate with the suffix '__pwt' (timestamps shuffled).
SURROGATES = tuple(key+'__pwt' for key in STAGE1_REAL)
SURROGATE_PARENT = dict(zip(SURROGATES, STAGE1_REAL))
# Eight synthetic graphs with known persistence: DAR = discrete autoregressive activity
# (a0 = no memory, a08 = strong memory), ad = activity-driven with or without memory;
# r1/r2 = two random realisations of each generator.
SYNTHETIC_GRAPHS = ('dar_a0_r1', 'dar_a08_r1', 'dar_a0_r2', 'dar_a08_r2',
         'ad_memoryless_r1', 'ad_memory_r1', 'ad_memoryless_r2', 'ad_memory_r2')
STAGE1_GRAPHS = STAGE1_REAL + SURROGATES + SYNTHETIC_GRAPHS
# The 16 real training sources; the eight real test sources are among them and
# are excluded from their own leave-one-source-out fold.
TRAINING_SOURCES = ('sp_hospital', 'sp_primaryschool', 'sp_highschool2013', 'sp_workplace',
         'sp_hypertext2009', 'snap_collegemsg', 'snap_email_eu', 'snap_mathoverflow',
         'snap_bitcoin_otc', 'nr_radoslaw_email', 'nr_digg_reply', 'jodie_wikipedia',
         'jodie_reddit', 'jodie_lastfm', 'jodie_mooc', 'copenhagen_bluetooth')

# ---------------------------------------------------------------- design
# The time axis of every graph is cut into W = 5 equal windows. rho_k is the share of
# interacting pairs that are active in at least k of the 5 windows; rho_2 is the target.
W = 5
# Sampling arms (how a partial observation is taken):
# R = node panel (random nodes, all their interactions), S = interaction-following walk,
# H = node panel that only sees the first 60% of time, B = random thinning of events.
# S_obs is an older walk variant that is kept for reproducibility but not reported.
ARMS = ('R', 'S', 'S_obs', 'H', 'B')
# Versioned identities; they key every random stream and every observation ID.
# Fixed labels of the arms. They are part of every random stream and every observation ID,
# so they must stay byte-identical; changing one would draw different observations.
ARM_ID = {'R': 'R-p888-access-v9-20260922',
          'S': 'S-interaction-p888-access-v10-20260923',
          'S_obs': 'S-obs-interaction-p888-access-v10-20260923',
          'H': 'H-p888-access-v9-20260922', 'B': 'B-p888-access-v9-20260922'}
# Budget: every arm is tuned so that it observes about 10% of all active (pair, window) cells.
COVERAGE_FRACTION = 0.10     # T = 0.10 * sum_e K_e expected observed active dyad-windows
BUDGET_TOLERANCE = 0.05      # relative tolerance of every arm's expectation around T
H_FRACTION = 0.60            # primary elapsed-time history fraction of arm H
H_SENSITIVITY = (0.40, 0.60, 0.80)
SAMPLER_DRAWS = 3            # test sampler draws per graph and arm
TRAINING_DRAWS = 5           # training sampler draws per graph and arm
LLM_REPEATS = 3
# Language-model configurations that get the same prompts.
CONFIGS = ('sol', 'deepseek', 'qwen_thinking', 'qwen_nonthinking')
QWEN_CONFIGS = ('qwen_thinking', 'qwen_nonthinking')


def parent_source(key):
    """A surrogate shares its parent's sampler streams (common random numbers)."""
    return SURROGATE_PARENT.get(key, key)


# Which block a graph belongs to: real, surrogate or synthetic.
def graph_stratum(key):
    if key in SURROGATE_PARENT: return 'surrogate'
    if key in TRAINING_SOURCES: return 'real'
    return 'synthetic'


def fold_for(key):
    """LOSO fold: a real source and its surrogate use the fold without the parent."""
    parent = parent_source(key)
    return parent if parent in STAGE1_REAL else 'synthetic'


def draws_for(arm, budget, domain='sample'):
    """Distinct sampler draws for one graph and arm.

    A saturated H panel (every node drawn) is deterministic, so it is drawn once;
    its model repeats are a different kind of repetition.
    """
    if arm == 'H' and budget['h_saturated']: return 1
    return TRAINING_DRAWS if domain in ('training', 'pool_train', 'pool_dev') else SAMPLER_DRAWS


# Arm label used in IDs and seeds; other budgets get a suffix so their streams differ.
def sampler_id(arm, fraction=COVERAGE_FRACTION):
    """Versioned sampler identity. Every budget other than the main 0.10 gets its own
    identity, hence its own random streams, observation IDs and request IDs."""
    if fraction == COVERAGE_FRACTION: return ARM_ID[arm]
    return f'{ARM_ID[arm]}-b{round(fraction*1000):03d}'


# ID of one observation: graph + arm label + draw number, e.g. sp_hospital__R-...__s1.
def observation_id(graph_id, arm, index, fraction=COVERAGE_FRACTION):
    return f'{graph_id}__{sampler_id(arm, fraction)}__s{index}'


# How many observations and model requests the design implies (used as a consistency check).
def planned_sizes(budgets):
    """Main/training observation and request counts derived from the calibrated budgets."""
    main = sum(draws_for(arm, budgets[g]) for g in STAGE1_GRAPHS for arm in ARMS)
    training = sum(draws_for(arm, budgets[g], 'training') for g in TRAINING_SOURCES for arm in ARMS)
    return {'main_observations': main, 'training_observations': training,
            'planned_calls': main*len(CONFIGS)*LLM_REPEATS,
            'qwen_calls': main*len(QWEN_CONFIGS)*LLM_REPEATS}


# ---------------------------------------------------------------- seeds
# Every random stream of the study is seed(domain, graph, arm, sample, repeat, config).
# SEEDS records each derived seed so that two different field tuples mapping to
# the same 63-bit seed are detected (domain separation / collision guard).
SEEDS = {}


def seed(domain, graph_id='', arm_id='', sample_index=0, repeat_index=0, config_id=''):
    fields = [MASTER_SEED, domain, graph_id, arm_id, sample_index, repeat_index, config_id]
    key = json.dumps(fields, separators=(',', ':'))
    value = int.from_bytes(hashlib.sha256(json.dumps(fields, separators=(',', ':'), ensure_ascii=False)
                                          .encode()).digest()[:8], 'big') % 2**63
    if SEEDS.get(value, key) != key:
        raise RuntimeError('seed collision')
    SEEDS[value] = key
    return value


# A numpy random generator seeded from the same labels as seed().
def rng(*args):
    return np.random.Generator(np.random.PCG64(seed(*args)))


# ---------------------------------------------------------------- I/O
# SHA-256 of a file's bytes.
def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def digest(value):
    """SHA256 of a canonical JSON encoding (used for blocks, prompts, payloads)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


# Write to a temporary file first and rename it, so a crash never leaves half a file.
def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'w') as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


# Same safe-write pattern for CSV tables; lists and dicts are stored as JSON text.
def write_csv(path, rows):
    """CSV with the union of row keys; nested values are JSON encoded."""
    import csv
    if not rows: return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    with open(tmp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    tmp.replace(path)


def fresh_directory(path):
    """Stages never resume: each writes into a new, empty directory."""
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'{path} is not empty; remove it to recompute this stage')
    path.mkdir(parents=True, exist_ok=True)
    return path


# Fingerprint of the scientific code, stored next to every prepared output so one can
# later check which code produced it.
def code_hashes():
    """Hashes of the scientific modules, recorded in every stage's inputs."""
    folder = ROOT/'src/study'
    files = sorted([*folder.glob('*.py'), *folder.glob('*.cpp'), *(ROOT/'config/prompts').glob('*.txt'),
                    ROOT/'config/study.yaml', ROOT/'config/datasets.yaml',
                    ROOT/'src/dataset_survey.py', ROOT/'src/dataset_audit.py'])
    return {str(p.relative_to(ROOT)): sha(p) for p in files}
