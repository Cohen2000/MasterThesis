"""v10 panel, design constants, deterministic seeds and I/O helpers."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DESIGN_VERSION = 'panel888-access-v10-20260923'
MASTER_SEED = 20260921

# Output tree of the current study. Every stage writes into its own subfolder.
RESULTS = ROOT/'results/panel888_v10'
PREPARED = RESULTS/'prepared'         # graphs, calibration, observations, requests
REFERENCES = RESULTS/'references'     # training pool, ExtraTrees folds, baseline predictions
DIAGNOSTICS = RESULTS/'diagnostics'   # offline sensitivity and diagnostic analyses
AUDIT = RESULTS/'audit'               # independent audits of the prepared study
QWEN = RESULTS/'qwen'                 # collected Qwen answers and their evaluation
BUILD = RESULTS/'build'               # compiled walk kernel (not an artifact)

# ---------------------------------------------------------------- panel
REAL_TEST = ('sp_hospital', 'sp_highschool2013', 'copenhagen_bluetooth', 'sp_workplace',
             'snap_email_eu', 'snap_collegemsg', 'snap_mathoverflow', 'nr_digg_reply')
SURROGATES = tuple(key+'__pwt' for key in REAL_TEST)
SURROGATE_PARENT = dict(zip(SURROGATES, REAL_TEST))
SYNTH = ('dar_a0_r1', 'dar_a08_r1', 'dar_a0_r2', 'dar_a08_r2',
         'ad_memoryless_r1', 'ad_memory_r1', 'ad_memoryless_r2', 'ad_memory_r2')
MAIN_KEYS = REAL_TEST + SURROGATES + SYNTH
# The 16 real training sources; the eight real test sources are among them and
# are excluded from their own leave-one-source-out fold.
TRAIN = ('sp_hospital', 'sp_primaryschool', 'sp_highschool2013', 'sp_workplace',
         'sp_hypertext2009', 'snap_collegemsg', 'snap_email_eu', 'snap_mathoverflow',
         'snap_bitcoin_otc', 'nr_radoslaw_email', 'nr_digg_reply', 'jodie_wikipedia',
         'jodie_reddit', 'jodie_lastfm', 'jodie_mooc', 'copenhagen_bluetooth')
# Reporting blocks: real, surrogate, and the four synthetic generator conditions.
STRATA = ('real', 'surrogate', 'dar_a0', 'dar_a08', 'ad_memoryless', 'ad_memory')

# ---------------------------------------------------------------- design
W = 5
ARMS = ('R', 'S', 'S_obs', 'H', 'B')
# Versioned identities; they key every random stream and every observation ID.
ARM_ID = {'R': 'R-p888-access-v9-20260922',
          'S': 'S-interaction-p888-access-v10-20260923',
          'S_obs': 'S-obs-interaction-p888-access-v10-20260923',
          'H': 'H-p888-access-v9-20260922', 'B': 'B-p888-access-v9-20260922'}
COVERAGE_FRACTION = 0.10     # T = 0.10 * sum_e K_e expected observed active dyad-windows
BUDGET_TOLERANCE = 0.05      # relative tolerance of every arm's expectation around T
H_FRACTION = 0.60            # primary elapsed-time history fraction of arm H
H_SENSITIVITY = (0.40, 0.60, 0.80)
SAMPLER_DRAWS = 3            # test sampler draws per graph and arm
TRAINING_DRAWS = 5           # training sampler draws per graph and arm
LLM_REPEATS = 3
CONFIGS = ('sol', 'deepseek', 'qwen_thinking', 'qwen_nonthinking')
QWEN_CONFIGS = ('qwen_thinking', 'qwen_nonthinking')
# Budget-sensitivity study (separate from the main study, which is fixed at 0.10).
BUDGET_GRID = (0.025, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
BUDGET_SENSITIVITY = ROOT/'results/panel888_budget_sensitivity'


def parent_source(key):
    """A surrogate shares its parent's sampler streams (common random numbers)."""
    return SURROGATE_PARENT.get(key, key)


def graph_stratum(key):
    if key in SURROGATE_PARENT: return 'surrogate'
    if key in TRAIN: return 'real'
    return 'synthetic'


def in_stratum(graph_id, stratum):
    """Reporting block of a main graph: real, surrogate or one synthetic condition."""
    if stratum in ('real', 'surrogate'): return graph_stratum(graph_id) == stratum
    return graph_id.startswith(stratum+'_r')


def fold_for(key):
    """LOSO fold: a real source and its surrogate use the fold without the parent."""
    parent = parent_source(key)
    return parent if parent in REAL_TEST else 'synthetic'


def draws_for(arm, budget, domain='sample'):
    """Distinct sampler draws for one graph and arm.

    A saturated H panel (every node drawn) is deterministic, so it is drawn once;
    its model repeats are a different kind of repetition.
    """
    if arm == 'H' and budget['h_saturated']: return 1
    return TRAINING_DRAWS if domain in ('training', 'pool_train', 'pool_dev') else SAMPLER_DRAWS


def sampler_id(arm, fraction=COVERAGE_FRACTION):
    """Versioned sampler identity. Every budget other than the main 0.10 gets its own
    identity, hence its own random streams, observation IDs and request IDs."""
    if fraction == COVERAGE_FRACTION: return ARM_ID[arm]
    return f'{ARM_ID[arm]}-b{round(fraction*1000):03d}'


def observation_id(graph_id, arm, index, fraction=COVERAGE_FRACTION):
    return f'{graph_id}__{sampler_id(arm, fraction)}__s{index}'


def planned_sizes(budgets):
    """Main/training observation and request counts derived from the calibrated budgets."""
    main = sum(draws_for(arm, budgets[g]) for g in MAIN_KEYS for arm in ARMS)
    training = sum(draws_for(arm, budgets[g], 'training') for g in TRAIN for arm in ARMS)
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


def rng(*args):
    return np.random.Generator(np.random.PCG64(seed(*args)))


# ---------------------------------------------------------------- I/O
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


def code_hashes():
    """Hashes of the scientific modules, recorded in every stage's inputs."""
    folder = ROOT/'src/main_experiment'
    files = sorted([*folder.glob('*.py'), *folder.glob('*.cpp'), *(ROOT/'config/main_experiment').glob('*.txt'),
                    ROOT/'config/study.yaml', ROOT/'config/datasets.yaml',
                    ROOT/'src/census.py', ROOT/'src/dataset_census.py'])
    return {str(p.relative_to(ROOT)): sha(p) for p in files}
