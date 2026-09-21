from pathlib import Path
import hashlib
import json
import os
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REAL_TEST = ('sp_hospital','sp_highschool2013','copenhagen_bluetooth','sp_workplace',
             'snap_email_eu','snap_collegemsg','snap_mathoverflow','nr_digg_reply')
SURROGATES = tuple(g+'__pwt' for g in REAL_TEST)
SURROGATE_PARENT = dict(zip(SURROGATES, REAL_TEST))
def parent_source(key): return SURROGATE_PARENT.get(key,key)
def graph_stratum(key): return 'surrogate' if key in SURROGATE_PARENT else 'real' if key in TRAIN else 'synthetic'
def fold_for(key):
    parent=parent_source(key)
    return parent if parent in REAL_TEST else 'synthetic'
TRAIN = ('sp_hospital','sp_primaryschool','sp_highschool2013','sp_workplace',
         'sp_hypertext2009','snap_collegemsg','snap_email_eu','snap_mathoverflow',
         'snap_bitcoin_otc','nr_radoslaw_email','nr_digg_reply','jodie_wikipedia',
         'jodie_reddit','jodie_lastfm','jodie_mooc','copenhagen_bluetooth')
ARMS = ('R','S','H','B')
# One final panel; earlier generations are development provenance only.
DESIGN_VERSION = 'panel888-pwt-srw-20260921'
PREVIOUS_DESIGN_VERSION = 'cells10-final-20260920'
MASTER_SEED = 20260921
ARCHIVED_ROOT = ROOT/'archive/pre_panel888_20260921/results'
PREVIOUS_RUN = ARCHIVED_ROOT/'main_experiment/cells10_final_20260920'
PREVIOUS_REVISION = ARCHIVED_ROOT/'baseline_revision_cells10_srw_htime60_20260920'
MATCHED_QUANTITY = 'expected_observed_active_dyad_windows'
COVERAGE_FRACTION = 0.10     # share of sum_e K_e every arm observes in expectation
BUDGET_FRACTION = 0.10       # event budget of the superseded designs; legacy variants only
BUDGET_TOLERANCE = 0.05      # unchanged relative tolerance for the matched expectation
CURRENT_RUN = 'results/main_experiment/panel888_pwt_srw_20260921'
CURRENT_REVISION = 'results/baseline_revision_panel888_pwt_srw_20260921'
SAMPLER_DRAWS = 3            # sampler draws per graph and arm, if the draw is random
TRAINING_DRAWS = 5
LLM_REPEATS = 3
CONFIGS = ('sol','deepseek','qwen_thinking','qwen_nonthinking')
QWEN_CONFIGS = ('qwen_thinking','qwen_nonthinking')
SYNTH = tuple(f'{family}_{mode}_r{r}' for family,modes in
              [('dar',('a0','a08')),('ad',('memoryless','memory'))]
              for r in (1,2) for mode in modes)
MAIN_KEYS = (*REAL_TEST,*SURROGATES,*SYNTH)
MAIN_GRAPHS = len(MAIN_KEYS)

# Arm H: uniform nodes, then a common elapsed-time suffix. Legacy constants are
# retained only for historical helper tests; they never define the current H.
H_VARIANT = 'uniform_nodes_time_suffix'
H_FRACTION = 0.60
H_SENSITIVITY = (0.40, 0.60, 0.80)
H_CAP = 5
LEGACY_H = 'H_suffix_v1'
LEGACY_ARMS = (LEGACY_H,)
# Versioned observation and sampler identities.
DESIGN_TAG = 'p888'
ARM_ID = {a:a+'-p888-20260921' for a in (*ARMS,LEGACY_H)}


def draws_for(arm,budget,domain='sample'):
    """Distinct sampler draws for one graph and arm.

    A saturated H sample (every node drawn) is deterministic, so repeating
    it would only duplicate one observation. It is carried once; model repeats of
    that single observation are a separate kind of repetition.
    """
    if arm=='H' and budget['h_saturated']: return 1
    return TRAINING_DRAWS if domain in ('training','pool_train','pool_dev') else SAMPLER_DRAWS


def observations_per_graph(budget,domain='sample'):
    return sum(draws_for(a,budget,domain) for a in ARMS)


def observation_id(graph_id,arm,index):
    return f'{graph_id}__{ARM_ID[arm]}__s{index}'


def planned_sizes(budgets,main_graphs,training_graphs):
    """Design sizes derived from the calibrated budgets, never from literals.

    budgets maps graph id -> budget dict. Main and training counts are returned
    separately because the four real test sources that are also training
    sources draw both domains.
    """
    main=sum(observations_per_graph(budgets[g]) for g in main_graphs)
    train=sum(observations_per_graph(budgets[g],'training') for g in training_graphs)
    return {'main_observations':main,'training_observations':train,
            'planned_calls':main*len(CONFIGS)*LLM_REPEATS,
            'qwen_calls':main*len(QWEN_CONFIGS)*LLM_REPEATS,
            'calls_per_observation':len(CONFIGS)*LLM_REPEATS}

SEEDS = {}

def seed(domain, graph_id='', arm_id='', sample_index=0, repeat_index=0, config_id=''):
    fields=[MASTER_SEED,domain,graph_id,arm_id,sample_index,repeat_index,config_id]
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
