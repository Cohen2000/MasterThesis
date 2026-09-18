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
# Design revision history, newest last. budget10-20261001 made the budget a fixed
# share of the full event archive. budget10-hrecent5-20260917 replaced arm H by a
# uniform sample of active dyads, each with its five most recent events.
# cells10-20260917 keeps every mechanism but matches the arms on the quantity the
# target is made of: the expected number of observed active dyad-windows is ten
# percent of all active dyad-windows of the full archive (sum_e K_e). The event
# budget matched event volume, which let H and B see 46 % and 37 % of the active
# dyad-windows of the real sources against 2.4 % for S. The date-like suffix of
# budget10-20261001 is a revision label, not a date.
DESIGN_VERSION = 'cells10-json-20260918'
PREVIOUS_DESIGN_VERSION = 'cells10-20260917'
MATCHED_QUANTITY = 'expected_observed_active_dyad_windows'
COVERAGE_FRACTION = 0.10     # share of sum_e K_e every arm observes in expectation
BUDGET_FRACTION = 0.10       # event budget of the superseded designs; legacy variants only
BUDGET_TOLERANCE = 0.05      # unchanged relative tolerance for the matched expectation
CURRENT_RUN = 'results/main_experiment/cells10_json_20260918'
CURRENT_REVISION = 'results/baseline_revision_cells10_json_20260918'
SAMPLER_DRAWS = 5            # sampler draws per graph and arm, if the draw is random
LLM_REPEATS = 3
CONFIGS = ('sol','deepseek','qwen_thinking','qwen_nonthinking')
QWEN_CONFIGS = ('qwen_thinking','qwen_nonthinking')
SYNTH = tuple(f'{family}_{mode}_r{r}' for family,modes in
              [('dar',('a0','a08')),('ad',('memoryless','memory'))]
              for r in (1,2) for mode in modes)
MAIN_GRAPHS = len(REAL_TEST)+len(SYNTH)

# Arm H of this design: a uniform dyad sample keeping the H_CAP most recent events
# of every sampled dyad. The superseded suffix-panel H stays available as a
# versioned development variant under its own arm code; it is never part of ARMS.
H_VARIANT = 'recent5'
H_CAP = 5
LEGACY_H = 'H_suffix_v1'
LEGACY_ARMS = (LEGACY_H,)
# Sampling identity stays c10: this revision changes inference and the training
# pool, not the main observation mechanism. Request IDs have a separate protocol
# suffix; old answers are never reusable under the revised generation contract.
DESIGN_TAG = 'c10'
ARM_ID = {'R':'R-c10','S':'S-c10','H':'H-recent5-c10','B':'B-c10',LEGACY_H:'H'}


def draws_for(arm,budget):
    """Distinct sampler draws for one graph and arm.

    A saturated H sample (every active dyad drawn) is deterministic, so repeating
    it would only duplicate one observation. It is carried once; model repeats of
    that single observation are a separate kind of repetition.
    """
    if arm=='H' and budget['h_saturated']: return 1
    return SAMPLER_DRAWS


def observations_per_graph(budget):
    return sum(draws_for(a,budget) for a in ARMS)


def observation_id(graph_id,arm,index):
    return f'{graph_id}__{ARM_ID[arm]}__s{index}'


def planned_sizes(budgets,main_graphs,training_graphs):
    """Design sizes derived from the calibrated budgets, never from literals.

    budgets maps graph id -> budget dict. Main and training counts are returned
    separately because the four real test sources that are also training
    sources draw both domains.
    """
    main=sum(observations_per_graph(budgets[g]) for g in main_graphs)
    train=sum(observations_per_graph(budgets[g]) for g in training_graphs)
    return {'main_observations':main,'training_observations':train,
            'planned_calls':main*len(CONFIGS)*LLM_REPEATS,
            'qwen_calls':main*len(QWEN_CONFIGS)*LLM_REPEATS,
            'calls_per_observation':len(CONFIGS)*LLM_REPEATS}

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
