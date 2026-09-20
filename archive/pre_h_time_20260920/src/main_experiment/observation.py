import math
import numpy as np
from .common import ARMS, ROOT, H_CAP, LEGACY_H, LEGACY_ARMS

KNOWN_ARMS=ARMS+LEGACY_ARMS
# One parameter line per arm; the names must stay distinct because the parser
# identifies the arm by which one is present. H carries its dyad-sample size; the
# legacy suffix variant keeps its panel size.
PARAMS={'R':'n_panel','S':'L','H':'n_dyads','B':'p',LEGACY_H:'n_panel_suffix'}
INTEGER_PARAM_ARMS=('R','S','H',LEGACY_H)
# Rule text per arm. Both H variants carry a version in the file name so neither
# can be read in place of the other; R, S and B are byte-identical to the
# previous design, which is what allows their answers to be reused.
RULE_FILES={'R':'rule_R.txt','S':'rule_S.txt','B':'rule_B.txt',
            'H':'rule_H_recent5_v2.txt',LEGACY_H:'rule_H_suffix_panel_v1.txt'}
HEADER='pattern,dyads,events'
H_HEADER='pattern,dyads,events,at_cap_dyads'
ALL_PATTERNS=[f'{p:05b}' for p in range(1,32)]
LEGACY_PATTERNS=['??'+f'{p:03b}' for p in range(1,8)]

FEATURE_VERSION='features-v3-hrecent5-20260917'
# Base block: absolute counts and indicators read directly from the serialized
# input. at_cap_{p} is the fourth H column and zero for every other arm. Derived
# block: scale-free transforms of the same input. None of them uses full-graph
# sizes, ground truth, true truncation, realised coverage, source names,
# generator parameters or generator families.
BASE_FEATURE_NAMES=(['N_obs','D_obs','M_obs']+[f'window_{i}' for i in range(1,6)]+
 [f'access_{i}' for i in range(1,6)]+[f'{p:05b}_{x}' for p in range(1,32) for x in ['dyads','events']]+
 [f'{p:05b}_at_cap' for p in range(1,32)]+
 [f'A_{i}' for i in range(1,6)]+[f'arm_{a}' for a in ARMS]+[PARAMS[a] for a in ARMS])
# corrector_rho_k is each arm's fixed simple reference: plug-in for R, the walk
# ratio for S, the bound midpoint for H and the homogeneous event corrector for B.
# The old three-to-five-window suffix correction appears nowhere in this block.
DERIVED_FEATURE_NAMES=([f'share_{p:05b}_dyads' for p in range(1,32)]+
 [f'share_{p:05b}_at_cap' for p in range(1,32)]+
 [f'share_window_{i}' for i in range(1,6)]+['events_per_dyad']+
 [f'plugin_rho_{k}' for k in range(2,6)]+[f'corrector_rho_{k}' for k in range(2,6)])
FEATURE_NAMES=BASE_FEATURE_NAMES+DERIVED_FEATURE_NAMES
assert len(BASE_FEATURE_NAMES)==119 and len(DERIVED_FEATURE_NAMES)==76
assert len(FEATURE_NAMES)==195 and len(set(FEATURE_NAMES))==195


def patterns_for(arm):
    return LEGACY_PATTERNS if arm==LEGACY_H else ALL_PATTERNS


def access_for(arm):
    return [0,0,1,1,1] if arm==LEGACY_H else [1]*5


def make(g,arm,budget,counts,traversals):
    occupied=counts.sum(1)>0
    per_dyad=counts.sum(1)
    patterns=(counts>0)@np.array([16,8,4,2,1])
    ds=np.bincount(patterns[occupied],minlength=32)
    es=np.zeros(32,dtype=np.int64); np.add.at(es,patterns[occupied],per_dyad[occupied])
    if arm=='H':
        # Observed events per dyad never exceed the cap, so "at cap" means exactly
        # H_CAP retrieved events: possibly truncated, not provably truncated.
        if per_dyad.max(initial=0)>H_CAP: raise ValueError('H dyad above the event cap')
        cs=np.bincount(patterns[occupied & (per_dyad==H_CAP)],minlength=32)
        table=[(f'{p:05b}',int(ds[p]),int(es[p]),int(cs[p])) for p in range(1,32)]
    elif arm==LEGACY_H:
        table=[('??'+f'{p:03b}',int(ds[p]),int(es[p])) for p in range(1,8)]
    else:
        table=[(f'{p:05b}',int(ds[p]),int(es[p])) for p in range(1,32)]
    A=None
    if arm=='S': A=np.bincount(g.K,weights=traversals/g.m,minlength=6)[1:6].tolist()
    parameter=(budget['legacy_suffix_panel']['n_panel_suffix'] if arm==LEGACY_H
               else budget[PARAMS[arm]])
    epw=counts.sum(0).tolist()
    obs={'arm':arm,'N_obs':int(len(np.unique(g.ends[occupied]))),'D_obs':int(occupied.sum()),
         'M_obs':int(counts.sum()),'Temporal_access':access_for(arm),
         'Events_per_window':([None,None]+epw[2:]) if arm==LEGACY_H else epw,
         'Walk_A':A,'parameter':parameter,'table':table}
    validate(obs)
    return obs


def validate(o):
    arm=o['arm']
    if arm not in KNOWN_ARMS: raise ValueError('arm')
    integer=lambda x:type(x) is int and x>=0
    for k in ['N_obs','D_obs','M_obs']:
        if not integer(o[k]): raise ValueError(k)
    if [r[0] for r in o['table']]!=patterns_for(arm): raise ValueError('table patterns/order')
    width=4 if arm=='H' else 3
    if any(len(r)!=width for r in o['table']): raise ValueError('table width')
    D=M=0
    for row in o['table']:
        pat,d,e=row[:3]; ones=pat.count('1')
        if not integer(d) or not integer(e) or (d==0)!=(e==0) or e<d*ones:
            raise ValueError('inconsistent table counts')
        if arm=='H':
            c=row[3]
            if not integer(c) or c>d: raise ValueError('at_cap_dyads')
            # At-cap dyads hold exactly H_CAP events; the others hold between
            # their number of active windows and H_CAP-1.
            if not (H_CAP*c+ones*(d-c)<=e<=H_CAP*c+(H_CAP-1)*(d-c)):
                raise ValueError('events inconsistent with the event cap')
        D+=d; M+=e
    if (D,M)!=(o['D_obs'],o['M_obs']): raise ValueError('table totals')
    access=access_for(arm)
    if o['Temporal_access']!=access or len(o['Events_per_window'])!=5: raise ValueError('access')
    for j,(a,e) in enumerate(zip(access,o['Events_per_window'])):
        if not a:
            if e is not None: raise ValueError('missing vs zero')
        else:
            if not integer(e): raise ValueError('window counts')
            active=sum(r[1] for r in o['table'] if r[0][j]=='1')
            if e<active or (active==0)!=(e==0): raise ValueError('window/table mismatch')
    if sum(e or 0 for e in o['Events_per_window'])!=M: raise ValueError('window sum')
    N=o['N_obs']
    if D==0:
        if N!=0 or M!=0: raise ValueError('empty counts')
    elif N<2 or N>2*D or D>N*(N-1)//2: raise ValueError('endpoint count')
    par=o['parameter']
    if arm in INTEGER_PARAM_ARMS:
        if not integer(par): raise ValueError('integer parameter')
        if arm in ('R',LEGACY_H) and N>par: raise ValueError('panel smaller than observed nodes')
        # Every sampled dyad is active and keeps at least one event, so the
        # number of listed dyads equals the sample size exactly.
        if arm=='H' and D!=par: raise ValueError('H lists every sampled dyad')
    elif not isinstance(par,(int,float)) or not 0<par<=1: raise ValueError('p')
    A=o['Walk_A']
    if arm=='S':
        if A is None or len(A)!=5 or any(not math.isfinite(x) or x<0 for x in A): raise ValueError('A')
        if bool(D)!=bool(sum(A)): raise ValueError('walk mass')
    elif A is not None: raise ValueError('inapplicable A')


def serialize(o):
    validate(o)
    fmt=lambda x:'NA' if x is None else format(x,'.17g') if type(x) is float else str(x)
    lines=['W=5','Temporal_access='+','.join(map(str,o['Temporal_access']))]
    lines += [f'{k}={o[k]}' for k in ['N_obs','D_obs','M_obs']]
    lines += ['Events_per_window='+','.join(map(fmt,o['Events_per_window'])),
              PARAMS[o['arm']]+'='+fmt(o['parameter']),
              'Walk_A='+('NA' if o['Walk_A'] is None else ','.join(map(fmt,o['Walk_A']))),
              H_HEADER if o['arm']=='H' else HEADER]
    lines += [','.join(map(str,row)) for row in o['table']]
    return '\n'.join(lines)


def parse(text):
    lines=text.splitlines()
    if lines[0]!='W=5' or lines[8] not in (HEADER,H_HEADER): raise ValueError('input block')
    fields=dict(line.split('=',1) for line in lines[1:8])
    params=set(fields)&set(PARAMS.values())
    if len(params)!=1: raise ValueError('parameter count')
    name=params.pop(); arm=next(a for a,p in PARAMS.items() if p==name)
    if (lines[8]==H_HEADER)!=(arm=='H'): raise ValueError('table header does not match the rule')
    rows=[]
    for line in lines[9:]:
        cells=line.split(',')
        rows.append((cells[0],*map(int,cells[1:])))
    o={'arm':arm,'parameter':int(fields[name]) if arm in INTEGER_PARAM_ARMS else float(fields[name]),
       **{k:int(fields[k]) for k in ['N_obs','D_obs','M_obs']},
       'Temporal_access':list(map(int,fields['Temporal_access'].split(','))),
       'Events_per_window':[None if x=='NA' else int(x) for x in fields['Events_per_window'].split(',')],
       'Walk_A':None if fields['Walk_A']=='NA' else list(map(float,fields['Walk_A'].split(','))),
       'table':rows}
    validate(o); return o


def features(o):
    # Local import: baselines imports validate from this module.
    from .baselines import plugin, corrector
    validate(o)
    if o['arm'] not in ARMS: raise ValueError('features exist only for the arms of the current design')
    table={r[0]:r[1:] for r in o['table']}
    get=lambda p:table.get(p,(0,0,0))
    f=[o[k] for k in ['N_obs','D_obs','M_obs']]+[x or 0 for x in o['Events_per_window']]+o['Temporal_access']
    f += [x for p in ALL_PATTERNS for x in get(p)[:2]]
    f += [(get(p)[2] if len(get(p))>2 else 0) for p in ALL_PATTERNS]
    f += o['Walk_A'] or [0]*5
    f += [int(o['arm']==a) for a in ARMS]
    f += [o['parameter'] if a==o['arm'] else 0 for a in ARMS]
    if len(f)!=len(BASE_FEATURE_NAMES): raise AssertionError('base feature count')
    # Derived block. An empty sample has D_obs = M_obs = 0; every ratio and every
    # model-based value is then coded as a plain zero rather than a division by zero.
    D=o['D_obs']; M=o['M_obs']
    f += [get(p)[0]/D if D else 0. for p in ALL_PATTERNS]
    f += [(get(p)[2] if len(get(p))>2 else 0)/D if D else 0. for p in ALL_PATTERNS]
    f += [(x or 0)/M if M else 0. for x in o['Events_per_window']]
    f += [M/D if D else 0.]
    if D:
        pi=plugin(o)
        try: co=corrector(o)
        except (ArithmeticError,FloatingPointError,OverflowError,ValueError): co=pi
    else:
        pi=co=[0.,0.,0.,0.]
    f += list(pi)+list(co)
    if len(f)!=len(FEATURE_NAMES): raise AssertionError('feature count')
    return np.array(f,float)


def messages(block):
    o=parse(block)
    spec=ROOT/'config/main_experiment'
    system=(spec/'system.txt').read_text().rstrip('\n')
    common=(spec/'user_prefix.txt').read_text().rstrip('\n')
    rule=(spec/RULE_FILES[o['arm']]).read_text().rstrip('\n')
    user=common+'\nSampling rule: '+rule+'\n'+block+'\nReturn the four full-archive estimates in the specified JSON format.'
    return [{'role':'system','content':system},{'role':'user','content':user}]
