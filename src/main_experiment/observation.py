import math
import numpy as np
from .common import ARMS, ROOT

# One parameter line per arm; the names must stay distinct because the parser
# identifies the arm by which one is present. H carries its own panel size.
PARAMS=dict(R='n_panel',S='L',H='n_panel_suffix',B='p')
INTEGER_PARAM_ARMS=('R','S','H')
# The first 88 entries are the original absolute counts and indicators. The 45
# derived entries that follow are scale-free transforms of exactly the same
# serialized observation input; none of them uses full-graph sizes, ground truth,
# realised coverage, source names, generator parameters or generator families.
BASE_FEATURE_NAMES=(['N_obs','D_obs','M_obs']+[f'window_{i}' for i in range(1,6)]+
 [f'access_{i}' for i in range(1,6)]+[f'{p:05b}_{x}' for p in range(1,32) for x in ['dyads','events']]+
 [f'A_{i}' for i in range(1,6)]+[f'arm_{a}' for a in ARMS]+['n_panel','L','n_panel_suffix','p'])
DERIVED_FEATURE_NAMES=([f'share_{p:05b}_dyads' for p in range(1,32)]+
 [f'share_window_{i}' for i in range(1,6)]+['events_per_dyad']+
 [f'plugin_rho_{k}' for k in range(2,6)]+[f'corrector_rho_{k}' for k in range(2,6)])
FEATURE_NAMES=BASE_FEATURE_NAMES+DERIVED_FEATURE_NAMES
assert len(BASE_FEATURE_NAMES)==88 and len(DERIVED_FEATURE_NAMES)==45
assert len(FEATURE_NAMES)==133

def make(g,arm,budget,counts,traversals):
    occupied=counts.sum(1)>0
    patterns=(counts>0)@np.array([16,8,4,2,1])
    ds=np.bincount(patterns[occupied],minlength=32)
    es=np.zeros(32,dtype=np.int64); np.add.at(es,patterns[occupied],counts.sum(1)[occupied])
    table=[(('??'+f'{p:03b}') if arm=='H' else f'{p:05b}',int(ds[p]),int(es[p]))
           for p in range(1,8 if arm=='H' else 32)]
    A=None
    if arm=='S': A=np.bincount(g.K,weights=traversals/g.m,minlength=6)[1:6].tolist()
    obs={'arm':arm,'N_obs':int(len(np.unique(g.ends[occupied]))),'D_obs':int(occupied.sum()),
         'M_obs':int(counts.sum()),'Temporal_access':[0,0,1,1,1] if arm=='H' else [1]*5,
         'Events_per_window':([None,None]+counts.sum(0)[2:].tolist()) if arm=='H' else counts.sum(0).tolist(),
         'Walk_A':A,'parameter':budget[PARAMS[arm]],'table':table}
    validate(obs)
    return obs

def validate(o):
    arm=o['arm']
    if arm not in ARMS: raise ValueError('arm')
    integer=lambda x:type(x) is int and x>=0
    for k in ['N_obs','D_obs','M_obs']:
        if not integer(o[k]): raise ValueError(k)
    expected=[('??'+f'{p:03b}') if arm=='H' else f'{p:05b}' for p in range(1,8 if arm=='H' else 32)]
    if [r[0] for r in o['table']]!=expected: raise ValueError('table patterns/order')
    D=M=0
    for pat,d,e in o['table']:
        if not integer(d) or not integer(e) or (d==0)!=(e==0) or e<d*pat.count('1'):
            raise ValueError('inconsistent table counts')
        D+=d; M+=e
    if (D,M)!=(o['D_obs'],o['M_obs']): raise ValueError('table totals')
    access=[0,0,1,1,1] if arm=='H' else [1]*5
    if o['Temporal_access']!=access or len(o['Events_per_window'])!=5: raise ValueError('access')
    for j,(a,e) in enumerate(zip(access,o['Events_per_window'])):
        if not a:
            if e is not None: raise ValueError('missing vs zero')
        else:
            if not integer(e): raise ValueError('window counts')
            active=sum(d for p,d,_ in o['table'] if p[j]=='1')
            if e<active or (active==0)!=(e==0): raise ValueError('window/table mismatch')
    if sum(e or 0 for e in o['Events_per_window'])!=M: raise ValueError('window sum')
    N=o['N_obs']
    if D==0:
        if N!=0 or M!=0: raise ValueError('empty counts')
    elif N<2 or N>2*D or D>N*(N-1)//2: raise ValueError('endpoint count')
    par=o['parameter']
    if arm in INTEGER_PARAM_ARMS:
        if not integer(par): raise ValueError('integer parameter')
        if arm in ('R','H') and N>par: raise ValueError('panel smaller than observed nodes')
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
              'pattern,dyads,events']
    lines += [f'{p},{d},{e}' for p,d,e in o['table']]
    return '\n'.join(lines)


def parse(text):
    lines=text.splitlines()
    if lines[0]!='W=5' or lines[8]!='pattern,dyads,events': raise ValueError('input block')
    fields=dict(line.split('=',1) for line in lines[1:8])
    params=set(fields)&set(PARAMS.values())
    if len(params)!=1: raise ValueError('parameter count')
    name=params.pop(); arm=next(a for a,p in PARAMS.items() if p==name)
    o={'arm':arm,'parameter':int(fields[name]) if arm in INTEGER_PARAM_ARMS else float(fields[name]),
       **{k:int(fields[k]) for k in ['N_obs','D_obs','M_obs']},
       'Temporal_access':list(map(int,fields['Temporal_access'].split(','))),
       'Events_per_window':[None if x=='NA' else int(x) for x in fields['Events_per_window'].split(',')],
       'Walk_A':None if fields['Walk_A']=='NA' else list(map(float,fields['Walk_A'].split(','))),
       'table':[(p,int(d),int(e)) for p,d,e in (line.split(',') for line in lines[9:])]}
    validate(o); return o


def features(o):
    # Local import: baselines imports validate from this module, so a module-level
    # import would be circular. The plug-in and the homogeneous corrector are used
    # here purely as fixed feature transforms, independently of whether either one
    # is the primary corrector.
    from .baselines import plugin, corrector
    validate(o); table={p.replace('?','0'):(d,e) for p,d,e in o['table']}
    f=[o[k] for k in ['N_obs','D_obs','M_obs']]+[x or 0 for x in o['Events_per_window']]+o['Temporal_access']
    f += [x for p in range(1,32) for x in table.get(f'{p:05b}',(0,0))]
    f += o['Walk_A'] or [0]*5
    f += [int(o['arm']==a) for a in ARMS]
    f += [o['parameter'] if a==o['arm'] else 0 for a in ARMS]
    if len(f)!=88: raise AssertionError('base feature count')
    # Derived block. An empty sample has D_obs = M_obs = 0; every ratio and every
    # model-based value is then coded as a plain zero rather than a division by
    # zero. Missing windows stay distinguishable through access_1..access_5.
    D=o['D_obs']; M=o['M_obs']
    f += [table.get(f'{p:05b}',(0,0))[0]/D if D else 0. for p in range(1,32)]
    f += [(x or 0)/M if M else 0. for x in o['Events_per_window']]
    f += [M/D if D else 0.]
    if D:
        pi=plugin(o)
        try: co=corrector(o)
        except (ArithmeticError,FloatingPointError,OverflowError,ValueError): co=pi
    else:
        pi=co=[0.,0.,0.,0.]
    f += list(pi)+list(co)
    if len(f)!=133: raise AssertionError('feature count')
    return np.array(f,float)


def messages(block):
    o=parse(block)
    spec=ROOT/'config/main_experiment'
    system=(spec/'system.txt').read_text().rstrip('\n')
    common=(spec/'user_prefix.txt').read_text().rstrip('\n')
    rule=(spec/f'rule_{o["arm"]}.txt').read_text().rstrip('\n')
    user=common+'\nSampling rule: '+rule+'\n'+block+'\nReturn the four full-archive estimates in the specified JSON format.'
    return [{'role':'system','content':system},{'role':'user','content':user}]
