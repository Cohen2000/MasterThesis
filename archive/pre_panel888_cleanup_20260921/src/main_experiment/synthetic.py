"""Algorithms fixed in docs/PROTOCOL_PANEL888_20260921.md before coding.

The frozen eight main-test instances are produced by generate_pair with the
frozen defaults; the training/development pool (see pool.py) uses the same two
families through generate_one with varied parameters and independent streams.
Parameter names are chosen so the frozen defaults reproduce bit-identically:
the activity-driven tail is carried as tail = gamma - 1 (default exactly 1.8)
because 2.8 - 1.0 is not the float 1.8 and would perturb every later draw.
"""
import hashlib
import numpy as np
import pandas as pd
from .common import rng, seed
from .data import canonical

def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

# Frozen main-test settings; also the defaults of the parameterised generators.
DAR_DEFAULTS=dict(N=500,E=5000,chi=.2,nu=1.)
AD_DEFAULTS=dict(N=500,tail=1.8,eps=.01,eta=1.,rounds=1000,c=1.)


def dar_latents(r,N,E,chi,nu):
    """Backbone and per-window latents shared by every alpha of one draw."""
    candidates=np.column_stack(np.triu_indices(N,1))
    edges=candidates[np.sort(r.choice(len(candidates),E,replace=False))]
    initial=r.random(E)<chi
    copy=r.random((4,E)); refresh=r.random((4,E))<chi
    counts=1+r.poisson(nu,(5,E))
    offsets=np.r_[0,np.cumsum(counts.ravel())]
    pos=r.random(int(offsets[-1]))
    return dict(edges=edges,initial=initial,copy=copy,refresh=refresh,counts=counts,
                offsets=offsets,pos=pos,E=E)


def dar_rows(L,alpha):
    """Binary copy-or-refresh recurrence, then the positive event layer."""
    E=L['E']; states=np.empty((5,E),bool); states[0]=L['initial']
    for j in range(1,5): states[j]=np.where(L['copy'][j-1]<alpha,states[j-1],L['refresh'][j-1])
    rows=[]
    for j,e in zip(*np.where(states)):
        ix=j*E+e; a,b=L['offsets'][ix:ix+2]
        for t in (j+L['pos'][a:b])/5: rows.append((int(L['edges'][e,0]),int(L['edges'][e,1]),float(t)))
    return rows,states


def ad_latents(r,N,tail,eps,eta,rounds):
    activities=eta*(eps**(-tail)+r.random(N)*(1-eps**(-tail)))**(-1/tail)
    return dict(activities=activities,activation=r.random((rounds,N)),
                decision=r.random((rounds,N)),partner=r.random((rounds,N)),N=N,rounds=rounds)


def ad_rows(L,mode,c):
    """Synchronous rounds; one undirected contact per dyad per round."""
    N=L['N']; rounds=L['rounds']
    memory=[set() for _ in range(N)]; rows=[]; mutual=0
    for t in range(rounds):
        contacts=set(); initiations=0
        for i in np.flatnonzero(L['activation'][t]<L['activities']):
            if mode=='memoryless':
                # Identical to indexing sorted([j for j in range(N) if j!=i]), without building it.
                k=int(L['partner'][t,i]*(N-1)); j=k if k<i else k+1
            else:
                old=memory[i]
                new=len(old)==0 or (len(old)<N-1 and L['decision'][t,i]<c/(len(old)+c))
                choices=[j for j in range(N) if j!=i and j not in old] if new else sorted(old)
                j=choices[int(L['partner'][t,i]*len(choices))]
            contacts.add((min(int(i),j),max(int(i),j))); initiations+=1
        mutual+=initiations-len(contacts)
        for i,j in sorted(contacts):
            rows.append((i,j,(t+.5)/rounds)); memory[i].add(j); memory[j].add(i)
    return rows,mutual


def generate_pair(family,replicate):
    """The frozen main-test pairs: two modes sharing every latent quantity."""
    pair_id=f'{family}_pair_r{replicate}'; r=rng('graph',pair_id)
    common={'family':family,'replicate':replicate,'seed':seed('graph',pair_id),
            'horizon':[0,1],'algorithm_document':'docs/PROTOCOL_PANEL888_20260921.md'}
    if family=='dar':
        L=dar_latents(r,**DAR_DEFAULTS)
        names=dict(backbone='edges',initial='initial',copy='copy',refresh='refresh',
                   counts='counts',positions='pos')
        common['shared_latents']={out:array_hash(L[k]) for out,k in names.items()}
        for mode,alpha in [('a0',0.),('a08',.8)]:
            rows,states=dar_rows(L,alpha)
            key=f'dar_{mode}_r{replicate}'
            g,x,q=canonical(key,pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))
            yield g,x,{**common,'mode':mode,'alpha':alpha,'cleaning':q,'states_sha256':array_hash(states)}
    elif family=='ad':
        L=ad_latents(r,**{k:AD_DEFAULTS[k] for k in ('N','tail','eps','eta','rounds')})
        common['shared_latents']={k:array_hash(L[k]) for k in ('activities','activation','decision','partner')}
        for mode in ['memoryless','memory']:
            rows,mutual=ad_rows(L,mode,AD_DEFAULTS['c'])
            key=f'ad_{mode}_r{replicate}'
            g,x,q=canonical(key,pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))
            yield g,x,{**common,'mode':mode,'mutual_initiations_collapsed':mutual,'cleaning':q,
                       'event_rule':'synchronous_unique_undirected_dyad_per_round_user_confirmed'}
    else: raise ValueError(family)


def generate_one(key,family,params,domain='pool'):
    """One pool graph from its own stream; no latent quantity is shared with any other graph."""
    r=rng(domain,key)
    meta={'family':family,'horizon':[0,1],'parameters':dict(params),
          'seed':seed(domain,key),'algorithm_document':'docs/PROTOCOL_PANEL888_20260921.md',
          'partition_stream':domain}
    if family=='dar':
        p={**DAR_DEFAULTS,**params}; alpha=p.pop('alpha')
        L=dar_latents(r,**{k:p[k] for k in ('N','E','chi','nu')})
        rows,states=dar_rows(L,alpha)
        extra={'states_sha256':array_hash(states),
               'latents_sha256':{k:array_hash(L[k]) for k in ('edges','initial','copy','refresh','counts','pos')}}
    elif family=='ad':
        p={**AD_DEFAULTS,**params}; mode=p.pop('mode')
        L=ad_latents(r,**{k:p[k] for k in ('N','tail','eps','eta','rounds')})
        rows,mutual=ad_rows(L,mode,p['c'])
        extra={'mutual_initiations_collapsed':mutual,
               'event_rule':'synchronous_unique_undirected_dyad_per_round_user_confirmed',
               'latents_sha256':{k:array_hash(L[k]) for k in ('activities','activation','decision','partner')}}
    else: raise ValueError(family)
    g,x,q=canonical(key,pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))
    return g,x,{**meta,**extra,'cleaning':q}
