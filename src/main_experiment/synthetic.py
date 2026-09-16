"""Algorithms fixed in docs/MAIN_EXPERIMENT_IMPLEMENTATION.md before coding."""
import hashlib
import numpy as np
import pandas as pd
from .common import rng, seed
from .data import canonical, save_graph

def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def generate_pair(family,replicate):
    pair_id=f'{family}_pair_r{replicate}'; r=rng('graph',pair_id)
    common={'family':family,'replicate':replicate,'seed':seed('graph',pair_id),
            'horizon':[0,1],'algorithm_document':'docs/MAIN_EXPERIMENT_IMPLEMENTATION.md'}
    if family=='dar':
        candidates=np.column_stack(np.triu_indices(500,1))
        edges=candidates[np.sort(r.choice(len(candidates),5000,replace=False))]
        initial=r.random(5000)<.2
        copy=r.random((4,5000)); refresh=r.random((4,5000))<.2
        counts=1+r.poisson(1,(5,5000))
        offsets=np.r_[0,np.cumsum(counts.ravel())]
        pos=r.random(int(offsets[-1]))
        common['shared_latents']={k:array_hash(v) for k,v in
             dict(backbone=edges,initial=initial,copy=copy,refresh=refresh,counts=counts,positions=pos).items()}
        for mode,alpha in [('a0',0.),('a08',.8)]:
            states=np.empty((5,5000),bool); states[0]=initial
            for j in range(1,5): states[j]=np.where(copy[j-1]<alpha,states[j-1],refresh[j-1])
            rows=[]
            for j,e in zip(*np.where(states)):
                ix=j*5000+e; a,b=offsets[ix:ix+2]
                for t in (j+pos[a:b])/5: rows.append((int(edges[e,0]),int(edges[e,1]),float(t)))
            key=f'dar_{mode}_r{replicate}'
            g,x,q=canonical(key,pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))
            yield g,x,{**common,'mode':mode,'alpha':alpha,'cleaning':q,'states_sha256':array_hash(states)}
    elif family=='ad':
        activities=(.01**(-1.8)+r.random(500)*(1-.01**(-1.8)))**(-1/1.8)
        activation=r.random((1000,500)); decision=r.random((1000,500)); partner=r.random((1000,500))
        common['shared_latents']={k:array_hash(v) for k,v in
            dict(activities=activities,activation=activation,decision=decision,partner=partner).items()}
        for mode in ['memoryless','memory']:
            memory=[set() for _ in range(500)]; rows=[]; mutual=0
            for t in range(1000):
                contacts=set(); initiations=0
                for i in np.flatnonzero(activation[t]<activities):
                    if mode=='memoryless': choices=[j for j in range(500) if j!=i]
                    else:
                        old=memory[i]
                        new=len(old)==0 or (len(old)<499 and decision[t,i]<1/(len(old)+1))
                        choices=[j for j in range(500) if j!=i and j not in old] if new else sorted(old)
                    j=choices[int(partner[t,i]*len(choices))]
                    contacts.add((min(int(i),j),max(int(i),j))); initiations+=1
                mutual+=initiations-len(contacts)
                for i,j in sorted(contacts):
                    rows.append((i,j,(t+.5)/1000)); memory[i].add(j); memory[j].add(i)
            key=f'ad_{mode}_r{replicate}'
            g,x,q=canonical(key,pd.DataFrame(rows,columns=['u','v','t']),horizon=(0,1))
            yield g,x,{**common,'mode':mode,'mutual_initiations_collapsed':mutual,'cleaning':q,
                       'event_rule':'synchronous_unique_undirected_dyad_per_round_user_confirmed'}
    else: raise ValueError(family)
