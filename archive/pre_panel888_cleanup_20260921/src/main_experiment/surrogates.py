"""P[w,t] on labeled canonical event records, permitting timestamp multiplicity.

Record i retains its parent dyad. Only timestamps are permuted. Never canonicalize
or deduplicate again: coincident events remain separate records for B thinning.
"""
from dataclasses import replace
import numpy as np
from .common import seed, rng, sha, read_json, write_json, DESIGN_VERSION
from .data import save_graph, load_graph


def shuffle(parent, index=0):
    domain='pwt_productive' if index==0 else 'pwt_null_diagnostic'
    t=rng(domain,parent.key,'',index).permutation(parent.t)
    cuts=np.array([parent.horizon[0]+(parent.horizon[1]-parent.horizon[0])*j/5 for j in range(1,5)])
    w=np.searchsorted(cuts,t,side='right').astype(np.int64)
    counts=np.bincount(parent.pair*5+w,minlength=parent.D*5).reshape(-1,5)
    return replace(parent,key=parent.key+'__pwt',t=t,w=w,counts=counts)


def collisions(g):
    order=np.lexsort((g.t,g.pair)); p=g.pair[order]; t=g.t[order]
    return int(np.sum((p[1:]==p[:-1]) & (t[1:]==t[:-1])))


def audit(parent,child):
    checks={
        'nodes':parent.N==child.N and np.array_equal(parent.u,child.u) and np.array_equal(parent.v,child.v),
        'support':np.array_equal(parent.ends,child.ends),
        'record_dyad_identity':np.array_equal(parent.pair,child.pair),
        'dyad_event_multiplicity':np.array_equal(parent.m,child.m),
        'total_events':parent.M==child.M,
        'timestamp_multiset':np.array_equal(np.sort(parent.t),np.sort(child.t)),
        'archive_bounds':parent.horizon==child.horizon,
        'count_representation':np.array_equal(child.counts,np.bincount(child.pair*5+child.w,minlength=child.D*5).reshape(-1,5))}
    if not all(checks.values()):raise AssertionError(checks)
    return {'checks':checks,'passed':True,'parent_collisions':collisions(parent),
            'surrogate_collisions':collisions(child),'events_removed_after_shuffle':0,
            'event_identity':'implicit zero-based index of parent canonical graph arrays; order unchanged',
            'representation':'labeled event multiset; simultaneous dyad records retain multiplicity'}


def prepare_surrogate(parent,out):
    out.mkdir(parents=True,exist_ok=True)
    g=shuffle(parent)
    report=audit(parent,g)
    if (out/'manifest.json').exists():
        stored=load_graph(out)
        for name in ('u','v','t','w','pair','ends','counts'):
            if not np.array_equal(getattr(g,name),getattr(stored,name)):raise ValueError('surrogate checkpoint changed')
    else:
        save_graph(out,g,{'design_version':DESIGN_VERSION,'parent':parent.key,
            'seed':seed('pwt_productive',parent.key),'null_model':'P[w,t]',
            'invariants':report,'selection':'single pre-fixed shuffle, no rejection/resampling'})
    write_json(out/'invariant_audit.json',report)
    return g
