from dataclasses import dataclass
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from dataset_census import parse_audited
from .common import ROOT, sha, write_json, atomic_npz, read_json, BUDGET_FRACTION

@dataclass
class Graph:
    key: str
    u: np.ndarray
    v: np.ndarray
    t: np.ndarray
    w: np.ndarray
    pair: np.ndarray
    ends: np.ndarray
    counts: np.ndarray
    horizon: tuple
    @property
    def N(self): return int(self.ends.max())+1
    @property
    def D(self): return len(self.ends)
    @property
    def M(self): return len(self.t)
    @property
    def K(self): return (self.counts>0).sum(axis=1)
    @property
    def truth(self): return [float(np.mean(self.K>=k)) for k in range(2,6)]
    @property
    def M_suffix(self):
        """Events in windows 3-5. The previous design used this as the budget."""
        return int(self.counts[:,2:].sum())
    @property
    def B(self):
        """Expected observed event volume: a fixed share of the full archive."""
        return int(round(BUDGET_FRACTION*self.M))
    @property
    def m(self): return self.counts.sum(axis=1)


def canonical(key, frame, proximity=False, horizon=None):
    x=frame[['u','v','t']].copy()
    if x[['u','v']].isna().any().any() or not np.isfinite(x.t.to_numpy(float)).all():
        raise ValueError('non-finite or missing canonical event')
    x.u=x.u.astype(str); x.v=x.v.astype(str)
    self_count=int((x.u==x.v).sum()); x=x[x.u!=x.v].copy()
    lo=np.minimum(x.u.to_numpy(),x.v.to_numpy()); hi=np.maximum(x.u.to_numpy(),x.v.to_numpy())
    x['u']=lo; x['v']=hi
    dup=int(x.duplicated(['u','v','t']).sum())
    if proximity: x=x.drop_duplicates(['u','v','t'])
    if x.empty: raise ValueError('empty full archive')
    x=x.sort_values(['u','v','t'],kind='stable').reset_index(drop=True)
    bounds=tuple(map(float,horizon or (x.t.min(),x.t.max())))
    if not bounds[0]<bounds[1]: raise ValueError('degenerate horizon')
    if x.t.min()<bounds[0] or x.t.max()>bounds[1]: raise ValueError('event outside fixed horizon')
    labels=np.unique(np.r_[x.u.to_numpy(),x.v.to_numpy()])
    u=np.searchsorted(labels,x.u).astype(np.int64); v=np.searchsorted(labels,x.v).astype(np.int64)
    ends,pair=np.unique(np.column_stack([u,v]),axis=0,return_inverse=True)
    t=x.t.to_numpy(float)
    cuts=np.array([bounds[0]+(bounds[1]-bounds[0])*j/5 for j in range(1,5)])
    w=np.searchsorted(cuts,t,side='right').astype(np.int64)
    counts=np.bincount(pair*5+w,minlength=len(ends)*5).reshape(-1,5).astype(np.int64)
    g=Graph(key,u,v,t,w,pair.astype(np.int64),ends,counts,bounds)
    # Graph-level validity, independent of whichever budget the design uses:
    # the suffix must contain events, otherwise arm H is degenerate. Budget
    # feasibility itself is checked in budget_parameters.
    if g.N<2 or g.D<1 or g.M_suffix<=0: raise ValueError(f'{key}: invalid full archive or empty suffix')
    return g,x,{'self_events_removed':self_count,'same_dyad_time_duplicates':dup,
                'duplicates_removed':dup if proximity else 0,'deduplicate_proximity':proximity}


def save_graph(out,g,manifest,frame=None):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    arrays={k:getattr(g,k) for k in ['u','v','t','w','pair','ends','counts']}
    atomic_npz(out/'graph.npz',**arrays,horizon=np.array(g.horizon))
    if frame is not None:
        # Stable sorted canonical export; source IDs and unmodified source timestamps.
        frame.to_csv(out/'canonical.csv',index=False,float_format='%.17g',lineterminator='\n')
        manifest['canonical_sha256']=sha(out/'canonical.csv')
    manifest.update(key=g.key,N_full=g.N,D_full=g.D,M_full=g.M,B=g.B,
                    horizon=list(g.horizon),truth=g.truth,events_per_window=g.counts.sum(0).tolist(),
                    graph_sha256=sha(out/'graph.npz'))
    write_json(out/'manifest.json',manifest)


def load_graph(out):
    out=Path(out); m=read_json(out/'manifest.json')
    if sha(out/'graph.npz')!=m['graph_sha256']: raise ValueError('graph checksum mismatch')
    with np.load(out/'graph.npz') as z:
        return Graph(m['key'],**{k:z[k] for k in ['u','v','t','w','pair','ends','counts']},horizon=tuple(z['horizon']))


def prepare_real(key,raw_dir,out):
    spec=yaml.safe_load((ROOT/'config/datasets.yaml').read_text())['datasets'][key]
    path=Path(raw_dir)/spec['file']
    raw,q=parse_audited(path,spec['format'],spec.get('census_audit',{}))
    g,x,clean=canonical(key,raw,proximity=key.startswith('sp_') or key=='copenhagen_bluetooth')
    manifest=dict(source_family=key,raw_file=path.name,raw_sha256=sha(path),
                  provider=spec['page_url'],format=spec['format'],parser=q,cleaning=clean,
                  timestamp_unit=spec['census_audit']['timestamp_unit'])
    if key=='copenhagen_bluetooth':
        original=Path(raw_dir)/'bt_symmetric.csv'
        if not original.exists(): raise FileNotFoundError('CNS original bt_symmetric.csv required')
        original_md5=hashlib.md5(original.read_bytes()).hexdigest()
        if original_md5!='98892459f73e774cf79e7977edfeee3e':
            raise ValueError('CNS original bytes differ from Figshare v1 file 14000795')
        bt=pd.read_csv(original)
        bt.columns=[c.lstrip('# ').strip() for c in bt.columns]
        bt=bt[bt.user_b>=0]
        source=bt.rename(columns={'user_a':'u','user_b':'v','timestamp':'t'})[['u','v','t']]
        _,cx,cq=canonical(key,source,proximity=True)
        # Compare numeric identities; parser may preserve textual integer IDs.
        a=x.copy(); b=cx.copy()
        for f in (a,b):
            f['u']=pd.to_numeric(f.u); f['v']=pd.to_numeric(f.v)
            f[['u','v']]=np.sort(f[['u','v']].to_numpy(),axis=1)
        a=a.sort_values(['u','v','t']).reset_index(drop=True)
        b=b.sort_values(['u','v','t']).reset_index(drop=True)
        same=np.array_equal(a.to_numpy(),b.to_numpy())
        manifest['copenhagen']={'article_id':7267433,'version':1,'file_id':14000795,
            'release_md5_verified':original_md5,'original_sha256':sha(original),'original_cleaning':cq,'legacy_export_matches':same,
            'transform_script_sha256':sha(Path(__file__))}
        if not same: raise ValueError('CNS original and local export differ after frozen cleaning')
        x=b  # Numeric min/max endpoints, sorted by (u,v,t), as defined for CNS.
    save_graph(out,g,manifest,x)
    return g
