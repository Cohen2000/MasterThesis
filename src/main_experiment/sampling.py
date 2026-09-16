import ctypes
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import ROOT, seed, rng, sha, write_json, read_json, atomic_npz

class Walk:
    def __init__(self,g,build_dir):
        build=Path(build_dir); build.mkdir(parents=True,exist_ok=True)
        src=Path(__file__).with_name('walk_kernel.cpp')
        lib=build/f'walk_{sha(src)[:16]}.so'
        if not lib.exists():
            tmp=lib.with_suffix('.tmp.so')
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
            tmp.replace(lib)
        self.lib=ctypes.CDLL(str(lib)); self.g=g
        fn=self.lib.walks
        fn.argtypes=[ctypes.c_int64,ctypes.c_int64]+[ctypes.c_void_p]*7+[ctypes.c_int64,ctypes.c_int64]+[ctypes.c_void_p]*4
        fn.restype=None
        u,v=g.ends.T; nodes=np.r_[u,v]; order=np.argsort(nodes,kind='stable')
        self.neighbors=np.ascontiguousarray(np.r_[v,u][order],dtype=np.int64)
        self.edges=np.ascontiguousarray(np.tile(np.arange(g.D),2)[order],dtype=np.int64)
        self.ptr=np.r_[0,np.cumsum(np.bincount(nodes,minlength=g.N))].astype(np.int64)
        weights=g.m[self.edges]; self.cum=np.cumsum(weights).astype(np.int64)
        starts=self.ptr[:-1]
        self.cum-=np.repeat(np.r_[0,np.cumsum(weights)[starts[1:]-1]],np.diff(self.ptr))
        mat=coo_matrix((np.ones(len(nodes)),(nodes,np.r_[v,u])),shape=(g.N,g.N)).tocsr()
        _,comp=connected_components(mat,directed=False)
        totals=np.bincount(comp[u],weights=g.m).astype(np.int64)
        self.component_volume=np.ascontiguousarray(totals[comp])

    def run(self,seeds,L,traversals=False):
        if not isinstance(L,int) or L<0: raise ValueError('invalid L')
        ss=np.asarray(seeds,dtype=np.uint64)
        delta=np.zeros(L+1,dtype=np.int64); volumes=np.zeros(len(ss),dtype=np.int64)
        counts=np.zeros((len(ss),self.g.D),dtype=np.int64) if traversals else None
        executed=np.zeros(len(ss),dtype=np.int64)
        args=[self.ptr,self.neighbors,self.edges,self.cum,self.g.m,self.component_volume,ss]
        self.lib.walks(self.g.N,self.g.D,*[a.ctypes.data for a in args],len(ss),L,
                       delta.ctypes.data,volumes.ctypes.data,
                       counts.ctypes.data if counts is not None else None,executed.ctypes.data)
        return np.cumsum(delta),volumes,counts,executed


def budget_parameters(g):
    B=g.B
    if B<=0 or B>g.M or g.N<2: raise ValueError('undefined budget')
    n=np.arange(g.N+1,dtype=np.int64)
    expected=g.M*(n.astype(float)*(n-1))/(g.N*(g.N-1))
    panel=int(np.argmin(abs(expected-B)))
    return {'B':B,'p':B/g.M,'n_panel':panel,'node_expected_events':float(expected[panel]),
            'node_relative_budget_error':float((expected[panel]-B)/B)}


def calibrate(g,out,build):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    engine=Walk(g,build); C=min(100*g.D,1_000_000); B=g.B
    seeds=[seed('walk_calibration',g.key,'S',i) for i in range(1,257)]
    calfile=out/'calibrated.json'; timefile=out/'timing.json'
    # Timing before full calibration; same path prefix, no extra research draw.
    # Skipped once calibration is complete: the estimate guards work that is already
    # done, and rerunning it would rewrite a finished checkpoint on every resume.
    if calfile.exists() and timefile.exists():
        print(f'{g.key}: walk pilot reused',flush=True)
    else:
        t=time.perf_counter(); engine.run(seeds[:8],min(C,2048)); dt=time.perf_counter()-t
        write_json(timefile,{'pilot_paths':8,'pilot_L':min(C,2048),'seconds':dt,
                   'upper_bound_seconds_linear':dt*(256+4096)*C/(8*min(C,2048))})
        print(f'{g.key}: walk pilot {dt:.3f}s',flush=True)
    if calfile.exists(): cal=read_json(calfile); L=cal['L']
    else:
        bound=1; start=time.perf_counter()
        while True:
            f=out/f'prefix_{bound}.npz'
            if f.exists():
                with np.load(f) as z: total=z['total']
            else:
                total,_,_,_=engine.run(seeds,bound)
                atomic_npz(f,total=total)
            if total[-1]>=256*B or bound==C: break
            bound=min(2*bound,C)
        reached=bool(total[-1]>=256*B)
        if not reached: L=C
        else:
            # Integer bisection on the monotone common-path prefix curve.
            lo,hi=0,bound
            while hi-lo>1:
                mid=(lo+hi)//2
                if total[mid]>=256*B: hi=mid
                else: lo=mid
            L=min((lo,hi),key=lambda l:(abs(int(total[l])-256*B),l))
        cal={'L':L,'C':C,'calibration_paths':256,'calibration_mean':float(total[L]/256),
             'search_limit_reached_without_budget':not reached,'seconds':time.perf_counter()-start}
        write_json(calfile,cal)
    # Fixed key order: a reloaded calibrated.json carries sorted keys, a freshly computed
    # one carries literal order, and the difference would reach budget_summary.csv columns.
    order=('L','C','calibration_paths','calibration_mean',
           'search_limit_reached_without_budget','seconds')
    if set(cal)!=set(order): raise ValueError('unexpected calibration keys')
    cal={k:cal[k] for k in order}
    # Validation checkpoints of 128 walks; no adaptation of L.
    volumes=[]
    def extend(target):
        for first in range(len(volumes)+1,target+1,128):
            last=min(first+127,target); ids=list(range(first,last+1))
            vs=[seed('walk_validation',g.key,'S',i) for i in ids]
            f=out/f'validation_{first}_{last}.json'
            if f.exists(): v=read_json(f)['volumes']
            else:
                _,v,_,executed=engine.run(vs,L)
                v=v.tolist(); write_json(f,{'volumes':v,'executed_transitions':int(executed.sum())})
            volumes.extend(v)
    extend(1024)
    mcse=lambda:float(np.std(volumes,ddof=1)/np.sqrt(len(volumes)))
    if mcse()/B>.01: extend(4096)
    mean=float(np.mean(volumes)); se=mcse()
    reasons=[]
    if cal['search_limit_reached_without_budget']: reasons.append('calibration_cap')
    if abs(mean-B)/B>.05: reasons.append('validation_mean_outside_5_percent')
    if se/B>.01: reasons.append('validation_mcse_above_1_percent')
    result={**budget_parameters(g),**cal,'validation_n':len(volumes),'validation_mean':mean,
            'validation_mcse':se,'validation_relative_error':(mean-B)/B,
            'budget_matched':not reasons,'unmatched_reasons':reasons}
    write_json(out/'budget.json',result)
    return result,engine


def draw(g,arm,index,domain,budget,walk=None):
    r=rng(domain,g.key,arm,index) if arm!='H' else None
    counts=None; re=None
    if arm=='R':
        panel=np.zeros(g.N,dtype=bool); panel[r.choice(g.N,budget['n_panel'],replace=False)]=True
        selected=panel[g.ends[:,0]] & panel[g.ends[:,1]]
        counts=g.counts*selected[:,None]
    elif arm=='S':
        _,_,rr,_=walk.run([seed(domain,g.key,arm,index)],int(budget['L']),True)
        re=rr[0]; counts=g.counts*(re>0)[:,None]
    elif arm=='H':
        counts=g.counts.copy(); counts[:,:2]=0
    elif arm=='B':
        keep=r.random(g.M)<budget['p']
        counts=np.bincount(g.pair[keep]*5+g.w[keep],minlength=g.D*5).reshape(-1,5)
    else: raise ValueError('unknown arm')
    return counts,re
