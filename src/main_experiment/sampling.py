import ctypes
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import (ROOT, seed, rng, sha, write_json, read_json, atomic_npz, BUDGET_FRACTION,
                     BUDGET_TOLERANCE, H_CAP, H_VARIANT, LEGACY_H, ARM_ID, draws_for)

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


def _panel_for(total,B,N):
    """Integer panel size whose expected observed volume is closest to the budget.

    A uniform node panel of size n includes a dyad with probability
    n(n-1)/(N(N-1)), independently of that dyad's activity, so the expected
    observed event count is that share of `total`.
    """
    n=np.arange(N+1,dtype=np.int64)
    expected=total*(n.astype(float)*(n-1))/(N*(N-1))
    panel=int(np.argmin(abs(expected-B)))
    return panel,float(expected[panel])


def _dyads_for(capped_total,B,D):
    """Integer dyad count in 1..D whose expected observed volume is closest to B.

    A uniform sample of d of the D active dyads contains each dyad with
    probability d/D, so the expected retained volume is exactly d*C/D with
    C = sum_e min(cap, m_e). Ties choose the smaller d. The choice uses only
    full-archive totals known to the experimenter, never a realised sample.
    """
    d=np.arange(1,D+1,dtype=np.int64)
    expected=d.astype(float)*capped_total/D
    k=int(np.argmin(abs(expected-B)))
    return int(d[k]),float(expected[k])


def recent_counts(counts,cap=H_CAP):
    """Window counts of the cap most recent events of every row.

    Windows are ordered in time and every event of window j is later than every
    event of window j-1 (cut points use side='right', so a timestamp on a cut
    belongs to the later window and ties never straddle two windows). The cap most
    recent events therefore fill the windows from window 5 backwards; which of
    several events inside one window are taken does not change any count.
    tests/frozen_main/test_hrecent5.py checks this against an explicit
    timestamp sort.
    """
    counts=np.asarray(counts,dtype=np.int64)
    out=np.zeros_like(counts)
    left=np.full(len(counts),cap,dtype=np.int64)
    for j in range(counts.shape[1]-1,-1,-1):
        take=np.minimum(counts[:,j],left)
        out[:,j]=take; left-=take
    return out


def reservoir_counts(counts,cap,r):
    """Window counts of min(cap, m_e) events drawn uniformly without replacement.

    Development comparison only: the same number of events per dyad as
    recent_counts, but a uniform subset of the dyad's history (a size-cap
    reservoir in the sense of Vitter 1985) instead of its most recent part. The
    multivariate hypergeometric draw is done one event at a time, which is exact.
    """
    rem=np.asarray(counts,dtype=np.int64).copy()
    out=np.zeros_like(rem)
    need=np.minimum(rem.sum(1),cap)
    for step in range(cap):
        active=need>step
        if not active.any(): break
        sub=rem[active]; tot=sub.sum(1)
        u=np.floor(r.random(len(sub))*tot).astype(np.int64)
        pick=(np.cumsum(sub,axis=1)<=u[:,None]).sum(1)
        idx=np.flatnonzero(active)
        rem[idx,pick]-=1; out[idx,pick]+=1
    return out


def h_parameters(g,B,cap=H_CAP):
    """Arm H parameters with each budget condition reported on its own.

    target_unreachable: even every active dyad keeps fewer than B events.
    saturated: the closest integer is d=D, so the sample is the whole population.
    within_tolerance: the unchanged 5 % rule on expected volume.
    These are not the same: a target can be unreachable and still within
    tolerance, and d=D can be the closest integer without the target being
    unreachable. The cap is a design constant and does not depend on the source.
    """
    capped=int(np.minimum(g.m,cap).sum())
    d,expected=_dyads_for(capped,B,g.D)
    rel=float((expected-B)/B)
    return {'h_variant':H_VARIANT,'h_cap':cap,'C_cap':capped,'capped_share':capped/g.M,
            'n_dyads':d,'h_dyad_share':d/g.D,'h_expected_events':expected,
            'h_relative_budget_error':rel,
            'h_target_unreachable':bool(capped<B),'h_saturated':bool(d==g.D),
            'h_within_tolerance':bool(abs(rel)<=BUDGET_TOLERANCE),
            'h_capped_dyad_share':float(np.mean(g.m>cap)),
            'h_at_cap_dyad_share':float(np.mean(g.m>=cap))}


def legacy_suffix_parameters(g,B):
    """Arm H of budget10-20261001, kept as a development variant."""
    suffix=g.M_suffix
    if suffix<B: panel_h,exp_h,reasons=g.N,float(suffix),['suffix_smaller_than_budget']
    else: (panel_h,exp_h),reasons=_panel_for(suffix,B,g.N),[]
    return {'M_suffix':suffix,'suffix_share':suffix/g.M,'n_panel_suffix':panel_h,
            'suffix_expected_events':exp_h,'suffix_relative_budget_error':float((exp_h-B)/B),
            'panel_unmatched_reasons':reasons}


def budget_parameters(g):
    """Budget and per-arm parameters for the fixed-share design.

    B is BUDGET_FRACTION of the full archive. Arm B keeps each event with exactly
    that probability, arm R uses a uniform node panel, arm H a uniform sample of
    active dyads with capped recent histories. Budget matching is judged per arm
    and over all arms; the walk part is added by calibrate().
    """
    B=g.B
    if B<=0 or B>g.M or g.N<2: raise ValueError('undefined budget')
    panel,exp_r=_panel_for(g.M,B,g.N)
    rel_r=float((exp_r-B)/B)
    h=h_parameters(g,B)
    return {'design_arm_H':H_VARIANT,'B':B,'p':BUDGET_FRACTION,'budget_fraction':BUDGET_FRACTION,
            'budget_tolerance':BUDGET_TOLERANCE,
            'n_panel':panel,'node_expected_events':exp_r,'node_relative_budget_error':rel_r,
            **h,'legacy_suffix_panel':legacy_suffix_parameters(g,B)}


def calibrate(g,out,build):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    params=budget_parameters(g)          # raises early on an undefined budget
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
    walk_reasons=[]
    if cal['search_limit_reached_without_budget']: walk_reasons.append('calibration_cap')
    if abs(mean-B)/B>BUDGET_TOLERANCE: walk_reasons.append('validation_mean_outside_5_percent')
    if se/B>.01: walk_reasons.append('validation_mcse_above_1_percent')
    # Complete budget matching, not only the walk: every arm is judged on its own
    # expected-volume deviation. B keeps each event with probability exactly p.
    by_arm={'R':abs(params['node_relative_budget_error'])<=BUDGET_TOLERANCE,
            'S':not walk_reasons,
            'H':params['h_within_tolerance'],
            'B':True}
    reasons=[f'S:{x}' for x in walk_reasons]
    if not by_arm['R']: reasons.append('R:node_panel_outside_5_percent')
    if not by_arm['H']: reasons.append('H:expected_volume_outside_5_percent')
    result={**params,**cal,'validation_n':len(volumes),'validation_mean':mean,
            'validation_mcse':se,'validation_relative_error':(mean-B)/B,
            'walk_budget_matched':not walk_reasons,'walk_unmatched_reasons':walk_reasons,
            'budget_matched_by_arm':by_arm,
            'budget_matched':all(by_arm.values()),'unmatched_reasons':reasons}
    write_json(out/'budget.json',result)
    return result,engine


def _panel_mask(g,r,size):
    panel=np.zeros(g.N,dtype=bool); panel[r.choice(g.N,size,replace=False)]=True
    return panel[g.ends[:,0]] & panel[g.ends[:,1]]


def sample_dyads(g,index,domain,budget):
    """Indices of the H dyad sample; uniform without replacement over E_full."""
    r=rng(domain,g.key,ARM_ID['H'],index)
    if budget['h_saturated']:
        if index!=1: raise ValueError('a saturated H sample is deterministic; only index 1 exists')
        return np.arange(g.D)
    return np.sort(r.choice(g.D,budget['n_dyads'],replace=False))


def draw(g,arm,index,domain,budget,walk=None):
    # The number of draws is set by the caller from draws_for(); the one index rule
    # enforced here is that a saturated H sample exists only once (sample_dyads).
    if index<1: raise ValueError('sample indices start at 1')
    counts=None; re=None
    if arm=='R':
        r=rng(domain,g.key,ARM_ID[arm],index)
        counts=g.counts*_panel_mask(g,r,budget['n_panel'])[:,None]
    elif arm=='S':
        _,_,rr,_=walk.run([seed(domain,g.key,ARM_ID[arm],index)],int(budget['L']),True)
        re=rr[0]; counts=g.counts*(re>0)[:,None]
    elif arm=='H':
        # Uniform over active dyads, drawn without looking at any event count, so
        # the sampled dyads' true profile is a simple random sample of E_full.
        # Each sampled dyad then keeps its H_CAP most recent events.
        sel=sample_dyads(g,index,domain,budget)
        counts=np.zeros_like(g.counts)
        counts[sel]=recent_counts(g.counts[sel],budget['h_cap'])
    elif arm==LEGACY_H:
        # budget10-20261001 arm H: uniform node panel, then only windows 3-5.
        r=rng(domain,g.key,ARM_ID[arm],index)
        counts=g.counts*_panel_mask(g,r,budget['legacy_suffix_panel']['n_panel_suffix'])[:,None]
        counts=counts.copy(); counts[:,:2]=0
    elif arm=='B':
        r=rng(domain,g.key,ARM_ID[arm],index)
        keep=r.random(g.M)<budget['p']
        counts=np.bincount(g.pair[keep]*5+g.w[keep],minlength=g.D*5).reshape(-1,5)
    else: raise ValueError('unknown arm')
    return counts,re
