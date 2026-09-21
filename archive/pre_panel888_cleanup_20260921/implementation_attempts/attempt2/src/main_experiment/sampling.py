from .common import parent_source
import ctypes
from pathlib import Path
import subprocess
import time
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from .common import (ROOT, seed, rng, sha, write_json, read_json, atomic_npz, BUDGET_FRACTION,
                     BUDGET_TOLERANCE, H_CAP, H_VARIANT, LEGACY_H, ARM_ID, draws_for,
                     COVERAGE_FRACTION, MATCHED_QUANTITY, DESIGN_VERSION, H_FRACTION)

class Walk:
    """Simple random walk: uniform start vertex, uniform current neighbor.

    volume selects what a first discovery adds to the recorded volume: 'events'
    (m_e, the superseded event budget) or 'cells' (K_e, the active dyad-windows
    the current design is matched on). Discovery weights never affect transitions.
    """
    def __init__(self,g,build_dir,volume='events'):
        build=Path(build_dir); build.mkdir(parents=True,exist_ok=True)
        src=Path(__file__).with_name('walk_kernel.cpp')
        lib=build/f'walk_{sha(src)[:16]}.so'
        if not lib.exists():
            tmp=lib.with_suffix('.tmp.so')
            subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(src),'-o',str(tmp)],check=True)
            tmp.replace(lib)
        self.lib=ctypes.CDLL(str(lib)); self.g=g
        fn=self.lib.walks
        fn.argtypes=[ctypes.c_int64,ctypes.c_int64]+[ctypes.c_void_p]*6+[ctypes.c_int64,ctypes.c_int64]+[ctypes.c_void_p]*4
        fn.restype=None
        u,v=g.ends.T; nodes=np.r_[u,v]; order=np.argsort(nodes,kind='stable')
        self.neighbors=np.ascontiguousarray(np.r_[v,u][order],dtype=np.int64)
        self.edges=np.ascontiguousarray(np.tile(np.arange(g.D),2)[order],dtype=np.int64)
        self.ptr=np.r_[0,np.cumsum(np.bincount(nodes,minlength=g.N))].astype(np.int64)
        mat=coo_matrix((np.ones(len(nodes)),(nodes,np.r_[v,u])),shape=(g.N,g.N)).tocsr()
        self.n_components,comp=connected_components(mat,directed=False)
        self.components=comp
        if volume not in ('events','cells'): raise ValueError(volume)
        self.volume=volume
        self.weight=np.ascontiguousarray(g.m if volume=='events' else g.K,dtype=np.int64)
        totals=np.bincount(comp[u],weights=self.weight).astype(np.int64)
        self.component_volume=np.ascontiguousarray(totals[comp])

    def run(self,seeds,L,traversals=False):
        if not isinstance(L,int) or L<0: raise ValueError('invalid L')
        ss=np.asarray(seeds,dtype=np.uint64)
        delta=np.zeros(L+1,dtype=np.int64); volumes=np.zeros(len(ss),dtype=np.int64)
        counts=np.zeros((len(ss),self.g.D),dtype=np.int64) if traversals else None
        executed=np.zeros(len(ss),dtype=np.int64)
        args=[self.ptr,self.neighbors,self.edges,self.weight,self.component_volume,ss]
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


def history_start(g,h=H_FRACTION):
    """Common query at archive end; time-based access, independent of W."""
    from decimal import Decimal
    if not 0<h<=1: raise ValueError('history fraction must be in (0,1]')
    start_fraction=float(Decimal(1)-Decimal(str(h)))
    return g.horizon[0]+start_fraction*(g.horizon[1]-g.horizon[0])


def history_counts(g,h=H_FRACTION):
    keep=g.t>=history_start(g,h)
    width=g.counts.shape[1]
    return np.bincount(g.pair[keep]*width+g.w[keep],minlength=g.D*width).reshape(g.D,width)


def h_parameters(g,T,h=H_FRACTION):
    """Calibrate pi(n)*sum J_e to T; no outcome-dependent choice of h."""
    recent=history_counts(g,h)
    visible=int((recent>0).sum())
    n,expected=_panel_for(visible,T,g.N)
    # If the suffix is empty the maximally accessible panel still yields zero.
    if not visible: n=g.N
    pi=n*(n-1)/(g.N*(g.N-1))
    rel=float((expected-T)/T)
    return {'h_variant':H_VARIANT,'history_fraction':h,'history_start':history_start(g,h),
            'query_time':float(g.horizon[1]),'J_total':visible,'n_panel_history':n,
            'h_node_share':n/g.N,'h_panel_dyad_inclusion':pi,'h_expected_cells':expected,
            'h_expected_events':float(pi*recent.sum()),
            'h_relative_budget_error':rel,'h_target_unreachable':bool(visible<T),
            'h_saturated':bool(n==g.N),'h_within_tolerance':bool(abs(rel)<=BUDGET_TOLERANCE)}


def bernoulli_p(g,T):
    """Retention probability whose expected observed active dyad-windows equal T.

    A cell (dyad, window) with n events is observed with probability 1-(1-p)^n,
    so the expectation is increasing in p; bisection to machine precision.
    """
    n=g.counts[g.counts>0].astype(float)
    f=lambda p: float(np.sum(-np.expm1(n*np.log1p(-p)))) if p<1 else float(len(n))
    lo,hi=0.,1.
    for _ in range(80):
        mid=(lo+hi)/2
        if f(mid)<T: lo=mid
        else: hi=mid
    p=hi
    return p,f(p)


def legacy_suffix_parameters(g,B):
    """Arm H of budget10-20261001, kept as a development variant (event budget)."""
    suffix=g.M_suffix
    if suffix<B: panel_h,exp_h,reasons=g.N,float(suffix),['suffix_smaller_than_budget']
    else: (panel_h,exp_h),reasons=_panel_for(suffix,B,g.N),[]
    return {'M_suffix':suffix,'suffix_share':suffix/g.M,'n_panel_suffix':panel_h,
            'suffix_expected_events':exp_h,'suffix_relative_budget_error':float((exp_h-B)/B),
            'panel_unmatched_reasons':reasons}


def budget_parameters(g):
    """Per-arm parameters of the cells10 design.

    Every arm is matched on the expected number of observed active dyad-windows,
    T = COVERAGE_FRACTION * sum_e K_e. R: node panel with pi(n)*W closest to T
    (a panel keeps each dyad, and hence each of its cells, with probability pi(n)).
    H: uniform node panel with time suffix, see h_parameters. B: retention probability p solved exactly.
    S: walk length calibrated by calibrate() on cells. Expected events and dyads
    are reported as descriptors; they are not matched. Only the experimenter uses
    full-archive quantities.
    """
    if g.B<=0 or g.B>g.M or g.N<2: raise ValueError('undefined budget')
    W=g.cells; T=COVERAGE_FRACTION*W
    panel,exp_r=_panel_for(W,T,g.N)
    pi=panel*(panel-1)/(g.N*(g.N-1))
    h=h_parameters(g,T)
    p,exp_b=bernoulli_p(g,T)
    return {'design_version':DESIGN_VERSION,'matched_quantity':MATCHED_QUANTITY,
            'design_arm_H':H_VARIANT,'coverage_fraction':COVERAGE_FRACTION,
            'active_dyad_windows':W,'T':T,'B':g.B,'budget_tolerance':BUDGET_TOLERANCE,
            'n_panel':panel,'node_expected_cells':exp_r,'node_relative_budget_error':float((exp_r-T)/T),
            'node_expected_events':pi*g.M,'node_dyad_share':pi,
            **h,
            'p':p,'bernoulli_expected_cells':exp_b,'bernoulli_relative_budget_error':float((exp_b-T)/T),
            'bernoulli_expected_events':p*g.M,
            'bernoulli_dyad_share':float(np.mean(-np.expm1(g.m*np.log1p(-p)))),
            'legacy_suffix_panel':legacy_suffix_parameters(g,g.B)}


def calibrate(g,out,build):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    params=budget_parameters(g)          # raises early on an undefined budget
    # SRW transition probabilities depend only on neighbor counts. Calibrate
    # discoveries on K_e with independent calibration/validation seed domains.
    engine=Walk(g,build,volume='cells'); C=min(100*g.D,1_000_000); B=params['T']
    seeds=[seed('walk_calibration_cells',g.key,ARM_ID['S'],i) for i in range(1,257)]
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
            vs=[seed('walk_validation_cells',g.key,ARM_ID['S'],i) for i in ids]
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
    ceiling=float(np.mean(engine.component_volume))
    if ceiling<B: walk_reasons.append('component_structural_ceiling_below_target')
    # Complete budget matching, not only the walk: every arm is judged on its own
    # expected-volume deviation. B keeps each event with probability exactly p.
    by_arm={'R':abs(params['node_relative_budget_error'])<=BUDGET_TOLERANCE,
            'S':not walk_reasons,
            'H':params['h_within_tolerance'],
            'B':abs(params['bernoulli_relative_budget_error'])<=BUDGET_TOLERANCE}
    reasons=[f'S:{x}' for x in walk_reasons]
    if not by_arm['R']: reasons.append('R:node_panel_outside_5_percent')
    if not by_arm['H']: reasons.append('H:expected_volume_outside_5_percent')
    if not by_arm['B']: reasons.append('B:expected_volume_outside_5_percent')
    result={**params,**cal,'walk_type':'simple_random_walk','walk_components':engine.n_components,
            'walk_expected_component_ceiling':ceiling,'walk_structural_target_unreachable':ceiling<B,
            'validation_n':len(volumes),'validation_mean':mean,
            'validation_mcse':se,'validation_relative_error':(mean-B)/B,
            'walk_budget_matched':not walk_reasons,'walk_unmatched_reasons':walk_reasons,
            'budget_matched_by_arm':by_arm,
            'budget_matched':all(by_arm.values()),'unmatched_reasons':reasons}
    write_json(out/'budget.json',result)
    return result,engine


def _panel_mask(g,r,size):
    panel=np.zeros(g.N,dtype=bool); panel[r.permutation(g.N)[:size]]=True
    return panel[g.ends[:,0]] & panel[g.ends[:,1]]


def history_panel_mask(g,index,domain,budget):
    """All full-archive dyads in the chosen node panel, including suffix-invisible ones.

    Common random ordering across h values is deliberate for paired sensitivity.
    Prefixes of a uniform permutation give uniform, nested panels at every size.
    """
    if budget['h_saturated'] and index!=1:
        raise ValueError('a saturated H panel has only one draw')
    r=rng(domain,parent_source(g.key),ARM_ID['H'],index)
    panel=np.zeros(g.N,dtype=bool)
    panel[r.permutation(g.N)[:budget['n_panel_history']]]=True
    return panel[g.ends[:,0]] & panel[g.ends[:,1]]


def draw(g,arm,index,domain,budget,walk=None):
    # The number of draws is set by the caller from draws_for(); the one index rule
    # enforced here is that a saturated H sample exists only once (sample_dyads).
    if index<1: raise ValueError('sample indices start at 1')
    counts=None; re=None
    if arm=='R':
        r=rng(domain,parent_source(g.key),ARM_ID[arm],index)
        counts=g.counts*_panel_mask(g,r,budget['n_panel'])[:,None]
    elif arm=='S':
        _,_,rr,_=walk.run([seed(domain,parent_source(g.key),ARM_ID[arm],index)],int(budget['L']),True)
        re=rr[0]; counts=g.counts*(re>0)[:,None]
    elif arm=='H':
        sel=history_panel_mask(g,index,domain,budget)
        counts=history_counts(g,budget['history_fraction'])*sel[:,None]
    elif arm==LEGACY_H:
        # budget10-20261001 arm H: uniform node panel, then only windows 3-5.
        r=rng(domain,parent_source(g.key),ARM_ID[arm],index)
        counts=g.counts*_panel_mask(g,r,budget['legacy_suffix_panel']['n_panel_suffix'])[:,None]
        counts=counts.copy(); counts[:,:2]=0
    elif arm=='B':
        r=rng(domain,parent_source(g.key),ARM_ID[arm],index)
        keep=r.random(g.M)<budget['p']
        counts=np.bincount(g.pair[keep]*5+g.w[keep],minlength=g.D*5).reshape(-1,5)
    else: raise ValueError('unknown arm')
    return counts,re
