"""Frozen synthetic training/development pool for the learned baseline.

The pool exists so the ExtraTrees reference is fitted on a diverse set of
mechanisms rather than on sixteen real sources alone. It adds no generator
family: both families are the ones already verified for the main experiment
(pool graphs use synthetic.generate_one).

Coverage is argued from generator semantics, not from proximity to the known
targets of the eight synthetic main-test instances. No parameter was chosen by
looking at the resulting rho_2, and the ranges below are fixed before any
result is computed.

Independence from every main test:
  * every pool graph is drawn on its own stream (domain 'pool', keyed by its id),
    disjoint from the main-test stream (domain 'graph', keyed by pair id);
  * pool graphs are generated one at a time, so no latent quantity is shared
    between any two graphs and the "paired variants stay in one partition" rule
    holds trivially;
  * training and development ids are disjoint, hence so are their streams;
  * no pool graph is derived from a main-test instance.
Real sources are governed by the existing leave-one-source-out rule; a held-out
real source is excluded from training with all of its derivations.
"""
from pathlib import Path
import numpy as np
from .common import ARMS, BUILD, COVERAGE_FRACTION, DESIGN_VERSION, digest, draws_for, observation_id, rng, seed, write_json
from .baselines import design_reference
from .observation import make, parse, serialize
from .sampling import calibrate, draw
from .synthetic import generate_one

POOL_VERSION='pool-v3-panel888-20260921'

# Held constant across the pool, with the reason each one is not varied.
CONSTANTS={
 'W':'5 windows; the estimand and the whole input contract are defined on W=5.',
 'horizon':'[0,1]; canonical rescaling, identical to the main-test instances.',
 'dar_event_layer':'1+Poisson(nu) events per active edge-window, placed uniformly '
                   'inside the window; the positive layer is what makes an active '
                   'window observable at all, so its shape stays fixed and only nu varies.',
 'dar_start':'stationary Bernoulli(chi) initial states; the DAR(1) recurrence has '
             'marginal chi for every alpha, so no burn-in is needed and none is used.',
 'ad_m':'one contact per activation. Extending the memory rule to m>1 would require '
        'an arbitrary decision about whether memory updates between the m picks '
        'within a round, which the source model does not fix; event volume is varied '
        'through eta and rounds instead.',
 'ad_activity_max':'1; activity is a per-round probability, so eta*x must stay <=1.',
 'ad_event_rule':'synchronous rounds, one undirected contact per dyad per round '
                 '(user-confirmed for the main experiment; unchanged here).',
 'graph_type':'undirected, self-loops removed, as in canonical().',
}

# Varied parameters: range, and why the range is the admissible one.
VARIED={
 'dar':{
  'N':{'grid':[200,300,500,800,1200],'why':'network size; spans a factor of six around '
       'the sizes seen in the real sources without making any single graph dominate runtime.'},
  'mean_degree':{'range':[4.0,24.0],'why':'backbone density as mean degree, converted to '
       'E=round(d*N/2) and capped at C(N,2). Below ~4 the backbone fragments and the '
       'single-component walk cannot traverse between components; above ~24 the dyad count grows without '
       'adding a new persistence regime.'},
  'chi':{'range':[0.04,0.45],'why':'per-window activity of a backbone edge. Below ~0.04 too '
       'few edges ever activate across five windows; above ~0.45 the profile saturates. '
       'The range is chosen to sweep the attainable persistence band, not to match any target.'},
  'alpha':{'range':[0.0,0.9],'why':'copy probability, i.e. memory. alpha=1 freezes the initial '
       'state so that K is 0 or 5, a degenerate limit that carries no information about '
       'intermediate persistence; 0.9 already produces very high persistence.'},
  'nu':{'range':[0.25,3.0],'why':'Poisson mean of the extra events, so mean events per active '
       'window is 1+nu in [1.25,4]. Keeps M/S informative for the event-layer correctors '
       'without making the event count dominate the dyad count.'},
 },
 'ad':{
  'N':{'grid':[200,300,500,800,1200],'why':'as for DAR.'},
  'tail':{'range':[0.8,2.4],'why':'activity distribution F(a) ~ a^-gamma on [eps,1] with '
       'gamma = 1 + tail, so gamma in [1.8,3.4]. Perra et al. report empirical exponents '
       'around 2-3; the band is widened slightly on both sides.'},
  'eps':{'range':[0.005,0.05],'why':'lower activity cut-off. It sets how many nodes are '
       'effectively inert and therefore how heterogeneous the dyad activity is.'},
  'eta':{'range':[0.2,1.0],'why':'activity rescaling. eta*x is a per-round probability, so '
       'eta<=1; below ~0.2 almost nothing happens within the round budget.'},
  'rounds':{'grid':[500,750,1000,1500,2000],'why':'event volume and how far the memory '
       'process is allowed to develop.'},
  'mode':{'values':['memoryless','memory'],'why':'the two published variants; memoryless gives '
       'low persistence, memory high, so the family covers both ends by construction.'},
  'c':{'range':[0.5,4.0],'why':'memory strength in p_new = c/(n+c); c=1 is the main-test value, '
       'and the band lets the reinforcement be clearly weaker or stronger. Ignored in '
       'memoryless mode, where it is recorded as null.'},
 },
}

# Every partition covers the grids; AD also balances the full N x rounds grid.
STRATA={
 'dar':{'axes':['chi','alpha'],'bins':[5,5],'train_per_cell':8,'dev_per_cell':2},
 'ad':{'axes':['mode','eta'],'bins':[2,5],'train_per_cell':20,'dev_per_cell':5},
}
COUNTS={'dar':{'train':200,'dev':50},'ad':{'train':200,'dev':50}}


def _edges(bounds,k):
    lo,hi=bounds; return np.linspace(lo,hi,k+1)


def draw_pool():
    """Label-blind stratification, with independently shuffled grids per partition.

    Balance N within each family/partition and all 25 N x rounds combinations
    within AD/partition. Assign these grids randomly to the continuous-parameter
    strata. Development is not the deterministic complement of training within
    a stratum; no joint parameter combination is deliberately held out.
    """
    specs=[]
    sizes=VARIED['dar']['N']['grid']
    ce=_edges(VARIED['dar']['chi']['range'],5)
    ae=_edges(VARIED['dar']['alpha']['range'],5)
    ee=_edges(VARIED['ad']['eta']['range'],5)
    for partition in ('train','dev'):
        r=rng('pool_definition',POOL_VERSION+':'+partition)
        dar_grid=sizes*(COUNTS['dar'][partition]//len(sizes))
        r.shuffle(dar_grid)
        dar_cursor=0
        ad_grid=[(n,k) for n in range(5) for k in range(5)]*(COUNTS['ad'][partition]//25)
        r.shuffle(ad_grid)
        ad_cursor=0
        for i in range(5):
            for j in range(5):
                count=STRATA['dar'][partition+'_per_cell']
                grid=dar_grid[dar_cursor:dar_cursor+count]
                dar_cursor+=count
                for N in grid:
                    d=float(r.uniform(*VARIED['dar']['mean_degree']['range']))
                    p={'N':N,'E':min(int(round(d*N/2)),N*(N-1)//2),'mean_degree':d,
                       'chi':float(r.uniform(ce[i],ce[i+1])),
                       'alpha':float(r.uniform(ae[j],ae[j+1])),
                       'nu':float(r.uniform(*VARIED['dar']['nu']['range']))}
                    specs.append({'family':'dar','parameters':p,'partition':partition,
                                  'stratum':{'chi_bin':i,'alpha_bin':j}})
        for mode in VARIED['ad']['mode']['values']:
            for j in range(5):
                count=STRATA['ad'][partition+'_per_cell']
                pairs=ad_grid[ad_cursor:ad_cursor+count]
                ad_cursor+=count
                for n,k in pairs:
                    p={'N':sizes[n],'rounds':VARIED['ad']['rounds']['grid'][k],
                       'mode':mode,'tail':float(r.uniform(*VARIED['ad']['tail']['range'])),
                       'eps':float(r.uniform(*VARIED['ad']['eps']['range'])),
                       'eta':float(r.uniform(ee[j],ee[j+1])),
                       'c':float(r.uniform(*VARIED['ad']['c']['range'])) if mode=='memory' else None}
                    specs.append({'family':'ad','parameters':p,'partition':partition,
                                  'stratum':{'mode':mode,'eta_bin':j}})
    counter={}
    for sp in specs:
        k=(sp['family'],sp['partition']); counter[k]=counter.get(k,0)+1
        sp['key']=f"{POOL_VERSION}_{sp['family']}_{sp['partition']}_{counter[k]:04d}"
        sp['seed']=seed('pool',sp['key'])
    return specs


def pool_definition():
    specs=draw_pool()
    got={(f,p):sum(s['family']==f and s['partition']==p for s in specs)
         for f in COUNTS for p in ('train','dev')}
    for f in COUNTS:
        for p in ('train','dev'):
            if got[(f,p)]!=COUNTS[f][p]: raise ValueError(f'pool count {f}/{p}={got[(f,p)]}')
    if len({s['key'] for s in specs})!=len(specs): raise ValueError('duplicate pool key')
    if len({s['seed'] for s in specs})!=len(specs): raise ValueError('pool seed collision')
    return {'version':POOL_VERSION,'constants':CONSTANTS,'varied':VARIED,'strata':STRATA,
            'counts':COUNTS,'n_graphs':len(specs),'graphs':specs}


def build_pool(out, specs, fraction=COVERAGE_FRACTION):
    """Generate every pool graph and its training/development observations.

    Graphs are cheap to regenerate from their seed and are not stored; each
    graph's truth, full budget and serialized observations are.
    """
    out = Path(out)
    for number, spec in enumerate(specs, 1):
        key = spec['key']
        domain = 'pool_'+spec['partition']
        g, budget, walk = pool_graph(spec, BUILD, fraction)
        rows = []
        for arm in ARMS:
            for index in range(1, draws_for(arm, budget, domain)+1):
                counts, traversals = draw(g, arm, index, domain, budget, walk)
                block = serialize(make(g, arm, budget, counts))
                if serialize(parse(block)) != block: raise ValueError('block round trip')
                rows.append({'id': observation_id(key, arm, index, fraction), 'graph_id': key, 'source_family': key,
                             'arm': arm, 'sample_index': index, 'domain': domain, 'block': block,
                             'block_sha256': digest(block), 'empty': parse(block)['D_obs'] == 0,
                             'budget_matched': budget['budget_matched_by_arm'][arm],
                             'design_reference': design_reference(g, traversals) if arm == 'S' else None})
        write_json(out/'observations'/f'{key}.json', {
            'key': key, 'family': spec['family'], 'partition': spec['partition'], 'stratum': spec['stratum'],
            'parameters': spec['parameters'], 'seed': spec['seed'], 'truth': list(g.truth),
            'N_full': g.N, 'D_full': g.D, 'M_full': g.M, 'active_dyad_windows': g.cells,
            'design_version': DESIGN_VERSION, 'budget': budget, 'observations': rows})
        if number % 50 == 0: print(f'pool: {number}/{len(specs)} graphs', flush=True)


def regenerate(spec):
    """Regenerate one pool graph from its specification (graphs are not stored)."""
    parameters = {k: v for k, v in spec['parameters'].items() if v is not None and k != 'mean_degree'}
    g, _, _ = generate_one(spec['key'], spec['family'], parameters, domain='pool')
    return g


def pool_graph(spec, build_dir, fraction=COVERAGE_FRACTION):
    """Regenerate one pool graph and calibrate its budget."""
    g = regenerate(spec)
    budget, walk, _ = calibrate(g, build_dir, fraction)
    return g, budget, walk
