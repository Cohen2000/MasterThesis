#!/usr/bin/env python3
"""Fixed diagnostic paths; does not alter production calibration or predictions."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.common import ROOT,CURRENT_RUN,CURRENT_REVISION,REAL_TEST,SYNTH,seed,sha,read_json,write_json
from main_experiment.data import load_graph
from main_experiment.sampling import Walk
from main_experiment.pipeline import csv_write
from main_experiment.integrity import code_binding,bind


def main():
    run=ROOT/CURRENT_RUN; out=ROOT/CURRENT_REVISION/'srw_diagnostics'; out.mkdir(parents=True,exist_ok=True)
    bind(out/'inputs.json',{'code':code_binding(),'script_sha256':sha(__file__),
        'walks':32,'lengths':'L and min(4L,1000000)',
        'budgets':{g:sha(run/'calibration'/g/'budget.json') for g in (*REAL_TEST,*SYNTH)}})
    components=[]; rows=[]; paths=[]
    for gid in (*REAL_TEST,*SYNTH):
        g=load_graph(run/'graphs'/gid); b=read_json(run/'calibration'/gid/'budget.json')
        w=Walk(g,run/'build',volume='cells'); comp=w.components
        nc=np.bincount(comp); ec=comp[g.ends[:,0]]; dc=np.bincount(ec,minlength=w.n_components)
        cells=np.bincount(ec,weights=g.K,minlength=w.n_components)
        pnode=nc/g.N; pedge=dc/g.D
        profiles=np.array([[np.mean(g.K[ec==c]>=k) for k in range(2,6)] for c in range(w.n_components)])
        mixture=pnode@profiles
        components.append({'graph_id':gid,'stratum':'real' if gid in REAL_TEST else 'synthetic',
            'components':w.n_components,'largest_node_share':float(pnode.max()),
            'largest_edge_share':float(pedge.max()),'largest_cell_share':float(cells.max()/g.cells),
            'node_vs_edge_component_total_variation':float(np.abs(pnode-pedge).sum()/2),
            'expected_ceiling_cells':float(pnode@cells),'target_cells':b['T'],
            'unreachable':bool(pnode@cells<b['T']),'L':b['L'],
            'validated_coverage':b['validation_mean']/g.cells,
            'validation_mcse_relative_to_target':b['validation_mcse']/b['T'],
            'budget_matched':b['walk_budget_matched'],
            'stationary_component_mixture_profile':mixture.tolist(),
            'stationary_mixture_signed_rho2':float(mixture[0]-g.truth[0]),
            'stationary_mixture_ProfileAE':float(np.mean(np.abs(mixture-g.truth)))})
        seeds=[seed('srw_diagnostic',gid,'S-srw-c10',i) for i in range(1,33)]
        for label,L in [('primary',b['L']),('longer',min(4*b['L'],1000000))]:
            values=[]
            # Small batches bound memory for large full-archive graphs.
            for first in range(0,32,4):
                _,volume,re,executed=w.run(seeds[first:first+4],int(L),True)
                assert (executed==L).all() and (re.sum(1)==L).all()
                for i,r in enumerate(re):
                    seen=r>0; cids=np.unique(ec[seen]); assert len(cids)==1
                    traversal=np.array([r[g.K>=k].sum()/L for k in range(2,6)])
                    plug=np.array([(g.K[seen]>=k).mean() for k in range(2,6)])
                    local=profiles[cids[0]]
                    rec={'graph_id':gid,'stratum':'real' if gid in REAL_TEST else 'synthetic','length':label,'L':L,
                        'path':first+i+1,'component':int(cids[0]),'traversal_rho2':float(traversal[0]),
                        'traversal_AE2':float(abs(traversal[0]-g.truth[0])),
                        'traversal_ProfileAE':float(np.mean(np.abs(traversal-g.truth))),
                        'finite_vs_local_stationary_rho2':float(traversal[0]-local[0]),
                        'finite_vs_local_stationary_ProfileAE':float(np.mean(np.abs(traversal-local))),
                        'plugin_AE2':float(abs(plug[0]-g.truth[0])),
                        'repeated_step_fraction':float(1-seen.sum()/L),
                        'distinct_dyad_coverage':float(seen.mean()),'cell_coverage':float(volume[i]/g.cells)}
                    paths.append(rec); values.append(rec)
            aggregate={'graph_id':gid,'stratum':values[0]['stratum'],'length':label,'L':L,'paths':32}
            for k in values[0]:
                if k in ('graph_id','stratum','length','L','path','component'): continue
                v=[r[k] for r in values]
                aggregate[k]=float(np.mean(v)); aggregate[k+'_MCSE']=float(np.std(v,ddof=1)/np.sqrt(32))
            aggregate['traversal_mean_bias_rho2']=aggregate['traversal_rho2']-g.truth[0]
            rows.append(aggregate)
    csv_write(out/'components.csv',components); csv_write(out/'paths.csv',paths); csv_write(out/'summary.csv',rows)
    report={'component_rows':components,'walk_rows':rows,
        'interpretation':'Local selection/exploration with complete histories. Finite-start/mixing and component bias remain; no unbiasedness claim.',
        'diagnostic_only':True,'production_length_changed':False}
    write_json(out/'report.json',report)
    print([(r['graph_id'],r['components'],r['budget_matched'],r['L']) for r in components])


if __name__=='__main__': main()
