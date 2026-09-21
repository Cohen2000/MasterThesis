"""Oracle-only decomposition; never imported by observations or predictors."""
import numpy as np


def profile_from_counts(counts):
    counts=np.asarray(counts)
    k=(counts>0).sum(1)
    k=k[k>0]
    return np.array([(k>=j).mean() for j in range(2,6)]) if len(k) else None


def decompose(full,censored,panel):
    """Separate node selection, dyad disappearance and lost visible-dyad windows."""
    seen=censored.sum(1)>0
    assert not seen[~panel].any()
    T=profile_from_counts(full); P=profile_from_counts(full[panel])
    Q=profile_from_counts(full[seen]); C=profile_from_counts(censored)
    profiles={'truth':T,'panel_full':P,'visible_full':Q,'censored':C}
    result={k:None if v is None else v.tolist() for k,v in profiles.items()}
    result.update(panel_dyads=int(panel.sum()),visible_dyads=int(seen.sum()),
                  lost_panel_dyads=int(panel.sum()-seen.sum()),
                  panel_cells=int((full[panel]>0).sum()),observed_cells=int((censored>0).sum()))
    if any(v is None for v in profiles.values()):
        result['defined']=False; return result
    parts={'node_selection':P-T,'dyad_disappearance':Q-P,'within_dyad_history':C-Q,
           'net_history':C-P,'total':C-T}
    np.testing.assert_allclose(parts['node_selection']+parts['net_history'],parts['total'],atol=1e-14)
    np.testing.assert_allclose(parts['dyad_disappearance']+parts['within_dyad_history'],parts['net_history'],atol=1e-14)
    result['defined']=True
    for name,v in parts.items():
        result[name]=v.tolist()
        result[name+'_rho2']=float(v[0]); result[name+'_abs_rho2']=float(abs(v[0]))
        result[name+'_profile_abs']=float(np.mean(np.abs(v)))
    return result
