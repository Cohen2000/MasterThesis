from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import sklearn
from .common import TRAIN, REAL_TEST, sha, digest, write_json, read_json
from .observation import parse, features, FEATURE_NAMES

PARAMETERS=dict(n_estimators=500,criterion='squared_error',max_features=1.0,
 min_samples_split=5,min_samples_leaf=1,max_depth=None,bootstrap=False,oob_score=False,
 random_state=0,min_weight_fraction_leaf=0.,min_impurity_decrease=0.,ccp_alpha=0.,
 warm_start=False,max_leaf_nodes=None,max_samples=None,monotonic_cst=None,n_jobs=1)

def fold_rows(rows,test_source):
    allowed=set(TRAIN)-({test_source} if test_source in REAL_TEST else set())
    selected=[r for r in rows if r['source_family'] in allowed]
    if {r['source_family'] for r in selected}!=allowed: raise ValueError('missing training source')
    weights=[]
    for r in selected:
        na=sum(x['source_family']==r['source_family'] and x['arm']==r['arm'] for x in selected)
        weights.append(1/(len(allowed)*4*na))
    return selected,np.array(weights)


def fit_folds(rows,truth,out,lockfile):
    out=Path(out); models={}; medians={}
    for test in [*REAL_TEST,'synthetic']:
        selected,weights=fold_rows(rows,test)
        sources=sorted({r['source_family'] for r in selected})
        median=np.median([truth[g] for g in sources],axis=0)
        X=np.array([features(parse(r['block'])) for r in selected]); y=np.array([truth[r['source_family']] for r in selected])
        inputs={'test_source':test,'sources':sources,'observations':[r['id'] for r in selected],
                'blocks_sha256':digest([r['block'] for r in selected]),'labels_sha256':digest(y.tolist()),
                'feature_names':FEATURE_NAMES,'feature_sha256':digest(FEATURE_NAMES),
                'features_sha256':digest(X.tolist()),'weights':weights.tolist(),
                'median':median.tolist(),'parameters':PARAMETERS,'sklearn_version':sklearn.__version__,
                'environment_lock_sha256':sha(lockfile)}
        folder=out/test; folder.mkdir(parents=True,exist_ok=True)
        mf=folder/'manifest.json'; model_file=folder/'model.pkl'
        if mf.exists():
            old=read_json(mf)
            if old['input_sha256']!=digest(inputs): raise ValueError('training resume inputs changed')
            if old['model_sha256']!=sha(model_file): raise ValueError('model checksum')
            with open(model_file,'rb') as f: model=pickle.load(f)
        else:
            model=ExtraTreesRegressor(**PARAMETERS).fit(X,y,sample_weight=weights)
            tmp=model_file.with_suffix('.tmp')
            with open(tmp,'wb') as f: pickle.dump(model,f,protocol=5)
            tmp.replace(model_file)
            write_json(mf,{**inputs,'input_sha256':digest(inputs),'model_sha256':sha(model_file),
                           'training_rows':len(selected),'independent_sources':len(sources)})
        models[test]=model; medians[test]=median
    return models,medians
