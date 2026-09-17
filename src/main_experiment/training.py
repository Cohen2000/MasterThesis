from pathlib import Path
import pickle
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import sklearn
from .common import TRAIN, REAL_TEST, sha, digest, write_json, read_json
from .observation import parse, features, FEATURE_NAMES, FEATURE_VERSION

PARAMETERS=dict(n_estimators=500,criterion='squared_error',max_features=1.0,
 min_samples_split=5,min_samples_leaf=1,max_depth=None,bootstrap=False,oob_score=False,
 random_state=0,min_weight_fraction_leaf=0.,min_impurity_decrease=0.,ccp_alpha=0.,
 warm_start=False,max_leaf_nodes=None,max_samples=None,monotonic_cst=None,n_jobs=1)

# Fixed block weights for the learned reference. The synthetic pool is far larger
# than the real block in row count, so without this the real sources would be
# drowned out by sheer volume.
BLOCK_WEIGHTS={'real':.50,'dar':.25,'ad':.25}
TRAINING_REVISION='baseline-revision-3-hrecent5-20260917'


def _group(r):
    return r.get('block_group','real')


def fold_rows(rows,test_source,pool_rows=None):
    """Rows for one fold, with weights 50% real / 25% DAR / 25% activity-driven.

    Within a block every graph or source carries the same total weight, within a
    graph the four arms carry the same weight, and within an arm the observations
    carry the same weight. A saturated H sample is one observation and carries the
    whole H weight of its graph; it is not duplicated. A held-out real source is removed with all of its rows;
    pool graphs are never derived from a real source, so nothing is removed there.
    """
    allowed=set(TRAIN)-({test_source} if test_source in REAL_TEST else set())
    selected=[r for r in rows if r['source_family'] in allowed]
    if {r['source_family'] for r in selected}!=allowed: raise ValueError('missing training source')
    if pool_rows: selected=selected+list(pool_rows)
    groups={}
    for r in selected: groups.setdefault(_group(r),set()).add(r['source_family'])
    if set(groups)-set(BLOCK_WEIGHTS): raise ValueError('unknown training block')
    share={g:BLOCK_WEIGHTS[g]/len(groups[g]) for g in groups}
    counts={}
    for r in selected:
        counts[(r['source_family'],r['arm'])]=counts.get((r['source_family'],r['arm']),0)+1
    weights=np.array([share[_group(r)]/(4*counts[(r['source_family'],r['arm'])]) for r in selected])
    # Normalising is a no-op once all three blocks are present and restores the
    # original scale when only the real block is.
    return selected,weights/weights.sum()


def fit_folds(rows,truth,out,lockfile,pool_rows=None,pool_truth=None):
    out=Path(out); models={}; medians={}
    labels=dict(truth); labels.update(pool_truth or {})
    for test in [*REAL_TEST,'synthetic']:
        selected,weights=fold_rows(rows,test,pool_rows)
        sources=sorted({r['source_family'] for r in selected})
        # The median baseline and the empty-sample replacement stay defined on the
        # real fold sources only; the pool must not move them.
        real_sources=sorted({r['source_family'] for r in selected if _group(r)=='real'})
        median=np.median([truth[g] for g in real_sources],axis=0)
        X=np.array([features(parse(r['block'])) for r in selected]); y=np.array([labels[r['source_family']] for r in selected])
        inputs={'test_source':test,'sources':sources,'real_sources':real_sources,
                'training_revision':TRAINING_REVISION,'n_features':len(FEATURE_NAMES),
                'feature_version':FEATURE_VERSION,
                'block_weights':BLOCK_WEIGHTS,
                'block_sources':{g:sorted({r['source_family'] for r in selected if _group(r)==g})
                                 for g in sorted({_group(r) for r in selected})},
                'observations':[r['id'] for r in selected],
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
