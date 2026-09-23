"""Leave-one-source-out ExtraTrees references.

Nine folds: one per real test source (that source is removed from training with
all of its rows; its surrogate is predicted by the same fold) and one
'synthetic' fold with all 16 real sources for the synthetic test instances.
Surrogates and main synthetic instances are never training data.

Two variants are fitted per fold: 'pooled' (real rows + independent synthetic
training pool, block weights .50 real / .25 DAR / .25 AD) and 'real_only'.
"""
from pathlib import Path
import pickle
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from .common import ARMS, REAL_TEST, TRAIN, digest, read_json, seed, sha, write_json
from .observation import FEATURE_NAMES, FEATURE_VERSION, features, parse
from .baselines import anchor_profile

FOLDS = (*REAL_TEST, 'synthetic')
# One versioned forest seed for every fold: seed('extratrees','all_folds') mod 2**32.
FOREST_SEED = seed('extratrees', 'all_folds') % 2**32
PARAMETERS = dict(n_estimators=500, criterion='squared_error', max_features=1.0, min_samples_split=5,
                  min_samples_leaf=1, max_depth=None, bootstrap=False, oob_score=False,
                  random_state=FOREST_SEED, min_weight_fraction_leaf=0., min_impurity_decrease=0.,
                  ccp_alpha=0., warm_start=False, max_leaf_nodes=None, max_samples=None,
                  monotonic_cst=None, n_jobs=1)
# Without block weights the 400 pool graphs would outweigh the 16 real sources by volume.
BLOCK_WEIGHTS = {'real': .50, 'dar': .25, 'ad': .25}


def fold_sources(fold):
    """Real training sources of a fold: all 16 minus the held-out test source."""
    return set(TRAIN)-({fold} if fold in REAL_TEST else set())


def fold_rows(rows, fold, pool_rows=None):
    """Rows and normalised weights of one fold.

    Within a block every graph carries the same weight, within a graph every arm,
    within an arm its observations. A saturated H draw is one observation
    and carries the whole H weight of its graph.
    """
    allowed = fold_sources(fold)
    selected = [r for r in rows if r['source_family'] in allowed]
    if {r['source_family'] for r in selected} != allowed: raise ValueError('missing training source')
    selected += list(pool_rows or [])
    # An empty draw (D_obs=0, only possible at low budget-sensitivity coverage) has
    # no released evidence at all: anchor_profile is undefined for it, and its
    # features() are already all zero, so it teaches the forest nothing.
    selected = [r for r in selected if not r.get('empty', False)]
    graphs_per_block = {}
    for r in selected:
        graphs_per_block.setdefault(r['block_group'], set()).add(r['source_family'])
    if set(graphs_per_block)-set(BLOCK_WEIGHTS): raise ValueError('unknown training block')
    graph_share = {b: BLOCK_WEIGHTS[b]/len(graphs) for b, graphs in graphs_per_block.items()}
    per_arm = {}
    for r in selected:
        per_arm[r['source_family'], r['arm']] = per_arm.get((r['source_family'], r['arm']), 0)+1
    weights = np.array([graph_share[r['block_group']]/(len(ARMS)*per_arm[r['source_family'], r['arm']]) for r in selected])
    # Normalising is a no-op with all three blocks and restores the scale for real_only.
    return selected, weights/weights.sum()


def fit_folds(rows, truth, out, pool_rows=None):
    """Fit and store one forest per fold. truth maps every training graph to its profile.

    Returns ({fold: model}, {fold: median profile of the fold's real sources}).
    """
    out = Path(out); models = {}; medians = {}
    for fold in FOLDS:
        selected, weights = fold_rows(rows, fold, pool_rows)
        real_sources = sorted(fold_sources(fold))
        median = np.median([truth[g] for g in real_sources], axis=0)
        X = np.array([features(parse(r['block'])) for r in selected])
        # Residual learning keeps the forest focused on cross-source deviations
        # from the released-information anchor; budgets are not training features.
        y = np.array([np.asarray(truth[r['source_family']]) - anchor_profile(parse(r['block']))
                      for r in selected])
        model = ExtraTreesRegressor(**PARAMETERS).fit(X, y, sample_weight=weights)
        folder = out/fold; folder.mkdir(parents=True, exist_ok=True)
        with open(folder/'model.pkl', 'wb') as f: pickle.dump(model, f, protocol=5)
        blocks = sorted({r['block_group'] for r in selected})
        write_json(folder/'manifest.json', {
            'test_source': fold, 'sources': sorted({r['source_family'] for r in selected}),
            'real_sources': real_sources,
            'block_sources': {b: sorted({r['source_family'] for r in selected if r['block_group'] == b}) for b in blocks},
            'block_weights': BLOCK_WEIGHTS, 'feature_version': FEATURE_VERSION, 'n_features': len(FEATURE_NAMES),
            'feature_sha256': digest(FEATURE_NAMES), 'observations': [r['id'] for r in selected],
            'blocks_sha256': digest([r['block'] for r in selected]), 'labels_sha256': digest(y.tolist()),
            'features_sha256': digest(X.tolist()), 'weights': weights.tolist(), 'median': median.tolist(),
            'parameters': PARAMETERS, 'sklearn_version': sklearn.__version__,
            'training_rows': len(selected), 'model_sha256': sha(folder/'model.pkl')})
        models[fold] = model; medians[fold] = median
    return models, medians


def load_models(folder):
    """{fold: model} of one variant directory, verifying each pickle's recorded hash."""
    models = {}
    for fold in FOLDS:
        manifest = read_json(Path(folder)/fold/'manifest.json')
        if sha(Path(folder)/fold/'model.pkl') != manifest['model_sha256']: raise ValueError(f'model checksum {fold}')
        with open(Path(folder)/fold/'model.pkl', 'rb') as f: models[fold] = pickle.load(f)
    return models
