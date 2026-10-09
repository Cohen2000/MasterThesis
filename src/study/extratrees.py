"""ExtraTrees: a forest of randomised trees that learns to correct a start estimate.

Training data per arm: samples of the 16 real training networks and of the 400 synthetic training
networks, with known truth. The forest predicts truth minus start estimate from the features of a
sample (sample.features). Start estimate, leaf size and features per split are chosen by leaving
out one real training network at a time, so a network is never used to tune or train the model
that predicts it. Extremely randomised trees: Geurts, Ernst & Wehenkel (2006); nested selection:
Cawley & Talbot (2010); folds by network: Roberts et al. (2017).
"""
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from .common import TRAIN, seed
from .estimators import mle, plugin, simple_b
from .sample import features

# Settings to choose from: the start estimate, the smallest leaf, the share of features per split.
GRID = [(start, leaf, share) for start in ('plugin', 'reference') for leaf in (1, 5, 20) for share in (.5, 1.)]


def reference(o):
    """The better start estimate: the observed share for R, the MLE for S, H and B."""
    if o['arm'] == 'R': return plugin(o)
    try:
        return mle(o)
    except (ArithmeticError, ValueError):
        if o['arm'] != 'B': raise
        return simple_b(o)


def row(o, network, kind, truth):
    """One sample for ExtraTrees: its features, both start estimates and the truth.
    kind is 'real', 'dar' or 'ad'; a sample without any pair gives no row."""
    if o['D_obs'] == 0: return None
    ref = reference(o)
    return {'network': network, 'kind': kind, 'truth': list(truth), 'plugin': plugin(o), 'reference': list(ref),
            'x': features(o, ref)}


def _weights(rows):
    """Real, DAR and AD networks weigh 50%, 25% and 25%; within a kind every network weighs the same."""
    share = {'real': .5, 'dar': .25, 'ad': .25}
    networks = {k: {r['network'] for r in rows if r['kind'] == k} for k in share}
    n = {}
    for r in rows: n[r['network']] = n.get(r['network'], 0)+1
    w = np.array([share[r['kind']]/len(networks[r['kind']])/n[r['network']] for r in rows])
    return w/w.sum()


def _fit(rows, start, leaf, share, random_state):
    """300 trees that predict truth minus start estimate."""
    X = np.array([r['x'] for r in rows])
    y = np.array([r['truth'] for r in rows])-np.array([r[start] for r in rows])
    forest = ExtraTreesRegressor(n_estimators=300, criterion='squared_error', min_samples_leaf=leaf, max_features=share,
                                 bootstrap=False, random_state=random_state, n_jobs=-1)
    return forest.fit(X, y, sample_weight=_weights(rows))


def _predict(forest, rows, start):
    return np.array([r[start] for r in rows])+forest.predict(np.array([r['x'] for r in rows]))


def select(rows, arm, fold, label='v10_et_nested'):
    """The setting with the smallest error of rho_2 when one more real network is left out in turn.
    `rows` are the training rows of the arm; `fold` is the network the final model must not see."""
    synthetic = [r for r in rows if r['kind'] != 'real']
    real = [r for r in rows if r['kind'] == 'real' and r['network'] != fold]
    scores = {}
    for start, leaf, share in GRID:
        errors = []
        for left_out in (n for n in TRAIN if n != fold):
            test = [r for r in real if r['network'] == left_out]
            forest = _fit(synthetic+[r for r in real if r['network'] != left_out], start, leaf, share,
                          seed(label, arm, fold, left_out, f'{start}:{leaf}:{int(share*100)}') % 2**32)
            errors.append(float(np.mean(np.abs(_predict(forest, test, start)[:, 0]-np.array([r['truth'][0] for r in test])))))
        scores[start, leaf, share] = float(np.mean(errors))
    return min(GRID, key=lambda c: (scores[c], *c))


def predict(rows, test, arm, fold, choice, label='v10_et_final'):
    """Fit the chosen setting on all training rows except those of `fold` and predict the test rows."""
    start, leaf, share = choice
    train = [r for r in rows if r['kind'] == 'real' and r['network'] != fold]+[r for r in rows if r['kind'] != 'real']
    return _predict(_fit(train, start, leaf, share, seed(label, arm, fold) % 2**32), test, start)
