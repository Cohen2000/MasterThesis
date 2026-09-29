"""Draws, the R/H panel release and cached per-observation features and MLE fits.

In plain words: the helpers every stage-2 task uses to draw an observation, turn it into
model input and features, and fit the MLE, identical to stage 1: an observation is drawn and serialized as in
prepare.observation_row; R/H blocks then release n_panel as in
scripts/add_panel_size.release. Features and MLE fits are pure functions of
the released block and are cached by its hash.
"""
import math
import numpy as np
from main_experiment.baselines import design_estimate, plugin
from main_experiment.common import digest, observation_id
from main_experiment.observation import FEATURE_NAMES, features, make, messages, parse, serialize
from main_experiment.sampling import draw
from main_experiment.shared_mle import fit as mle_fit
from .core import MLE_ANCHOR_ARMS, PANEL_ARMS, Cache

REF_START = FEATURE_NAMES.index('anchor_rho_2')
PANEL_FIELD = {'R': 'n_panel', 'H': 'n_panel_history'}


# ID of the R/H observation with the panel size shown ('-panel-release' is added to the arm label).
def released_id(hidden_id):
    stem, sample = hidden_id.rsplit('__', 1)
    if not sample.startswith('s'): raise ValueError('unexpected sampler ID')
    return stem+'-panel-release__'+sample


def release(block, panel):
    """Add the realised panel size to an R/H block (the panel release)."""
    old = parse(block)
    if old['arm'] not in PANEL_ARMS or 'n_panel' in old or type(panel) is not int or panel < 1:
        raise ValueError('bad panel release')
    new = {**old, 'n_panel': panel}
    released = serialize(new)
    if parse(released) != new or released.replace(f'n_panel={panel}\n', '') != block:
        raise AssertionError('panel release changed more than n_panel')
    return released


# Draw one observation and return it without (hidden) and with (released) the panel size.
def draw_block(g, arm, index, domain, budget, walk):
    """(hidden block, released block, counts) of one sampler draw."""
    counts, traversals = draw(g, arm, index, domain, budget, walk)
    obs = make(g, arm, budget, counts, traversals)
    hidden = serialize(obs)
    parsed = parse(hidden)
    if not np.array_equal(features(obs), features(parsed)): raise AssertionError('serialization changes features')
    block = release(hidden, int(budget[PANEL_FIELD[arm]])) if arm in PANEL_ARMS else hidden
    return hidden, block, counts


def observation_record(g, arm, index, domain, budget, hidden, block, counts, stratum):
    """Stored observation in the common schema; only `block`/`messages` are model inputs."""
    hidden_id = observation_id(g.key, arm, index, budget['coverage_fraction'])
    oid = released_id(hidden_id) if arm in PANEL_ARMS else hidden_id
    parsed = parse(block)
    prompt = messages(block)
    record = {'id': oid, 'graph_id': g.key, 'source_family': g.key, 'stratum': stratum,
              'parent_source': g.key, 'arm': arm, 'sample_index': index, 'domain': domain,
              'empty': parsed['D_obs'] == 0, 'block': block, 'block_sha256': digest(block),
              'messages': prompt, 'prompt_sha256': digest(prompt), 'truth': g.truth,
              'budget_matched': budget['budget_matched_by_arm'][arm],
              'budget_matched_all_arms': budget['budget_matched'],
              'internal_evaluation': {'observed_event_fraction': parsed['M_obs']/g.M,
                                      'observed_dyad_fraction': parsed['D_obs']/g.D,
                                      'observed_cell_fraction': int((counts > 0).sum())/g.cells}}
    if arm in PANEL_ARMS:
        record.update(n_panel=int(budget[PANEL_FIELD[arm]]), paired_hidden_id=hidden_id,
                      hidden_block_sha256=digest(hidden))
    return record


# Numeric ExtraTrees features of one block (cached by the block's hash).
def feature_vector(block, cache=None):
    cache = cache or Cache('features')
    return np.asarray(cache.get(digest(block), lambda: features(parse(block)).tolist()), float)


# Starting estimate that ExtraTrees corrects: the MLE for the walk arms, the stage-1 anchor otherwise.
def anchored(block, arm, x):
    """Four-estimator rule: S/S_obs anchor on the shared MLE instead of the design estimator."""
    if arm not in MLE_ANCHOR_ARMS: return x
    x = np.array(x, float)
    x[REF_START:REF_START+4] = mle(block)['rho']
    return x


def et_row(block, oid, source, arm, domain, family, fold, truth, cache=None):
    """ET cache row (same fields as extratrees.cache) and its feature vector."""
    x = anchored(block, arm, feature_vector(block, cache))
    o = parse(block)
    if o['D_obs'] == 0: raise ValueError('empty observations carry no ET row')
    row = {'id': oid, 'source': source, 'arm': arm, 'domain': domain, 'family': family, 'fold': fold,
           'truth': list(truth), 'plugin': plugin(o), 'reference': x[REF_START:REF_START+4].tolist(),
           'block_sha256': digest(block)}
    return row, x


def _finite(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


# Shared MLE of one block (see src/main_experiment/shared_mle.py), cached.
def mle(block, cache=None):
    """Shared-MLE fit of one block (cached): prediction, fallback and diagnostics."""
    def compute():
        r = mle_fit(parse(block))
        return {'rho': [float(v) for v in r.rho], 'fallback_used': bool(r.fallback_used),
                'fit_status': r.fit_status,
                'flags': {k: _finite(v) if not isinstance(v, (bool, np.bool_)) else bool(v)
                          for k, v in r.flags.items()}}
    return (cache or Cache('mle')).get(digest(block), compute)


# All untrained estimators of one block, as stored in offline.json.
def offline_predictions(block, median, cache=None):
    """plugin, median, design (S arms) and MLE for one released block."""
    o = parse(block)
    out = {'plugin': plugin(o), 'median': list(median)}
    if o['arm'] in ('S', 'S_obs'): out['design'] = design_estimate(o)
    fit = mle(block, cache)
    out['mle'] = fit['rho']
    return out, fit


def training_truths(stage1):
    """Full-archive truth of each of the 16 real training sources (frozen stage-1 training draws)."""
    from main_experiment.common import TRAIN, read_json
    truth = {}
    for path in sorted((stage1/'prepared/observations/training').glob('*.json')):
        r = read_json(path)
        if truth.setdefault(r['source_family'], r['truth']) != r['truth']: raise ValueError('training truth mismatch')
    if set(truth) != set(TRAIN): raise ValueError('incomplete real training sources')
    return truth


# The 'median' baseline: median true profile of the training sources outside the fold.
def fold_median(truths, fold):
    """Training-median profile of a fold (all 16 sources for the synthetic fold)."""
    from main_experiment.common import TRAIN
    return np.median([truths[s] for s in TRAIN if s != fold], axis=0).tolist()
