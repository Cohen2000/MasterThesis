"""ET replicate training draws and the fixed ET test set.

Replicate k >= 1 redraws every real training observation (domain training_r<k>)
and every pool-train observation on the fixed pool graphs (domain pool_train_r<k>)
with the unchanged samplers, budgets and R/H panel release. Before drawing, each
task re-creates one frozen stage-1 draw with the current code and fails if the block
differs, so the replicates use exactly the same procedure.

In plain words: to see how much ExtraTrees results depend on its random training data,
the training observations are drawn again (replicates 1..10) and the forests refitted.
"""
import tempfile
from pathlib import Path
import numpy as np
from study.common import digest, draws_for, fold_for, observation_id, read_json, write_json
from study.observation import parse
from study.data import load_graph
from study.training_pool import regenerate
from study.walk import Walk
from .core import ARMS, MLE_ANCHOR_ARMS, STAGE2_SOURCES, PANEL_RUN, PANEL_ARMS, STAGE1, Cache, stream_domain
from .observe import REF_START, anchored, draw_block, et_row, feature_vector


def _write(out, rows, X):
    for arm in sorted(X):
        write_json(out/f'rows_{arm}.json', [r for r in rows if r['arm'] == arm])
        np.save(out/f'X_{arm}.npy', np.asarray(X[arm], float))


# Stage 'draw_real': new training observations of one real source and arm for replicate k.
def draw_real(task, out, _inputs):
    """All training draws of one real source and arm for replicate k."""
    k, source, arm = task.params['k'], task.params['source'], task.params['arm']
    g = load_graph(STAGE1/'prepared/graphs'/source)
    budget = read_json(STAGE1/'prepared/calibration'/f'{source}.json')
    walk = Walk(g, Path(tempfile.mkdtemp(prefix='v11ext_walk_'))) if arm in ('S', 'S_obs') else None
    # Code-equivalence gate: draw 1 of the stage-1 training domain must be reproduced bit for bit.
    sealed = read_json(STAGE1/'prepared/observations/training'/f'{observation_id(source, arm, 1)}.json')
    hidden, _, _ = draw_block(g, arm, 1, 'training', budget, walk)
    if hidden != sealed['block']: raise AssertionError(f'{source}/{arm}: current code does not reproduce the stage-1 training draw')
    domain = stream_domain('training', k)
    cache = Cache('features')
    rows, X = [], {}
    for index in range(1, draws_for(arm, budget, 'training')+1):
        _, block, _ = draw_block(g, arm, index, domain, budget, walk)
        if parse(block)['D_obs'] == 0: continue        # the stage-1 cache skips empty observations
        row, x = et_row(block, observation_id(source, arm, index), source, arm, 'real_train', 'real', fold_for(source), g.truth, cache)
        row.update(replicate=k, sample_index=index)
        rows.append(row); X.setdefault(arm, []).append(x)
    _write(out, rows, X)


# The fixed list of pool-train graphs (regenerated from their seeds, never stored).
def pool_specs():
    specs = [s for s in read_json(STAGE1/'references/pool_definition.json')['graphs'] if s['partition'] == 'train']
    if len(specs) != 400: raise ValueError('expected 400 pool-train graphs')
    return specs


# Stage 'draw_pool': new training observations of a chunk of pool graphs for replicate k.
def draw_pool(task, out, _inputs):
    """All arms and training draws of one chunk of pool-train graphs for replicate k."""
    k, first, last = task.params['k'], task.params['first'], task.params['last']
    domain = stream_domain('pool_train', k)
    cache = Cache('features')
    rows, X = [], {}
    for spec in pool_specs()[first:last]:
        stored = read_json(STAGE1/'references/pool/observations'/f'{spec["key"]}.json')
        g = regenerate(spec)
        if [g.N, g.D, g.M, g.truth] != [stored['N_full'], stored['D_full'], stored['M_full'], stored['truth']]:
            raise AssertionError(f'{spec["key"]}: regenerated pool graph differs from stage 1')
        budget = stored['budget']
        walk = Walk(g, Path(tempfile.mkdtemp(prefix='v11ext_walk_')))
        sealed = {(r['arm'], r['sample_index']): r for r in stored['observations']}
        for arm in ARMS:
            if k == 1:   # code-equivalence gate, once per graph and arm
                hidden, _, _ = draw_block(g, arm, 1, 'pool_train', budget, walk)
                if hidden != sealed[arm, 1]['block']: raise AssertionError(f'{spec["key"]}/{arm}: stage-1 pool draw not reproduced')
            for index in range(1, draws_for(arm, budget, 'pool_train')+1):
                _, block, _ = draw_block(g, arm, index, domain, budget, walk)
                if parse(block)['D_obs'] == 0: continue
                row, x = et_row(block, sealed[arm, index]['id'], spec['key'], arm, 'pool_train',
                                spec['family'], 'synthetic', stored['truth'], cache)
                row.update(replicate=k, sample_index=index)
                rows.append(row); X.setdefault(arm, []).append(x)
    _write(out, rows, X)


# ---------------------------------------------------------------- fixed test set
def stage1_cache(arm):
    """The frozen stage-1 ET cache of an arm: R/H from the panel-release run, others from stage 1."""
    folder = PANEL_RUN/'et_run/et' if arm in PANEL_ARMS else STAGE1/'et'
    rows = read_json(folder/'rows.json')
    X = np.load(folder/'features.npy', mmap_mode='r')
    if len(rows) != len(X): raise ValueError('ET cache shape mismatch')
    return rows, X


# Stage 'testset': the observations ExtraTrees is scored on (identical for every replicate).
def testset(task, out, inputs):
    """ET test rows per arm: the 360 stage-1 test observations plus the stage-2 draws.

    The stage-1 features are taken from the frozen caches; recomputing them from the
    released blocks with the current code is recorded as a code-equivalence check.
    """
    blocks = {}
    for path in (STAGE1/'prepared/observations/sample').glob('*.json'):
        r = read_json(path); blocks[r['id']] = r['block']
    for path in (PANEL_RUN/'mainexp/run/observations/sample').glob('*.json'):
        r = read_json(path); blocks[r['id']] = r['block']
    cache = Cache('features')
    check = {}
    for arm in ARMS:
        rows, X = stage1_cache(arm)
        ids = [i for i, r in enumerate(rows) if r['arm'] == arm and r['domain'] == 'main']
        test = [{**rows[i], 'group': 'v11', 'block_sha256': digest(blocks[rows[i]['id']])} for i in ids]
        x = np.asarray(X[ids], float)
        recomputed = np.array([feature_vector(blocks[r['id']], cache) for r in test])
        check[arm] = {'rows': len(test), 'max_abs_feature_difference': float(np.max(np.abs(recomputed-x)))}
        if arm in MLE_ANCHOR_ARMS:
            x = np.array([anchored(blocks[r['id']], arm, v) for r, v in zip(test, x)])
            for r, v in zip(test, x): r['reference'] = v[REF_START:REF_START+4].tolist()
        for source in STAGE2_SOURCES:
            folder = inputs[f'source:{source}']
            new = [r for r in read_json(folder/'et_rows.json') if r['arm'] == arm]
            test += [{**r, 'group': 'new4'} for r in new]
            x = np.vstack([x, np.load(folder/f'X_{arm}.npy')])
        if len(test) != len(x) or len({r['id'] for r in test}) != len(test): raise ValueError('test set identity')
        write_json(out/f'rows_{arm}.json', test)
        np.save(out/f'X_{arm}.npy', x)
    if sum(v['rows'] for v in check.values()) != 360: raise ValueError('expected 360 stage-1 test ET rows')
    write_json(out/'feature_check.json', check)
    worst = max(v['max_abs_feature_difference'] for v in check.values())
    print('TESTSET stage-1 feature recomputation max |diff|', worst, flush=True)
    if worst > 1e-6: raise AssertionError('current code does not reproduce the stage-1 ET features')


# Stage 'anchor0': replicate-0 training rows of the walk arms with the MLE as starting estimate.
def anchor0(task, out, _inputs):
    """Replicate-0 training rows of an MLE-anchored arm: the frozen stage-1 cache with
    the design anchor replaced by the shared MLE of the same block."""
    arm = task.params['arm']
    if arm not in MLE_ANCHOR_ARMS: raise ValueError('anchor0 is only for MLE-anchored arms')
    rows, X = stage1_cache(arm)
    ids = [i for i, r in enumerate(rows) if r['arm'] == arm and r['domain'] in ('real_train', 'pool_train')]
    blocks = {}
    for i in ids:
        r = rows[i]
        if r['domain'] == 'real_train' and r['id'] not in blocks:
            blocks[r['id']] = read_json(STAGE1/'prepared/observations/training'/f'{r["id"]}.json')['block']
        elif r['domain'] == 'pool_train' and r['id'] not in blocks:
            for o in read_json(STAGE1/'references/pool/observations'/f'{r["source"]}.json')['observations']:
                if o['arm'] == arm: blocks[o['id']] = o['block']
    out_rows, out_X = [], []
    for i in ids:
        r = dict(rows[i])
        x = anchored(blocks[r['id']], arm, X[i])
        r['reference'] = x[REF_START:REF_START+4].tolist()
        r['block_sha256'] = digest(blocks[r['id']])
        out_rows.append(r); out_X.append(x)
    write_json(out/f'rows_{arm}.json', out_rows)
    np.save(out/f'X_{arm}.npy', np.asarray(out_X, float))
    print('ANCHOR0', arm, len(out_rows), flush=True)
