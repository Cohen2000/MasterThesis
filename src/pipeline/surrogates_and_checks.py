"""Panel of 12 real sources, 12 paired surrogates and 8 synthetic graphs (arms R/S/H/B).

In plain words: the stage-2 tasks that complete the panel. Stages:
  surrogate  P[w,t] surrogate of each stage-2 source (main_experiment.surrogates), its own
             10% calibration and the parent's sampler streams (common random numbers)
  testset_sur / train_sur   ET test rows of the stage-2 surrogates, predicted by the existing
             fits: pickled production models, or refits with the stored choices and seeds
             whose predictions on the existing test rows must reproduce exactly
  qwen_sur   Qwen bundle of the stage-2 surrogate observations (same protocol)
  walkdiag   the 1000-walk diagnostic of scripts/check_random_walk.py plus coverage measures
  history    temporal-truncation decomposition of the H arm
  api_freeze the frozen API observations of the stage-2 originals and surrogates
"""
from dataclasses import replace
import json
import pickle
import sys
import tempfile
from pathlib import Path
import numpy as np
from main_experiment.common import ROOT, draws_for, read_json, seed, write_csv, write_json
from main_experiment.data import load_graph
from main_experiment.observation import make, parse, serialize
from main_experiment.sampling import calibrate
from main_experiment.shared_mle import fit as mle_fit, fit_profile_from_counts
from main_experiment.surrogates import prepare_surrogate
from main_experiment.walk import AUDIT_ARM_ID, Walk
from . import core
from .core import PIPELINE_DIR, STAGE2_FOLD, STAGE2_SOURCES, RADOSLAW, PANEL_RUN, STAGE1
from .observe import draw_block, et_row, fold_median, observation_record, offline_predictions, training_truths

ARMS4 = ('R', 'S', 'H', 'B')                 # reported arms; S_obs is drawn for reproducibility only
SURROGATES = tuple(s+'__pwt' for s in STAGE2_SOURCES)
QWEN_SUR_DIR = PIPELINE_DIR/'qwen_sur'
_OWN = ['pipeline/core.py', 'pipeline/observe.py', 'pipeline/surrogates_and_checks.py']
core.STAGE_CODE.update({
    'surrogate': _OWN, 'testset_sur': _OWN, 'api_freeze': _OWN, 'history': _OWN,
    'train_sur': _OWN+['pipeline/replicates.py', 'pipeline/et.py', '../scripts/extratrees.py'],
    'qwen_sur': _OWN+['pipeline/qwen.py', '../scripts/run_qwen_engine.py'],
    'walkdiag': _OWN+['../scripts/check_random_walk.py']})


# Source family: a surrogate 'x__pwt' belongs to family 'x'.
def family(key):
    return key[:-5] if key.endswith('__pwt') else key


# Prepared graph of a stage-2 source (or the reused stage-1 graph of nr_radoslaw_email).
def parent_graph(source, inputs):
    if source == RADOSLAW: return load_graph(STAGE1/'prepared/graphs'/source)
    return load_graph(inputs[f'source:{source}']/'graph')


def graph_of(key, inputs):
    """Any panel graph: stage-1 graphs from the frozen preparation, stage-2 graphs from their task."""
    if key.endswith('__pwt') and family(key) in STAGE2_SOURCES:
        return load_graph(inputs[f'surrogate:{family(key)}']/'graph')
    if key in STAGE2_SOURCES: return parent_graph(key, inputs)
    return load_graph(STAGE1/'prepared/graphs'/key)


# ---------------------------------------------------------------- surrogates
# Stage 'surrogate': build the surrogate, calibrate its arms, draw its test observations.
def surrogate(task, out, inputs):
    parent = task.params['source']
    g = prepare_surrogate(parent_graph(parent, inputs), out/'graph')      # audits every invariant
    budget, walk, validation = calibrate(g, Path(tempfile.mkdtemp(prefix='v11ext_walk_')))
    write_json(out/'calibration.json', budget)
    streams = replace(g, key=parent)      # the surrogate draws on its parent's sampler streams
    fold = STAGE2_FOLD[parent]
    median = fold_median(training_truths(STAGE1), fold)
    offline, rows, X = [], [], {}
    for arm in ARMS4:
        for index in range(1, draws_for(arm, budget, 'sample')+1):
            hidden, block, counts = draw_block(streams, arm, index, 'sample', budget, walk)
            record = observation_record(g, arm, index, 'sample', budget, hidden, block, counts, 'surrogate')
            record.update(group='surrogate', fold=fold, parent_source=parent)
            if record['empty']: raise ValueError(f'{record["id"]}: empty main observation')
            write_json(out/'observations'/f'{record["id"]}.json', record)
            predictions, fit = offline_predictions(block, median)
            offline += [{'id': record['id'], 'method': m, 'prediction': p, **({'mle_fit': fit} if m == 'mle' else {})}
                        for m, p in predictions.items() if m != 'design']
            row, x = et_row(block, record['id'], g.key, arm, 'main', 'surrogate', fold, g.truth)
            rows.append({**row, 'group': 'surrogate_ext'}); X.setdefault(arm, []).append(x)
    write_json(out/'offline.json', offline)
    write_json(out/'et_rows.json', rows)
    for arm, xs in X.items(): np.save(out/f'X_{arm}.npy', np.asarray(xs, float))
    write_json(out/'summary.json', {'source': g.key, 'parent': parent, 'fold': fold, 'N': g.N, 'D': g.D, 'M': g.M,
                                    'truth': g.truth, 'events_per_window': g.counts.sum(0).tolist(),
                                    'budget_matched_by_arm': {a: budget['budget_matched_by_arm'][a] for a in ARMS4},
                                    'unmatched_reasons': budget['unmatched_reasons'], 'h_saturated': budget['h_saturated'],
                                    'n_panel': budget['n_panel'], 'n_panel_history': budget['n_panel_history'],
                                    'L': budget['L'], 'p': budget['p'], 'observations': len(rows)})
    print('SURROGATE', g.key, 'N', g.N, 'D', g.D, 'rho2', round(g.truth[0], 4), 'matched', budget['budget_matched'], flush=True)


# Stage 'testset_sur': ExtraTrees test rows of the stage-2 surrogates.
def testset_sur(task, out, inputs):
    for arm in ARMS4:
        rows, X = [], []
        for parent in STAGE2_SOURCES:
            folder = inputs[f'surrogate:{parent}']
            rows += [r for r in read_json(folder/'et_rows.json') if r['arm'] == arm]
            X.append(np.load(folder/f'X_{arm}.npy'))
        write_json(out/f'rows_{arm}.json', rows)
        np.save(out/f'X_{arm}.npy', np.vstack(X))


def train_sur(task, out, inputs):
    """Predict the new surrogate rows of one (replicate, arm, fold) with the existing fit."""
    from . import et
    k, arm, fold = task.params['k'], task.params['arm'], task.params['fold']
    existing = read_json(inputs[f'train:{k}:{arm}:{fold}']/'predictions.json')
    rows_new = [r for r in read_json(inputs['testset_sur']/f'rows_{arm}.json') if r['fold'] == fold]
    X_new = np.load(inputs['testset_sur']/f'X_{arm}.npy')[[i for i, r in enumerate(
        read_json(inputs['testset_sur']/f'rows_{arm}.json')) if r['fold'] == fold]]
    choice = existing['choice']
    x_new, _, base_new = et.extratrees.arrays(rows_new, X_new, np.arange(len(rows_new)), choice['anchor'])
    production = k == 0 and fold in et.STAGE1_FOLDS and arm not in core.MLE_ANCHOR_ARMS
    if production:
        folder = PANEL_RUN/'et_run/et' if arm in core.PANEL_ARMS else STAGE1/'et'
        with open(folder/'models'/arm/fold/'model.pkl', 'rb') as f: model = pickle.load(f)
        check = {'model': 'pickled production model'}
    else:
        rows, X = et.training(k, arm, inputs)
        train_ids = np.array([i for i, r in enumerate(rows) if r['domain'] == 'pool_train' or
                              (r['domain'] == 'real_train' and r['source'] != fold)], dtype=int)
        x_train, truth, base = et.extratrees.arrays(rows, X, train_ids, choice['anchor'])
        model = et.extratrees.model(choice['min_samples_leaf'], choice['max_features'],
                               seed(core.stream_domain('et_final', k), arm, fold) % 2**32, et.cpus())
        model.fit(x_train, truth-base, sample_weight=et.extratrees.block_weights(rows, train_ids))
        old_rows = read_json(inputs['testset']/f'rows_{arm}.json')
        old_ids = [i for i, r in enumerate(old_rows) if r['fold'] == fold]
        x_old, _, base_old = et.extratrees.arrays(old_rows, np.load(inputs['testset']/f'X_{arm}.npy'), np.array(old_ids), choice['anchor'])
        stored = np.array([r['prediction'] for r in existing['observations']])
        diff = float(np.max(np.abs(base_old+model.predict(x_old)-stored)))
        # Parallel tree averaging may change the last bit (max 1.7e-16 observed).
        if [old_rows[i]['id'] for i in old_ids] != [r['id'] for r in existing['observations']] or diff > 1e-12:
            raise AssertionError(f'{task.name}: refit does not reproduce the existing fit ({diff})')
        check = {'model': 'refit with stored choice and seed', 'reproduced_existing_rows': len(old_ids)}
    pred = base_new+model.predict(x_new)
    write_json(out/'predictions.json', {'replicate': k, 'arm': arm, 'fold': fold, 'choice': choice,
                                        'train_sources': existing['train_sources'], 'check': check,
                                        'observations': [{'id': r['id'], 'group': r['group'], 'prediction': list(map(float, p))}
                                                         for r, p in zip(rows_new, pred)]})


# Stage 'qwen_sur': Qwen bundle for the stage-2 surrogate observations.
def qwen_sur(task, out, inputs):
    from . import qwen
    rows = [read_json(p) for parent in STAGE2_SOURCES
            for p in sorted((inputs[f'surrogate:{parent}']/'observations').glob('*.json'))]
    requests = qwen.requests_for(rows)
    for r in rows: write_json(out/'run/observations/sample'/f'{r["id"]}.json', r)
    (out/'run/requests.jsonl').write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in requests))
    qwen.install(out, QWEN_SUR_DIR)
    print('QWEN_SUR', len(rows), 'observations', len(requests), 'requests', flush=True)


# ---------------------------------------------------------------- API freeze
API_FIELDS = ('id', 'graph_id', 'stratum', 'arm', 'sample_index', 'block', 'block_sha256', 'messages', 'prompt_sha256')


# Stage 'api_freeze': write the observations the paid API models receive, with their hashes.
def api_freeze(task, out, inputs):
    """R/S/H/B observations of the stage-2 originals (unchanged) and surrogates, in the API schema."""
    rows = []
    for parent in STAGE2_SOURCES:
        for folder in (inputs[f'source:{parent}'], inputs[f'surrogate:{parent}']):
            for p in sorted((folder/'observations').glob('*.json')):
                r = read_json(p)
                if r['arm'] in ARMS4: rows.append({k: r[k] for k in API_FIELDS})
    if len(rows) != 96 or len({r['id'] for r in rows}) != 96: raise ValueError('expected 96 API observations')
    for r in rows: write_json(out/'observations'/f'{r["id"]}.json', r)
    frozen = [(r['id'], r['block_sha256'], r['prompt_sha256']) for r in sorted(rows, key=lambda r: r['id'])]
    write_json(out/'API_FREEZE_EXT.json', {
        'observations': 96, 'graphs': sorted({r['graph_id'] for r in rows}), 'arms': list(ARMS4), 'draws': 3,
        'observation_hash_manifest_sha256': digest_list(frozen),
        'hashes': {i: {'block_sha256': b, 'prompt_sha256': p} for i, b, p in frozen}})


# One fingerprint over all (ID, block hash, prompt hash) triples of a frozen observation set.
def digest_list(frozen):
    import hashlib
    return hashlib.sha256(json.dumps(frozen, separators=(',', ':')).encode()).hexdigest()


# ---------------------------------------------------------------- walk diagnostic
# Stage 'walkdiag': the walk audit for one graph (see scripts/check_random_walk.py).
def walkdiag(task, out, inputs):
    """scripts/check_random_walk.audit_graph (1000 walks, unchanged criteria) plus MLE bias/variance,
    effective sample size, revisits and component/community coverage of the same walks."""
    import networkx as nx
    sys.path.insert(0, str(ROOT/'scripts'))
    import check_random_walk as audit
    key = task.params['graph']
    g = graph_of(key, inputs)
    build = Path(tempfile.mkdtemp(prefix='v11ext_walk_'))
    row = audit.audit_graph(g, build)
    # The audit infers the stratum from the stage-1 panel; restate it and re-evaluate the unchanged gate rule.
    row['stratum'] = 'surrogate' if key.endswith('__pwt') else 'real'
    shift = row['stationary_shift_rho2']
    row['gate_applicable'] = bool(abs(shift) > .05)
    row['gate_pass'] = bool(not row['gate_applicable'] or
                            (abs(row['design_S_rho2_bias']) <= .1*abs(row['plugin_rho2_bias']) and
                             row['design_S_rho2_rmse'] <= .5*row['plugin_rho2_rmse']))
    row['not_correctable_at_this_budget'] = bool(row['gate_applicable'] and not row['gate_pass'])
    walk = Walk(g, build)
    L = row['L']
    G = nx.Graph()
    G.add_weighted_edges_from((int(a), int(b), float(w)) for (a, b), w in zip(g.ends, g.m))
    communities = nx.community.louvain_communities(G, weight='weight', seed=seed('v11_walk_communities', key) % 2**32)
    community = np.full(g.N, -1); community_cells = np.zeros(len(communities))
    for c, members in enumerate(communities): community[list(members)] = c
    cells_node = g.K
    np.add.at(community_cells, community[g.ends[:, 0]], cells_node)
    comp_cells = np.bincount(walk.components[g.ends[:, 0]], weights=g.K)
    mle_values, cover_comp, cover_comm, comm_touched, mle_failures = [], [], [], [], 0
    seeds = [seed('v10_walk_audit', g.key, AUDIT_ARM_ID, i) for i in range(1, audit.PATHS+1)]
    for first in range(0, audit.PATHS, audit.BATCH):
        _, _, counts, _ = walk.run(seeds[first:first+audit.BATCH], L, True)
        for c in counts:
            seen = c > 0
            # The S block exactly as released (rounded weights), then the production MLE.
            block = serialize(make(g, 'S', {'L': L}, g.counts*seen[:, None], c))
            try:
                mle_values.append(mle_fit(parse(block)).rho)
            except ValueError:   # e.g. only K=5 dyads discovered: reported as a failure, not repaired
                mle_failures += 1
            start = walk.components[g.ends[seen, 0][0]]
            cover_comp.append(float(g.K[seen].sum()/comp_cells[start]))
            touched = np.unique(community[np.r_[g.ends[seen, 0], g.ends[seen, 1]]])
            comm_touched.append(len(touched)/len(communities))
            cover_comm.append(float(g.K[seen].sum()/community_cells[touched].sum()))
    a = np.asarray(mle_values); truth = np.array(g.truth)
    for j in range(4):
        row[f'mle_S_rho{j+2}_bias'] = float(a[:, j].mean()-truth[j])
        row[f'mle_S_rho{j+2}_sd'] = float(a[:, j].std(ddof=1))
        row[f'mle_S_rho{j+2}_rmse'] = float(np.sqrt(np.mean((a[:, j]-truth[j])**2)))
    row['mle_S_failures'] = mle_failures        # bias/SD/RMSE above use the other walks
    largest = comp_cells.max()/comp_cells.sum()
    row.update(components=int(walk.n_components), largest_component_cell_share=float(largest),
               communities=len(communities), community_modularity=float(nx.community.modularity(G, communities)),
               walk_cell_share_of_start_component=float(np.mean(cover_comp)),
               communities_touched_share=float(np.mean(comm_touched)),
               walk_cell_share_of_touched_communities=float(np.mean(cover_comm)))
    write_json(out/'walk.json', row)
    print('WALKDIAG', key, 'L', L, 'bias plugin/design/mle', round(row['plugin_rho2_bias'], 4),
          round(row['design_S_rho2_bias'], 4), round(row['mle_S_rho2_bias'], 4), 'gate', row['gate_pass'], flush=True)


# ---------------------------------------------------------------- history decomposition
WINDOWS = {'last60': (2, 3, 4), 'first60': (0, 1, 2)}


def history_rows(g, h_blocks):
    """Numerator/denominator losses of temporal truncation and the resulting errors.

    Population level (every dyad, no node sampling): K_e counts active windows among all five,
    J_e among the three retained ones. rho_k = #(K>=k)/#(K>=1); the truncated plugin is
    #(J>=k)/#(J>=1) = rho_k (1-numerator loss)/(1-denominator loss). The truncated MLE fits the
    J histogram and extrapolates to W=5. Sampled level: plugin and MLE of the actual H draws
    (last 60% plus the node panel), which adds node sampling to the truncation.
    """
    truth = np.array(g.truth)
    active = g.counts > 0
    K = active.sum(1)
    rows = []
    for window, cols in WINDOWS.items():
        J = active[:, cols].sum(1)
        hist = np.bincount(J, minlength=4).astype(float)
        plugin = np.array([np.mean(J[J >= 1] >= k) for k in range(2, 6)])
        mle = np.array(fit_profile_from_counts(hist.tolist(), 3).rho)
        for k in range(2, 6):
            num_loss = 1-(J >= k).sum()/max(1, (K >= k).sum())
            den_loss = 1-(J >= 1).sum()/(K >= 1).sum()
            rows.append({'graph_id': g.key, 'level': 'population', 'window': window, 'k': k,
                         'rho_k': truth[k-2], 'numerator_loss': float(num_loss), 'denominator_loss': float(den_loss),
                         'loss_gap': float(num_loss-den_loss),
                         'plugin': float(plugin[k-2]), 'plugin_error': float(plugin[k-2]-truth[k-2]),
                         'mle': float(mle[k-2]), 'mle_error': float(mle[k-2]-truth[k-2])})
    for block in h_blocks:
        o = parse(block)
        D = o['D_obs']
        plugin = np.array([sum(r[1] for r in o['table'] if r[0].count('1') >= k)/D for k in range(2, 6)])
        mle = np.array(fit_profile_from_counts(
            [0.]+[sum(r[1] for r in o['table'] if r[0].count('1') == j) for j in range(1, 4)], 3).rho)
        for k in range(2, 6):
            rows.append({'graph_id': g.key, 'level': 'H_sampled', 'window': 'last60', 'k': k, 'rho_k': truth[k-2],
                         'plugin': float(plugin[k-2]), 'plugin_error': float(plugin[k-2]-truth[k-2]),
                         'mle': float(mle[k-2]), 'mle_error': float(mle[k-2]-truth[k-2])})
    return rows


# Stage 'history': how much of arm H's error comes from seeing only 60% of time, measured
# on the complete graphs without any node sampling (HISTORY tables).
def history(task, out, inputs):
    from main_experiment.common import REAL_TEST
    keys = [*REAL_TEST, *STAGE2_SOURCES]
    keys += [k+'__pwt' for k in keys]
    blocks = {}
    for p in (PANEL_RUN/'mainexp/run/observations/sample').glob('*__H-*.json'):
        r = read_json(p); blocks.setdefault(r['graph_id'], []).append(r['block'])
    for parent in STAGE2_SOURCES:
        for folder in (inputs[f'source:{parent}'], inputs[f'surrogate:{parent}']):
            for p in (folder/'observations').glob('*__H-*.json'):
                r = read_json(p); blocks.setdefault(r['graph_id'], []).append(r['block'])
    rows = []
    for key in keys:
        if len(blocks.get(key, [])) not in (1, 3): raise ValueError(f'{key}: H draws missing')
        rows += [{**r, 'group': 'surrogate' if key.endswith('__pwt') else 'real', 'family': family(key)}
                 for r in history_rows(graph_of(key, inputs), sorted(blocks[key]))]
    write_csv(out/'HISTORY.csv', rows)
    print('HISTORY', len(keys), 'graphs', len(rows), 'rows', flush=True)

