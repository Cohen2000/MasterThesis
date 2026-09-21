#!/usr/bin/env python3
"""Independent audit of the prepared study before any inference.

Re-derives from the stored graphs and the seed rule (not by trusting the stage
code paths) and asserts:
  1. checksums of the prepared artifacts;
  2. graph truths from unique (dyad, window) occurrences; CNS provenance;
  3. P[w,t] surrogate invariants and the exact productive permutation;
  4. budgets: T, R/H panel sizes, H time cutoff, B retention p, walk validation;
  5. every main and training observation re-drawn from its parent-keyed stream
     (common random numbers) and serialized byte-identically; prompts;
  6. common random numbers of parent/surrogate pairs for R, H, S and B;
  7. ExtraTrees folds: forest seed, LOSO exclusion, no surrogate or main synthetic
     graph in training, medians, and every stored reference prediction;
  8. requests: counts per configuration, IDs, payload/prompt hashes, frozen
     payloads, dispatch flags, no reuse of earlier request IDs;
  9. prompt templates: required semantics, byte identity with the last freeze;
 10. seeds: every stream of the study enumerated, no collision, no overlap with
     the previous generation;
 11. diagnostics present with their fixed sizes.
Writes results/panel888/audit/offline_audit.json and complete_seed_manifest.jsonl.
"""
import json
import sys
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from main_experiment.baselines import PRIMARY_REFERENCE, corrector, plugin
from main_experiment.common import (ARM_ID, ARMS, AUDIT, BUDGET_TOLERANCE, BUILD, CONFIGS, COVERAGE_FRACTION,
                                    DESIGN_VERSION, DIAGNOSTICS, LLM_REPEATS, MAIN_KEYS, MASTER_SEED, PREPARED,
                                    REAL_TEST, REFERENCES, ROOT, SEEDS, SURROGATES, SYNTH, TRAIN, digest, draws_for,
                                    fold_for, observation_id, planned_sizes, read_json, read_jsonl, seed, sha,
                                    write_json)
from main_experiment.data import CNS_ORIGINAL_MD5, load_graph
from main_experiment.observation import FEATURE_NAMES, features, make, messages, parse, serialize
from main_experiment.pool import pool_definition
from main_experiment.requests import validate_request
from main_experiment.sampling import Walk
from main_experiment.training import FOLDS, FOREST_SEED, load_models

PREVIOUS = ROOT/'archive/pre_panel888_20260921'
PREVIOUS_RUN = PREVIOUS/'results/main_experiment/cells10_final_20260920'
PROMPT_FILES = ('system.txt', 'user_prefix.txt', 'rule_R.txt', 'rule_S.txt', 'rule_H_time_v3.txt', 'rule_B.txt')
UNCHANGED_SINCE_FREEZE = ('system.txt', 'user_prefix.txt', 'rule_R.txt', 'rule_H_time_v3.txt', 'rule_B.txt')
WALK_PATHS = 1000


def generator(*fields):
    """PCG64 stream from the seed rule, built here instead of through common.rng."""
    return np.random.Generator(np.random.PCG64(seed(*fields)))


# ---------------------------------------------------------------- 1-2
def audit_checksums():
    checksums = read_json(PREPARED/'checksums.json')
    for name, h in checksums.items():
        assert sha(PREPARED/name) == h, f'checksum {name}'
    return len(checksums)


def audit_graph(g, manifest):
    windows = np.unique(g.pair*5+g.w)                  # unique (dyad, window) occurrences
    K = np.bincount(windows//5, minlength=g.D)
    assert [float(np.mean(K >= k)) for k in range(2, 6)] == manifest['truth'] == g.truth
    assert g.counts.sum() == g.M and manifest['active_dyad_windows'] == len(windows)
    assert (g.ends[:, 0] < g.ends[:, 1]).all() and len(np.unique(g.ends, axis=0)) == g.D


def audit_copenhagen():
    manifest = read_json(PREPARED/'graphs/copenhagen_bluetooth/manifest.json')
    assert manifest['copenhagen']['legacy_export_matches']
    assert manifest['copenhagen']['release_md5_verified'] == CNS_ORIGINAL_MD5
    export = pd.read_csv(PREPARED/'graphs/copenhagen_bluetooth/canonical.csv')
    assert (export.u < export.v).all()
    assert export.equals(export.sort_values(['u', 'v', 't']).reset_index(drop=True))


# ---------------------------------------------------------------- 3
def audit_surrogate(parent, surrogate):
    assert parent.N == surrogate.N and np.array_equal(parent.ends, surrogate.ends)
    assert np.array_equal(parent.u, surrogate.u) and np.array_equal(parent.v, surrogate.v)
    assert np.array_equal(parent.pair, surrogate.pair)                         # record keeps its dyad
    assert np.array_equal(parent.m, surrogate.m) and parent.M == surrogate.M   # no record removed
    assert np.array_equal(np.sort(parent.t), np.sort(surrogate.t)) and parent.horizon == surrogate.horizon
    assert np.array_equal(generator('pwt_productive', parent.key).permutation(parent.t), surrogate.t)

    def collisions(g):
        order = np.lexsort((g.t, g.pair))
        return int(np.sum((np.diff(g.pair[order]) == 0) & (np.diff(g.t[order]) == 0)))
    return {'parent': parent.key, 'events': parent.M, 'parent_collisions': collisions(parent),
            'surrogate_collisions': collisions(surrogate), 'rho_parent': parent.truth, 'rho_surrogate': surrogate.truth}


# ---------------------------------------------------------------- 4
def audit_budget(g, b):
    cells = g.cells; T = COVERAGE_FRACTION*cells
    assert b['active_dyad_windows'] == cells and abs(b['T']-T) < 1e-9
    n = np.arange(g.N+1); pi = n*(n-1)/(g.N*(g.N-1))
    assert b['n_panel'] == int(np.argmin(np.abs(pi*cells-T)))
    cutoff = g.horizon[0]+.4*(g.horizon[1]-g.horizon[0])            # h = .60
    keep = g.t >= cutoff
    J = len(np.unique(g.pair[keep]*5+g.w[keep]))
    n_h = int(np.argmin(np.abs(pi*J-T))) if J else g.N
    assert (b['J_total'], b['n_panel_history'], b['history_start']) == (J, n_h, cutoff) and b['history_fraction'] == .6
    assert b['h_saturated'] == (n_h == g.N) and b['h_target_unreachable'] == (J < T)
    relative = float((pi[n_h]*J-T)/T)
    assert abs(relative-b['h_relative_budget_error']) < 1e-12
    cell_events = g.counts[g.counts > 0]
    assert abs(float(np.sum(1-(1-b['p'])**cell_events))-T)/T < 1e-9
    volumes = read_json(PREPARED/'calibration'/f'{g.key}_validation_volumes.json')
    assert len(volumes) == b['validation_n'] in (1024, 4096)
    assert float(np.mean(volumes)) == b['validation_mean']
    assert float(np.std(volumes, ddof=1)/np.sqrt(len(volumes))) == b['validation_mcse']
    by_arm = {'R': bool(abs(b['node_relative_budget_error']) <= BUDGET_TOLERANCE), 'S': b['walk_budget_matched'],
              'H': bool(abs(relative) <= BUDGET_TOLERANCE),
              'B': bool(abs(b['bernoulli_relative_budget_error']) <= BUDGET_TOLERANCE)}
    assert b['budget_matched_by_arm'] == by_arm and b['budget_matched'] == all(by_arm.values())
    return {'graph_id': g.key, 'T': T, **{f'matched_{a}': v for a, v in by_arm.items()},
            'relative_errors': {'R': b['node_relative_budget_error'], 'S': b['validation_relative_error'],
                                'H': relative, 'B': b['bernoulli_relative_budget_error']}}


# ---------------------------------------------------------------- 5-6
def redraw(g, arm, index, domain, b, walk):
    """Observed counts re-derived with an explicit parent-keyed stream."""
    parent = g.key.removesuffix('__pwt')
    stream = (domain, parent, ARM_ID[arm], index)
    if arm in ('R', 'H'):
        size = b['n_panel'] if arm == 'R' else b['n_panel_history']
        chosen = np.zeros(g.N, bool); chosen[generator(*stream).permutation(g.N)[:size]] = True
        dyads = chosen[g.ends[:, 0]] & chosen[g.ends[:, 1]]
        keep = dyads[g.pair] & (g.t >= b['history_start'] if arm == 'H' else True)
    elif arm == 'B':
        keep = generator(*stream).random(g.M) < b['p']
    else:
        traversals = walk.run([seed(*stream)], int(b['L']), True)[2][0]
        return g.counts*(traversals > 0)[:, None], traversals
    return np.bincount(g.pair[keep]*5+g.w[keep], minlength=g.D*5).reshape(-1, 5), None


def audit_observations(graphs, budgets):
    counts = Counter(); walks = {}
    for domain in ('sample', 'training'):
        for path in sorted((PREPARED/'observations'/domain).glob('*.json')):
            row = read_json(path); g = graphs[row['graph_id']]; b = budgets[row['graph_id']]
            assert path.stem == observation_id(row['graph_id'], row['arm'], row['sample_index'])
            assert 1 <= row['sample_index'] <= draws_for(row['arm'], b, domain)
            if row['arm'] == 'S' and g.key not in walks: walks[g.key] = Walk(g, BUILD)
            c, traversals = redraw(g, row['arm'], row['sample_index'], domain, b, walks.get(g.key))
            assert serialize(make(g, row['arm'], b, c)) == row['block']
            if row['arm'] == 'S':                        # internal reference, recomputed from the traversal log
                degree = np.bincount(g.ends.ravel(), minlength=g.N)
                w = traversals/(degree[g.ends[:, 0]]*degree[g.ends[:, 1]])
                np.testing.assert_allclose(row['design_reference'], [w[g.K >= k].sum()/w.sum() for k in range(2, 6)],
                                           rtol=1e-12)
            else:
                assert row['design_reference'] is None
            assert digest(row['block']) == row['block_sha256'] and serialize(parse(row['block'])) == row['block']
            assert messages(row['block']) == row['messages'] and digest(row['messages']) == row['prompt_sha256']
            assert len(features(parse(row['block']))) == len(FEATURE_NAMES) == 129
            assert row['truth'] == g.truth and row['design_version'] == DESIGN_VERSION
            counts[domain] += 1
            # Every arm, S included, shows only deduplicated dyads: no walk statistics anywhere.
            assert 'Walk_A' not in row['block'] and 'Auxiliary' not in row['messages'][1]['content']
    return dict(counts)


def audit_common_random_numbers(graphs, budgets):
    """Parent and surrogate share node permutations (R/H), walk streams (S) and uniforms (B)."""
    rows = []
    for parent in REAL_TEST:
        g, s = graphs[parent], graphs[parent+'__pwt']
        bg, bs = budgets[parent], budgets[parent+'__pwt']
        for index in range(1, 4):
            permutation = generator('sample', parent, ARM_ID['R'], index).permutation(g.N)
            # Nested panels: the smaller R panel is contained in the larger one.
            small, large = sorted((bg['n_panel'], bs['n_panel']))
            assert set(permutation[:small]) <= set(permutation[:large])
            L = min(bg['L'], bs['L'])
            walk_seed = seed('sample', parent, ARM_ID['S'], index)
            a = Walk(g, BUILD).run([walk_seed], L, True)[2]
            b = Walk(s, BUILD).run([walk_seed], L, True)[2]
            assert np.array_equal(a, b)                 # identical support, identical walk prefix
            uniforms = generator('sample', parent, ARM_ID['B'], index).random(g.M)
            kept_g, kept_s = uniforms < bg['p'], uniforms < bs['p']
            assert np.all(kept_g <= kept_s) if bg['p'] <= bs['p'] else np.all(kept_s <= kept_g)
        rows.append({'parent': parent, 'n_panel': [bg['n_panel'], bs['n_panel']], 'L': [bg['L'], bs['L']],
                     'n_panel_history': [bg['n_panel_history'], bs['n_panel_history']], 'p': [bg['p'], bs['p']]})
    return rows


# ---------------------------------------------------------------- 7
def audit_models_and_references(truths):
    models = {v: load_models(REFERENCES/f'models_{v}') for v in ('pooled', 'real_only')}
    folds = []
    for variant in models:
        for fold in FOLDS:
            m = read_json(REFERENCES/f'models_{variant}'/fold/'manifest.json')
            allowed = set(TRAIN)-({fold} if fold in REAL_TEST else set())
            assert m['parameters']['random_state'] == FOREST_SEED == 1858608657
            assert set(m['real_sources']) == allowed and fold not in m['sources']
            assert not set(m['sources']) & (set(SURROGATES) | set(SYNTH))
            assert all('p888-20260921' in oid for oid in m['observations'])
            assert all(truths.get(oid.split('__')[0]) is None or oid.split('__')[0] in allowed
                       for oid in m['observations'])
            if variant == 'pooled':
                assert len(m['block_sources']['dar']) == len(m['block_sources']['ad']) == 200
                assert not any('_dev_' in s for s in m['sources'])
            else:
                assert set(m['block_sources']) == {'real'}
            np.testing.assert_array_equal(m['median'], np.median([truths[s] for s in sorted(allowed)], axis=0))
            folds.append({'variant': variant, 'fold': fold, 'model_sha256': m['model_sha256'],
                          'training_rows': m['training_rows'], 'random_state': m['parameters']['random_state']})
    primary = read_json(REFERENCES/'primary_baselines.json')['observations']
    observations = {p.stem: read_json(p) for p in (PREPARED/'observations/sample').glob('*.json')}
    assert set(primary) == set(observations)
    for oid, row in observations.items():
        entry = primary[oid]; o = parse(row['block']); fold = fold_for(row['graph_id'])
        assert entry['fold'] == fold and entry['block_sha256'] == row['block_sha256'] and entry['truth'] == row['truth']
        assert entry['primary_corrector'] == entry[PRIMARY_REFERENCE[row['arm']]]
        for variant in models:
            expected = (models[variant][fold].predict([features(o)])[0] if o['D_obs']
                        else read_json(REFERENCES/f'models_{variant}'/fold/'manifest.json')['median'])
            np.testing.assert_array_equal(entry['extratrees_'+variant]['prediction'], expected)
        if o['D_obs']:
            assert entry['plugin']['prediction'] == plugin(o)
            if row['arm'] != 'H': assert entry['corrector']['prediction'] == corrector(o)
        for method in ('plugin', 'corrector', 'median', 'extratrees_pooled', 'extratrees_real_only'):
            values = entry[method]['prediction']
            assert all(0 <= v <= 1 for v in values) and all(a >= b for a, b in zip(values, values[1:]))
    return folds


# ---------------------------------------------------------------- 8-9
def audit_requests(design):
    requests = read_jsonl(PREPARED/'requests.jsonl')
    ids = [r['id'] for r in requests]
    assert len(requests) == design['planned_calls'] == len(set(ids))
    per_config = Counter(r['config_id'] for r in requests)
    assert all(per_config[c] == design['main_observations']*LLM_REPEATS for c in CONFIGS)
    slots = Counter((r['observation_id'], r['config_id'], r['repeat_index']) for r in requests)
    assert max(slots.values()) == 1
    observations = {p.stem: read_json(p) for p in (PREPARED/'observations/sample').glob('*.json')}
    for r in requests:
        validate_request(r)
        o = observations[r['observation_id']]
        assert r['payload'].get('messages', r['payload'].get('input')) == o['messages']
        assert r['id'].endswith('__'+DESIGN_VERSION) and not r['started'] and not r['mock']
        assert r['status'] == ('skipped_empty' if o['empty'] else 'not_started')
        qwen = r['config_id'].startswith('qwen')
        assert r['production_dispatch_enabled'] is qwen and r['requires_technical_release'] is (not qwen)
        if qwen: assert r['payload']['structured_output']['json_object'] is True
    previous = {r['id'] for r in read_jsonl(PREVIOUS_RUN/'requests.jsonl')}
    assert not previous & set(ids)
    return {'requests': len(requests), 'per_config': dict(per_config), 'previous_request_ids_reused': 0,
            'requests_sha256': sha(PREPARED/'requests.jsonl')}


def audit_prompt_templates():
    spec = ROOT/'config/main_experiment'
    text = {f: (spec/f).read_text() for f in PROMPT_FILES}
    for f in UNCHANGED_SINCE_FREEZE:
        assert sha(spec/f) == sha(PREVIOUS/'config/main_experiment'/f), f'template changed: {f}'
    required = {
        'system.txt': ['Return your final answer as one JSON object with exactly the keys rho_2, rho_3, rho_4, rho_5'],
        'user_prefix.txt': ['W=5', 'E_full', 'K_e', 'rho_2 >= rho_3 >= rho_4 >= rho_5',
                            'The full numbers of vertices, dyads, and events are unknown',
                            'Temporal_access indicates temporal accessibility only',
                            '? = temporally inaccessible window',
                            'Dyads without any observed event are omitted, so the all-zero pattern is not listed'],
        'rule_R.txt': ['Uniform node panel', 'Accessible zeros for listed dyads therefore indicate true inactive windows.'],
        'rule_S.txt': ['Degree-biased random walk', 'initial vertex is drawn uniformly from V_full',
                       'with probability d_v / (sum of d_x over all neighbors x of u)',
                       'Each observed dyad is listed once', 'traversal counts and vertex degrees are not reported',
                       'Exactly L transitions are taken, with no burn-in, no restart, and no stopping'],
        'rule_H_time_v3.txt': ['common query is made at the full archive end (normalized time 1)', 't >= 1-History_fraction'],
        'rule_B.txt': ['Bernoulli event sampling',
                       'An observed 0 in an accessible window can therefore be a false negative']}
    for f, phrases in required.items():
        for phrase in phrases: assert phrase in text[f], (f, phrase)
    forbidden = {'user_prefix.txt': ['Walk_A', '10%', 'corrector', 'baseline'],
                 'rule_S.txt': ['Walk_A', 'stationary', 'corrector', 'uniformly among']}
    for f, phrases in forbidden.items():
        for phrase in phrases: assert phrase.lower() not in text[f].lower(), (f, phrase)
    return {f: sha(spec/f) for f in PROMPT_FILES}


# ---------------------------------------------------------------- 10
def audit_seeds():
    """Register every stream of the study; common.seed raises on any collision."""
    for r in read_json(PREPARED/'seed_manifest.json'):
        assert r['fields'][0] == MASTER_SEED and seed(*r['fields'][1:]) == r['seed']
    seed('extratrees', 'all_folds')
    for spec in pool_definition()['graphs']:
        key = spec['key']
        budget = read_json(REFERENCES/'pool/observations'/f'{key}.json')['budget']
        seed('pool', key)
        for i in range(1, 257): seed('walk_calibration_cells', key, ARM_ID['S'], i)
        for i in range(1, budget['validation_n']+1): seed('walk_validation_cells', key, ARM_ID['S'], i)
        domain = 'pool_'+spec['partition']
        for arm in ARMS:
            for i in range(1, draws_for(arm, budget, domain)+1): seed(domain, key, ARM_ID[arm], i)
    for key in MAIN_KEYS:
        for i in range(1, WALK_PATHS+1): seed('walk_diagnostic', key, ARM_ID['S'], i)
    for key in REAL_TEST:
        seed('pwt_productive', key)
        for i in range(1, 100): seed('pwt_null_diagnostic', key, '', i)
    AUDIT.mkdir(parents=True, exist_ok=True)
    with open(AUDIT/'complete_seed_manifest.jsonl', 'w') as f:
        for value, fields in sorted(SEEDS.items()):
            f.write(json.dumps({'seed': value, 'fields': json.loads(fields)}, separators=(',', ':'))+'\n')
    previous = {r['seed'] for r in read_json(PREVIOUS_RUN/'seed_manifest.json')}
    assert not set(SEEDS) & previous
    domains = Counter(json.loads(v)[1] for v in SEEDS.values())
    return {'unique_streams': len(SEEDS), 'domains': dict(domains), 'collisions': 0,
            'previous_generation_overlap': 0, 'manifest_sha256': sha(AUDIT/'complete_seed_manifest.jsonl'),
            'crn_aliases': {s: s.removesuffix('__pwt') for s in SURROGATES},
            'alias_scope': 'sampler streams R/S/H/B keyed by the parent source by design'}


# ---------------------------------------------------------------- 11
def audit_diagnostics():
    null = read_json(DIAGNOSTICS/'null_model/report.json')
    assert null['null_shuffles'] == 8*99 and null['all_invariants_passed']
    windows = read_json(DIAGNOSTICS/'windows/report.json')
    assert windows['rows'] == 24*19 and windows['census_graphs'] == 24
    assert len(pd.read_csv(DIAGNOSTICS/'history/sources.csv')) == 24*3
    assert len(pd.read_csv(DIAGNOSTICS/'walk/paths.csv')) == 24*WALK_PATHS*2
    for name in ('decomposition', 'mixture_bounds'): assert (DIAGNOSTICS/name/'report.json').exists()
    return {name: sha(DIAGNOSTICS/name/'report.json')
            for name in ('decomposition', 'history', 'walk', 'null_model', 'windows', 'mixture_bounds')}


def main():
    report = read_json(PREPARED/'report.json')
    assert report['offline_ready'] and report['design_version'] == DESIGN_VERSION
    checksums = audit_checksums()
    keys = (*TRAIN, *SYNTH, *SURROGATES)
    graphs = {k: load_graph(PREPARED/'graphs'/k) for k in keys}
    budgets = {k: read_json(PREPARED/'calibration'/f'{k}.json') for k in keys}
    for k in keys: audit_graph(graphs[k], read_json(PREPARED/'graphs'/k/'manifest.json'))
    audit_copenhagen()
    design = planned_sizes(budgets)
    assert (design['main_observations'], design['training_observations'], design['planned_calls']) == (288, 320, 3456)
    result = {
        'design_version': DESIGN_VERSION, 'checksums': checksums, 'design': design,
        'surrogates': [audit_surrogate(graphs[p], graphs[p+'__pwt']) for p in REAL_TEST],
        'budgets': [audit_budget(graphs[k], budgets[k]) for k in keys],
        'deterministic_h_graphs': sorted(k for k in MAIN_KEYS if budgets[k]['h_saturated']),
        'observations_rederived': audit_observations(graphs, budgets),
        'common_random_numbers': audit_common_random_numbers(graphs, budgets),
        'models': audit_models_and_references({k: graphs[k].truth for k in TRAIN}),
        'requests': audit_requests(design),
        'prompt_templates': audit_prompt_templates(),
        'seeds': audit_seeds(),
        'diagnostics': audit_diagnostics(),
        'sol_deepseek_started': 0, 'verified': True}
    write_json(AUDIT/'offline_audit.json', result)
    print(json.dumps({k: result[k] for k in ('design', 'observations_rederived', 'requests', 'deterministic_h_graphs',
                                             'verified')}, indent=1))
    print('seeds:', result['seeds']['unique_streams'], 'unique streams, no collision')


if __name__ == '__main__':
    main()
