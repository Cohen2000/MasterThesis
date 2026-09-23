"""Stage 1: graphs, surrogates, budgets, observations, prompts and requests.

Strictly offline. For each of the 16 real training sources, the eight synthetic
main instances and the eight surrogates, this stage builds the canonical graph,
calibrates the four arms to T, draws the observations (test draws for the 24
main graphs, training draws for the 16 real sources), renders the prompts,
checks their token sizes and writes the request manifest of all four model
configurations. Nothing is sent to any model.
"""
import importlib.metadata
import json
import platform
import subprocess
import numpy as np
from .common import (ARMS, CONFIGS, DESIGN_VERSION, LLM_REPEATS, MAIN_KEYS, MASTER_SEED, PREPARED, BUILD,
                     ROOT, SEEDS, SURROGATE_PARENT, SURROGATES, SYNTH, TRAIN, code_hashes, digest, draws_for,
                     fresh_directory, graph_stratum, observation_id, parent_source, planned_sizes, read_json,
                     sha, write_csv, write_json)
from .baselines import design_reference
from .data import load_graph, prepare_real, save_graph
from .observation import FEATURE_VERSION, features, make, messages, parse, serialize
from .requests import EXECUTION_POLICY, planned
from .sampling import calibrate, draw
from .surrogates import prepare_surrogate
from .synthetic import generate_pair
from .token_sizes import TOKEN_COUNTS, TokenCounters


def build_graph(key, out, raw_dir):
    """Canonical graph of one real source, synthetic instance or surrogate."""
    folder = out/'graphs'/key
    if key in TRAIN:
        return prepare_real(key, raw_dir, folder)
    if key in SURROGATES:
        return prepare_surrogate(load_graph(out/'graphs'/SURROGATE_PARENT[key]), folder)
    # Synthetic instances come in pairs sharing all latents (a0/a08, memoryless/memory);
    # the pair is generated once, when its first member is requested.
    if not (folder/'manifest.json').exists():
        family, replicate = key.split('_')[0], int(key[-1])
        for g, frame, meta in generate_pair(family, replicate):
            save_graph(out/'graphs'/g.key, g, meta, frame)
    return load_graph(folder)


def observation_row(g, arm, index, domain, budget, counts, traversals):
    """One stored observation. Only `block`/`messages` are model input; `truth` and the
    S design_reference (from the traversal log) are internal and never shown."""
    obs = make(g, arm, budget, counts, traversals)
    block = serialize(obs)
    parsed = parse(block)
    if not np.array_equal(features(obs), features(parsed)): raise AssertionError('serialization changes features')
    prompt = messages(block)
    return {'id': observation_id(g.key, arm, index, budget['coverage_fraction']), 'graph_id': g.key, 'source_family': g.key,
            'stratum': graph_stratum(g.key), 'parent_source': parent_source(g.key), 'arm': arm,
            'sample_index': index, 'domain': domain, 'empty': parsed['D_obs'] == 0,
            'block': block, 'block_sha256': digest(block), 'messages': prompt, 'prompt_sha256': digest(prompt),
            'truth': g.truth, 'design_reference': None,
            'design_version': DESIGN_VERSION,
            'deterministic_draw': draws_for(arm, budget, domain) == 1,
            'budget_matched': budget['budget_matched_by_arm'][arm],
            'budget_matched_all_arms': budget['budget_matched'],
            'internal_evaluation': {'observed_event_fraction': parsed['M_obs']/g.M,
                                    'observed_dyad_fraction': parsed['D_obs']/g.D,
                                    'observed_cell_fraction': int((counts > 0).sum())/g.cells}}


def domains_of(key):
    """Test draws for the 24 main graphs, training draws for the 16 real sources."""
    return (['training'] if key in TRAIN else [])+(['sample'] if key in MAIN_KEYS else [])


def record_environment(out):
    packages = sorted(f'{d.metadata["Name"]}=={d.version}' for d in importlib.metadata.distributions())
    (out/'environment.lock.txt').write_text('\n'.join(packages)+'\n')
    def git(*a):
        try:
            return subprocess.check_output(['git', *a], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None
    commit = git('rev-parse', 'HEAD')
    write_json(out/'environment.json', {
        'python': platform.python_version(), 'platform': platform.platform(),
        'git_commit_at_run': commit, 'git_worktree_dirty_at_run': bool(git('status', '--porcelain')) if commit else None,
        'lock_sha256': sha(out/'environment.lock.txt'), 'inference_performed': False,
        'compiler': subprocess.check_output(['g++', '--version'], text=True).splitlines()[0]})


def run(out=PREPARED, raw_dir=ROOT/'data/raw', tokenizers=ROOT/'data/tokenizers'):
    out = fresh_directory(out)
    write_json(out/'inputs.json', {'code': code_hashes(), 'master_seed': MASTER_SEED,
                                   'design_version': DESIGN_VERSION, 'feature_version': FEATURE_VERSION})
    record_environment(out)
    budgets = {}; data_rows = []; main_rows = []; training_rows = []
    for key in (*TRAIN, *SYNTH, *SURROGATES):
        g = build_graph(key, out, raw_dir)
        budget, walk, validation_volumes = calibrate(g, BUILD)
        budgets[key] = budget
        write_json(out/'calibration'/f'{key}.json', budget)
        write_json(out/'calibration'/f'{key}_validation_volumes.json', validation_volumes)
        data_rows.append(read_json(out/'graphs'/key/'manifest.json'))
        print(f'{key}: N={g.N} D={g.D} M={g.M} cells={g.cells} T={budget["T"]:.1f} n={budget["n_panel"]} '
              f'L={budget["L"]} n_H={budget["n_panel_history"]} p={budget["p"]:.5f} '
              f'H_saturated={budget["h_saturated"]} matched={budget["budget_matched"]}', flush=True)
        for domain in domains_of(key):
            for arm in ARMS:
                for index in range(1, draws_for(arm, budget, domain)+1):
                    counts, traversals = draw(g, arm, index, domain, budget, walk)
                    row = observation_row(g, arm, index, domain, budget, counts, traversals)
                    write_json(out/'observations'/domain/f'{row["id"]}.json', row)
                    (training_rows if domain == 'training' else main_rows).append(row)
    write_csv(out/'data_summary.csv', data_rows)
    write_csv(out/'budget_summary.csv', [{'graph_id': k, **b} for k, b in budgets.items()])

    counters = TokenCounters(tokenizers)
    sizes = [{'id': row['id'], **counters.count(row['messages'])} for row in main_rows]
    write_csv(out/'prompt_sizes.csv', sizes)
    write_json(out/'tokenizer_manifest.json', {name: read_json(tokenizers/name/'provenance.json')
                                               for name in ('qwen', 'deepseek')})

    requests = planned(main_rows)
    write_json(out/'execution_policy.json', EXECUTION_POLICY)
    with open(out/'requests.jsonl', 'w') as f:
        for r in requests: f.write(json.dumps(r, separators=(',', ':'), allow_nan=False, sort_keys=True)+'\n')
    write_json(out/'seed_manifest.json', [{'seed': s, 'fields': json.loads(v)} for s, v in sorted(SEEDS.items())])

    design = planned_sizes(budgets)
    report = {'design_version': DESIGN_VERSION, 'feature_version': FEATURE_VERSION,
              'matched_quantity': 'expected_observed_active_dyad_windows', 'coverage_fraction': 0.10,
              'planned_observations': design['main_observations'], 'prepared_observations': len(main_rows),
              'planned_training_observations': design['training_observations'],
              'training_observations': len(training_rows),
              'deterministic_h_graphs': sorted(k for k, b in budgets.items() if b['h_saturated']),
              'h_target_unreachable_graphs': sorted(k for k, b in budgets.items() if b['h_target_unreachable']),
              'budget_unmatched_graphs': {k: b['unmatched_reasons'] for k, b in budgets.items() if not b['budget_matched']},
              'planned_logical_calls': design['planned_calls'], 'planned_qwen_calls': design['qwen_calls'],
              'request_manifest_rows': len(requests),
              'requests_per_config': {c: sum(r['config_id'] == c for r in requests) for c in CONFIGS},
              'empty_observations': sum(r['empty'] for r in main_rows),
              'prompt_sizes_checked': len(sizes),
              'max_prompt_tokens': {k: max(s[k] for s in sizes) for k in TOKEN_COUNTS},
              'started_calls': 0, 'llm_study_conducted': False}
    report['offline_ready'] = (len(main_rows) == design['main_observations'] and
                               len(training_rows) == design['training_observations'] and
                               len(sizes) == len(main_rows) and len(requests) == design['planned_calls'] and
                               len(requests) == len(main_rows)*len(CONFIGS)*LLM_REPEATS)
    write_json(out/'report.json', report)
    files = {str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*'))
             if p.is_file() and 'build' not in p.relative_to(out).parts and p.name != 'checksums.json'}
    write_json(out/'checksums.json', files)
    print(json.dumps(report, indent=2), flush=True)
    if not report['offline_ready']: raise SystemExit('prepared study incomplete')
    return report


def prepared_graph(key):
    """A main graph and its budget as written by this stage."""
    return load_graph(PREPARED/'graphs'/key), read_json(PREPARED/'calibration'/f'{key}.json')
