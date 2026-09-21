"""Stages 2-4: synthetic training pool, ExtraTrees folds, reference predictions.

  pool        400 training + 100 development synthetic graphs (pool.py) with
              five training draws per arm.
  train       nine LOSO folds x {pooled, real_only} forests (training.py).
  references  development check of every reference on the 100 development
              graphs (synthetic fold), and every reference prediction for the
              288 main observations (primary_baselines.json), which the LLM
              evaluation compares against.
"""
import time
from pathlib import Path
import numpy as np
from .baselines import METHODS, PRIMARY_REFERENCE, PRIMARY_REFERENCE_NAME, all_references
from .common import (ARMS, DESIGN_VERSION, PREPARED, STRATA, code_hashes, fold_for, graph_stratum, in_stratum,
                     read_json, sha, write_csv, write_json)
from .observation import parse
from .pool import COUNTS, build_pool, pool_definition
from .training import FOLDS, fit_folds, load_models

VARIANTS = ('pooled', 'real_only')


# ---------------------------------------------------------------- inputs
def real_training_rows(prepared=PREPARED):
    rows = [read_json(p) for p in sorted((prepared/'observations/training').glob('*.json'))]
    for r in rows: r['block_group'] = 'real'
    return rows, {r['source_family']: r['truth'] for r in rows}


def pool_rows(out, partition):
    """Observations, truths and graph records of one pool partition."""
    rows = []; truth = {}; graphs = {}
    for path in sorted((out/'pool/observations').glob('*.json')):
        graph = read_json(path)
        if graph['partition'] != partition: continue
        truth[graph['key']] = graph['truth']; graphs[graph['key']] = graph
        rows += [{**r, 'block_group': graph['family']} for r in graph['observations']]
    expected = sum(c[partition] for c in COUNTS.values())
    if len(truth) != expected: raise ValueError(f'incomplete pool {partition}: {len(truth)}/{expected}')
    return rows, truth, graphs


def fold_references(out):
    models = {v: load_models(out/f'models_{v}') for v in VARIANTS}
    medians = {fold: read_json(out/'models_pooled'/fold/'manifest.json')['median'] for fold in FOLDS}
    return models, medians


def error_record(prediction, truth):
    e = np.asarray(prediction, float)-np.asarray(truth, float)
    return {'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0])}


# ---------------------------------------------------------------- stages
def stage_pool(out):
    definition = pool_definition()
    write_json(out/'pool_definition.json', definition)
    start = time.perf_counter()
    build_pool(out/'pool', definition['graphs'])
    return {'graphs': definition['n_graphs'], 'seconds': time.perf_counter()-start}


def stage_train(out, prepared=PREPARED):
    real, real_truth = real_training_rows(prepared)
    pool, pool_truth, _ = pool_rows(out, 'train')
    fit_folds(real, {**real_truth, **pool_truth}, out/'models_pooled', pool)
    fit_folds(real, real_truth, out/'models_real_only')
    return {'real_rows': len(real), 'pool_rows': len(pool), 'pool_graphs': len(pool_truth), 'folds': list(FOLDS)}


def stage_references(out):
    models, medians = fold_references(out)
    return {'development': development_check(out, models, medians), 'main': main_references(out, models, medians)}


# ---------------------------------------------------------------- development check
def per_graph_mean(records, metric):
    """Mean per graph, so every development graph or main source counts once."""
    values = {}
    for r in records: values.setdefault(r['graph_id'], []).append(r[metric])
    return {g: float(np.mean(v)) for g, v in values.items()}


def mean_and_se(values):
    a = np.asarray(list(values), float)
    if len(a) < 2: return (float(a.mean()) if len(a) else float('nan')), float('nan')
    return float(a.mean()), float(a.std(ddof=1)/np.sqrt(len(a)))


def graph_level_rows(records, group_field, groups):
    """MAE2/ProfileMAE with graph-level SE, and paired differences to plugin and corrector."""
    rows = []
    for group in groups:
        for arm in ARMS:
            for method in METHODS:
                select = [r for r in records if group in ('all', r[group_field]) and r['arm'] == arm]
                chosen = [r for r in select if r['method'] == method]
                if not chosen: continue
                ae = per_graph_mean(chosen, 'AE2')
                row = {group_field: group, 'arm': arm, 'method': method, 'graphs': len(ae)}
                for metric, label in (('AE2', 'MAE2'), ('ProfileAE', 'ProfileMAE'), ('signed_rho2', 'signed_rho2')):
                    row[label], row[label+'_se'] = mean_and_se(per_graph_mean(chosen, metric).values())
                row['worst_graph_AE2'] = max(ae.values())
                for reference in ('plugin', 'corrector'):
                    if method == reference: continue
                    base = per_graph_mean([r for r in select if r['method'] == reference], 'AE2')
                    common = sorted(set(ae) & set(base))
                    row[f'paired_vs_{reference}'], row[f'paired_vs_{reference}_se'] = mean_and_se(ae[g]-base[g] for g in common)
                rows.append(row)
    return rows


def mixture_diagnostics(records):
    fits = [r for r in records if r['method'] == 'mixture' and r['status'] != 'empty_sample']
    status = {}
    for r in fits: status[r['status']] = status.get(r['status'], 0)+1
    by_status = [{'status': s, 'n': sum(r['status'] == s for r in fits),
                  'MAE2': float(np.mean([r['AE2'] for r in fits if r['status'] == s]))} for s in sorted(status)]
    return {'fits': len(fits), 'status_counts': status, 'fallbacks_to_homogeneous': sum(bool(r['fallback']) for r in fits),
            'seconds_total': float(sum(r['seconds'] for r in fits)), 'MAE2_by_status': by_status}


def development_check(out, models, medians):
    """Every reference on the 100 development graphs, predicted by the synthetic fold."""
    rows, truth, graphs = pool_rows(out, 'dev')
    fold = {v: models[v]['synthetic'] for v in VARIANTS}
    records = []
    for r in rows:
        predictions = all_references(parse(r['block']), fold, medians['synthetic'])
        for method, p in predictions.items():
            records.append({'graph_id': r['graph_id'], 'family': graphs[r['graph_id']]['family'], 'arm': r['arm'],
                            'sample_index': r['sample_index'], 'method': method,
                            'budget_matched': bool(r['budget_matched']), **error_record(p['prediction'], truth[r['graph_id']]),
                            'status': p['status'], 'fallback': p.get('fallback', ''), 'seconds': p.get('seconds', 0.)})
    write_csv(out/'development_observations.csv', records)
    matched = [r for r in records if r['budget_matched']]
    summary = {'rows': graph_level_rows(records, 'family', ('all', 'dar', 'ad')),
               'budget_matched_rows': graph_level_rows(matched, 'family', ('all',)),
               'mixture': mixture_diagnostics(records)}
    write_json(out/'development_summary.json', summary)
    write_csv(out/'development_summary.csv', summary['rows'])
    return {'observations': len(rows), 'graphs': len(truth), 'mixture': summary['mixture']}


# ---------------------------------------------------------------- main panel
def main_references(out, models, medians, prepared=PREPARED):
    """All reference predictions for the 288 main observations; the primary one named."""
    observations = [read_json(p) for p in sorted((prepared/'observations/sample').glob('*.json'))]
    manifests = {k: read_json(PREPARED/'graphs'/k/'manifest.json') for k in {o['graph_id'] for o in observations}}
    entries = {}; records = []
    for row in observations:
        o = parse(row['block']); g = row['graph_id']; fold = fold_for(g)
        predictions = all_references(o, {v: models[v][fold] for v in VARIANTS}, medians[fold])
        entry = {m: {k: p[k] for k in ('prediction', 'status', 'fallback', 'flags') if k in p} for m, p in predictions.items()}
        entry.update(primary_corrector=dict(entry[PRIMARY_REFERENCE[row['arm']]]),
                     primary_corrector_name=PRIMARY_REFERENCE_NAME[row['arm']],
                     arm=row['arm'], stratum=graph_stratum(g), graph_id=g, sample_index=row['sample_index'],
                     fold=fold, deterministic_draw=row['deterministic_draw'], truth=row['truth'],
                     block_sha256=row['block_sha256'])
        entries[row['id']] = entry
        D_full = manifests[g]['D_full']
        observed_windows = sum(r[0].count('1')*r[1] for r in o['table'])
        for method, p in predictions.items():
            records.append({'graph_id': g, 'stratum': graph_stratum(g), 'arm': row['arm'],
                            'sample_index': row['sample_index'], 'method': method,
                            **error_record(p['prediction'], row['truth']),
                            'status': p['status'], 'fallback': p.get('fallback', ''),
                            'D_obs': o['D_obs'], 'M_obs': o['M_obs'], 'dyad_coverage': o['D_obs']/D_full,
                            'window_coverage': observed_windows/manifests[g]['active_dyad_windows'],
                            'event_coverage': o['M_obs']/manifests[g]['M_full'], 'budget_matched': row['budget_matched']})
    write_json(out/'primary_baselines.json', {
        'design_version': DESIGN_VERSION, 'protocol': 'docs/PROTOCOL_PANEL888_20260921.md',
        'primary_corrector_by_arm': PRIMARY_REFERENCE, 'primary_corrector_meaning': PRIMARY_REFERENCE_NAME,
        'primary_trained_reference': 'extratrees_pooled', 'observations': entries})
    write_csv(out/'main_observations.csv', records)
    for r in records:
        r['evidence_block'] = next(s for s in STRATA if in_stratum(r['graph_id'], s))
    rows = graph_level_rows(records, 'evidence_block', STRATA)
    coverage = []
    for block in STRATA:
        for arm in ARMS:
            chosen = [r for r in records if r['evidence_block'] == block and r['arm'] == arm and r['method'] == 'plugin']
            coverage.append({'evidence_block': block, 'arm': arm,
                             **{f'{k}_median': float(np.median([r[k] for r in chosen]))
                                for k in ('dyad_coverage', 'window_coverage', 'event_coverage', 'D_obs', 'M_obs')},
                             'D_obs_min': int(min(r['D_obs'] for r in chosen))})
    write_json(out/'main_summary.json', {'rows': rows, 'coverage': coverage})
    write_csv(out/'main_summary.csv', rows)
    write_csv(out/'main_coverage.csv', coverage)
    return {'observations': len(observations), 'records': len(records)}


def run(out, stage):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    stages = {'pool': stage_pool, 'train': stage_train, 'references': stage_references}
    marker = out/f'{stage}.json'
    if marker.exists(): raise FileExistsError(f'{marker} exists; remove {out} to recompute')
    started = time.perf_counter()
    report = stages[stage](out)
    report.update(stage=stage, design_version=DESIGN_VERSION, code=code_hashes(),
                  prepared_requests_sha256=sha(PREPARED/'requests.jsonl'), seconds=time.perf_counter()-started)
    write_json(marker, report)
    print(stage, {k: v for k, v in report.items() if k != 'code'}, flush=True)
    return report
