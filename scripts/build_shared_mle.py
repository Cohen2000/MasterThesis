#!/usr/bin/env python3
"""Offline evaluation of the shared ZT Beta-Binomial MLE (main_experiment.shared_mle).

Does not touch or recompute any existing reference (plugin/corrector/ExtraTrees/
mixture): it only reads the already-prepared observations of the main study
(results/panel888) and of every budget-sensitivity level
(results/panel888_budget_sensitivity), and writes its own output tree,
results/panel888_shared_mle/, which is the "new result directory" of this task.

  main_and_budget   shared_mle.fit on every main and budget-level observation
  adequacy          three fixed model-adequacy diagnostics on development/
                    training material only (never on the eight held-out test
                    sources): (A) full-data representation, (B) H-like
                    extrapolation, (C) B-like observation, as specified for
                    this task. The model form is not tuned on their result.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from budget_sensitivity import folder                                            # noqa: E402
from main_experiment.common import (BUDGET_GRID, H_FRACTION, REFERENCES, ROOT, STRATA, TRAIN, in_stratum, read_json,
                                    write_csv, write_json)
from main_experiment.observation import access_for, parse
from main_experiment.pool import pool_definition, regenerate
from main_experiment.prepare import build_graph
from main_experiment.references import per_graph_mean, pool_rows
from main_experiment.shared_mle import fit, fit_profile_from_counts

OUT = ROOT/'results/panel888_shared_mle'


def error_record(prediction, truth):
    e = np.asarray(prediction, float)-np.asarray(truth, float)
    return {'AE2': float(abs(e[0])), 'ProfileAE': float(np.mean(np.abs(e))), 'signed_rho2': float(e[0])}


# ---------------------------------------------------------------- main + budget
def observation_rows(fraction):
    out = folder(fraction)
    for path in sorted((out/'observations/sample').glob('*.json')):
        yield read_json(path)


def fit_rows(fraction):
    rows = []; empty = 0; failed = 0
    for row in observation_rows(fraction):
        o = parse(row['block'])
        if o['D_obs'] == 0: empty += 1; continue
        try: result = fit(o)
        except ValueError: failed += 1; continue
        rows.append({'budget': fraction, 'graph_id': row['graph_id'], 'arm': row['arm'],
                     'evidence_block': next(s for s in STRATA if in_stratum(row['graph_id'], s)),
                     'fit_status': result.fit_status, 'fallback_used': result.fallback_used,
                     'objective': result.objective, **error_record(result.rho, row['truth'])})
    return rows, empty, failed


def summarize(rows, group_field):
    """Graph-level-mean MAE2/ProfileMAE per (group, arm); no source disappears silently."""
    out = []
    for group in sorted({r[group_field] for r in rows}) if rows else []:
        for arm in sorted({r['arm'] for r in rows}):
            chosen = [r for r in rows if r[group_field] == group and r['arm'] == arm]
            if not chosen: continue
            ae = per_graph_mean(chosen, 'AE2'); pe = per_graph_mean(chosen, 'ProfileAE')
            out.append({group_field: group, 'arm': arm, 'graphs': len(ae),
                        'MAE2': float(np.mean(list(ae.values()))), 'ProfileMAE': float(np.mean(list(pe.values()))),
                        'fallback_count': sum(r['fallback_used'] for r in chosen)})
    return out


def main_and_budget():
    report = {}; all_rows = []
    for fraction in BUDGET_GRID:
        rows, empty, failed = fit_rows(fraction)
        all_rows += rows
        report[f'b{round(fraction*1000):03d}'] = {'fraction': fraction, 'observations': len(rows),
                                                   'empty_sample': empty, 'failed': failed,
                                                   'fallback_used': sum(r['fallback_used'] for r in rows)}
    write_csv(OUT/'main_and_budget_observations.csv', all_rows)
    write_json(OUT/'main_and_budget_summary.json', {'by_budget_arm': summarize(all_rows, 'budget'),
                                                     'by_evidence_block_arm': summarize(all_rows, 'evidence_block')})
    return report


# ---------------------------------------------------------------- adequacy diagnostics
def development_training_graphs(graph_cache=ROOT/'results/.shared_mle_graph_cache'):
    """Real training sources (rebuilt directly from data/raw, not from any
    prepared-study output -- this needs no cluster artifact at all) plus the
    100 synthetic development graphs -- never a held-out test source.
    """
    out = [(f'train:{key}', build_graph(key, graph_cache, ROOT/'data/raw')) for key in TRAIN]
    for spec in pool_definition()['graphs']:
        if spec['partition'] == 'dev': out.append((f"dev:{spec['key']}", regenerate(spec)))
    return out


def adequacy_full_data(graphs):
    """(A) Fit as if the complete 5-window history were observed; compare to the true full-data profile."""
    records = []
    for label, g in graphs:
        counts = [0]*6
        for k in g.K: counts[k] += 1
        result = fit_profile_from_counts(counts, 5)
        records.append({'graph': label, 'alpha': result.alpha, 'beta': result.beta, 'fit_status': result.fit_status,
                        'fallback_used': result.fallback_used, 'objective': result.objective,
                        **error_record(result.rho, g.truth)})
    return records


def adequacy_h_like(graphs, h=H_FRACTION):
    """(B) Hide the early windows the main H access pattern hides; fit visible-only; extrapolate to W=5."""
    mask = np.array(access_for('H', h), dtype=bool); m = int(mask.sum())
    records = []
    for label, g in graphs:
        restricted = (g.counts[:, mask] > 0).sum(1)
        counts = [0]*(m+1)
        for x in restricted:
            if x > 0: counts[x] += 1
        result = fit_profile_from_counts(counts, m)
        records.append({'graph': label, 'visible_windows': m, 'alpha': result.alpha, 'beta': result.beta,
                        'fit_status': result.fit_status, 'fallback_used': result.fallback_used,
                        'objective': result.objective, **error_record(result.rho, g.truth)})
    return records


def adequacy_b_like():
    """(C) The shared B model (Beta activity + ZTP(lambda) + Binomial(p) thinning) on dev-pool B observations."""
    rows, truth, _ = pool_rows(REFERENCES, 'dev')
    records = []
    for r in rows:
        if r['arm'] != 'B': continue
        o = parse(r['block'])
        if o['D_obs'] == 0: continue
        try: result = fit(o)
        except ValueError: continue
        records.append({'graph_id': r['graph_id'], 'fit_status': result.fit_status,
                        'fallback_used': result.fallback_used, 'lam': result.lam,
                        **error_record(result.rho, truth[r['graph_id']])})
    return records


def adequacy():
    graphs = development_training_graphs()
    a = adequacy_full_data(graphs); b = adequacy_h_like(graphs); c = adequacy_b_like()
    write_csv(OUT/'adequacy_full_data.csv', a)
    write_csv(OUT/'adequacy_h_like.csv', b)
    write_csv(OUT/'adequacy_b_like.csv', c)
    def summary(records):
        if not records: return {'n': 0}
        return {'n': len(records), 'MAE2': float(np.mean([r['AE2'] for r in records])),
                'ProfileMAE': float(np.mean([r['ProfileAE'] for r in records])),
                'fallback_used': sum(r['fallback_used'] for r in records)}
    report = {'full_data_representation': summary(a), 'h_like_extrapolation': summary(b), 'b_like_observation': summary(c)}
    write_json(OUT/'adequacy_summary.json', report)
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {'adequacy': adequacy(), 'main_and_budget': main_and_budget()}
    write_json(OUT/'report.json', report)
    print(report, flush=True)


if __name__ == '__main__':
    main()
