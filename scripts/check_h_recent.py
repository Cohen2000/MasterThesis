#!/usr/bin/env python3
"""Offline acceptance check of arm H (budget10-hrecent5-20260917), 20 draws per main graph.

No LLM call and no model fit. Full histories are used for evaluation only.

Per main graph:
  population   exact quantities over all active dyads: capped share, dyads whose
               observed window count J is below K (information actually lost),
               the expected plug-in profile rho^J (under a fixed-size uniform
               sample E[plug-in] = rho^J exactly), the expected upper profile rho^U,
               and the SRS variances (1-d/D) S^2/d of oracle, plug-in and midpoint.
  draws        20 independent dyad samples on the check stream (one for a
               saturated graph, whose sample is deterministic): selection =
               oracle - truth, history = plug-in - oracle, total, bound midpoint
               error, coverage of truth by [L,U], width, realised volume, at-cap
               share, and the existing baselines (plug-in, midpoint, fold median,
               pooled and real-only ExtraTrees).
  mechanism    on the same sampled dyads, min(5,m_e) events drawn uniformly
               without replacement instead of the most recent ones (identical
               volume by construction): lost information and plug-in error. The
               recent-cap bounds are NOT applied to these samples.
"""
import argparse, csv, json, math, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import (REAL_TEST, SYNTH, H_CAP, ARM_ID, rng, read_json, write_json)
from main_experiment.data import load_graph
from main_experiment.sampling import budget_parameters, recent_counts, reservoir_counts
from main_experiment.observation import make, serialize, parse, features
from main_experiment.baselines import plugin, h_bounds, h_midpoint

DRAWS = 20
KS = (2, 3, 4, 5)
WEAK_HISTORY = 0.02     # |expected history loss on rho_2| below this is reported as a limit case


def profile(K):
    return np.array([np.mean(K >= k) for k in KS])


def srs_var(values, d, D):
    """Variance of the mean of a size-d simple random sample without replacement."""
    if d >= D: return np.zeros(values.shape[1])
    return (1 - d / D) * np.var(values, axis=0, ddof=1) / d


def population(g, b):
    K = g.K
    obs = recent_counts(g.counts)
    J = (obs > 0).sum(1)
    first = np.argmax(obs > 0, axis=1) + 1
    atcap = obs.sum(1) == H_CAP
    U = np.where(atcap, J + first - 1, J)
    ind = lambda x: np.stack([x >= k for k in KS], 1).astype(float)
    IK, IJ, IU = ind(K), ind(J), ind(U)
    mid = (IJ + IU) / 2
    d, D = b['n_dyads'], g.D
    rho, rhoJ, rhoU = IK.mean(0), IJ.mean(0), IU.mean(0)
    return {'truth': rho, 'rhoJ': rhoJ, 'rhoU': rhoU, 'mid': mid.mean(0),
            'history': rhoJ - rho, 'mid_bias': mid.mean(0) - rho, 'width': rhoU - rhoJ,
            'sd_selection': np.sqrt(srs_var(IK, d, D)), 'sd_plugin': np.sqrt(srs_var(IJ, d, D)),
            'sd_mid': np.sqrt(srs_var(mid, d, D)),
            'share_capped': float(np.mean(g.m > H_CAP)), 'share_at_cap': float(np.mean(atcap)),
            'share_J_lt_K': float(np.mean(J < K)),
            'share_K2_seen_below_2': float(np.mean((K >= 2) & (J < 2)) / max(np.mean(K >= 2), 1e-300)),
            'mean_lost_windows': float(np.mean(K - J)),
            'truth_outside_population_bounds': bool(np.any(rho > rhoU + 1e-12))}


def load_models(rev):
    models, medians = {}, {}
    for fold in [*REAL_TEST, 'synthetic']:
        for kind in ('pooled', 'real_only'):
            with open(rev / f'models_{kind}' / fold / 'model.pkl', 'rb') as f:
                models[(fold, kind)] = pickle.load(f)
        medians[fold] = read_json(rev / 'models_pooled' / fold / 'manifest.json')['median']
    return models, medians


def mean_se(x):
    x = np.asarray(x, float)
    if len(x) < 2: return float(x.mean()), 0.
    return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='results/main_experiment/hrecent5_20260917')
    ap.add_argument('--revision', default='results/baseline_revision_hrecent5_20260917')
    ap.add_argument('--out', default='results/h_recent5_check_20260917')
    a = ap.parse_args()
    run, rev, out = Path(a.run), Path(a.revision), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    models, medians = load_models(rev)
    t0 = time.time()
    pop_rows, draw_rows, graph_rows = [], [], []
    for key in [*REAL_TEST, *SYNTH]:
        g = load_graph(run / 'graphs' / key)
        b = read_json(run / 'calibration' / key / 'budget.json')
        if {k: b[k] for k in ('n_dyads', 'C_cap')} != {k: budget_parameters(g)[k] for k in ('n_dyads', 'C_cap')}:
            raise ValueError('stored budget differs from recomputation')
        real = key in REAL_TEST
        fold = key if real else 'synthetic'
        pop = population(g, b)
        # Population value of the uniform-subset comparison: twenty full passes.
        res_rho, res_lost = [], []
        for i in range(DRAWS):
            rr = rng('h_check_reservoir_population', key, ARM_ID['H'], i + 1)
            c = reservoir_counts(g.counts, H_CAP, rr)
            Jr = (c > 0).sum(1)
            res_rho.append(profile(Jr)); res_lost.append(np.mean(Jr < g.K))
        pop['rhoJ_reservoir'] = np.mean(res_rho, 0)
        pop['history_reservoir'] = pop['rhoJ_reservoir'] - pop['truth']
        pop['share_J_lt_K_reservoir'] = float(np.mean(res_lost))
        n_draws = 1 if b['h_saturated'] else DRAWS
        prow = {'graph_id': key, 'stratum': 'real' if real else 'synthetic', 'D': g.D, 'M': g.M, 'B': g.B,
                'C_cap': b['C_cap'], 'n_dyads': b['n_dyads'], 'dyad_share': b['h_dyad_share'],
                'expected_volume': b['h_expected_events'], 'relative_budget_error': b['h_relative_budget_error'],
                'target_unreachable': b['h_target_unreachable'], 'saturated': b['h_saturated'],
                'within_tolerance': b['h_within_tolerance'], 'draws': n_draws}
        for name in ('share_capped', 'share_at_cap', 'share_J_lt_K', 'share_J_lt_K_reservoir',
                     'share_K2_seen_below_2', 'mean_lost_windows', 'truth_outside_population_bounds'):
            prow[name] = pop[name]
        for name in ('truth', 'rhoJ', 'rhoU', 'mid', 'history', 'history_reservoir', 'mid_bias', 'width',
                     'sd_selection', 'sd_plugin', 'sd_mid'):
            for k, v in zip(KS, pop[name]): prow[f'{name}_{k}'] = float(v)
        prow['weak_history_rho2'] = abs(pop['history'][0]) < WEAK_HISTORY
        # rho_2 alone can hide a large loss on rho_3..rho_5 (DAR alpha=0.8,
        # activity-driven with memory), so the profile mean is flagged separately.
        prow['history_profile'] = float(np.mean(pop['history']))
        prow['weak_history_profile'] = abs(prow['history_profile']) < WEAK_HISTORY
        pop_rows.append(prow)
        rows = []
        for ix in range(1, n_draws + 1):
            if b['h_saturated']:
                sel = np.arange(g.D)
            else:
                r = rng('h_check', key, ARM_ID['H'], ix)
                sel = np.sort(r.choice(g.D, b['n_dyads'], replace=False))
            counts = np.zeros_like(g.counts)
            counts[sel] = recent_counts(g.counts[sel])
            o = parse(serialize(make(g, 'H', b, counts, None)))
            K = g.K[sel]
            oracle = profile(K)
            plug = np.array(plugin(o)); lo, hi = map(np.array, h_bounds(o)); mid = np.array(h_midpoint(o))
            if np.any(oracle < lo - 1e-12) or np.any(oracle > hi + 1e-12):
                raise AssertionError('sample bounds violated')
            x = features(o).reshape(1, -1)
            et = models[(fold, 'pooled')].predict(x)[0]; eo = models[(fold, 'real_only')].predict(x)[0]
            rr = rng('h_check_reservoir', key, ARM_ID['H'], ix)
            rc = reservoir_counts(g.counts[sel], H_CAP, rr)
            if not np.array_equal(rc.sum(1), counts[sel].sum(1)): raise AssertionError('volume differs')
            Jr = (rc > 0).sum(1); Jc = (counts[sel] > 0).sum(1)
            res = profile(Jr)
            truth = pop['truth']
            row = {'graph_id': key, 'stratum': prow['stratum'], 'draw': ix,
                   'M_obs': o['M_obs'], 'volume_ratio': o['M_obs'] / g.B,
                   'at_cap_share': sum(t[3] for t in o['table']) / o['D_obs'],
                   'lost_share': float(np.mean(Jc < K)), 'lost_share_reservoir': float(np.mean(Jr < K)),
                   'covers_truth_2': bool(lo[0] - 1e-12 <= truth[0] <= hi[0] + 1e-12),
                   'covers_truth_all': bool(np.all(lo - 1e-12 <= truth) and np.all(truth <= hi + 1e-12)),
                   'width_2': float(hi[0] - lo[0])}
            for k, i in zip(KS, range(4)):
                row[f'selection_{k}'] = float(oracle[i] - truth[i])
                row[f'history_{k}'] = float(plug[i] - oracle[i])
                row[f'total_{k}'] = float(plug[i] - truth[i])
                row[f'mid_error_{k}'] = float(mid[i] - truth[i])
                row[f'history_reservoir_{k}'] = float(res[i] - oracle[i])
                row[f'total_reservoir_{k}'] = float(res[i] - truth[i])
            for name, pred in (('plugin', plug), ('midpoint', mid), ('median', np.array(medians[fold])),
                               ('extratrees_pooled', et), ('extratrees_real_only', eo),
                               ('plugin_reservoir', res)):
                row[f'AE2_{name}'] = float(abs(pred[0] - truth[0]))
                row[f'ProfileAE_{name}'] = float(np.mean(np.abs(pred - truth)))
            rows.append(row)
        draw_rows += rows
        grow = {'graph_id': key, 'stratum': prow['stratum'], 'draws': n_draws,
                'deterministic': b['h_saturated']}
        for name in ('selection_2', 'history_2', 'total_2', 'mid_error_2', 'history_reservoir_2',
                     'total_reservoir_2', 'volume_ratio', 'at_cap_share', 'lost_share',
                     'lost_share_reservoir', 'width_2', 'covers_truth_2', 'covers_truth_all',
                     'AE2_plugin', 'AE2_midpoint', 'AE2_median', 'AE2_extratrees_pooled',
                     'AE2_extratrees_real_only', 'AE2_plugin_reservoir',
                     'ProfileAE_plugin', 'ProfileAE_midpoint', 'ProfileAE_extratrees_pooled'):
            m, se = mean_se([r[name] for r in rows])
            grow[name] = m; grow[name + '_mcse'] = se
        for name in ('selection_2', 'history_2', 'total_2', 'mid_error_2'):
            grow['abs_' + name] = float(np.mean([abs(r[name]) for r in rows]))
        # Selection: expectation zero exactly; compare with the analytic SRS spread.
        emp_sd = float(np.std([r['selection_2'] for r in rows], ddof=1)) if n_draws > 1 else 0.
        grow['selection_2_z'] = (grow['selection_2'] / grow['selection_2_mcse']
                                 if grow['selection_2_mcse'] > 0 else None)
        grow['selection_sd_empirical'] = emp_sd
        grow['selection_sd_analytic'] = prow['sd_selection_2']
        grow['selection_sd_ratio'] = emp_sd / prow['sd_selection_2'] if prow['sd_selection_2'] > 0 else None
        grow['expected_history_2'] = prow['history_2']
        grow['expected_midpoint_bias_2'] = prow['mid_bias_2']
        for name, ref in (('midpoint', 'plugin'), ('extratrees_pooled', 'plugin'),
                          ('plugin_reservoir', 'plugin'), ('midpoint', 'extratrees_pooled')):
            m, se = mean_se([r[f'AE2_{name}'] - r[f'AE2_{ref}'] for r in rows])
            grow[f'paired_{name}_minus_{ref}'] = m; grow[f'paired_{name}_minus_{ref}_mcse'] = se
        grow['weak_history_rho2'] = prow['weak_history_rho2']
        graph_rows.append(grow)
        print(f'{key}: d={b["n_dyads"]}/{g.D} sat={b["h_saturated"]} hist2={prow["history_2"]:+.4f} '
              f'plugAE2={grow["AE2_plugin"]:.4f} midAE2={grow["AE2_midpoint"]:.4f} '
              f'ET={grow["AE2_extratrees_pooled"]:.4f} res_hist2={prow["history_reservoir_2"]:+.4f}', flush=True)
    strata = []
    for stratum, keys in (('real', REAL_TEST), ('dar_a0', [k for k in SYNTH if k.startswith('dar_a0_')]),
                          ('dar_a08', [k for k in SYNTH if k.startswith('dar_a08_')]),
                          ('ad_memoryless', [k for k in SYNTH if k.startswith('ad_memoryless_')]),
                          ('ad_memory', [k for k in SYNTH if k.startswith('ad_memory_')])):
        sel = [r for r in graph_rows if r['graph_id'] in keys]
        pop = [r for r in pop_rows if r['graph_id'] in keys]
        row = {'stratum': stratum, 'graphs': len(sel)}
        for name in ('AE2_plugin', 'AE2_midpoint', 'AE2_median', 'AE2_extratrees_pooled',
                     'AE2_extratrees_real_only', 'AE2_plugin_reservoir', 'selection_2', 'history_2',
                     'total_2', 'mid_error_2', 'abs_selection_2', 'abs_history_2', 'abs_total_2',
                     'covers_truth_2', 'width_2', 'volume_ratio', 'lost_share', 'lost_share_reservoir',
                     'paired_midpoint_minus_plugin', 'paired_extratrees_pooled_minus_plugin',
                     'paired_plugin_reservoir_minus_plugin'):
            row[name] = float(np.mean([r[name] for r in sel]))
            if name + '_mcse' in sel[0]:
                row[name + '_mcse'] = math.sqrt(sum(r[name + '_mcse'] ** 2 for r in sel)) / len(sel)
            if len(sel) > 1:
                row[name + '_between_source_se'] = float(np.std([r[name] for r in sel], ddof=1) / math.sqrt(len(sel)))
        row['expected_history_2'] = float(np.mean([p['history_2'] for p in pop]))
        row['expected_history_reservoir_2'] = float(np.mean([p['history_reservoir_2'] for p in pop]))
        row['share_J_lt_K'] = float(np.mean([p['share_J_lt_K'] for p in pop]))
        strata.append(row)

    def dump(name, rows):
        with open(out / name, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
            w.writeheader(); w.writerows(rows)
    dump('population.csv', pop_rows); dump('draws.csv', draw_rows)
    dump('graphs.csv', graph_rows); dump('strata.csv', strata)
    summary = {
        'design': 'budget10-hrecent5-20260917', 'draws_per_graph': DRAWS,
        'check_streams': ['h_check', 'h_check_reservoir', 'h_check_reservoir_population'],
        'deterministic_graphs': [p['graph_id'] for p in pop_rows if p['saturated']],
        'unreachable_graphs': [p['graph_id'] for p in pop_rows if p['target_unreachable']],
        'outside_tolerance_graphs': [p['graph_id'] for p in pop_rows if not p['within_tolerance']],
        'weak_history_graphs': [p['graph_id'] for p in pop_rows if p['weak_history_rho2']],
        'weak_history_profile_graphs': [p['graph_id'] for p in pop_rows if p['weak_history_profile']],
        'weak_history_threshold': WEAK_HISTORY,
        'max_abs_selection_z': max((abs(r['selection_2_z']) for r in graph_rows if r['selection_2_z'] is not None), default=None),
        'selection_sd_ratio_range': [min(r['selection_sd_ratio'] for r in graph_rows if r['selection_sd_ratio']),
                                     max(r['selection_sd_ratio'] for r in graph_rows if r['selection_sd_ratio'])],
        'sample_bounds_violations': 0,
        'strata': strata, 'seconds': time.time() - t0}
    write_json(out / 'summary.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'strata'}, indent=1))


if __name__ == '__main__':
    main()
