#!/usr/bin/env python3
"""Descriptive budget and coverage sweep. Changes nothing in the main design.

Authorised by the user on 2026-09-17 after the H revision was specified. It is a
post-hoc description of how the observation budget, and for arm H the event cap,
move coverage and plug-in / fixed-reference errors on the fourteen main graphs.
No LLM call, no ExtraTrees fit, no selection of any design parameter from it.

Part A, arm H only: budget share x event cap. Exact population expectations
(E[plug-in] = rho^J under a fixed-size uniform dyad sample, SRS variances) plus
Monte-Carlo absolute errors and bound coverage from a few draws.
Part B, all four arms: budget share with the design's own mechanisms (cap 5 for H,
walk length recalibrated per budget with 256 paths), plug-in and the fixed simple
reference of each arm (plug-in for R, walk ratio for S, bound midpoint for H,
homogeneous and mixture corrector for B), and three kinds of coverage.
"""
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from main_experiment.common import REAL_TEST, SYNTH, rng, seed, write_json
from main_experiment.data import load_graph
from main_experiment.sampling import (Walk, recent_counts, _panel_for, _dyads_for, _panel_mask)
from main_experiment.observation import make, parse, serialize
from main_experiment.baselines import plugin, corrector, h_bounds, h_midpoint, activity, bisect
from main_experiment import mixtures

KS = (2, 3, 4, 5)
FRACTIONS_H = (0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50)
CAPS = (1, 2, 3, 5, 10, 20, None)
FRACTIONS_ARMS = (0.02, 0.05, 0.10, 0.20, 0.30)
DRAWS_H = 20
DRAWS_ARMS = 10


def ind(x):
    return np.stack([x >= k for k in KS], 1).astype(float)


def capped(counts, cap):
    return counts.copy() if cap is None else recent_counts(counts, cap)


def part_a(g, stratum):
    rows = []
    K = g.K; IK = ind(K); truth = IK.mean(0)
    for cap in CAPS:
        obs = capped(g.counts, cap)
        J = (obs > 0).sum(1)
        C = int(obs.sum())
        if cap is None:
            U = J
        else:
            first = np.argmax(obs > 0, axis=1) + 1
            U = np.where(obs.sum(1) == cap, J + first - 1, J)
        IJ, IU = ind(J), ind(U); mid = (IJ + IU) / 2
        for f in FRACTIONS_H:
            B = f * g.M
            d, exp = _dyads_for(C, B, g.D)
            frac = d / g.D
            var = lambda X: (1 - frac) * np.var(X, axis=0, ddof=1) / d if d < g.D else np.zeros(4)
            ae_p, ae_m, cover, width, sel = [], [], [], [], []
            n = 1 if d == g.D else DRAWS_H
            for i in range(n):
                s = np.arange(g.D) if d == g.D else rng('sweep_h', g.key, f'cap{cap}_f{f}', i + 1).choice(g.D, d, replace=False)
                p = IJ[s].mean(0); m = mid[s].mean(0); lo, hi = IJ[s].mean(0), IU[s].mean(0)
                ae_p.append(abs(p[0] - truth[0])); ae_m.append(abs(m[0] - truth[0]))
                cover.append(lo[0] <= truth[0] <= hi[0]); width.append(hi[0] - lo[0])
                sel.append(IK[s].mean(0)[0] - truth[0])
            rows.append({'graph_id': g.key, 'stratum': stratum, 'cap': 'full' if cap is None else cap,
                         'budget_fraction': f, 'C_cap_share': C / g.M, 'n_dyads': d, 'dyad_coverage': frac,
                         'saturated': d == g.D, 'target_unreachable': C < B,
                         'relative_budget_error': (exp - B) / B,
                         'truth_2': truth[0], 'expected_plugin_bias_2': IJ.mean(0)[0] - truth[0],
                         'expected_plugin_profile_bias': float(np.mean(IJ.mean(0) - truth)),
                         'expected_midpoint_bias_2': mid.mean(0)[0] - truth[0],
                         'sd_plugin_2': math.sqrt(var(IJ)[0]), 'sd_midpoint_2': math.sqrt(var(mid)[0]),
                         'sd_selection_2': math.sqrt(var(IK)[0]),
                         'share_J_lt_K': float(np.mean(J < K)),
                         'MAE2_plugin_mc': float(np.mean(ae_p)), 'MAE2_midpoint_mc': float(np.mean(ae_m)),
                         'coverage_truth_2_mc': float(np.mean(cover)), 'width_2_mc': float(np.mean(width)),
                         'draws': n})
    return rows


def calibrate_L(engine, B, C, seeds):
    bound = 1
    while True:
        total, _, _, _ = engine.run(seeds, bound)
        if total[-1] >= len(seeds) * B or bound == C: break
        bound = min(2 * bound, C)
    target = len(seeds) * B
    if total[-1] < target: return C, False
    lo, hi = 0, bound
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if total[mid] >= target: hi = mid
        else: lo = mid
    return min((lo, hi), key=lambda l: (abs(int(total[l]) - target), l)), True


def coverage(g, counts):
    seen = counts.sum(1) > 0
    return {'dyad_coverage': float(seen.mean()),
            'window_coverage': float((counts > 0).sum() / (g.counts > 0).sum()),
            'event_coverage': float(counts.sum() / g.M)}


def b_candidate(o):
    D = o['D_obs']; S = sum(r[0].count('1') * r[1] for r in o['table']); M = o['M_obs']
    mu = activity(S / D, 5); mean = M / S; p = o['parameter']
    r = 0. if mean <= 1 else bisect(lambda x: 1. if x == 0 else x / -math.expm1(-x), mean, 0., mean)
    fit = mixtures.fit_events(o, mu, r / p)
    if mixtures.is_unreliable(fit.status, fit.flags): return corrector(o), True
    return fit.prediction, False


def part_b(g, stratum, build):
    rows = []
    truth = np.array(g.truth)
    engine = Walk(g, build)
    C_walk = min(100 * g.D, 1_000_000)
    for f in FRACTIONS_ARMS:
        B = int(round(f * g.M))
        if B <= 0: continue
        n_panel, _ = _panel_for(g.M, B, g.N)
        d, _ = _dyads_for(int(np.minimum(g.m, 5).sum()), B, g.D)
        cal = [seed('sweep_walk_calibration', g.key, f'f{f}', i) for i in range(1, 257)]
        L, reached = calibrate_L(engine, B, C_walk, cal)
        params = {'n_panel': n_panel, 'L': int(L), 'n_dyads': d, 'p': f}
        for arm in ('R', 'S', 'H', 'B'):
            n = 1 if (arm == 'H' and d == g.D) else DRAWS_ARMS
            for i in range(1, n + 1):
                re = None
                r = rng('sweep_arms', g.key, f'{arm}_f{f}', i)
                if arm == 'R':
                    counts = g.counts * _panel_mask(g, r, n_panel)[:, None]
                elif arm == 'S':
                    _, _, rr, _ = engine.run([seed('sweep_walk', g.key, f'f{f}', i)], int(L), True)
                    re = rr[0]; counts = g.counts * (re > 0)[:, None]
                elif arm == 'H':
                    sel = np.arange(g.D) if d == g.D else np.sort(r.choice(g.D, d, replace=False))
                    counts = np.zeros_like(g.counts); counts[sel] = recent_counts(g.counts[sel])
                else:
                    keep = r.random(g.M) < f
                    counts = np.bincount(g.pair[keep] * 5 + g.w[keep], minlength=g.D * 5).reshape(-1, 5)
                cov = coverage(g, counts)
                if cov['dyad_coverage'] == 0: continue
                o = parse(serialize(make(g, arm, params, counts, re)))
                preds = {'plugin': plugin(o), 'reference': corrector(o)}
                fallback = None
                if arm == 'B':
                    preds['mixture'], fallback = b_candidate(o)
                for method, p in preds.items():
                    p = np.array(p)
                    rows.append({'graph_id': g.key, 'stratum': stratum, 'arm': arm, 'budget_fraction': f,
                                 'draw': i, 'deterministic': n == 1, 'method': method,
                                 'parameter': params[{'R': 'n_panel', 'S': 'L', 'H': 'n_dyads', 'B': 'p'}[arm]],
                                 'walk_budget_reached': reached if arm == 'S' else None,
                                 **cov, 'AE2': float(abs(p[0] - truth[0])), 'signed_2': float(p[0] - truth[0]),
                                 'ProfileAE': float(np.mean(np.abs(p - truth))),
                                 'mixture_fallback': fallback if method == 'mixture' else None})
    return rows


def aggregate(rows, keys, values):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(r[k] for k in keys), {}).setdefault(r['graph_id'], []).append(r)
    out = []
    for key, per in sorted(groups.items(), key=lambda kv: str(kv[0])):
        row = dict(zip(keys, key)); row['graphs'] = len(per)
        for v in values:
            means = [np.mean([x[v] for x in rs]) for rs in per.values()]
            row[v] = float(np.mean(means))
            if len(means) > 1: row[v + '_between_graph_se'] = float(np.std(means, ddof=1) / math.sqrt(len(means)))
        out.append(row)
    return out


def dump(path, rows):
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='results/main_experiment/hrecent5_20260917')
    ap.add_argument('--out', default='results/budget_sweep_20260917')
    ap.add_argument('--part', choices=['a', 'b', 'both'], default='both')
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out); out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    graphs = [(k, 'real' if k in REAL_TEST else k.rsplit('_r', 1)[0]) for k in [*REAL_TEST, *SYNTH]]
    if a.part in ('a', 'both'):
        rows = []
        for key, stratum in graphs:
            rows += part_a(load_graph(run / 'graphs' / key), stratum)
            print(f'A {key} {time.time()-t0:.0f}s', flush=True)
        dump(out / 'h_budget_cap_grid.csv', rows)
        grouped = [dict(r, stratum_group='real' if r['stratum'] == 'real' else 'synthetic') for r in rows]
        dump(out / 'h_budget_cap_summary.csv',
             aggregate(grouped, ['stratum_group', 'cap', 'budget_fraction'],
                       ['dyad_coverage', 'saturated', 'target_unreachable', 'expected_plugin_bias_2',
                        'expected_midpoint_bias_2', 'sd_plugin_2', 'share_J_lt_K', 'MAE2_plugin_mc',
                        'MAE2_midpoint_mc', 'coverage_truth_2_mc', 'width_2_mc']))
    if a.part in ('b', 'both'):
        rows = []
        for key, stratum in graphs:
            rows += part_b(load_graph(run / 'graphs' / key), stratum, run / 'build')
            print(f'B {key} {time.time()-t0:.0f}s', flush=True)
        dump(out / 'arms_budget_draws.csv', rows)
        grouped = [dict(r, stratum_group='real' if r['stratum'] == 'real' else 'synthetic') for r in rows]
        dump(out / 'arms_budget_summary.csv',
             aggregate(grouped, ['stratum_group', 'arm', 'method', 'budget_fraction'],
                       ['AE2', 'signed_2', 'ProfileAE', 'dyad_coverage', 'window_coverage', 'event_coverage']))
    write_json(out / 'sweep_manifest.json', {
        'status': 'descriptive post-hoc sweep; the main design is not changed by it',
        'fractions_h': FRACTIONS_H, 'caps': [c if c is not None else 'full' for c in CAPS],
        'fractions_arms': FRACTIONS_ARMS, 'draws_h': DRAWS_H, 'draws_arms': DRAWS_ARMS,
        'streams': ['sweep_h', 'sweep_arms', 'sweep_walk', 'sweep_walk_calibration'],
        'walk_calibration': '256 paths, doubling and integer bisection as in the design; no validation walks',
        'seconds': time.time() - t0})


if __name__ == '__main__':
    main()
