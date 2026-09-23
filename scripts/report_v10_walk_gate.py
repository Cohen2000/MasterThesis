#!/usr/bin/env python3
"""Render amended v10 walk evidence from the cluster CSV and confirmation JSON."""
import csv
import json
from pathlib import Path

out = Path(__file__).resolve().parents[1] / 'docs/results/panel888_v10_walk_gate_20260923'
csv_path = out / 'walk_gate.csv'
csv_path.write_text(csv_path.read_text())  # normalise CRLF for the committed copy
rows = list(csv.DictReader(csv_path.open()))
if len(rows) != 24: raise ValueError('incomplete 24-graph audit')
applicable = [r for r in rows if r['gate_applicable'] == 'True']
failed = [r for r in applicable if r['gate_pass'] != 'True']
fmt = lambda x: f'{float(x):+.4f}'
lines = [
    '# Amended v10 interaction-walk gate',
    '',
    'Cluster job 7143961; 1,000 independent walks per graph at the calibrated 10% length.',
    'Post-hoc amendment: the original absolute-bias threshold ignored Monte Carlo error and the',
    'first-order bias of a finite-sample Hájek ratio. The walk and plain S estimator were unchanged.',
    '',
    'For each real or surrogate source with absolute stationary interaction-walk shift above 0.05,',
    'the amended gate requires absolute S bias at most 10% of absolute plugin bias and S RMSE',
    'at most half the plugin RMSE. Failures remain in the study and are flagged as',
    '**not correctable at this budget**.',
    '',
    '| Source | Shift | Ratio ESS | Plugin bias | S bias ± MCSE | Predicted first-order bias | S / plugin bias | S / plugin RMSE | Gate |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---|',
]
for r in applicable:
    rmse_ratio = float(r['design_S_rho2_rmse']) / float(r['plugin_rho2_rmse'])
    share = float(r['design_bias_share_of_plugin_bias'])
    lines.append(f"| {r['graph_id']} | {fmt(r['stationary_shift_rho2'])} | "
                 f"{float(r['ratio_ess_mean']):.1f} | "
                 f"{fmt(r['plugin_rho2_bias'])} | {fmt(r['design_S_rho2_bias'])} ± "
                 f"{float(r['design_S_rho2_bias_mcse']):.4f} | "
                 f"{fmt(r['first_order_ratio_bias_mean'])} | {share:.3f} | {rmse_ratio:.3f} | "
                 f"{'pass' if r['gate_pass'] == 'True' else 'not correctable at this budget'} |")
lines += ['', f'{len(failed)} of {len(applicable)} applicable sources failed the amended gate.', '',
          '## Confirmation runs', '',
          'Cluster array job 7143568; each condition used 1,000 walks at the original calibrated L.',
          'The strength-start condition samples the initial vertex proportional to event strength;',
          'the 4L condition retains the uniform start.', '',
          '| Source | Uniform L bias ± MCSE | Strength-start L bias ± MCSE | Uniform 4L bias ± MCSE |',
          '|---|---:|---:|---:|']
for key in ('sp_hospital', 'sp_hospital__pwt', 'sp_highschool2013__pwt'):
    c = json.loads((out / 'confirmation' / f'{key}.json').read_text())
    a, b = c['conditions']
    lines.append(f"| {key} | {fmt(c['uniform_L_bias'])} ± {c['uniform_L_mcse']:.4f} | "
                 f"{fmt(a['design_S_rho2_bias'])} ± {a['design_S_rho2_bias_mcse']:.4f} | "
                 f"{fmt(b['design_S_rho2_bias'])} ± {b['design_S_rho2_bias_mcse']:.4f} |")
lines += ['', 'The CSV and JSON give all 24 graph rows, stationary rho_2..rho_5 targets,',
          'bias, SD and RMSE for plugin, S and S_obs, weight ESS, revisit and inclusion diagnostics.', '']
(out / 'WALK_GATE.md').write_text('\n'.join(lines))
