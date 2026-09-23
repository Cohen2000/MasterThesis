#!/usr/bin/env python3
"""Render the fixed v10 walk gate directly from the cluster audit CSV."""
import csv
from pathlib import Path

out = Path(__file__).resolve().parents[1] / 'docs/results/panel888_v10_walk_gate_20260923'
# The shared CSV writer emits CRLF; normalise the committed copy for diff hygiene.
csv_path = out / 'walk_gate.csv'
csv_path.write_text(csv_path.read_text())
rows = list(csv.DictReader(csv_path.open()))
if len(rows) != 24:
    raise ValueError('incomplete 24-graph audit')
applicable = [r for r in rows if r['gate_applicable'] == 'True']
failed = [r for r in applicable if r['gate_pass'] != 'True']
fmt = lambda x: f'{float(x):+.4f}'
lines = [
    '# v10 interaction-walk gate',
    '',
    'Cluster job 7143357; 1,000 independent walks per graph at its recalibrated 10% length.',
    'The 24-graph audit failed the prespecified gate. No v10 offline study or Qwen production was submitted.',
    '',
    'Gate applies to each real or surrogate source with absolute stationary interaction-walk shift above 0.05:',
    'absolute mean S design bias must be at most 0.01 and S design RMSE at most half the plugin RMSE.',
    '',
    '| Source | Stationary shift | Plugin bias | Plugin RMSE | S bias | S RMSE | S RMSE / plugin | Gate |',
    '|---|---:|---:|---:|---:|---:|---:|---|',
]
for r in applicable:
    ratio = float(r['design_S_rho2_rmse']) / float(r['plugin_rho2_rmse'])
    lines.append(f"| {r['graph_id']} | {fmt(r['stationary_shift_rho2'])} | "
                 f"{fmt(r['plugin_rho2_bias'])} | {float(r['plugin_rho2_rmse']):.4f} | "
                 f"{fmt(r['design_S_rho2_bias'])} | {float(r['design_S_rho2_rmse']):.4f} | "
                 f"{ratio:.3f} | {'pass' if r['gate_pass'] == 'True' else 'FAIL'} |")
lines += ['', f'{len(failed)} of {len(applicable)} applicable sources failed.',
          'The CSV and JSON contain all 24 graph rows, rho_2..rho_5 stationary targets for uniform, degree, and interaction weights,',
          'bias, SD, RMSE for plugin, S, and S_obs, and revisit, effective sample size, and inclusion diagnostics.', '']
(out / 'WALK_GATE.md').write_text('\n'.join(lines))
