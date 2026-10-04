"""Replace marked Markdown tables with values from frozen analysis data."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis_metrics import METHODS, noise_cases, noise_components

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/analysis/data'
FINAL = ROOT/'docs/results/final'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] + ['---:']*(len(headers)-1)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def write_tables(pred, features):
    # Import lazily: analysis_figures imports this module only after drawing.
    from analysis_figures import ARMS, METHODS as LABELS, short
    doc = ROOT/'docs/analysis/ANALYSIS.md'
    text = doc.read_text()
    order = features.sort_values('rho2', ascending=False).index
    truth = json.loads((FINAL/'TRUTH.json').read_text())
    settings = {'copenhagen_bluetooth': 'Phone proximity', 'reality_mining': 'Phone proximity',
                'lkml_reply': 'Email replies', 'nr_radoslaw_email': 'Email', 'snap_email_eu': 'Email',
                'nr_digg_reply': 'Online replies', 'snap_collegemsg': 'Messages', 'snap_mathoverflow': 'Online replies',
                'sp_highschool2013': 'Face-to-face', 'sp_hospital': 'Face-to-face', 'sp_malawi': 'Face-to-face',
                'sp_workplace': 'Face-to-face'}
    fmt = lambda v: f'{v:.1f}'
    percent = lambda v: f'{100*v:.1f} %'
    blocks = {}
    blocks['network_specs'] = table(['Network', 'Interactions', 'Nodes', 'Pairs', 'Events', 'Events/pair'],
        [[short(s), settings[s], *[f'{features.loc[s, c]:,.0f}' for c in ('nodes', 'pairs', 'events')],
          fmt(features.loc[s, 'events_per_pair'])] for s in order])
    blocks['network_profiles'] = table(['Network', 'True ρ₂', 'True ρ₃', 'True ρ₄', 'True ρ₅', 'Effective pairs', 'Early-only pairs'],
        [[short(s), *[percent(v) for v in truth[s]], f'{features.loc[s, "effective_pairs"]:,.0f}',
          percent(features.loc[s, 'only_early_pairs'])] for s in order])

    tokens = pd.read_csv(DATA/'under_the_hood.csv').set_index(['method', 'arm']).median_reasoning_tokens
    errors = pd.read_csv(FINAL/'SUMMARY.csv').query("group == 'real'").set_index(['method', 'arm']).MAE_2*100
    blocks['reasoning'] = table(['Method', 'Sampler', 'Median reasoning tokens', 'Error (pp)'],
        [[LABELS[m], a, f'{tokens[(m, a)]:,.0f}', fmt(errors[(m, a)])]
         for m in ['gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking'] for a in ARMS])

    cases = noise_cases(pred)
    cases.to_csv(DATA/'noise_by_network.csv', index=False, float_format='%.8g')
    components = noise_components(cases)
    components.to_csv(DATA/'noise_components.csv', index=False, float_format='%.8g')
    c = cases[cases.target.eq(2)].set_index(['method', 'source', 'arm'])
    pieces = ['#### MLE and ExtraTrees: all 12 networks\n\n' +
              table(['Method', 'R', 'S', 'H', 'B', 'Mean of R/S/H/B'],
                    [[LABELS[m], *['0.0']*5] for m in ('mle', 'et')])]
    for m in METHODS[2:]:
        rows = []
        for s in order:
            vals = [c.loc[(m, s, a), 'answer_sd'] for a in ARMS]
            cells = [fmt(v) + ('†' if m not in ('mle', 'et') and c.loc[(m, s, a), 'complete_answer_samples'] < 3 else '')
                     for a, v in zip(ARMS, vals)]
            rows.append([short(s), percent(features.loc[s, 'rho2']), *cells, fmt(np.mean(vals))])
        pieces.append(f'#### {LABELS[m]}\n\n' + table(['Network', 'True ρ₂', 'R', 'S', 'H', 'B', 'Mean of R/S/H/B'], rows))
    blocks['answer_spread'] = '\n\n'.join(pieces)
    pieces = []
    for m in METHODS:
        rows = []
        for s in order:
            vals = [c.loc[(m, s, a), 'between_sd'] for a in ARMS]
            rows.append([short(s), percent(features.loc[s, 'rho2']), *map(fmt, vals), fmt(np.mean(vals))])
        pieces.append(f'#### {LABELS[m]}\n\n' + table(['Network', 'True ρ₂', 'R', 'S', 'H', 'B', 'Mean of R/S/H/B'], rows))
    blocks['sample_spread'] = '\n\n'.join(pieces)

    comp = components[components.target.eq(2)].set_index(['method', 'arm'])
    blocks['noise_components'] = table(['Method', 'Sampler', 'Networks', 'Between samples', 'Answer noise', 'Adjusted sample noise'],
        [[LABELS[m], a, f'{comp.loc[(m, a), "networks"]:.0f}/12',
          *[fmt(comp.loc[(m, a), col]) for col in ('between_sd', 'answer_sd')],
          fmt(comp.loc[(m, a), 'sample_sd_estimate']) + ('†' if comp.loc[(m, a), 'sample_variance_estimate'] < 0 else '')]
         for m in METHODS for a in ARMS])

    blocks['noise_profile'] = table(['Method', 'Target', 'Between samples', 'Same sample'],
        [[LABELS[m], f'ρ{k}', fmt(g.between_sd.mean()), fmt(g.answer_sd.mean())]
         for m in METHODS for k in range(2, 6)
         for g in [cases[cases.method.eq(m) & cases.target.eq(k)].groupby('arm')[['between_sd', 'answer_sd']].median()]])
    training = pd.read_csv(FINAL/'VARIABILITY_TRAINING.csv').query("group == 'real'").set_index('arm')
    blocks['training'] = table(['ExtraTrees: new training fit', 'R', 'S', 'H', 'B'],
                              [['Spread (SD, pp)', *[fmt(training.loc[a, 'median_observation_SD_rho2']*100) for a in ARMS]]])

    for key, value in blocks.items():
        pattern = re.compile(rf'(<!-- table:{key} -->\n).*?(<!-- /table:{key} -->)', re.S)
        text, n = pattern.subn(lambda m: m[1] + value + '\n' + m[2], text)
        if n != 1: raise ValueError(f'expected one Markdown placeholder for {key}, found {n}')
    doc.write_text(text)


if __name__ == '__main__':
    write_tables(pd.read_csv(FINAL/'PREDICTIONS.csv'), pd.read_csv(DATA/'network_features.csv', index_col=0))
