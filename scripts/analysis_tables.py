"""Replace marked Markdown tables with values from frozen analysis data."""
import re
from pathlib import Path

import pandas as pd

from analysis_metrics import noise_cases, noise_components, variance_shares

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/analysis/data'
FINAL = ROOT/'docs/results/final'


def table(headers, rows, left=1):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---']*left + ['---:']*(len(headers)-left)) + ' |']
                     + ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def write_tables(pred, features):
    # Import lazily: analysis_figures imports this module only after drawing.
    from analysis_figures import ARMS, METHODS as LABELS, short
    doc = ROOT/'docs/analysis/ANALYSIS.md'
    text = doc.read_text()
    order = features.sort_values('rho2', ascending=False).index
    settings = {'copenhagen_bluetooth': 'Phone proximity', 'reality_mining': 'Phone proximity',
                'lkml_reply': 'Email replies', 'nr_radoslaw_email': 'Email', 'snap_email_eu': 'Email',
                'nr_digg_reply': 'Online replies', 'snap_collegemsg': 'Messages', 'snap_mathoverflow': 'Online replies',
                'sp_highschool2013': 'Face-to-face', 'sp_hospital': 'Face-to-face', 'sp_malawi': 'Face-to-face',
                'sp_workplace': 'Face-to-face'}
    fmt = lambda v: f'{v:.1f}'
    percent = lambda v: f'{100*v:.1f} %'
    blocks = {}
    blocks['network_specs'] = table(['Network', 'Interactions', 'Nodes', 'Pairs', 'Events', 'Events/pair', 'True ρ₂'],
        [[short(s), settings[s], *[f'{features.loc[s, c]:,.0f}' for c in ('nodes', 'pairs', 'events')],
          fmt(features.loc[s, 'events_per_pair']), percent(features.loc[s, 'rho2'])] for s in order], left=2)

    tokens = pd.read_csv(DATA/'under_the_hood.csv').set_index(['method', 'arm']).median_reasoning_tokens
    errors = pd.read_csv(FINAL/'SUMMARY.csv').query("group == 'real'").set_index(['method', 'arm']).MAE_2*100
    # One row per model, with the four samplers nested under one shared header.
    rows = []
    for m in ['gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking']:
        cells = [f'{int(tokens[(m, a)] + .5):,} / {fmt(errors[(m, a)])}' for a in ARMS]
        rows.append('<tr><th scope="row">' + LABELS[m] + '</th>' +
                    ''.join('<td>' + cell + '</td>' for cell in cells) + '</tr>')
    blocks['reasoning'] = '\n'.join([
        '<table>', '<thead>',
        '<tr><th rowspan="2" scope="col">LLM</th><th colspan="4" scope="colgroup">Median reasoning tokens / error (pp)</th></tr>',
        '<tr>' + ''.join(f'<th scope="col">{a}</th>' for a in ARMS) + '</tr>',
        '</thead>', '<tbody>', *rows, '</tbody>', '</table>'])

    shares = variance_shares(pd.read_csv(FINAL/'PER_SOURCE.csv'), ['mle', 'et', 'gpt_6_sol', 'gpt_6_sol_tools', 'deepseek_flash', 'qwen_thinking'])
    blocks['network_or_method'] = table(['Sampler', 'Network', 'Method', 'Both together'],
        [[f'**{ARMS[a]}**', *[f'{100*shares.loc[a, part]:.0f} %' for part in ('network', 'method', 'both')]] for a in ARMS])

    # What makes a network hard: rank correlations of the typical error with a network property, from relations.csv.
    rel = pd.read_csv(DATA/'relations.csv').fillna('').set_index(['relation', 'group', 'method', 'arm']).value
    def signed(relation, group, arm):
        """A rank correlation with its sign; '–' where the property hardly differs between the networks."""
        if (relation, group, '', arm) not in rel.index: return '–'
        value = rel[(relation, group, '', arm)]
        return '0.00' if abs(value) < .005 else f'{value:+.2f}'.replace('-', '−')
    own = {'R': 'nodes', 'S': 'events per pair', 'H': 'early-late mismatch', 'B': 'true rho_2'}
    shown = {'nodes': 'Nodes', 'events per pair': 'Events per pair', 'early-late mismatch': 'Early–late mismatch', 'true rho_2': 'True ρ₂'}
    sets = ('real', 'time-shuffled twin', 'synthetic', 'all 32')
    blocks['hard_properties'] = table(['Property', 'Sampler', 'Real', 'Twins', 'Synthetic', 'All 32'],
        [['Effective pairs', a, *[signed('typical error vs effective pairs', g, a) for g in sets]] for a in ARMS]
        + [[f'**{shown[own[a]]}**', f'**{a}**', *[signed(f'typical error vs {own[a]}', g, a) for g in sets]] for a in ARMS], left=2)
    at_level = {'R': lambda k: 'nodes', 'S': lambda k: 'events per pair', 'H': lambda k: f'early-late mismatch of rho_{k}',
                'B': lambda k: f'true rho_{k}'}
    words = {'R': 'nodes', 'S': 'events per pair', 'H': 'early–late mismatch of that level', 'B': 'true share of that level'}
    blocks['levels'] = table(['Sampler', 'Property', 'ρ₂', 'ρ₃', 'ρ₄', 'ρ₅'],
        [[f'**{a}**', words[a], *[' / '.join(signed(f'typical error of rho_{k} vs {at_level[a](k)}', g, a) for g in sets[:2])
                                 for k in range(2, 6)]] for a in ARMS], left=2)

    cases = noise_cases(pred)
    cases.to_csv(DATA/'noise_by_network.csv', index=False, float_format='%.8g')
    components = noise_components(cases)
    components.to_csv(DATA/'noise_components.csv', index=False, float_format='%.8g')

    for key, value in blocks.items():
        pattern = re.compile(rf'(<!-- table:{key} -->\n).*?(<!-- /table:{key} -->)', re.S)
        text, n = pattern.subn(lambda m: m[1] + value + '\n' + m[2], text)
        if n != 1: raise ValueError(f'expected one Markdown placeholder for {key}, found {n}')
    doc.write_text(text)


if __name__ == '__main__':
    write_tables(pd.read_csv(FINAL/'PREDICTIONS.csv'), pd.read_csv(DATA/'network_features.csv', index_col=0))
